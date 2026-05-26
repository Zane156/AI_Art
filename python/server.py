"""
AITermUI Backend Server v2
Features:
- Multi-model inference (PixArt-Sigma, SD3-Medium, SD3.5-Large)
- LoRA training with progress tracking
- Agent with tool-calling (DeepSeek, OpenAI, SiliconFlow, Zhipu, Moonshot, Bailian)
- Auto-labeling (WD14 / CLIP)
- System diagnostics & error tracking
"""
import os, sys, json, time, threading, urllib.request, urllib.error, re, traceback, hashlib, shutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from collections import deque

PORT = 17860
OUTPUT_DIR = "outputs"
MAX_ERROR_LOG = 200

# ---- Paths ----
_THIS_FILE = os.path.abspath(__file__)
_BACKEND_DIR = os.path.dirname(_THIS_FILE)
BUILD_DIR = os.path.dirname(_BACKEND_DIR)
# Try to find the real project root (account for CMake copy to build/)
if os.path.basename(BUILD_DIR) == "build":
    PROJECT_ROOT = os.path.dirname(BUILD_DIR)
else:
    PROJECT_ROOT = BUILD_DIR

# ---- Error Log ----
_error_log = deque(maxlen=MAX_ERROR_LOG)
_error_log_lock = threading.Lock()

def log_error(source, msg, exc=None):
    entry = {
        "time": time.time(),
        "source": source,
        "message": str(msg)[:500],
    }
    if exc:
        entry["traceback"] = traceback.format_exc()[:2000]
    with _error_log_lock:
        _error_log.append(entry)
    print(f"[ERROR] {source}: {msg}")

# ---- API Provider Registry ----
API_PROVIDERS = {
    "deepseek":    {"name": "DeepSeek",       "endpoint": "https://api.deepseek.com/v1/chat/completions",              "model": "deepseek-chat"},
    "openai":      {"name": "OpenAI",         "endpoint": "https://api.openai.com/v1/chat/completions",                 "model": "gpt-4o"},
    "siliconflow": {"name": "SiliconFlow",    "endpoint": "https://api.siliconflow.cn/v1/chat/completions",             "model": "deepseek-ai/DeepSeek-V3"},
    "zhipu":       {"name": "ZhipuAI",        "endpoint": "https://open.bigmodel.cn/api/paas/v4/chat/completions",      "model": "glm-4-flash"},
    "moonshot":    {"name": "Moonshot",       "endpoint": "https://api.moonshot.cn/v1/chat/completions",                "model": "moonshot-v1-8k"},
    "bailian":     {"name": "Bailian",        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "model": "qwen-plus"},
}

# ---- Model Registry ----
MODEL_CONFIGS = {
    "PixArt-Sigma":  {"aliases": ["PixArt-Sigma", "pixart", "pixart-sigma"],
                      "path": os.path.join(PROJECT_ROOT, "models", "PixArt-Sigma"),
                      "vram": 7,  "max_size": 1024,  "expected_size_gb": 7,
                      "download_source": "hf-mirror",
                      "hf_repo": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                      "pipeline_class": "PixArtSigmaPipeline"},
    "SD3-Medium":    {"aliases": ["SD3-Medium", "sd3", "sd3-medium"],
                      "path": os.path.join(PROJECT_ROOT, "models", "SD3-Medium"),
                      "vram": 11, "max_size": 1024,  "expected_size_gb": 11,
                      "download_source": "modelscope",
                      "ms_repo": "AI-ModelScope/stable-diffusion-3-medium-diffusers",
                      "hf_repo": "stabilityai/stable-diffusion-3-medium-diffusers",
                      "pipeline_class": "StableDiffusion3Pipeline"},
    "SD3.5-Large":   {"aliases": ["SD3.5-Large", "sd35", "sd3.5-large"],
                      "path": os.path.join(PROJECT_ROOT, "models", "SD3.5-Large"),
                      "vram": 22, "max_size": 1024,  "expected_size_gb": 22,
                      "download_source": "hf-mirror",
                      "_vpn_required": True,
                      "hf_repo": "stabilityai/stable-diffusion-3.5-large-diffusers",
                      "pipeline_class": "StableDiffusion3Pipeline"},
}

def resolve_model(name):
    """Resolve a model name (supports aliases)."""
    for key, cfg in MODEL_CONFIGS.items():
        if name in cfg["aliases"] or name == key:
            return key, cfg
    return None, None

# ---- Task Tracking ----
_tasks = {}
_tasks_lock = threading.Lock()
_task_counter = 0

# ---- Pipeline Cache ----
_pipe_cache = {}
_pipe_lock = threading.Lock()

# ---- Agent Tool Definitions ----
AGENT_TOOLS = [
    {
        "name": "list_models",
        "description": "列出所有AI模型及其状态（已下载/未下载/已加载），包括显存需求",
        "parameters": {}
    },
    {
        "name": "get_system_info",
        "description": "获取系统信息：GPU型号、显存总量/可用量、Python版本、PyTorch版本、CUDA状态、当前加载的模型",
        "parameters": {}
    },
    {
        "name": "download_model",
        "description": "从HuggingFace下载模型（自动使用hf-mirror镜像加速）",
        "parameters": {
            "model_name": {"type": "string", "description": "模型名称: PixArt-Sigma / SD3-Medium / SD3.5-Large"},
            "repo_id": {"type": "string", "description": "HuggingFace仓库ID，可选。留空则使用默认仓库"},
            "target_dir": {"type": "string", "description": "目标目录，可选。留空则使用默认路径"}
        }
    },
    {
        "name": "generate_image",
        "description": "使用AI模型生成图像",
        "parameters": {
            "model": {"type": "string", "description": "模型名称"},
            "prompt": {"type": "string", "description": "提示词 (英文效果更好)"},
            "steps": {"type": "integer", "description": "推理步数 (默认25)"},
            "guidance": {"type": "number", "description": "引导系数 (默认4.5)"},
            "width": {"type": "integer", "description": "宽度 (默认1024)"},
            "height": {"type": "integer", "description": "高度 (默认1024)"},
            "seed": {"type": "integer", "description": "随机种子(-1=随机)"}
        }
    },
    {
        "name": "start_training",
        "description": "开始LoRA训练。必须先准备好图片数据集",
        "parameters": {
            "base_model": {"type": "string", "description": "底膜名称"},
            "data_dir": {"type": "string", "description": "训练图片目录 (含图片和.txt标注文件)"},
            "output_dir": {"type": "string", "description": "LoRA输出目录"},
            "steps": {"type": "integer", "description": "训练步数 (默认500)"},
            "learning_rate": {"type": "number", "description": "学习率 (默认1e-4)"},
            "rank": {"type": "integer", "description": "LoRA秩 (默认16)"},
            "batch_size": {"type": "integer", "description": "批次大小 (默认1)"}
        }
    },
    {
        "name": "auto_label",
        "description": "为训练图片自动生成标注文件(.txt)。扫描目录中所有图片，为缺少标注的图片自动生成caption",
        "parameters": {
            "data_dir": {"type": "string", "description": "图片目录路径"},
            "method": {"type": "string", "description": "标注方法: 'filename'(使用文件名) / 'simple'(简单通用标签)，默认'simple'"}
        }
    },
    {
        "name": "scan_dataset",
        "description": "扫描训练数据集，统计图片数量、标注状态、尺寸分布",
        "parameters": {
            "data_dir": {"type": "string", "description": "数据目录路径"}
        }
    },
    {
        "name": "check_task",
        "description": "查询任务进度（生成或训练）",
        "parameters": {
            "task_id": {"type": "string", "description": "任务ID"}
        }
    },
    {
        "name": "diagnose",
        "description": "运行全面诊断，检查环境、模型、依赖等所有组件状态，发现并报告问题",
        "parameters": {}
    },
    {
        "name": "get_error_log",
        "description": "获取最近的错误日志，用于诊断问题",
        "parameters": {
            "count": {"type": "integer", "description": "获取最近的N条错误 (默认20)"}
        }
    },
    {
        "name": "cancel_task",
        "description": "取消正在运行的任务",
        "parameters": {
            "task_id": {"type": "string", "description": "任务ID"}
        }
    },
]

# ====================================================================
# MODEL LOADING
# ====================================================================

def detect_model_format(model_path):
    """Detect if model is in diffusers format or single-file checkpoint."""
    if not os.path.isdir(model_path):
        return None
    # Check for diffusers format indicators
    has_model_index = os.path.isfile(os.path.join(model_path, "model_index.json"))
    has_config = os.path.isfile(os.path.join(model_path, "config.json"))
    # Check for single-file safetensors
    safetensors = [f for f in os.listdir(model_path) if f.endswith('.safetensors') and not f.endswith('.incomplete')]
    if has_model_index and has_config:
        return "diffusers"
    elif safetensors:
        return "single_file"
    elif os.path.isdir(os.path.join(model_path, "transformer")) or \
         os.path.isdir(os.path.join(model_path, "unet")):
        return "diffusers"  # Subfolder-based diffusers format
    # Empty or partial download
    visible = [f for f in os.listdir(model_path) if not f.startswith('.')]
    if not visible:
        return "empty"
    return "incomplete"

def check_model_integrity(model_path, model_key=""):
    """Check model completeness — not just folder existence."""
    if not os.path.isdir(model_path):
        return {"status": "missing", "message": "模型未下载"}

    # Check for stale download temp files
    try:
        items = os.listdir(model_path)
    except:
        return {"status": "incomplete", "message": "无法读取模型目录"}

    temp_files = [f for f in items if f.endswith('.download')]
    if temp_files:
        # Clean up temp files for a clean retry
        for tf in temp_files:
            try: os.remove(os.path.join(model_path, tf))
            except: pass
        return {"status": "incomplete", "message": "下载未完成 (可重新下载)"}

    fmt = detect_model_format(model_path)
    if fmt == "empty":
        return {"status": "incomplete", "message": "模型目录为空"}

    if fmt is None:
        return {"status": "missing", "message": "模型目录不存在"}

    if fmt == "incomplete":
        return {"status": "incomplete", "message": "模型文件不完整，请重新下载"}

    # For diffusers format: check key files exist
    if fmt == "diffusers":
        required = ["model_index.json"]
        missing = [f for f in required if not os.path.isfile(os.path.join(model_path, f))]
        if missing:
            return {"status": "incomplete",
                    "message": f"缺少关键文件: {', '.join(missing)}"}

        # Check subdirectories have content
        for sub in ["transformer", "unet", "text_encoder", "vae"]:
            sub_path = os.path.join(model_path, sub)
            if os.path.isdir(sub_path):
                sub_files = [f for f in os.listdir(sub_path) if not f.startswith('.')]
                if not sub_files:
                    return {"status": "incomplete",
                            "message": f"子目录 {sub} 为空"}

        # Total size sanity check (expect at least 50MB for any model)
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for f in files:
                if not f.endswith('.download'):
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except:
                        pass
        if total_size < 50 * 1024 * 1024:
            return {"status": "incomplete",
                    "message": f"模型太小 ({total_size/1e6:.0f}MB)，可能未完整下载"}
        if total_size > 0:
            return {"status": "complete",
                    "message": f"模型就绪 ({total_size/1e9:.1f}GB)"}

    # For single_file format
    if fmt == "single_file":
        safetensors = [f for f in items if f.endswith('.safetensors') and not f.endswith('.incomplete')]
        if not safetensors:
            return {"status": "incomplete", "message": "模型文件不完整"}
        total_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in safetensors)
        return {"status": "complete",
                "message": f"模型就绪 ({total_size/1e9:.1f}GB)"}

    # Check for download leftovers: .incomplete files
    incomplete = [f for f in items if f.endswith('.incomplete')]
    if incomplete:
        return {"status": "incomplete", "message": "下载未完成 (可重新下载)"}

    return {"status": "complete", "message": "模型就绪"}

def get_pipeline(model_name):
    """Load or retrieve cached diffusers pipeline."""
    key_name, info = resolve_model(model_name)
    if not info:
        return None, f"未知模型: {model_name}"
    model_path = info["path"]
    if not os.path.isdir(model_path):
        return None, f"模型未下载: {model_path} 不存在"

    fmt = detect_model_format(model_path)
    if fmt == "unknown":
        return None, f"模型目录格式无法识别: {model_path}"
    if fmt in ("incomplete", "empty"):
        return None, f"模型未完全下载，请点击 [一键下载模型] 下载: {model_path}"

    with _pipe_lock:
        if key_name in _pipe_cache:
            return _pipe_cache[key_name], None

        # Free old model's GPU memory before loading new one
        if _pipe_cache:
            old_key = list(_pipe_cache.keys())[0]
            old_pipe = _pipe_cache.pop(old_key)
            del old_pipe
            _pipe_cache.clear()
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

        try:
            import torch
            torch.backends.cudnn.benchmark = True

            pipe_cls_name = info["pipeline_class"]
            print(f"[Loading] {key_name} format={fmt} from {model_path} ...")

            if fmt == "single_file":
                # Load from single safetensors file
                safetensors = sorted(
                    [f for f in os.listdir(model_path) if f.endswith('.safetensors')],
                    key=lambda x: os.path.getsize(os.path.join(model_path, x))
                )
                if not safetensors:
                    return None, "单文件模型目录中没有 .safetensors 文件"
                ckpt_path = os.path.join(model_path, safetensors[0])
                print(f"[Loading] Single file: {ckpt_path} ({os.path.getsize(ckpt_path)/1e9:.1f} GB)")

                if pipe_cls_name == "StableDiffusion3Pipeline":
                    from diffusers import StableDiffusion3Pipeline
                    pipe = StableDiffusion3Pipeline.from_single_file(
                        ckpt_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
                elif pipe_cls_name == "PixArtSigmaPipeline":
                    from diffusers import PixArtSigmaPipeline
                    pipe = PixArtSigmaPipeline.from_single_file(
                        ckpt_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
                else:
                    return None, f"单文件加载不支持的 pipeline: {pipe_cls_name}"
            else:
                # Load from diffusers format
                if pipe_cls_name == "PixArtSigmaPipeline":
                    from diffusers import PixArtSigmaPipeline
                    pipe = PixArtSigmaPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
                elif pipe_cls_name == "StableDiffusion3Pipeline":
                    from diffusers import StableDiffusion3Pipeline
                    pipe = StableDiffusion3Pipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                    )
                else:
                    return None, f"不支持的 pipeline: {pipe_cls_name}"

            pipe = pipe.to("cuda")
            # Enable CPU offload only for 8GB cards
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
            if vram <= 10:
                try:
                    pipe.enable_model_cpu_offload()
                    print(f"[Loaded] {key_name} with CPU offload (VRAM={vram}GB)")
                except Exception:
                    print(f"[Loaded] {key_name} without offload")
            else:
                print(f"[Loaded] {key_name} OK (VRAM={vram}GB)")

            _pipe_cache.clear()
            _pipe_cache[key_name] = pipe
            return pipe, None

        except Exception as e:
            log_error("model_load", f"{key_name}: {e}", e)
            return None, f"加载模型失败: {str(e)}"

# ====================================================================
# INFERENCE
# ====================================================================

def run_inference(pipe, prompt, steps, guidance, seed, width, height, output_dir, progress_cb=None):
    """Run image generation with progress callback."""
    import torch
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
    actual_seed = seed if seed >= 0 else torch.Generator("cuda").seed()

    print(f"[Generating] '{prompt[:80]}' steps={steps} {width}x{height} seed={actual_seed}")
    t0 = time.time()

    # Build call kwargs (only standard parameters, no callback yet)
    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
    }
    if seed >= 0:
        call_kwargs["generator"] = generator

    # Try inference with progressive fallback for callback APIs
    result = None
    if progress_cb:
        # Strategy 1: callback_on_step_end (SD3 / newer diffusers)
        try:
            result = pipe(**call_kwargs, callback_on_step_end=progress_cb)
        except TypeError:
            # Strategy 2: callback+callback_steps (SD1/2 / older API)
            try:
                result = pipe(**call_kwargs, callback=progress_cb, callback_steps=1)
            except TypeError:
                # Strategy 3: no callback at all
                result = pipe(**call_kwargs)
    else:
        result = pipe(**call_kwargs)

    elapsed = time.time() - t0

    os.makedirs(output_dir, exist_ok=True)
    fname = f"gen_{int(time.time())}_{width}x{height}.png"
    out_path = os.path.join(output_dir, fname)
    result.images[0].save(out_path)

    info = f"{width}x{height} | {steps}步 | {elapsed:.1f}秒 | seed={actual_seed}"
    print(f"[Done] {out_path} ({info})")
    return out_path, info

# ====================================================================
# LORA TRAINING
# ====================================================================

class ImageDataset:
    """Simple image-text dataset for LoRA training."""
    def __init__(self, data_dir, size=512):
        import torch
        from torch.utils.data import Dataset
        from PIL import Image
        self.data_dir = data_dir
        self.size = size
        self.items = []
        for f in sorted(os.listdir(data_dir)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                img_path = os.path.join(data_dir, f)
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                caption = ""
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as tf:
                        caption = tf.read().strip()
                if not caption:
                    caption = os.path.splitext(f)[0].replace('_', ' ')
                self.items.append((img_path, caption))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        from PIL import Image
        import torch
        from torchvision import transforms
        img_path, caption = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return {"image": transform(img), "caption": caption}

def run_lora_training(task_id, base_model_name, data_dir, output_dir,
                       steps=500, lr=1e-4, rank=16, batch_size=1, size=512):
    """Run LoRA training with progress tracking."""
    try:
        import torch
        from torch.utils.data import DataLoader

        _tasks[task_id]["phase"] = "loading_model"
        _tasks[task_id]["message"] = "正在加载底膜..."

        pipe, err = get_pipeline(base_model_name)
        if err:
            _tasks[task_id]["result"] = {"success": False, "error_msg": err}
            _tasks[task_id]["done"] = True
            return

        key_name, _ = resolve_model(base_model_name)
        if key_name == "SD3.5-Large" and batch_size > 1:
            batch_size = 1  # Force batch=1 for large models
            _tasks[task_id]["message"] = "SD3.5-Large 强制 batch_size=1"

        _tasks[task_id]["phase"] = "preparing_data"
        _tasks[task_id]["message"] = "正在准备数据集..."

        dataset = ImageDataset(data_dir, size=size)
        if len(dataset) < 5:
            _tasks[task_id]["result"] = {
                "success": False,
                "error_msg": f"数据集图片太少 ({len(dataset)}张)，至少需要5张"
            }
            _tasks[task_id]["done"] = True
            return

        # Print dataset info
        captions = [item["caption"] for i in range(min(5, len(dataset)))
                    for item in [dataset[i]]]
        sample_info = f"数据集: {len(dataset)}张图片, {size}x{size}\n示例标注:\n" + \
                      "\n".join(f"  - {c[:80]}" for c in captions)
        print(f"[Training] {sample_info}")

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare LoRA
        _tasks[task_id]["phase"] = "preparing_lora"
        _tasks[task_id]["message"] = "正在配置 LoRA..."

        from peft import LoraConfig, get_peft_model, TaskType

        # Target the UNet (or transformer for PixArt)
        if hasattr(pipe, 'unet'):
            target = pipe.unet
        elif hasattr(pipe, 'transformer'):
            target = pipe.transformer
        else:
            _tasks[task_id]["result"] = {"success": False, "error_msg": "无法找到 UNet/Transformer"}
            _tasks[task_id]["done"] = True
            return

        # Find linear layers to target
        target_modules = []
        for name, mod in target.named_modules():
            if isinstance(mod, torch.nn.Linear):
                # Get the leaf module name
                leaf = name.split('.')[-1]
                if leaf not in target_modules:
                    target_modules.append(leaf)

        if not target_modules:
            target_modules = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj"]
        target_modules = target_modules[:8]  # Limit to avoid explosion

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        try:
            target = get_peft_model(target, lora_config)
        except ValueError as e:
            _tasks[task_id]["result"] = {
                "success": False,
                "error_msg": f"LoRA配置失败 (target_modules={target_modules}): {e}"
            }
            _tasks[task_id]["done"] = True
            return

        target.train()
        trainable = sum(p.numel() for p in target.parameters() if p.requires_grad)
        total = sum(p.numel() for p in target.parameters())
        print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/max(total,1):.1f}%)")

        # Optimizer
        optimizer = torch.optim.AdamW(target.parameters(), lr=lr, weight_decay=0.01)

        # Text encoder for encoding captions
        if hasattr(pipe, 'tokenizer'):
            tokenizer = pipe.tokenizer
        else:
            _tasks[task_id]["result"] = {"success": False, "error_msg": "无法找到 tokenizer"}
            _tasks[task_id]["done"] = True
            return

        scheduler = pipe.scheduler
        text_encoder = None
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            text_encoder = pipe.text_encoder
        elif hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            text_encoder = pipe.text_encoder_2
        elif hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
            text_encoder = pipe.text_encoder_3

        vae = None
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            vae = pipe.vae

        # Freeze text encoder and VAE
        if text_encoder:
            text_encoder.eval()
            for p in text_encoder.parameters():
                p.requires_grad = False
        if vae:
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False

        # Training loop
        _tasks[task_id]["phase"] = "training"
        _tasks[task_id]["total"] = steps
        _tasks[task_id]["step"] = 0
        _tasks[task_id]["message"] = "训练中..."

        global_step = 0
        losses = []
        t_start = time.time()

        for epoch in range(max(1, steps // len(dataloader) + 1)):
            for batch in dataloader:
                if global_step >= steps:
                    break

                images = batch["image"].to("cuda", dtype=torch.float16)
                captions = batch["caption"]
                bsz = images.shape[0]

                # Encode captions
                with torch.no_grad():
                    tokens = tokenizer(
                        captions, padding="max_length",
                        max_length=77, truncation=True, return_tensors="pt"
                    ).to("cuda")
                    if text_encoder:
                        encoder_hidden_states = text_encoder(tokens.input_ids)[0]
                    else:
                        encoder_hidden_states = tokens.input_ids

                    # Encode images to latents
                    if vae:
                        with torch.no_grad():
                            latents = vae.encode(images).latent_dist.sample()
                            latents = latents * vae.config.scaling_factor
                    else:
                        latents = images

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                noise_pred = target(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)

                # Handle tuple output from PEFT
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                global_step += 1
                _tasks[task_id]["step"] = global_step

                if global_step % 10 == 0 or global_step == steps:
                    avg_loss = sum(losses[-50:]) / min(len(losses), 50)
                    elapsed = time.time() - t_start
                    eta = (elapsed / global_step) * (steps - global_step) if global_step > 0 else 0
                    _tasks[task_id]["message"] = (
                        f"训练中 {global_step}/{steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"ETA={eta:.0f}s"
                    )
                    print(f"[Training] step={global_step}/{steps} loss={avg_loss:.4f}")

            if global_step >= steps:
                break

        elapsed_total = time.time() - t_start

        # Save LoRA weights
        _tasks[task_id]["phase"] = "saving"
        _tasks[task_id]["message"] = "保存 LoRA 权重..."

        os.makedirs(output_dir, exist_ok=True)
        lora_path = os.path.join(output_dir, f"lora_{key_name.lower()}_r{rank}_s{steps}.safetensors")

        # Save PEFT model
        target.save_pretrained(output_dir)
        print(f"[Training] LoRA saved to {output_dir}")

        final_loss = sum(losses) / len(losses) if losses else 0
        result = {
            "success": True,
            "message": (
                f"训练完成!\n"
                f"• 模型: {key_name}\n"
                f"• 步数: {steps}\n"
                f"• 最终 loss: {final_loss:.4f}\n"
                f"• 用时: {elapsed_total:.0f}秒\n"
                f"• 输出: {output_dir}\n"
                f"• 数据集: {len(dataset)}张图片"
            ),
            "output_dir": output_dir,
            "loss": round(final_loss, 4),
            "steps": steps,
            "elapsed": round(elapsed_total, 1),
            "dataset_size": len(dataset),
        }
        _tasks[task_id]["result"] = result
        _tasks[task_id]["done"] = True
        _tasks[task_id]["step"] = steps

    except Exception as e:
        log_error("lora_training", str(e), e)
        _tasks[task_id]["result"] = {"success": False, "error_msg": f"训练错误: {str(e)}"}
        _tasks[task_id]["done"] = True

# ====================================================================
# AUTO LABELING
# ====================================================================

def auto_label_dataset(data_dir, method="simple"):
    """Generate caption files for images in a directory."""
    if not os.path.isdir(data_dir):
        return {"success": False, "message": f"目录不存在: {data_dir}"}

    images = [f for f in sorted(os.listdir(data_dir))
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]

    if not images:
        return {"success": False, "message": "目录中没有图片"}

    labeled = 0
    skipped = 0
    details = []

    for fname in images:
        img_path = os.path.join(data_dir, fname)
        txt_path = os.path.splitext(img_path)[0] + '.txt'

        # Skip if already has a non-empty label
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                existing = f.read().strip()
            if existing:
                skipped += 1
                continue

        if method == "filename":
            # Use filename as caption (replace _ with space, remove extension)
            caption = os.path.splitext(fname)[0].replace('_', ' ').replace('-', ' ')
            # Remove common prefixes/suffixes
            caption = re.sub(r'\b(IMG|DSC|image|photo|pic)\d*\b', '', caption, flags=re.IGNORECASE).strip()
        else:
            # "simple" method: generate a tag-based label
            tags = []
            name_lower = fname.lower()
            # Character/style detection from filename
            if any(k in name_lower for k in ['anime', '二次元', '2d']):
                tags.append("anime style")
            if any(k in name_lower for k in ['realistic', '写实', '3d']):
                tags.append("realistic")
            if any(k in name_lower for k in ['portrait', '肖像', 'port']):
                tags.append("portrait")
            if any(k in name_lower for k in ['landscape', '风景', 'scenery']):
                tags.append("landscape")
            if any(k in name_lower for k in ['nsfw', 'r18']):
                tags.append("nsfw")
            # Default tags
            if not tags:
                tags.append("high quality")
            tags.append("masterpiece")
            caption = ", ".join(tags)

        if not caption:
            caption = "high quality"

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        labeled += 1
        if labeled <= 5:
            details.append(f"  {fname} -> {caption}")

    return {
        "success": True,
        "message": f"标注完成: 新增 {labeled} 个, 跳过 {skipped} 个 (已有标注)",
        "labeled": labeled,
        "skipped": skipped,
        "total": len(images),
        "samples": details,
    }

# ====================================================================
# DIAGNOSTICS
# ====================================================================

def run_diagnostics():
    """Comprehensive system diagnostics."""
    results = []
    issues = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results.append({"check": "Python版本", "status": "ok", "detail": py_ver})

    # PyTorch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        results.append({"check": "CUDA可用", "status": "ok" if cuda_ok else "error",
                        "detail": f"PyTorch {torch.__version__}"})
        if not cuda_ok:
            issues.append("CUDA不可用，无法进行图像生成")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            vram_free = torch.cuda.mem_get_info()[0] // (1024**3)
            results.append({"check": "GPU", "status": "ok",
                            "detail": f"{gpu_name} ({vram_total}GB 总 / {vram_free}GB 可用)"})
            if vram_total < 8:
                issues.append(f"显存仅{vram_total}GB，可能无法加载模型")
    except Exception as e:
        results.append({"check": "PyTorch/CUDA", "status": "error", "detail": str(e)})
        issues.append("PyTorch 或 CUDA 配置有问题")

    # Libraries
    for lib in ["diffusers", "peft", "transformers", "PIL", "accelerate"]:
        try:
            m = __import__(lib)
            ver = getattr(m, "__version__", "ok")
            results.append({"check": f"库: {lib}", "status": "ok", "detail": str(ver)})
        except ImportError:
            results.append({"check": f"库: {lib}", "status": "error", "detail": "未安装"})
            issues.append(f"{lib} 未安装")

    # Models
    for name, cfg in MODEL_CONFIGS.items():
        path = cfg["path"]
        exists = os.path.isdir(path)
        if exists:
            fmt = detect_model_format(path)
            size_gb = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(path)
                for f in filenames
            ) / (1024**3)
            results.append({
                "check": f"模型: {name}", "status": "ok",
                "detail": f"{fmt}格式, {size_gb:.1f}GB, 需{cfg['vram']}GB显存"
            })
        else:
            results.append({
                "check": f"模型: {name}", "status": "warning",
                "detail": f"未下载 (需{cfg['vram']}GB显存)"
            })

    # Port check
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_free = s.connect_ex(('127.0.0.1', PORT)) != 0
    s.close()
    results.append({"check": f"端口{PORT}", "status": "ok", "detail": "可用" if port_free else "已被占用"})

    return {
        "checks": results,
        "issues": issues,
        "healthy": len(issues) == 0,
        "project_root": PROJECT_ROOT,
    }

# ====================================================================
# AGENT SYSTEM
# ====================================================================

AGENT_SYSTEM_PROMPT = """你是 AITermUI 的内置 AI 助手。AITermUI 是一个本地 AI 图像生成工具（终端 UI），你运行在用户的 GPU 电脑上。

你的能力：
1. 你可以通过调用工具来执行实际操作（下载模型、生成图像、训练LoRA、标注数据等）
2. 你可以诊断系统问题、查看错误日志
3. 你可以查看模型状态、任务进度
4. 你可以帮助用户配置和优化参数

重要规则：
- 调用工具时，必须使用格式: [TOOL_CALL]{"tool": "工具名", "params": {...}}[/TOOL_CALL]
- 一次只能调用一个工具，等待工具结果后再决定下一步
- 如果用户的问题可以通过工具解决，主动调用工具
- 如果发现错误，先尝试用 diagnose 或 get_error_log 诊断
- 使用中文回复，简洁直接
- 不要编造信息，不确定的事情使用工具查询
- 如果工具执行失败，分析原因并给出建议

当前环境:
- GPU: 由工具 get_system_info 获取
- 项目目录: """ + PROJECT_ROOT

def call_agent_single(api_key, provider_id, messages):
    """Single LLM API call."""
    if provider_id not in API_PROVIDERS:
        provider_id = "deepseek"
    p = API_PROVIDERS[provider_id]

    body = json.dumps({
        "model": p["model"],
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.7,
    })

    req = urllib.request.Request(
        p["endpoint"],
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

def execute_agent_tool(tool_name, params):
    """Execute an agent tool and return the result."""
    try:
        if tool_name == "list_models":
            result = []
            for name, cfg in MODEL_CONFIGS.items():
                exists = os.path.isdir(cfg["path"])
                loaded = name in _pipe_cache
                fmt = detect_model_format(cfg["path"]) if exists else "N/A"
                result.append({
                    "name": name, "available": exists,
                    "loaded": loaded, "format": fmt,
                    "vram_gb": cfg["vram"], "max_size": cfg["max_size"],
                })
            return {"success": True, "models": result}

        elif tool_name == "get_system_info":
            import torch
            cuda_ok = torch.cuda.is_available()
            info = {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "pytorch": torch.__version__,
                "cuda_available": cuda_ok,
                "loaded_model": list(_pipe_cache.keys()),
                "active_tasks": len(_tasks),
            }
            if cuda_ok:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                vram_free = torch.cuda.mem_get_info()[0] // (1024**3)
                info["vram_total_gb"] = vram_total
                info["vram_free_gb"] = vram_free
            return {"success": True, "info": info}

        elif tool_name == "download_model":
            model_name = params.get("model_name", "")
            repo_id = params.get("repo_id", "")
            target_dir = params.get("target_dir", "")

            key_name, cfg = resolve_model(model_name)
            if not cfg:
                return {"success": False, "message": f"未知模型: {model_name}。可用: {list(MODEL_CONFIGS.keys())}"}
            if not repo_id:
                repo_id = cfg["hf_repo"]
            if not target_dir:
                target_dir = cfg["path"]

            from huggingface_hub import snapshot_download
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            os.makedirs(target_dir, exist_ok=True)

            print(f"[Download] {repo_id} -> {target_dir}")
            snapshot_download(
                repo_id=repo_id, local_dir=target_dir,
                local_dir_use_symlinks=False, resume_download=True,
                max_workers=4,
            )
            return {"success": True, "message": f"模型 {key_name} 下载完成到 {target_dir}"}

        elif tool_name == "generate_image":
            model = params.get("model", "PixArt-Sigma")
            prompt = params.get("prompt", "")
            steps = int(params.get("steps", 25))
            guidance = float(params.get("guidance", 4.5))
            width = int(params.get("width", 1024))
            height = int(params.get("height", 1024))
            seed = int(params.get("seed", -1))

            if not prompt.strip():
                return {"success": False, "message": "提示词不能为空"}

            pipe, err = get_pipeline(model)
            if err:
                return {"success": False, "message": err}

            from diffusers.utils import make_image_grid
            out_path, info = run_inference(pipe, prompt, steps, guidance, seed, width, height, OUTPUT_DIR)
            return {"success": True, "message": f"生成完成!\n{info}\n保存至: {out_path}",
                    "image_path": out_path, "info": info}

        elif tool_name == "start_training":
            base_model = params.get("base_model", "PixArt-Sigma")
            data_dir = params.get("data_dir", "")
            output_dir = params.get("output_dir", os.path.join(PROJECT_ROOT, "outputs", "lora"))
            steps = int(params.get("steps", 500))
            lr = float(params.get("learning_rate", 1e-4))
            rank = int(params.get("rank", 16))
            batch_size = int(params.get("batch_size", 1))

            if not os.path.isdir(data_dir):
                return {"success": False, "message": f"数据目录不存在: {data_dir}"}

            global _task_counter
            with _tasks_lock:
                _task_counter += 1
                tid = str(_task_counter)
                _tasks[tid] = {"step": 0, "total": steps, "done": False,
                               "phase": "init", "message": "初始化训练...", "result": None}

            threading.Thread(
                target=run_lora_training,
                args=(tid, base_model, data_dir, output_dir, steps, lr, rank, batch_size),
                daemon=True
            ).start()

            return {"success": True, "message": f"训练已启动!\n• 模型: {base_model}\n• 数据: {data_dir}\n• 步数: {steps}\n• 输出: {output_dir}\n任务ID: {tid}",
                    "task_id": tid}

        elif tool_name == "auto_label":
            data_dir = params.get("data_dir", "")
            method = params.get("method", "simple")
            return auto_label_dataset(data_dir, method)

        elif tool_name == "scan_dataset":
            data_dir = params.get("data_dir", "")
            if not os.path.isdir(data_dir):
                return {"success": False, "message": f"目录不存在: {data_dir}"}

            images = sorted([f for f in os.listdir(data_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])
            txts = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.txt')])

            from PIL import Image
            sizes = {}
            for img_name in images[:100]:  # Sample first 100
                try:
                    im = Image.open(os.path.join(data_dir, img_name))
                    sz = f"{im.size[0]}x{im.size[1]}"
                    sizes[sz] = sizes.get(sz, 0) + 1
                except:
                    pass

            has_labels = sum(1 for img in images
                           if os.path.exists(os.path.splitext(os.path.join(data_dir, img))[0] + '.txt'))

            return {
                "success": True,
                "total_images": len(images),
                "labeled": has_labels,
                "unlabeled": len(images) - has_labels,
                "txt_files": len(txts),
                "size_distribution": sizes,
                "sample_files": images[:10],
            }

        elif tool_name == "check_task":
            tid = params.get("task_id", "")
            with _tasks_lock:
                task = _tasks.get(tid)
            if not task:
                return {"success": False, "message": f"任务 {tid} 不存在"}
            return {
                "success": True,
                "done": task["done"],
                "phase": task.get("phase", "?"),
                "step": task["step"],
                "total": task.get("total", 0),
                "message": task.get("message", ""),
                "result": task.get("result"),
            }

        elif tool_name == "diagnose":
            return run_diagnostics()

        elif tool_name == "get_error_log":
            count = int(params.get("count", 20))
            with _error_log_lock:
                recent = list(_error_log)[-count:]
            return {
                "success": True,
                "count": len(recent),
                "total_logged": len(_error_log),
                "errors": recent,
            }

        elif tool_name == "cancel_task":
            tid = params.get("task_id", "")
            with _tasks_lock:
                if tid in _tasks:
                    _tasks[tid]["done"] = True
                    _tasks[tid]["result"] = {"success": False, "error_msg": "用户取消"}
                    return {"success": True, "message": f"任务 {tid} 已取消"}
            return {"success": False, "message": f"任务 {tid} 不存在"}

        else:
            return {"success": False, "message": f"未知工具: {tool_name}"}

    except Exception as e:
        log_error("agent_tool", f"{tool_name}: {e}", e)
        return {"success": False, "message": f"工具执行错误: {str(e)}"}

def run_agent_loop(api_key, provider_id, user_message, screen, status, max_rounds=5):
    """Multi-turn agent loop with tool calling."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"[当前页面: {screen}] [状态: {status}]\n\n可用工具:\n" +
         json.dumps(AGENT_TOOLS, ensure_ascii=False, indent=2) +
         f"\n\n用户消息: {user_message}"},
    ]

    final_reply = ""
    tool_calls_made = []

    for round_idx in range(max_rounds):
        try:
            resp_text = call_agent_single(api_key, provider_id, messages)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                em = json.loads(err_body).get("error", {}).get("message", err_body[:200])
            except:
                em = err_body[:200]
            return {"success": False, "reply": f"供应商 {provider_id.upper()} 返回错误 [{e.code}]: {em}\n请确认: ①密钥有效且未过期 ②供应商选择正确 ③账户余额充足"}, []
        except Exception as e:
            return {"success": False, "reply": f"连接失败: {str(e)}"}, []

        # Check for tool calls
        tool_pattern = r'\[TOOL_CALL\](.+?)\[/TOOL_CALL\]'
        matches = list(re.finditer(tool_pattern, resp_text, re.DOTALL))

        if not matches:
            # No tool call, this is the final reply
            final_reply = resp_text
            break

        # Process all tool calls in this response
        for match in matches:
            tool_json_str = match.group(1).strip()
            try:
                tool_call = json.loads(tool_json_str)
                tool_name = tool_call.get("tool", "")
                params = tool_call.get("params", {})
            except json.JSONDecodeError:
                final_reply = resp_text  # Invalid JSON, treat as final reply
                break

            if not tool_name:
                continue

            # Execute tool
            result = execute_agent_tool(tool_name, params)
            tool_calls_made.append({"tool": tool_name, "params": params, "result": result})

            # Format tool result as JSON string (compact)
            result_str = json.dumps(result, ensure_ascii=False)

            # Add to conversation
            messages.append({"role": "assistant", "content": resp_text})
            messages.append({"role": "user",
                "content": f"[工具执行结果 - {tool_name}]\n{result_str}\n\n请基于此结果继续回答或调用下一个工具。"})

        # If we processed tool calls, continue loop
        if matches:
            continue
        else:
            final_reply = resp_text
            break

    if not final_reply:
        final_reply = "已执行所需工具。如有其他问题请随时提问。"

    return {"success": True, "reply": final_reply}, tool_calls_made

# ====================================================================
# HTTP HANDLER
# ====================================================================

class RequestHandler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/sys_info":
            try:
                import torch
                vram = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
                gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            except:
                vram, gpu = 8, "NVIDIA GeForce RTX 4060"
            self._send_json({
                "vram_gb": vram,
                "has_cuda": torch.cuda.is_available() if 'torch' in sys.modules else False,
                "gpu_name": gpu,
                "project_root": PROJECT_ROOT,
                "providers": list(API_PROVIDERS.keys()),
                "loaded_model": list(_pipe_cache.keys()) if _pipe_cache else [],
            })

        elif path == "/scan_models":
            result = []
            for name, cfg in MODEL_CONFIGS.items():
                integrity = check_model_integrity(cfg["path"], name)
                loaded = name in _pipe_cache
                result.append({
                    "name": name, "path": cfg["path"],
                    "status": integrity["status"],
                    "message": integrity["message"],
                    "vram_gb": cfg["vram"],
                    "loaded": loaded,
                })
            self._send_json({"models": result})

        elif path == "/model_status":
            qs = parse_qs(parsed.query)
            model = qs.get("model", [""])[0]
            key_name, cfg = resolve_model(model)
            if not cfg:
                self._send_json({"error": f"未知模型: {model}"}, 404)
                return
            integrity = check_model_integrity(cfg["path"], key_name)
            loaded = key_name in _pipe_cache
            self._send_json({
                "name": key_name,
                "status": integrity["status"],
                "message": integrity["message"],
                "loaded": loaded,
                "path": cfg["path"],
                "vram_gb": cfg["vram"],
            })

        elif path == "/task_status":
            qs = parse_qs(parsed.query)
            tid = qs.get("id", [""])[0]
            with _tasks_lock:
                task = _tasks.get(tid)
            if task:
                self._send_json({
                    "step": task["step"], "total": task.get("total", 0),
                    "done": task["done"], "phase": task.get("phase", "running"),
                    "message": task.get("message", ""),
                })
            else:
                self._send_json({"error": "task not found"}, 404)

        elif path == "/task_result":
            qs = parse_qs(parsed.query)
            tid = qs.get("id", [""])[0]
            with _tasks_lock:
                task = _tasks.get(tid)
            if task and task["done"]:
                result = task["result"].copy() if task["result"] else {}
                del _tasks[tid]
                self._send_json(result)
            elif task:
                self._send_json({"done": False, "step": task["step"], "total": task.get("total", 0)})
            else:
                self._send_json({"error": "task not found"}, 404)

        elif path == "/diagnose":
            self._send_json(run_diagnostics())

        elif path == "/error_log":
            qs = parse_qs(parsed.query)
            count = int(qs.get("count", [20])[0])
            with _error_log_lock:
                recent = list(_error_log)[-count:]
            self._send_json({"count": len(recent), "errors": recent})

        elif path == "/shutdown":
            self._send_json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.5), server.shutdown())).start()

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json({"error": f"JSON解析失败: {e}"}, 400)
            return

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/generate":
            self._handle_generate(data)
        elif path == "/train_lora":
            self._handle_train(data)
        elif path == "/agent_chat":
            self._handle_agent(data)
        elif path == "/download_model":
            self._handle_download(data)
        elif path == "/auto_label":
            self._handle_auto_label(data)
        elif path == "/scan_dataset":
            self._handle_scan_dataset(data)
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        pass  # Suppress HTTP access logs

    # ---- POST Handlers ----

    def _handle_generate(self, data):
        model_name = data.get("model", "PixArt-Sigma")
        prompt = data.get("prompt", "")
        steps = int(data.get("steps", 25))
        guidance = float(data.get("guidance", 4.5))
        seed_val = data.get("seed", -1)
        try:
            seed = int(seed_val)
        except (ValueError, TypeError):
            seed = -1
        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        output_dir = data.get("output_dir", OUTPUT_DIR)

        if not prompt.strip():
            self._send_json({"success": False, "error_msg": "提示词不能为空"})
            return

        key_name, _ = resolve_model(model_name)
        if not key_name:
            self._send_json({"success": False, "error_msg": f"未知模型: {model_name}"})
            return

        global _task_counter
        with _tasks_lock:
            _task_counter += 1
            tid = str(_task_counter)
            _tasks[tid] = {"step": 0, "total": steps, "done": False,
                           "phase": "loading", "result": None, "message": "加载模型..."}

        def progress_cb(step_idx, timestep, latents):
            if tid in _tasks:
                _tasks[tid]["step"] = step_idx + 1

        def run_gen():
            try:
                import torch
                pipe, err = get_pipeline(model_name)
                if err:
                    _tasks[tid]["result"] = {"success": False, "error_msg": err}
                    _tasks[tid]["done"] = True
                    return

                _tasks[tid]["phase"] = "running"
                _tasks[tid]["message"] = "推理中..."
                out_path, info = run_inference(
                    pipe, prompt, steps, guidance, seed,
                    width, height, output_dir, progress_cb
                )
                _tasks[tid]["result"] = {"success": True, "image_path": out_path, "info": info}
            except Exception as e:
                log_error("generate", str(e), e)
                _tasks[tid]["result"] = {"success": False, "error_msg": str(e)}
            _tasks[tid]["done"] = True
            _tasks[tid]["step"] = steps

        threading.Thread(target=run_gen, daemon=True).start()
        self._send_json({"success": True, "task_id": tid, "message": "推理启动中..."})

    def _handle_train(self, data):
        base_model = data.get("base_model", "")
        data_dir = data.get("data_dir", "")
        output_dir = data.get("output_dir", os.path.join(PROJECT_ROOT, "outputs", "lora"))
        steps = int(data.get("steps", 500))
        lr = float(data.get("learning_rate", 1e-4))
        rank = int(data.get("rank", 16))
        batch_size = int(data.get("batch_size", 1))

        if not data_dir:
            self._send_json({"success": False, "error_msg": "请选择数据文件夹"})
            return
        if not os.path.isdir(data_dir):
            self._send_json({"success": False, "error_msg": f"数据文件夹不存在: {data_dir}"})
            return

        # Validate dataset
        imgs = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        if len(imgs) < 5:
            self._send_json({
                "success": False,
                "error_msg": f"图片太少 ({len(imgs)}张)，至少需要5张。请添加更多图片到 {data_dir}"
            })
            return

        global _task_counter
        with _tasks_lock:
            _task_counter += 1
            tid = str(_task_counter)
            _tasks[tid] = {"step": 0, "total": steps, "done": False,
                           "phase": "init", "message": "初始化训练...", "result": None}

        threading.Thread(
            target=run_lora_training,
            args=(tid, base_model, data_dir, output_dir, steps, lr, rank, batch_size),
            daemon=True
        ).start()

        self._send_json({
            "success": True, "task_id": tid,
            "message": f"训练已启动\n• 数据: {len(imgs)}张图片\n• 步数: {steps}\n• 输出: {output_dir}"
        })

    def _handle_agent(self, data):
        message = data.get("message", "")
        api_key = data.get("api_key", "").strip()
        provider = data.get("provider", "deepseek").strip()
        screen = data.get("screen", "unknown")
        status = data.get("status", "unknown")

        if not api_key or len(api_key) < 15:
            self._send_json({"success": False, "reply": f"API密钥无效 (长度 {len(api_key)})，请检查: 确保已粘贴完整密钥（以 sk- 开头）"})
            return
        if not message.strip():
            self._send_json({"success": False, "reply": "请输入问题。"})
            return

        result, tool_calls = run_agent_loop(api_key, provider, message, screen, status)

        self._send_json({
            "success": result.get("success", False),
            "reply": result.get("reply", "无响应"),
            "tool_calls": [{"tool": tc["tool"], "summary": tc["result"].get("message", str(tc["result"]))[:200]}
                          for tc in tool_calls],
        })

    def _handle_download(self, data):
        model_name = data.get("model", "")
        repo_id = data.get("repo", "")
        target_dir = data.get("target_dir", "")

        if not model_name:
            self._send_json({"success": False, "error_msg": "需要 model 参数"})
            return

        key_name, cfg = resolve_model(model_name)
        if not cfg:
            self._send_json({"success": False, "error_msg": f"未知模型: {model_name}"})
            return

        if not repo_id:
            repo_id = cfg["hf_repo"]
        if not target_dir:
            target_dir = cfg["path"]
        expected_gb = cfg.get("expected_size_gb", 0)

        # Clean stale temp files from previous attempts
        if os.path.isdir(target_dir):
            for f in os.listdir(target_dir):
                if f.endswith('.incomplete') or f.endswith('.download'):
                    try: os.remove(os.path.join(target_dir, f))
                    except: pass

        # Prevent duplicate downloads
        with _tasks_lock:
            for tid, task in list(_tasks.items()):
                tresult = task.get("result") or {}
                if task.get("phase") == "downloading" and tresult.get("model") == key_name:
                    self._send_json({"success": True, "task_id": tid,
                                     "message": f"{key_name} 正在下载中..."})
                    return

        # Create task
        global _task_counter
        with _tasks_lock:
            _task_counter += 1
            task_id = f"download_{_task_counter}"
            _tasks[task_id] = {
                "done": False, "step": 0, "total": 100,
                "phase": "preparing", "message": "准备下载...",
                "result": {"model": key_name, "action": "download", "target": target_dir,
                          "expected_gb": expected_gb}
            }

        import time as _time

        def _mark_failed(msg):
            with _tasks_lock:
                t = _tasks.get(task_id)
                if t:
                    t["done"] = True; t["phase"] = "failed"; t["message"] = msg[:120]
            print(f"[Download] FAILED: {msg}")

        def _do_modelscope_download():
            """Download via ModelScope (Alibaba CDN).
            Same approach as hf-mirror: HubApi.get_model_files for file list +
            requests streaming download with chunk-level progress updates."""
            import requests

            ms_repo = cfg.get("ms_repo", repo_id)

            with _tasks_lock:
                t = _tasks.get(task_id)
                if t: t["message"] = f"获取 {key_name} 文件列表 (ModelScope)..."

            # --- Step 1: Get recursive file list ---
            try:
                from modelscope.hub.api import HubApi
                api = HubApi()
                entries = api.get_model_files(ms_repo, recursive=True)
                files = []
                total_expected_size = 0
                for e in entries:
                    if e.get("Type") == "blob":
                        path = e.get("Path", "")
                        size = e.get("Size", 0)
                        if path:
                            files.append((path, size))
                            total_expected_size += size
            except Exception as e:
                _mark_failed(f"无法连接 ModelScope: {e}")
                return

            if not files:
                _mark_failed("ModelScope 空文件列表")
                return

            total_files = len(files)
            expected_gb_real = total_expected_size / 1e9
            print(f"[Download] {key_name}: {total_files} files, {expected_gb_real:.1f}GB via ModelScope")

            with _tasks_lock:
                t = _tasks.get(task_id)
                if t:
                    t["total"] = total_files
                    t["phase"] = "downloading"
                    t["message"] = f"{key_name}: 0/{total_files} 文件"

            # --- Step 2: Download each file ---
            start_time = _time.time()
            total_bytes = 0
            last_report = 0.0

            def _report_progress(fi, force=False):
                nonlocal last_report
                now = _time.time()
                if not force and now - last_report < 2:
                    return
                last_report = now
                elapsed = now - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0
                if speed > 1e6:
                    speed_s = f"{speed/1e6:.1f} MB/s"
                elif speed > 1e3:
                    speed_s = f"{speed/1e3:.0f} KB/s"
                else:
                    speed_s = "计算中..."
                if expected_gb_real > 0:
                    pct = int(min(total_bytes / (expected_gb_real * 1e9) * 100, 99))
                    size_s = f"{total_bytes/1e9:.2f}/{expected_gb_real:.1f}GB"
                    if speed > 0:
                        remain = (expected_gb_real * 1e9 - total_bytes) / speed
                        m, s = divmod(int(remain), 60)
                        if m >= 60:
                            h, m = divmod(m, 60)
                            eta = f" 剩余{h}h{m}m"
                        elif m > 0:
                            eta = f" 剩余{m}m{s}s"
                        else:
                            eta = f" 剩余{s}s"
                    else:
                        eta = ""
                else:
                    pct = int((fi + 1) / total_files * 100)
                    size_s = f"{total_bytes/1e6:.0f}MB"
                    eta = ""
                with _tasks_lock:
                    t = _tasks.get(task_id)
                    if t:
                        t["step"] = pct
                        t["total"] = 100
                        t["message"] = f"{key_name}: {fi+1}/{total_files}文件 {size_s} {speed_s}{eta}"

            for fi, (fname, fsize) in enumerate(files):
                with _tasks_lock:
                    t = _tasks.get(task_id)
                    if not t or t.get("done"):
                        return

                local_path = os.path.join(target_dir, fname)

                # Skip if already complete (size matches)
                try:
                    if os.path.exists(local_path) and abs(os.path.getsize(local_path) - fsize) < 100:
                        total_bytes += os.path.getsize(local_path)
                        _report_progress(fi, force=True)
                        continue
                except OSError:
                    pass

                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                file_url = f"https://www.modelscope.cn/api/v1/models/{ms_repo}/repo?Revision=master&FilePath={fname}"
                ok = False
                last_err = ""
                file_start_bytes = total_bytes
                for retry in range(3):
                    try:
                        r = requests.get(file_url, timeout=(30, 600), stream=True)
                        r.raise_for_status()
                        tmp = local_path + ".download"
                        with open(tmp, 'wb') as fout:
                            for chunk in r.iter_content(65536):
                                if chunk:
                                    fout.write(chunk)
                                    total_bytes += len(chunk)
                                    _report_progress(fi)
                        os.replace(tmp, local_path)
                        ok = True
                        break
                    except Exception as e:
                        last_err = str(e)
                        if retry < 2:
                            _time.sleep(5 * (retry + 1))
                        try:
                            os.remove(local_path + ".download")
                        except:
                            pass
                        total_bytes = file_start_bytes

                if not ok:
                    _mark_failed(f"下载中断: {fname} ({last_err[:80]})\n支持断点续传，请重试")
                    return

                _report_progress(fi, force=True)

            # --- Done ---
            fmt = detect_model_format(target_dir)
            with _tasks_lock:
                t = _tasks.get(task_id)
                if t:
                    t["done"] = True
                    t["step"] = 100
                    t["total"] = 100
                    t["phase"] = "complete"
                    t["message"] = f"下载完成: {key_name} ({total_bytes/1e9:.1f}GB)"
            with _pipe_lock:
                _pipe_cache.pop(key_name, None)
            print(f"[Download] {key_name} DONE via ModelScope ({total_bytes/1e9:.1f}GB)")

        def _download_worker():
            src = cfg.get("download_source", "hf-mirror")
            if src == "modelscope":
                _do_modelscope_download()
                return

            import requests

            # --- Step 1: Get file list ---
            with _tasks_lock:
                t = _tasks.get(task_id)
                if t: t["message"] = f"获取 {key_name} 文件列表..."

            files = []
            api_mirror = ""
            for mirror in ["https://hf-mirror.com", "https://huggingface.co"]:
                try:
                    r = requests.get(f"{mirror}/api/models/{repo_id}", timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    for sib in data.get("siblings", []):
                        rfn = sib.get("rfilename", "")
                        if rfn: files.append(rfn)
                    api_mirror = mirror
                    print(f"[Download] {key_name}: {len(files)} files via {mirror}")
                    break
                except Exception as e:
                    print(f"[Download] {mirror} API failed: {e}")

            if not files:
                _mark_failed("无法连接下载服务器，请检查网络后重试")
                return

            total_files = len(files)
            with _tasks_lock:
                t = _tasks.get(task_id)
                if t:
                    t["total"] = total_files
                    t["phase"] = "downloading"
                    t["message"] = f"{key_name}: 0/{total_files} 文件"

            # --- Step 2: Download each file ---
            start_time = _time.time()
            total_bytes = 0
            last_report = 0.0

            def _report_progress(fi, force=False):
                nonlocal last_report
                now = _time.time()
                if not force and now - last_report < 2:
                    return
                last_report = now
                elapsed = now - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0
                if speed > 1e6:
                    speed_s = f"{speed/1e6:.1f} MB/s"
                elif speed > 1e3:
                    speed_s = f"{speed/1e3:.0f} KB/s"
                else:
                    speed_s = "0 KB/s"
                if expected_gb > 0:
                    pct = int(min(total_bytes / (expected_gb * 1e9) * 100, 99))
                    size_s = f"{total_bytes/1e9:.2f}/{expected_gb:.0f}GB"
                    if speed > 0:
                        remain = (expected_gb * 1e9 - total_bytes) / speed
                        m, s = divmod(int(remain), 60)
                        if m >= 60:
                            h, m = divmod(m, 60)
                            eta = f" 剩余{h}h{m}m"
                        elif m > 0:
                            eta = f" 剩余{m}m{s}s"
                        else:
                            eta = f" 剩余{s}s"
                    else:
                        eta = ""
                else:
                    pct = int((fi + 1) / total_files * 100)
                    size_s = f"{total_bytes/1e6:.0f}MB"
                    eta = ""
                with _tasks_lock:
                    t = _tasks.get(task_id)
                    if t:
                        t["step"] = pct
                        t["total"] = 100
                        t["message"] = f"{key_name}: {fi+1}/{total_files}文件 {size_s} {speed_s}{eta}"

            for fi, fname in enumerate(files):
                with _tasks_lock:
                    t = _tasks.get(task_id)
                    if not t or t.get("done"):
                        return

                file_url = f"{api_mirror}/{repo_id}/resolve/main/{fname}"
                local_path = os.path.join(target_dir, fname)

                # Skip already-downloaded files
                try:
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        total_bytes += os.path.getsize(local_path)
                        continue
                except OSError:
                    pass

                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # Download with up to 3 retries
                ok = False
                last_err = ""
                file_start_bytes = total_bytes
                for retry in range(3):
                    try:
                        r = requests.get(file_url, timeout=(30, 600), stream=True)
                        r.raise_for_status()
                        tmp = local_path + ".download"
                        with open(tmp, 'wb') as fout:
                            for chunk in r.iter_content(65536):
                                if chunk:
                                    fout.write(chunk)
                                    total_bytes += len(chunk)
                                    _report_progress(fi)
                        os.replace(tmp, local_path)
                        ok = True
                        break
                    except Exception as e:
                        last_err = str(e)
                        if retry < 2:
                            _time.sleep(5 * (retry + 1))
                        if os.path.exists(local_path + ".download"):
                            try: os.remove(local_path + ".download")
                            except: pass
                        total_bytes = file_start_bytes

                if not ok:
                    _mark_failed(f"下载中断: {fname} （{last_err[:60]}）\n支持断点续传，请重试")
                    return

                _report_progress(fi, force=True)

            # --- Done ---
            fmt = detect_model_format(target_dir)
            with _tasks_lock:
                t = _tasks.get(task_id)
                if t:
                    t["done"] = True
                    t["step"] = 100
                    t["total"] = 100
                    t["phase"] = "complete"
                    t["message"] = f"下载完成: {key_name} ({total_bytes/1e9:.1f}GB)"
            with _pipe_lock:
                _pipe_cache.pop(key_name, None)
            print(f"[Download] {key_name} DONE: {total_bytes/1e9:.1f}GB")

        threading.Thread(target=_download_worker, daemon=True).start()
        vpn_note = "\n需要 VPN 连接国际网络" if cfg.get("_vpn_required") else ""
        self._send_json({
            "success": True,
            "task_id": task_id,
            "message": f"开始下载 {key_name} (预计 {expected_gb} GB, 共 ? 个文件){vpn_note}"
        })

    def _handle_auto_label(self, data):
        data_dir = data.get("data_dir", "")
        method = data.get("method", "simple")
        result = auto_label_dataset(data_dir, method)
        self._send_json(result)

    def _handle_scan_dataset(self, data):
        data_dir = data.get("data_dir", "")
        if not os.path.isdir(data_dir):
            self._send_json({"success": False, "message": f"目录不存在: {data_dir}"})
            return

        images = sorted([f for f in os.listdir(data_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))])

        from PIL import Image
        sizes = {}
        for img_name in images[:100]:
            try:
                im = Image.open(os.path.join(data_dir, img_name))
                sz = f"{im.size[0]}x{im.size[1]}"
                sizes[sz] = sizes.get(sz, 0) + 1
            except:
                pass

        has_labels = sum(1 for img in images
                       if os.path.exists(os.path.splitext(os.path.join(data_dir, img))[0] + '.txt'))

        self._send_json({
            "success": True,
            "total_images": len(images),
            "labeled": has_labels,
            "unlabeled": len(images) - has_labels,
            "size_distribution": sizes,
            "sample_files": images[:10],
        })


def parent_watcher(parent_pid):
    """Daemon thread: exit if parent process is gone."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x00100000
    hProcess = kernel32.OpenProcess(SYNCHRONIZE, False, int(parent_pid))
    if not hProcess:
        print(f"[Watcher] Cannot open parent PID={parent_pid}, exiting")
        os._exit(0)
    print(f"[Watcher] Monitoring parent PID={parent_pid}")
    while True:
        ret = kernel32.WaitForSingleObject(hProcess, 5000)
        if ret == 0:  # WAIT_OBJECT_0 = parent died
            print(f"[Watcher] Parent PID={parent_pid} exited, shutting down")
            kernel32.CloseHandle(hProcess)
            os._exit(0)
        # ret == 0x102 = WAIT_TIMEOUT, keep watching


# ====================================================================
# SERVER MAIN
# ====================================================================

server = None

def main():
    global server
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Parse --parent-pid for auto-cleanup
    parent_pid = None
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--parent-pid="):
            parent_pid = a.split("=", 1)[1]
            break
    if parent_pid:
        threading.Thread(target=parent_watcher, args=(parent_pid,), daemon=True).start()
    
    server = HTTPServer(("127.0.0.1", PORT), RequestHandler)
    print(f"AITermUI Backend v2 running on http://127.0.0.1:{PORT}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    main()
