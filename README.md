# AI_Art — 基于 FTXUI 的 AI 绘画终端工具

## 项目简介

一个 C++ 终端 UI 程序，通过 FTXUI 库提供图形化操作界面，底层调用 Python 后端进行 AI 图像生成和 LoRA 训练。支持 PixArt-Sigma、SD3-Medium、SD3.5-Large 三种模型。

## 选题动机

这次作业要求使用 **FTXUI**（C++ 终端 UI 库）。我之前接触过本地 AI 生图，但更多都是依靠 AI 给我代码直接复制粘贴来下载模型并调用模型。虽然接触过一些网上的 UI，但感觉琳琅满目的，所以借着这次作业我想设计一套自己的 UI。模型的选择方面，我之前用的都是基于 **UNet** 的模型，但最近和 AI 聊天了解到，现在更先进的生成模型架构是 **DiT（Diffusion Transformer，用 Transformer 替代 UNet 做去噪）**——PixArt-Sigma 和 SD3 系列就是基于 DiT 的开源模型。正好趁这个机会，一边试用 DiT 架构的模型，一边做个带 UI 的工具，**自己能用，作业也能交**。

Python 部分是不得已而为之——模型加载、推理、LoRA 训练都必须在 Python 生态（PyTorch / Diffusers）里跑，C++ 没有对应的库支持。所以 Python 后端由 AI 一键生成，我的工作重心在 C++ 前端。

## 我做了什么 & AI 做了什么

| 部分 | 负责人 | 具体内容 |
|------|--------|---------|
| **UI 设计** | 我 | 整体界面大致的布局（通过prompt让ai构建的）、交互逻辑设计 |
| **CMake 构建** | 我 | 研究 CMake 配置、FTXUI 依赖管理、编译调试 |
| **方向确定** | 我 | 决定做 AI 绘画工具、选 FTXUI 作为 UI 库 |
| **C++ 前端代码** | AI（我监督修改） | generate_screen、agent_panel、backend_ipc 等模块的功能实现 |
| **Python 后端** | AI（一键生成） | 模型加载、推理、LoRA 训练、Agent 对话等 |

**本项目重点是 FTXUI 库的使用和终端 UI 的设计实现**——包括多页面切换、菜单交互、进度条轮询、键盘事件处理等。C++ 与 Python 的跨语言通信通过 JSON + HTTP 管道实现，这部分也是研究和学习的内容。

## 项目结构

```
AI_Art_Submit/
├── cpp/                      # C++ 前端（FTXUI 终端界面）
│   ├── src/                  # 源代码
│   │   ├── main.cpp          # 程序入口 + 主界面布局
│   │   ├── generate_screen.cpp   # AI 生成页面（模型选择、参数配置）
│   │   ├── train_screen.cpp      # LoRA 训练页面
│   │   ├── agent_panel.cpp       # AI 助手侧边面板
│   │   └── backend_ipc.cpp       # 与 Python 后端的进程通信
│   └── include/              # 头文件
├── python/
│   └── server.py             # Python 后端（模型加载/推理/训练）
├── CMakeLists.txt            # CMake 构建配置
├── requirements.txt          # Python 依赖列表
├── 启动.bat                  # 一键启动脚本
├── api.txt.example           # API Key 配置模板
├── models/                   # 模型文件夹（从百度网盘下载放入）
│   ├── PixArt-Sigma/
│   └── SD3-Medium/
└── README.md                 # 本文件
```

## 模型下载

程序需要模型文件才能运行，有两种获取方式：

### 方式一：程序内一键下载（推荐）

启动程序后，在**生图页面**的模型选择菜单中点击 **[下载模型]** 按钮，程序会自动下载对应模型到 `models/` 目录，支持断点续传。

### 方式二：百度网盘（国内更快）

> **百度网盘链接**：https://pan.baidu.com/s/1BfypSFGzLcFe4ngWeGR8qA?pwd=3tb8  
> **提取码**：3tb8

下载后解压，将模型文件夹放入项目根目录的 `models/` 目录下即可。

### 模型列表

| 模型 | 大小 | 状态 |
|------|------|------|
| PixArt-Sigma | ~7 GB | ✅ 可用 |
| SD3-Medium | ~10 GB | ✅ 可用 |
| SD3.5-Large | ~16 GB | ⚠️ 模型文件有问题，暂不可用 |

无论哪种方式，放好后双击 `启动.bat`，程序会自动检测并识别模型。

```
models/
├── PixArt-Sigma/     # 下载后放入
├── SD3-Medium/       # 下载后放入
└── SD3.5-Large/      # 暂不可用
```

## 启动方法

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 编译 C++ 前端

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

CMake 会自动从 GitHub 拉取 FTXUI 库源码并编译。如果遇到网络问题（国内 git clone 超时），可以手动 clone FTXUI 仓库后指定路径：

```bash
cmake .. -DFETCHCONTENT_SOURCE_DIR_FTXUI="你的FTXUI路径"
```

### 3. 运行

双击 `启动.bat`，或：

```bash
cd build
Release\aitermui.exe
```

## 踩坑记录

- **FTXUI FetchContent 国内超时**：CMake 在 `FetchContent_MakeAvailable` 阶段需要 git clone FTXUI 源码，国内网络不稳定。解决方案是手动 clone 后通过 `FETCHCONTENT_SOURCE_DIR_FTXUI` 指定本地路径。
- **MSVC UTF-8 编码**：C++ 源码含中文字符串时 MSVC 需要 `/utf-8` 编译选项（已在 CMakeLists.txt 中添加）。
- **模型下载断点续传**：大模型文件（7-22GB）下载中断后需要支持续传，通过逐文件下载 + 临时文件标记实现。
- **GPU 显存泄漏**：切换模型时旧 pipeline 未显式释放 GPU 显存，导致 OOM。通过 `del pipe` + `torch.cuda.empty_cache()` + `gc.collect()` 三步解决。
