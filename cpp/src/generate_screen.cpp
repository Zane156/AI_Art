#include "generate_screen.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/dom/elements.hpp"
#include "backend_ipc.hpp"
#include "file_dialog.hpp"
#include "json_util.hpp"
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace ftxui;

// Helper: paste from clipboard into a string
static bool PasteFromClipboard(std::string& target) {
    if (!OpenClipboard(nullptr)) return false;
    HANDLE h = GetClipboardData(CF_UNICODETEXT);
    if (!h) { CloseClipboard(); return false; }
    wchar_t* w = (wchar_t*)GlobalLock(h);
    if (!w) { CloseClipboard(); return false; }
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (len > 1) {
        std::string text(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w, -1, &text[0], len, nullptr, nullptr);
        target += text;
    }
    GlobalUnlock(h);
    CloseClipboard();
    return true;
}

struct GState {
    int model_index = 0;
    std::vector<std::string> model_names = {
        " PixArt-Sigma       [显存: 8GB+]  ",
        " SD3-Medium         [显存: 11GB+] ",
        " SD3.5-Large        [显存: 22GB+] ",
    };
    std::vector<std::string> model_keys = {"PixArt-Sigma", "SD3-Medium", "SD3.5-Large"};
    std::vector<bool> model_downloaded = {false, false, false};
    std::vector<std::string> model_status_msg = {"", "", ""};
    bool use_lora = false;
    float lora_weight = 0.8f;
    std::string lora_path;
    std::string prompt;
    int steps = 25;
    float cfg = 4.5f;
    std::string seed_str = "-1";
    int size_idx = 0;
    std::vector<std::string> size_names = {" 1024x1024 ", " 768x768 "};
    std::string output_dir = "outputs";

    // Generation state (atomic for ints, mutex for strings)
    bool generating = false;             // only set from main thread
    std::string task_id;                 // only set from main thread
    std::atomic<int> prog_cur{0};
    std::atomic<int> prog_total{0};
    std::mutex state_mutex;              // protects out_path, result_info, status_msg, download variables
    std::string out_path;
    std::string result_info;
    std::string status_msg;
    std::atomic<bool> poll_active{false};

    // Model download
    bool scanning_models = false;
    bool downloading_model = false;
    std::string download_msg;
};

Component CreateGenerateScreen(
    std::shared_ptr<BackendIPC> backend,
    std::function<void()> on_switch_to_train,
    std::function<void(const std::string&)> on_status_change,
    std::function<void()> on_ui_refresh)
{
    auto st = std::make_shared<GState>();

    // ---- Scan model status on startup ----
    st->scanning_models = true;
    std::thread([st, backend, on_ui_refresh] {
        for (int i = 0; i < 3; i++) {
            auto resp = backend->CheckModelStatus(st->model_keys[i]);
        auto sts = JsonGet(resp, "status");
        st->model_downloaded[i] = (sts == "complete");
        st->model_status_msg[i] = JsonGet(resp, "message");
        }
        st->scanning_models = false;
        on_ui_refresh();
    }).detach();

    // ---- Components ----
    auto model_menu        = Menu(&st->model_names, &st->model_index);
    // Custom model path feature removed
    auto lora_checkbox     = Checkbox(" 启用 LoRA", &st->use_lora);
    auto lora_path_input   = Input(&st->lora_path, "选择 .safetensors 文件...");
    auto browse_lora_btn   = Button(" [浏览文件...] ", [st] {
        std::string file = FileDialog::OpenFile(L"Safetensors", L"*.safetensors");
        if (!file.empty()) st->lora_path = file;
    });
    auto lora_weight_slider = Slider("权重:", &st->lora_weight, 0.0f, 2.0f, 0.05f);
    auto prompt_input       = Input(&st->prompt, "请输入提示词 (Ctrl+V 粘贴)...");
    prompt_input = CatchEvent(prompt_input, [&st](Event e) {
        if (e == Event::Character('\x16')) { PasteFromClipboard(st->prompt); return true; }
        return false;
    });
    auto steps_slider       = Slider("步数:", &st->steps, 10, 50, 1);
    auto cfg_slider         = Slider("引导:", &st->cfg, 1.0f, 15.0f, 0.5f);
    auto seed_input         = Input(&st->seed_str, "-1 = 随机");
    auto size_menu          = Menu(&st->size_names, &st->size_idx);
    auto output_dir_input   = Input(&st->output_dir, "保存路径...");
    auto browse_out_btn     = Button(" [选择保存位置...] ", [st] {
        std::string f = FileDialog::OpenFolder("选择图片保存位置");
        if (!f.empty()) st->output_dir = f;
    });

    auto generate_btn = Button(" [ 开始生成 ] ", [st, backend, on_status_change, on_ui_refresh] {
        if (st->generating) return;
        if (st->prompt.empty()) {
            st->result_info = "请先输入提示词";
            return;
        }
        st->generating = true;
        st->prog_cur = 0;
        st->prog_total = st->steps;
        {
            std::lock_guard<std::mutex> lk(st->state_mutex);
            st->out_path = "";
            st->result_info = "";
            st->status_msg = "正在加载模型...";
        }
        on_status_change("正在生成...");

        int w = (st->size_idx == 0) ? 1024 : 768;
        std::string mn;
        if      (st->model_index == 0) mn = "PixArt-Sigma";
        else if (st->model_index == 1) mn = "SD3-Medium";
        else if (st->model_index == 2) mn = "SD3.5-Large";
        else                           mn = "PixArt-Sigma";  // safety (unreachable)

        // Safely parse seed (protect against non-numeric input)
        int seed_val = -1;
        try { seed_val = std::stoi(st->seed_str.empty() ? "-1" : st->seed_str); }
        catch (...) { seed_val = -1; st->seed_str = "-1"; }

        // Build JSON safely with JsonPair/JsonPairNum
        std::ostringstream json;
        json << std::fixed << std::setprecision(2) << "{";
        JsonPair(json, "model", mn);                    json << ",";
        JsonPair(json, "prompt", st->prompt);            json << ",";
        JsonPairNum(json, "steps", st->steps);           json << ",";
        JsonPairNum(json, "guidance", st->cfg);          json << ",";
        JsonPairNum(json, "seed", seed_val);              json << ",";
        JsonPairNum(json, "width", w);                   json << ",";
        JsonPairNum(json, "height", w);                  json << ",";
        JsonPair(json, "output_dir", st->output_dir);
        json << "}";

        std::string resp = backend->Generate(json.str());

        st->task_id = JsonGet(resp, "task_id");
        if (st->task_id.empty()) {
            st->generating = false;
            st->result_info = JsonGet(resp, "error_msg");
            {
                std::lock_guard<std::mutex> lk(st->state_mutex);
                st->status_msg = "生成失败";
            }
            on_status_change("失败: " + st->result_info);
            return;
        }

        {
            std::lock_guard<std::mutex> lk(st->state_mutex);
            st->status_msg = "推理中...";
        }
        st->poll_active = true;

        // Poll progress in background thread (thread-safe via atomics + mutex)
        std::thread([st, backend, on_status_change, on_ui_refresh]() {
            while (st->poll_active) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (!st->poll_active) break;

                auto s = backend->CheckTaskStatus(st->task_id);
                auto phase = JsonGet(s, "phase");

                // Update atomic progress counters
                {
                    auto step = JsonGet(s, "step");
                    auto total = JsonGet(s, "total");
                    if (!step.empty()) {
                        int cur = std::atoi(step.c_str());
                        st->prog_cur = (std::min)(cur, st->prog_total.load());
                    }
                    if (!total.empty()) {
                        st->prog_total = std::atoi(total.c_str());
                    }
                }

                // Update status message — phase-aware, single write point
                {
                    std::lock_guard<std::mutex> lk(st->state_mutex);
                    if (phase == "downloading") {
                        int cur = st->prog_cur.load();
                        st->status_msg = "正下载模型... " + std::to_string(cur) + "%";
                    } else if (phase == "loading") {
                        st->status_msg = "正加载模型管道...";
                    } else if (st->prog_cur.load() == 0) {
                        st->status_msg = "管线就绪，正初始化推理...";
                    } else if (st->prog_cur.load() >= st->prog_total.load() && st->prog_total.load() > 0) {
                        st->status_msg = "推理完成，正在保存图片...";
                    } else {
                        st->status_msg = "推理中 " + std::to_string(st->prog_cur.load())
                                       + "/" + std::to_string(st->prog_total.load());
                    }
                }

                on_ui_refresh();  // force redraw from background thread

                bool done = (s.find("\"done\":true") != std::string::npos);
                if (done) {
                    auto r = backend->GetTaskResult(st->task_id);
                    st->poll_active = false;
                    st->generating = false;
                    st->prog_cur = st->prog_total.load();

                    if (r.find("\"success\":true") != std::string::npos) {
                        {
                            std::lock_guard<std::mutex> lk(st->state_mutex);
                            st->out_path = JsonGet(r, "image_path");
                            st->result_info = JsonGet(r, "info");
                        }
                        on_status_change("生成完成!");
                    } else {
                        auto err = JsonGet(r, "error_msg");
                        {
                            std::lock_guard<std::mutex> lk(st->state_mutex);
                            if (!err.empty()) {
                                st->result_info = "失败: " + err;
                            } else if (!r.empty()) {
                                st->result_info = "生成结束，请查看输出";
                            } else {
                                st->result_info = "生成完成（结果获取失败）";
                            }
                        }
                        on_status_change(st->result_info);
                    }
                    on_ui_refresh();
                    break;
                }
            }
        }).detach();
    });

    // ---- Download model button ----
    auto download_btn = Button(" [ 一键下载模型 ] ", [st, backend, on_ui_refresh] {
        if (st->downloading_model) return;
        st->downloading_model = true;
        st->download_msg = "正在启动下载...";
        st->prog_cur = 0;
        st->prog_total = 0;
        on_ui_refresh();

        std::string mn = st->model_keys[st->model_index];
        std::thread([st, backend, mn, on_ui_refresh] {
            auto resp = backend->DownloadModel(mn, "", "");
            auto succ = JsonGet(resp, "success");
            if (succ != "true" && succ != "True") {
                auto msg = JsonGet(resp, "message");
                if (msg.empty()) msg = JsonGet(resp, "error_msg");
                {
                    std::lock_guard<std::mutex> lk(st->state_mutex);
                    st->download_msg = "启动失败: " + msg;
                }
                st->downloading_model = false;
                on_ui_refresh();
                return;
            }

            auto task_id = JsonGet(resp, "task_id");
            if (task_id.empty()) {
                std::lock_guard<std::mutex> lk(st->state_mutex);
                st->download_msg = "下载已启动 (后台运行)";
                st->downloading_model = false;
                on_ui_refresh();
                return;
            }

            // Poll download progress
            while (st->downloading_model) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                if (!st->downloading_model) break;

                auto status = backend->CheckTaskStatus(task_id);
                auto step = JsonGet(status, "step");
                auto total = JsonGet(status, "total");
                auto msg = JsonGet(status, "message");
                auto phase = JsonGet(status, "phase");

                {
                    std::lock_guard<std::mutex> lk(st->state_mutex);
                    if (!msg.empty()) st->download_msg = msg;
                    if (!step.empty()) st->prog_cur = std::atoi(step.c_str());
                    if (!total.empty()) st->prog_total = std::atoi(total.c_str());
                }

                bool done = (status.find("\"done\":true") != std::string::npos);
                if (done) {
                    bool ok = (phase == "complete" || phase.find("complete") == 0);
                    st->downloading_model = false;
                    if (ok) {
                        st->model_downloaded[st->model_index] = true;
                        {
                            std::lock_guard<std::mutex> lk(st->state_mutex);
                            st->download_msg = "下载完成!";
                            st->prog_cur = 100;
                        }
                    } else {
                        {
                            std::lock_guard<std::mutex> lk(st->state_mutex);
                            st->download_msg = msg.empty() ? ("下载" + phase) : msg;
                        }
                    }
                    on_ui_refresh();
                    break;
                }
                on_ui_refresh();
            }
        }).detach();
    });

    // ---- Container ----
    auto container = Container::Vertical({
        model_menu,
        lora_checkbox, lora_path_input, browse_lora_btn, lora_weight_slider,
        prompt_input, steps_slider, cfg_slider, seed_input, size_menu,
        output_dir_input, browse_out_btn,
        generate_btn, download_btn,
    });

    return Renderer(container, [st,
        model_menu,
        lora_checkbox, lora_path_input, browse_lora_btn, lora_weight_slider,
        prompt_input, steps_slider, cfg_slider, seed_input, size_menu,
        output_dir_input, browse_out_btn, generate_btn, download_btn
    ] {
        Elements all;

        // Model selection
        all.push_back(text(" 模型选择") | bold | color(Color::Cyan));
        all.push_back(model_menu->Render());


        // Model download status
        if (st->scanning_models) {
            all.push_back(text(" 正在检查模型状态...") | dim | color(Color::Yellow));
        } else if (st->downloading_model) {
            int pct = st->prog_cur.load();
            all.push_back(text(" " + st->download_msg) | color(Color::Yellow));
            int bar_w = 30;
            int filled = pct * bar_w / 100;
            if (filled > bar_w) filled = bar_w;
            if (filled < 0) filled = 0;
            std::string bar = std::string(filled, '#') + std::string(bar_w - filled, '-');
            all.push_back(hbox({
                text("  [" + bar + "] ") | color(Color::CyanLight),
                text(std::to_string(pct) + "%") | bold,
            }));
        } else if (st->model_index < 3 && !st->model_downloaded[st->model_index]) {
            // Model not downloaded and not downloading
            if (!st->download_msg.empty()) {
                // Show last download result (error)
                all.push_back(text(" " + st->download_msg) | color(Color::Red));
            }
            if (!st->model_status_msg[st->model_index].empty()) {
                all.push_back(text(" " + st->model_status_msg[st->model_index]) | color(Color::Yellow));
            }
            all.push_back(text(" 模型未就绪") | color(Color::Red));
            all.push_back(download_btn->Render() | center);
        } else if (st->model_index < 3 && st->model_downloaded[st->model_index]) {
            std::string dmsg = st->download_msg;
            all.push_back(text(" " + (dmsg.empty() ? "已下载" : dmsg)) | color(Color::Green));
        }

        // LoRA
        all.push_back(separator());
        all.push_back(text(" LoRA 设置") | bold | color(Color::Yellow));
        all.push_back(lora_checkbox->Render());
        if (st->use_lora) {
            all.push_back(hbox({text(" 权重: "), lora_weight_slider->Render() | flex,
                               text(" " + std::to_string(st->lora_weight).substr(0, 4))}));
            all.push_back(hbox({text(" 文件: "), lora_path_input->Render() | flex}));
            all.push_back(browse_lora_btn->Render());
        }

        // Parameters
        all.push_back(separator());
        all.push_back(text(" 参数设置") | bold | color(Color::Green));
        all.push_back(hbox({steps_slider->Render() | flex, text(" " + std::to_string(st->steps))}));
        all.push_back(hbox({cfg_slider->Render() | flex, text(" " + std::to_string(st->cfg).substr(0, 3))}));
        all.push_back(hbox({text(" 种子: "), seed_input->Render() | size(WIDTH, LESS_THAN, 14)}));
        all.push_back(hbox({text(" 尺寸: "), size_menu->Render()}));

        // Prompt (with multiline support for long text)
        all.push_back(separator());
        all.push_back(text(" 提示词") | bold | color(Color::Magenta));
        all.push_back(text("  (需要英文！PixArt/SD3 不支持中文)") | dim | color(Color::Yellow));
        all.push_back(text("  格式参考: 主体描述 + 风格 + 场景 + 质量词") | dim);
        all.push_back(text("  示例: a cute cat, anime style, cherry blossom garden, masterpiece, best quality") | dim | color(Color::Grey50));
        all.push_back(prompt_input->Render() | size(WIDTH, GREATER_THAN, 50));
        // Show prompt preview if long
        if (st->prompt.size() > 60) {
            all.push_back(paragraph(st->prompt) | dim | color(Color::Grey50));
        }

        // Output path
        all.push_back(separator());
        all.push_back(text(" 保存位置") | bold | color(Color::Blue));
        all.push_back(hbox({text(" 路径: "), output_dir_input->Render() | flex}));
        all.push_back(browse_out_btn->Render());

        // Generate button / status
        all.push_back(separator());
        if (st->generating) {
            // Status shown below in progress section (no duplicate here)
        } else {
            all.push_back(generate_btn->Render() | center);
        }

        // Progress / Loading / Result indicator
        int cur = st->prog_cur.load();
        int tot = st->prog_total.load();
        if (st->generating || cur > 0) {
            all.push_back(separator());
            std::string sm;
            {
                std::lock_guard<std::mutex> lk(st->state_mutex);
                sm = st->status_msg;
            }
            bool isLoading = (cur == 0 && st->generating);
            if (isLoading) {
                all.push_back(text(" " + sm) | color(Color::Yellow) | center);
            } else {
                float pct = (tot > 0) ? (float)cur / tot : 0.0f;
                int bar_w = 40;
                int filled = (int)(pct * bar_w);
                std::string bar = std::string(filled, '#') + std::string(bar_w - filled, '-');
                all.push_back(text(" " + sm + "  [" + bar + "]") | center);
            }
        }

        // Result display (mutex-protected strings)
        {
            std::lock_guard<std::mutex> lk(st->state_mutex);
            if (!st->out_path.empty()) {
                all.push_back(separator());
                all.push_back(text(" 完成! " + st->result_info) | color(Color::Green) | bold);
                all.push_back(text(" 保存至: " + st->out_path) | color(Color::Yellow));
            } else if (!st->result_info.empty()) {
                all.push_back(separator());
                bool is_fail = (st->result_info.find("失败") != std::string::npos 
                             || st->result_info.find("错误") != std::string::npos);
                all.push_back(text(" " + st->result_info) | color(is_fail ? Color::Red : Color::Green) | bold);
            }
        }

        return vbox(std::move(all)) | yframe | flex;
    });
}
