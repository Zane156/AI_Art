#include "train_screen.hpp"
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

using namespace ftxui;

struct TState {
    int base_model_idx = 0;
    std::vector<std::string> base_models = {
        " PixArt-Sigma       [显存: 8GB+]  ",
        " SD3-Medium         [显存: 11GB+] ",
        " SD3.5-Large        [显存: 22GB+] ",
    };
    std::string data_dir;
    std::string output_dir = "outputs/lora";
    int train_steps = 500;
    float lr = 1e-4f;
    int lora_rank = 16;
    int batch_size = 1;

    // Training state
    bool is_training = false;
    std::string task_id;
    std::atomic<int> prog_cur{0};
    std::atomic<int> prog_total{0};
    std::atomic<bool> poll_active{false};
    std::mutex state_mutex;
    std::string status_msg;
    std::string result_info;

    // Dataset info
    int dataset_images = 0;
    int dataset_labeled = 0;
    bool dataset_scanned = false;
};

Component CreateTrainScreen(
    std::shared_ptr<BackendIPC> backend,
    std::function<void()> on_switch_to_generate,
    std::function<void(const std::string&)> on_status_change,
    std::function<void()> on_ui_refresh)
{
    auto st = std::make_shared<TState>();

    // ---- Components ----
    auto model_menu       = Menu(&st->base_models, &st->base_model_idx);
    auto data_dir_input   = Input(&st->data_dir, "选择训练数据文件夹...");
    auto output_dir_input = Input(&st->output_dir, "选择模型保存路径...");
    auto browse_data_btn  = Button(" [浏览数据文件夹...] ", [st] {
        std::string f = FileDialog::OpenFolder("选择训练数据文件夹");
        if (!f.empty()) { st->data_dir = f; st->dataset_scanned = false; }
    });
    auto browse_out_btn   = Button(" [浏览保存路径...] ", [st] {
        std::string f = FileDialog::OpenFolder("选择模型保存路径");
        if (!f.empty()) st->output_dir = f;
    });

    // Scan dataset
    auto scan_data_btn = Button(" [ 扫描数据 ] ", [st, backend, on_ui_refresh] {
        if (st->data_dir.empty()) return;
        std::ostringstream json;
        json << "{";
        JsonPair(json, "data_dir", st->data_dir);
        json << "}";
        std::string resp = backend->HttpPost("/scan_dataset", json.str());
        auto imgs = JsonGet(resp, "total_images");
        auto labeled = JsonGet(resp, "labeled");
        if (!imgs.empty()) st->dataset_images = std::atoi(imgs.c_str());
        if (!labeled.empty()) st->dataset_labeled = std::atoi(labeled.c_str());
        st->dataset_scanned = true;
        on_ui_refresh();
    });

    // Auto-label
    auto auto_label_btn = Button(" [ 自动打标 ] ", [st, backend, on_ui_refresh] {
        if (st->data_dir.empty()) return;
        std::ostringstream json;
        json << "{";
        JsonPair(json, "data_dir", st->data_dir);
        json << ",";
        JsonPair(json, "method", "simple");
        json << "}";
        std::string resp = backend->HttpPost("/auto_label", json.str());
        auto labeled = JsonGet(resp, "labeled");
        st->dataset_labeled += labeled.empty() ? 0 : std::atoi(labeled.c_str());
        st->dataset_scanned = true;
        on_ui_refresh();
    });

    auto steps_slider = Slider("训练步数:", &st->train_steps, 100, 5000, 100);
    auto lr_slider    = Slider("学习率:", &st->lr, 1e-5f, 1e-3f, 1e-5f);
    auto rank_slider  = Slider("LoRA Rank:", &st->lora_rank, 8, 128, 8);
    auto batch_slider = Slider("批次大小:", &st->batch_size, 1, 4, 1);

    // Start training
    auto start_btn = Button(" [ 开始训练 ] ", [st, backend, on_status_change, on_ui_refresh] {
        if (st->is_training) return;
        if (st->data_dir.empty()) {
            st->result_info = "请先选择训练数据文件夹";
            return;
        }
        if (st->dataset_images < 5) {
            st->result_info = "图片不足 (需要至少5张)，请扫描数据确认";
            return;
        }

        st->is_training = true;
        st->prog_cur = 0;
        st->prog_total = st->train_steps;
        {
            std::lock_guard<std::mutex> lk(st->state_mutex);
            st->status_msg = "启动训练...";
            st->result_info = "";
        }
        on_status_change("训练中...");

        std::string mn = (st->base_model_idx == 0) ? "PixArt-Sigma"
            : (st->base_model_idx == 1) ? "SD3-Medium" : "SD3.5-Large";

        std::ostringstream json;
        json << std::fixed << std::setprecision(6) << "{";
        JsonPair(json, "base_model", mn);          json << ",";
        JsonPair(json, "data_dir", st->data_dir);  json << ",";
        JsonPair(json, "output_dir", st->output_dir); json << ",";
        JsonPairNum(json, "steps", st->train_steps);   json << ",";
        JsonPairNum(json, "learning_rate", st->lr);    json << ",";
        JsonPairNum(json, "rank", st->lora_rank);      json << ",";
        JsonPairNum(json, "batch_size", st->batch_size);
        json << "}";

        std::string resp = backend->TrainLoRA(json.str());
        st->task_id = JsonGet(resp, "task_id");

        if (st->task_id.empty()) {
            st->is_training = false;
            auto err = JsonGet(resp, "error_msg");
            st->result_info = err.empty() ? "训练启动失败" : err;
            {
                std::lock_guard<std::mutex> lk(st->state_mutex);
                st->status_msg = "失败";
            }
            on_status_change("训练失败: " + st->result_info);
            return;
        }

        {
            std::lock_guard<std::mutex> lk(st->state_mutex);
            st->status_msg = "训练初始化中...";
        }
        st->poll_active = true;

        // Poll progress in background
        std::thread([st, backend, on_status_change, on_ui_refresh]() {
            while (st->poll_active) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                if (!st->poll_active) break;

                auto s = backend->CheckTaskStatus(st->task_id);
                auto step = JsonGet(s, "step");
                auto total = JsonGet(s, "total");
                auto msg = JsonGet(s, "message");

                if (!step.empty()) st->prog_cur = std::atoi(step.c_str());
                if (!total.empty()) st->prog_total = std::atoi(total.c_str());

                {
                    std::lock_guard<std::mutex> lk(st->state_mutex);
                    if (!msg.empty()) st->status_msg = JsonUnescape(msg);
                }
                on_ui_refresh();

                bool done = (s.find("\"done\":true") != std::string::npos);
                if (done) {
                    auto r = backend->GetTaskResult(st->task_id);
                    st->poll_active = false;
                    st->is_training = false;

                    if (r.find("\"success\":true") != std::string::npos) {
                        auto out = JsonGet(r, "output_dir");
                        auto loss = JsonGet(r, "loss");
                        auto ds = JsonGet(r, "dataset_size");
                        auto elapsed = JsonGet(r, "elapsed");
                        std::string info = "训练完成!";
                        if (!loss.empty()) info += " Loss: " + loss;
                        if (!elapsed.empty()) info += " | " + elapsed + "秒";
                        if (!out.empty()) info += " | 输出: " + out;
                        st->result_info = info;
                        on_status_change("训练完成!");
                    } else {
                        auto err = JsonGet(r, "error_msg");
                        st->result_info = err.empty() ? "训练失败" : err;
                        on_status_change("训练失败");
                        {
                            std::lock_guard<std::mutex> lk(st->state_mutex);
                            st->status_msg = "失败";
                        }
                    }
                    on_ui_refresh();
                    break;
                }
            }
        }).detach();
    });

    auto go_gen_btn = Button(" [ 切换到图像生成 ] ", on_switch_to_generate);

    auto container = Container::Vertical({
        model_menu, data_dir_input, browse_data_btn, scan_data_btn, auto_label_btn,
        output_dir_input, browse_out_btn,
        steps_slider, lr_slider, rank_slider, batch_slider,
        start_btn, go_gen_btn,
    });

    return Renderer(container, [st,
        model_menu, data_dir_input, browse_data_btn, scan_data_btn, auto_label_btn,
        output_dir_input, browse_out_btn,
        steps_slider, lr_slider, rank_slider, batch_slider,
        start_btn, go_gen_btn
    ] {
        Elements all;

        all.push_back(text(" LoRA 模型训练") | bold | center | color(Color::Cyan));
        all.push_back(separator());

        // === Model selection ===
        all.push_back(text(" 底膜选择") | bold | color(Color::Cyan));
        all.push_back(model_menu->Render());
        all.push_back(text(" 推荐: PixArt-Sigma (刚好8GB显存)") | dim);

        // === Data preparation ===
        all.push_back(separator());
        all.push_back(text(" 数据准备") | bold | color(Color::Yellow));
        all.push_back(hbox({text(" 数据路径: "), data_dir_input->Render() | flex}));
        all.push_back(browse_data_btn->Render());
        all.push_back(hbox({
            scan_data_btn->Render(),
            text(" "),
            auto_label_btn->Render(),
        }));

        // Dataset status
        if (st->dataset_scanned) {
            bool enough = st->dataset_images >= 5;
            auto stats_color = enough ? Color::Green : Color::Red;
            all.push_back(hbox({
                text(" ") | size(WIDTH, EQUAL, 2),
                text(std::to_string(st->dataset_images) + " 张图片") | color(stats_color),
                text(" | "),
                text(std::to_string(st->dataset_labeled) + " 个标注") | color(stats_color),
                text(" | "),
                text(enough ? "足够训练" : "需要≥5张") | color(stats_color) | bold,
            }));
        }

        all.push_back(text(""));
        all.push_back(text(" 标注格式: 与图片同名的 .txt 文件") | dim);
        all.push_back(text(" 自动打标: 生成通用标签 (可自己修改)") | dim);

        // === Output path ===
        all.push_back(hbox({text(" 保存路径: "), output_dir_input->Render() | flex}));
        all.push_back(browse_out_btn->Render());

        // === Hyperparameters ===
        all.push_back(separator());
        all.push_back(text(" 超参数") | bold | color(Color::Green));
        all.push_back(text(" 推荐: 500步 / lr=1e-4 / Rank=16 / Batch=1") | dim);
        all.push_back(hbox({steps_slider->Render() | flex, text(" " + std::to_string(st->train_steps))}));
        all.push_back(hbox({lr_slider->Render() | flex, text(" " + std::to_string(st->lr).substr(0, 7))}));
        all.push_back(hbox({rank_slider->Render() | flex, text(" " + std::to_string(st->lora_rank))}));
        all.push_back(hbox({batch_slider->Render() | flex, text(" " + std::to_string(st->batch_size))}));

        // === Training button / progress ===
        all.push_back(separator());
        if (st->is_training) {
            int cur = st->prog_cur.load();
            int tot = st->prog_total.load();
            std::string sm;
            {
                std::lock_guard<std::mutex> lk(st->state_mutex);
                sm = st->status_msg;
            }
            float pct = (tot > 0) ? (float)cur / tot : 0.0f;
            int bar_w = 40;
            int filled = (int)(pct * bar_w);
            std::string bar = std::string(filled, '#') + std::string(bar_w - filled, '-');
            all.push_back(text(" " + sm) | color(Color::Yellow) | center);
            all.push_back(text(" [" + bar + "] " + std::to_string(cur) + "/" + std::to_string(tot)) | center);
        } else {
            all.push_back(start_btn->Render() | center);
        }

        // === Result ===
        if (!st->result_info.empty()) {
            all.push_back(separator());
            bool ok = st->result_info.find("完成") != std::string::npos;
            all.push_back(text(" " + st->result_info) | color(ok ? Color::Green : Color::Red) | bold);
            if (ok) {
                all.push_back(text(" 可在图像生成页面加载 LoRA") | color(Color::Cyan) | bold);
            }
            all.push_back(go_gen_btn->Render() | center);
        }

        return vbox(std::move(all)) | yframe | flex;
    });
}
