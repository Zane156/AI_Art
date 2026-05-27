#include "agent_panel.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/dom/elements.hpp"
#include "backend_ipc.hpp"
#include "json_util.hpp"
#include <memory>
#include <string>
#include <deque>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <thread>
#include <chrono>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace ftxui;

static bool PasteFromClipboard(std::string& target) {
    if (!OpenClipboard(nullptr)) return false;
    HANDLE h = GetClipboardData(CF_UNICODETEXT);
    if (!h) { CloseClipboard(); return false; }
    wchar_t* w = (wchar_t*)GlobalLock(h);
    if (!w) { CloseClipboard(); return false; }
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (len > 1) {
        char* buf = (char*)malloc(len);
        WideCharToMultiByte(CP_UTF8, 0, w, -1, buf, len, nullptr, nullptr);
        target += buf;
        free(buf);
    }
    GlobalUnlock(h);
    CloseClipboard();
    return true;
}

struct AState {
    std::deque<ChatEntry> chat_history;
    std::string chat_input_text;
    std::string api_key_text;
    std::mutex chat_mutex;

    bool waiting_reply = false;
    bool api_tested = false;
    bool testing_api = false;
    std::string test_result;

    int scroll_offset = 0;
    static constexpr int MAX_VISIBLE = 10;
    bool user_scrolled = false;
    int last_size = 0;
};

static void TestAPIKey(
    std::shared_ptr<BackendIPC> backend,
    std::shared_ptr<AState> st,
    std::function<void()> on_ui_refresh)
{
    st->testing_api = true;
    st->test_result = "测试中...";
    on_ui_refresh();

    std::thread([st, backend, on_ui_refresh] {
        std::string key = st->api_key_text;
        while (!key.empty() && (key.back() == ' ' || key.back() == '\n' || key.back() == '\r')) key.pop_back();
        while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) key.erase(0, 1);

        if (key.size() < 15 || key.rfind("sk-", 0) != 0) {
            st->test_result = "密钥格式错误 (需 sk- 开头)";
            st->testing_api = false;
            on_ui_refresh();
            return;
        }

        std::ostringstream body;
        body << "{";
        JsonPair(body, "message", "ping");
        body << ",";
        JsonPair(body, "api_key", key);
        body << ",";
        JsonPair(body, "provider", "deepseek");
        body << ",";
        JsonPair(body, "screen", "agent");
        body << ",";
        JsonPair(body, "status", "testing");
        body << "}";

        std::string resp = backend->ChatWithAgent(body.str());

        if (resp.empty()) {
            st->test_result = "后端无响应 - 请检查后端是否启动";
            st->api_tested = false;
        } else {
            std::string succ = JsonGet(resp, "success");
            if (succ == "true" || succ == "True") {
                st->test_result = "DeepSeek API 验证通过";
                st->api_tested = true;
                backend->SetApiKey(key);
            } else {
                std::string reply = JsonGet(resp, "reply");
                std::string err = JsonGet(resp, "error");
                st->test_result = reply.empty() ? (err.empty() ? "验证失败" : err) : reply;
                st->api_tested = false;
            }
        }

        st->testing_api = false;
        on_ui_refresh();
    }).detach();
}

static void SendAgentMessage(
    std::shared_ptr<BackendIPC> backend,
    std::shared_ptr<AState> st,
    const std::string& user_msg,
    std::function<std::string()> get_current_screen,
    std::function<std::string()> get_status_text,
    std::function<void()> on_ui_refresh)
{
    std::string key = st->api_key_text;
    while (!key.empty() && (key.back() == ' ' || key.back() == '\n' || key.back() == '\r')) key.pop_back();
    while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) key.erase(0, 1);

    std::ostringstream body;
    body << "{";
    JsonPair(body, "message", user_msg);
    body << ",";
    JsonPair(body, "api_key", key);
    body << ",";
    JsonPair(body, "provider", "deepseek");
    body << ",";
    JsonPair(body, "screen", get_current_screen());
    body << ",";
    JsonPair(body, "status", get_status_text());
    body << "}";

    std::string response = backend->ChatWithAgent(body.str());

    std::string succ = JsonGet(response, "success");
    std::string reply = JsonGet(response, "reply");

    {
        std::lock_guard<std::mutex> lk(st->chat_mutex);
        if (succ == "true" || succ == "True") {
            st->chat_history.push_back({ChatEntry::AgentReply, reply, ""});
        } else {
            st->chat_history.push_back({ChatEntry::ErrorMsg,
                reply.empty() ? "无响应 - 请检查网络或 API Key" : reply, ""});
        }
        st->waiting_reply = false;
    }
    on_ui_refresh();
}

Component CreateAgentPanel(
    std::shared_ptr<BackendIPC> backend,
    std::function<std::string()> get_current_screen,
    std::function<std::string()> get_status_text,
    std::function<void()> on_ui_refresh)
{
    auto st = std::make_shared<AState>();

    st->api_key_text = backend->GetApiKey();

    {
        std::lock_guard<std::mutex> lk(st->chat_mutex);
        st->chat_history.push_back({ChatEntry::SystemMsg,
            "欢迎使用 AI Art Agent\n"
            "1. 输入 DeepSeek API Key (sk-...)\n"
            "2. 点击 [测试API] 验证密钥\n"
            "3. 验证通过后即可对话", ""});
    }

    // ---- Components ----
    auto chat_input = Input(&st->chat_input_text, "输入消息...");
    chat_input = CatchEvent(chat_input, [st](Event e) {
        if (e == Event::Character('\x16')) { PasteFromClipboard(st->chat_input_text); return true; }
        return false;
    });

    auto api_input = Input(&st->api_key_text, "sk-...");
    api_input = CatchEvent(api_input, [st](Event e) {
        if (e == Event::Character('\x16')) { PasteFromClipboard(st->api_key_text); return true; }
        return false;
    });

    auto test_btn = Button("测试API", [st, backend, on_ui_refresh] {
        if (st->testing_api) return;
        TestAPIKey(backend, st, on_ui_refresh);
    });

    auto clear_key_btn = Button("清空密钥", [st, on_ui_refresh, backend] {
        st->api_key_text.clear();
        st->api_tested = false;
        st->test_result.clear();
        backend->SetApiKey("");
        on_ui_refresh();
    });

    auto clear_chat_btn = Button("清空聊天", [st, on_ui_refresh] {
        std::lock_guard<std::mutex> lk(st->chat_mutex);
        st->chat_history.clear();
        st->chat_history.push_back({ChatEntry::SystemMsg,
            "已清空。请先验证 API Key。", ""});
        st->scroll_offset = 0;
        st->user_scrolled = false;
        on_ui_refresh();
    });

    auto send_btn = Button("发送", [st, backend, get_current_screen, get_status_text, on_ui_refresh] {
        if (st->chat_input_text.empty()) return;
        if (st->waiting_reply) return;
        if (!st->api_tested) {
            std::lock_guard<std::mutex> lk(st->chat_mutex);
            st->chat_history.push_back({ChatEntry::SystemMsg,
                "请先测试 API Key 通过后再发送消息。", ""});
            on_ui_refresh();
            return;
        }

        std::string msg = std::move(st->chat_input_text);
        st->chat_input_text.clear();

        {
            std::lock_guard<std::mutex> lk(st->chat_mutex);
            st->chat_history.push_back({ChatEntry::UserMsg, msg, ""});
            st->waiting_reply = true;
        }
        on_ui_refresh();

        std::thread([st, backend, msg, get_current_screen, get_status_text, on_ui_refresh] {
            SendAgentMessage(backend, st, msg, get_current_screen, get_status_text, on_ui_refresh);
        }).detach();
    });

    auto btn_scroll_up = Button("^", [st, on_ui_refresh] {
        int total = (int)st->chat_history.size();
        int max_offset = std::max(0, total - AState::MAX_VISIBLE);
        st->scroll_offset = std::min(st->scroll_offset + 5, max_offset);
        st->user_scrolled = (st->scroll_offset > 0);
        on_ui_refresh();
    });

    auto btn_scroll_dn = Button("v", [st, on_ui_refresh] {
        st->scroll_offset = std::max(st->scroll_offset - 5, 0);
        if (st->scroll_offset == 0) st->user_scrolled = false;
        on_ui_refresh();
    });

    // Container: ALL interactive components must be here for events to work
    auto container = Container::Vertical({
        Container::Horizontal({ test_btn, clear_key_btn, clear_chat_btn }),
        api_input,
        chat_input,
        Container::Horizontal({ send_btn, btn_scroll_up, btn_scroll_dn }),
    });

    return Renderer(container, [st,
        chat_input, send_btn, api_input,
        test_btn, clear_key_btn, clear_chat_btn,
        btn_scroll_up, btn_scroll_dn
    ] {
        Elements all;

        // ===== TITLE BAR =====
        all.push_back(hbox({
            text(" Agent ") | bold | color(Color::Cyan),
            text(" ") | flex,
            clear_chat_btn->Render() | size(WIDTH, EQUAL, 10),
        }));
        all.push_back(separator());

        // ===== API KEY ROW =====
        all.push_back(hbox({
            text(" 密钥: ") | dim,
            api_input->Render() | flex,
            separator(),
            test_btn->Render(),
            separator(),
            clear_key_btn->Render(),
        }));

        // Test result
        if (!st->test_result.empty()) {
            all.push_back(separator());
            auto c = st->api_tested ? Color::Green : (st->testing_api ? Color::Yellow : Color::Red);
            all.push_back(hbox({
                text("  ") | color(c) | bold,
                text(st->test_result) | color(c),
            }));
        }

        all.push_back(separator());

        // ===== CHAT AREA =====
        {
            std::lock_guard<std::mutex> lk(st->chat_mutex);
            int total = (int)st->chat_history.size();

            if (total > st->last_size && !st->user_scrolled) {
                st->scroll_offset = 0;
            }
            st->last_size = total;

            int max_offset = std::max(0, total - AState::MAX_VISIBLE);
            if (st->scroll_offset > max_offset) st->scroll_offset = max_offset;
            if (st->scroll_offset < 0) st->scroll_offset = 0;

            int first_visible = max_offset - st->scroll_offset;
            int last_visible  = std::min(first_visible + AState::MAX_VISIBLE, total);

            if (total == 0) {
                all.push_back(text(" 暂无消息") | dim | center);
            } else {
                // Scroll bar
                if (total > AState::MAX_VISIBLE) {
                    all.push_back(hbox({
                        text(" [" + std::to_string(first_visible + 1) + "-"
                             + std::to_string(last_visible) + "/" + std::to_string(total) + "]") | dim,
                        separator(),
                        btn_scroll_up->Render(),
                        btn_scroll_dn->Render(),
                    }));
                    all.push_back(separator());
                }

                // Render visible messages
                for (int i = last_visible - 1; i >= first_visible; i--) {
                    auto& e = st->chat_history[i];
                    switch (e.type) {
                    case ChatEntry::UserMsg:
                        all.push_back(hbox({
                            text(" 你 ") | bold | color(Color::YellowLight),
                        }));
                        all.push_back(paragraph(e.text));
                        all.push_back(separator());
                        break;
                    case ChatEntry::AgentReply:
                        all.push_back(hbox({
                            text(" AI ") | bold | color(Color::CyanLight),
                        }));
                        all.push_back(paragraph(e.text));
                        all.push_back(separator());
                        break;
                    case ChatEntry::ToolCall:
                        all.push_back(hbox({
                            text(" >> ") | bold | color(Color::Cyan),
                            text(e.detail) | bold,
                        }));
                        if (!e.text.empty())
                            all.push_back(paragraph(e.text) | dim);
                        all.push_back(separator());
                        break;
                    case ChatEntry::ToolResult:
                        all.push_back(hbox({
                            text(" OK ") | color(Color::Green),
                            text(e.text) | dim,
                        }));
                        all.push_back(separator());
                        break;
                    case ChatEntry::ErrorMsg:
                        all.push_back(paragraph(e.text) | color(Color::Red));
                        all.push_back(separator());
                        break;
                    case ChatEntry::SystemMsg:
                        all.push_back(paragraph(e.text) | dim | color(Color::Grey50));
                        all.push_back(separator());
                        break;
                    }
                }
            }
        }

        // ===== FOOTER =====
        all.push_back(separator());
        if (st->waiting_reply) {
            all.push_back(hbox({
                text(" AI 思考中...") | dim | color(Color::Yellow) | blink,
            }));
            all.push_back(separator());
        }
        all.push_back(hbox({
            chat_input->Render() | flex,
            separator(),
            send_btn->Render() | size(WIDTH, EQUAL, 6),
        }));

        return vbox(all) | flex | borderEmpty;
    });
}
