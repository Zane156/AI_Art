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

// ====================================================================
// Ctrl+V clipboard paste
// ====================================================================
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

// ====================================================================
// State
// ====================================================================
struct AState {
    std::deque<ChatEntry> chat_history;
    std::string chat_input_text;
    std::string api_key_text;
    std::mutex chat_mutex;

    bool waiting_reply = false;
    bool api_tested = false;
    bool testing_api = false;
    std::string test_result;

    // Scroll state: offset from bottom (0 = show newest)
    int scroll_offset = 0;
    static constexpr int MAX_VISIBLE = 12;
    bool user_scrolled = false;  // true if user manually scrolled up
    int last_size = 0;           // track for auto-scroll on new messages
};

// ====================================================================
// Test DeepSeek API key
// ====================================================================
static void TestAPIKey(
    std::shared_ptr<BackendIPC> backend,
    std::shared_ptr<AState> st,
    std::function<void()> on_ui_refresh)
{
    st->testing_api = true;
    st->test_result = "  测试中...";
    on_ui_refresh();

    std::thread([st, backend, on_ui_refresh] {
        std::string key = st->api_key_text;
        while (!key.empty() && (key.back() == ' ' || key.back() == '\n' || key.back() == '\r')) key.pop_back();
        while (!key.empty() && (key.front() == ' ' || key.front() == '\t')) key.erase(0, 1);

        if (key.size() < 15 || key.find("sk-") != 0) {
            st->test_result = "  x 密钥格式错误 (需要 sk- 开头)";
            st->testing_api = false;
            on_ui_refresh();
            return;
        }

        std::ostringstream body;
        body << "{";
        JsonPair(body, "message", "test");
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
        std::string succ = JsonGet(resp, "success");

        if (succ == "true" || succ == "True") {
            st->test_result = "  v DeepSeek API 验证通过";
            st->api_tested = true;
            backend->SetApiKey(key);
        } else {
            std::string err = JsonGet(resp, "reply");
            if (err.empty()) err = JsonGet(resp, "error");
            st->test_result = "  x " + (err.empty() ? "验证失败" : err);
            st->api_tested = false;
        }

        st->testing_api = false;
        on_ui_refresh();
    }).detach();
}

// ====================================================================
// Send agent message via backend
// ====================================================================
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

// ====================================================================
// CreateAgentPanel
// ====================================================================
Component CreateAgentPanel(
    std::shared_ptr<BackendIPC> backend,
    std::function<std::string()> get_current_screen,
    std::function<std::string()> get_status_text,
    std::function<void()> on_ui_refresh)
{
    auto st = std::make_shared<AState>();

    // Load saved key
    st->api_key_text = backend->GetApiKey();

    // Welcome message
    {
        std::lock_guard<std::mutex> lk(st->chat_mutex);
        st->chat_history.push_back({ChatEntry::SystemMsg,
            "欢迎使用 AI Art Agent\n"
            "1. 输入 DeepSeek API Key (sk-...)\n"
            "2. 点击 [ 测试API ] 验证密钥\n"
            "3. 验证通过后即可开始对话", ""});
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

    auto test_btn = Button(" 测试API ", [st, backend, on_ui_refresh] {
        if (st->testing_api) return;
        TestAPIKey(backend, st, on_ui_refresh);
    });

    auto save_key_btn = Button(" 保存密钥 ", [st, backend] {
        backend->SetApiKey(st->api_key_text);
    });

    auto clear_btn = Button(" 清空聊天 ", [st, on_ui_refresh] {
        std::lock_guard<std::mutex> lk(st->chat_mutex);
        st->chat_history.clear();
        st->chat_history.push_back({ChatEntry::SystemMsg,
            "已清空。请先确保 API Key 已验证。", ""});
        on_ui_refresh();
    });

    auto send_btn = Button(" 发送 ", [st, backend, get_current_screen, get_status_text, on_ui_refresh] {
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

    auto btn_scroll_up = Button(" ^ ", [st, on_ui_refresh] {
        int total = (int)st->chat_history.size();
        int max_offset = std::max(0, total - AState::MAX_VISIBLE);
        st->scroll_offset = std::min(st->scroll_offset + 5, max_offset);
        st->user_scrolled = (st->scroll_offset > 0);
        on_ui_refresh();
    });

    auto btn_scroll_dn = Button(" v ", [st, on_ui_refresh] {
        st->scroll_offset = std::max(st->scroll_offset - 5, 0);
        if (st->scroll_offset == 0) st->user_scrolled = false;
        on_ui_refresh();
    });

    // Container includes ALL interactive components
    auto container = Container::Vertical({
        Container::Horizontal({ test_btn, save_key_btn, clear_btn, btn_scroll_up, btn_scroll_dn }),
        api_input,
        chat_input,
        send_btn,
    });

    return Renderer(container, [st,
        chat_input, send_btn, api_input,
        test_btn, save_key_btn, clear_btn,
        btn_scroll_up, btn_scroll_dn
    ] {
        Elements header, chat, footer;

        // ===== HEADER =====
        header.push_back(hbox({
            text(" Agent") | bold | color(Color::Cyan),
            text(" ") | flex,
            clear_btn->Render(),
        }));
        header.push_back(separator());

        // API key input row
        header.push_back(hbox({
            text(" Key: ") | dim,
            api_input->Render() | flex,
            test_btn->Render(),
        }));
        header.push_back(hbox({
            save_key_btn->Render(),
        }));

        // Test result
        if (!st->test_result.empty()) {
            auto c = st->api_tested ? Color::Green : (st->testing_api ? Color::Yellow : Color::Red);
            header.push_back(hbox({
                text(st->test_result) | color(c),
                text(" ") | flex,
                clear_btn->Render(),
            }));
        }

        // ===== CHAT (offset-based scrolling, newest-first) =====
        {
            std::lock_guard<std::mutex> lk(st->chat_mutex);
            int total = (int)st->chat_history.size();

            // Auto-scroll to bottom on new messages (unless user scrolled up)
            if (total > st->last_size && !st->user_scrolled) {
                st->scroll_offset = 0;
            }
            st->last_size = total;

            // Clamp offset
            int max_offset = std::max(0, total - AState::MAX_VISIBLE);
            if (st->scroll_offset > max_offset) st->scroll_offset = max_offset;
            if (st->scroll_offset < 0) st->scroll_offset = 0;

            int first_visible = max_offset - st->scroll_offset;
            int last_visible = std::min(first_visible + AState::MAX_VISIBLE, total);

            if (total == 0) {
                chat.push_back(text(" 暂无消息") | dim | center);
            } else {
                // Scroll indicator bar
                if (st->scroll_offset > 0 || total > AState::MAX_VISIBLE) {
                    std::string indicator = "  [消息 " + std::to_string(first_visible + 1)
                        + "-" + std::to_string(last_visible) + " / " + std::to_string(total) + "]";
                    if (st->scroll_offset > 0) {
                        indicator += "  ^ 往上翻";
                    }
                    if (first_visible + AState::MAX_VISIBLE < total) {
                        indicator += "  v 往下翻";
                    }
                    chat.push_back(hbox({
                        text(indicator) | dim | color(Color::Grey50),
                        text(" ") | flex,
                        btn_scroll_up->Render(),
                        btn_scroll_dn->Render(),
                    }));
                    chat.push_back(separator());
                }

                // Render only visible messages (newest-first within window)
                for (int i = last_visible - 1; i >= first_visible; i--) {
                    auto& e = st->chat_history[i];
                    switch (e.type) {
                    case ChatEntry::UserMsg:
                        chat.push_back(separator());
                        chat.push_back(hbox({
                            text("  你") | bold | color(Color::YellowLight),
                        }));
                        chat.push_back(paragraph(e.text));
                        break;
                    case ChatEntry::AgentReply:
                        chat.push_back(separator());
                        chat.push_back(hbox({
                            text("  AI") | bold | color(Color::CyanLight),
                        }));
                        chat.push_back(paragraph(e.text));
                        break;
                    case ChatEntry::ToolCall:
                        chat.push_back(hbox({
                            text("  >> ") | bold | color(Color::Cyan),
                            text(e.detail) | bold,
                        }));
                        if (!e.text.empty())
                            chat.push_back(paragraph(e.text) | dim);
                        break;
                    case ChatEntry::ToolResult:
                        chat.push_back(hbox({
                            text("  OK ") | color(Color::Green),
                            text(e.text) | dim,
                        }));
                        break;
                    case ChatEntry::ErrorMsg:
                        chat.push_back(separator());
                        chat.push_back(paragraph(e.text) | color(Color::Red));
                        break;
                    case ChatEntry::SystemMsg:
                        chat.push_back(separator());
                        chat.push_back(paragraph(e.text) | dim | color(Color::Grey50));
                        break;
                    }
                }
            }
        }

        // ===== FOOTER =====
        if (st->waiting_reply) {
            footer.push_back(separator());
            footer.push_back(hbox({
                text("  AI 思考中") | dim | color(Color::Yellow),
                text("...") | blink,
            }));
        }
        footer.push_back(separator());
        footer.push_back(hbox({
            chat_input->Render() | flex,
            send_btn->Render(),
        }));

        return vbox({
            vbox(std::move(header)),
            separator(),
            vbox(std::move(chat)) | flex,
            separator(),
            vbox(std::move(footer)),
        }) | flex | borderEmpty;
    });
}
