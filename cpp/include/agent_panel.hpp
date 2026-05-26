#pragma once
#include "ftxui/component/component.hpp"
#include "backend_ipc.hpp"
#include <functional>
#include <vector>
#include <string>

struct ChatEntry {
    enum Type { UserMsg, AgentReply, ToolCall, ToolResult, ErrorMsg, SystemMsg };
    Type type;
    std::string text;
    std::string detail;
};

ftxui::Component CreateAgentPanel(
    std::shared_ptr<BackendIPC> backend,
    std::function<std::string()> get_current_screen,
    std::function<std::string()> get_status_text,
    std::function<void()> on_ui_refresh);
