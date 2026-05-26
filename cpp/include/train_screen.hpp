#pragma once
#include "ftxui/component/component.hpp"
#include "backend_ipc.hpp"

ftxui::Component CreateTrainScreen(
    std::shared_ptr<BackendIPC> backend,
    std::function<void()> on_switch_to_generate,
    std::function<void(const std::string&)> on_status_change,
    std::function<void()> on_ui_refresh);
