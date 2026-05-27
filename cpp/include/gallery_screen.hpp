#pragma once
#include "ftxui/component/component.hpp"
#include <memory>
#include <vector>
#include <string>
#include <filesystem>

struct GalleryEntry {
    std::string filename;
    std::string path;
    std::string date;
    uint64_t size_bytes = 0;
    int width = 0;
    int height = 0;
    bool is_png = false;
};

ftxui::Component CreateGalleryScreen(
    std::function<void()> on_switch_to_generate,
    std::function<void()> on_ui_refresh);
