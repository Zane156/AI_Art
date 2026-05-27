#include "gallery_screen.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/dom/elements.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using namespace ftxui;
namespace fs = std::filesystem;

// ─── PNG header reader ──────────────────────────────────────────────────────
static bool read_png_size(const std::string& path, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char sig[8];
    f.read(sig, 8);
    if (sig[0] != (char)0x89 || sig[1] != 'P' || sig[2] != 'N' || sig[3] != 'G') return false;
    // Skip IHDR length + "IHDR"
    f.seekg(16);
    unsigned char buf[8];
    f.read((char*)buf, 8);
    if (!f) return false;
    w = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    h = (buf[4] << 24) | (buf[5] << 16) | (buf[6] << 8) | buf[7];
    return (w > 0 && h > 0 && w < 65536 && h < 65536);
}

// ─── Format helpers ─────────────────────────────────────────────────────────
static std::string fmt_size(uint64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int i = 0;
    double s = (double)bytes;
    while (s >= 1024.0 && i < 3) { s /= 1024.0; i++; }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(i == 0 ? 0 : 1) << s << " " << units[i];
    return oss.str();
}

static std::string fmt_time(const std::string& iso) {
    // Convert ISO write time to readable format
    if (iso.length() < 19) return iso;
    // "2026-05-27 16:30:00"  → keep as-is or simplify
    return iso.substr(0, 16).replace(10, 1, " ");  // actually string replace is different
    // Simple: just return first 16 chars with space
}

// ─── Scan outputs directory ──────────────────────────────────────────────────
static std::vector<GalleryEntry> scan_outputs(const std::string& dir) {
    std::vector<GalleryEntry> entries;
    std::error_code ec;
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return entries;

    for (const auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        // Lowercase extension
        std::string ext_lower = ext;
        std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
        if (ext_lower != ".png" && ext_lower != ".jpg" && ext_lower != ".jpeg" && ext_lower != ".webp")
            continue;

        GalleryEntry ge;
        ge.path = entry.path().string();
        ge.filename = entry.path().filename().string();
        ge.size_bytes = entry.file_size(ec);
        ge.is_png = (ext_lower == ".png");

        // Read PNG dimensions
        if (ge.is_png) {
            read_png_size(ge.path, ge.width, ge.height);
        }

        // Format write time
        auto ftime = entry.last_write_time(ec);
        if (!ec) {
            auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            auto tt = std::chrono::system_clock::to_time_t(sctp);
            std::tm tm;
            localtime_s(&tm, &tt);
            std::ostringstream oss;
            oss << std::setw(2) << std::setfill('0') << tm.tm_mon + 1 << "-"
                << std::setw(2) << std::setfill('0') << tm.tm_mday << " "
                << std::setw(2) << std::setfill('0') << tm.tm_hour << ":"
                << std::setw(2) << std::setfill('0') << tm.tm_min;
            ge.date = oss.str();
        } else {
            ge.date = "???";
        }

        entries.push_back(ge);
    }

    // Sort by date descending (newest first)
    std::sort(entries.begin(), entries.end(), [](const GalleryEntry& a, const GalleryEntry& b) {
        return a.date > b.date;
    });

    return entries;
}

// ─── Main component ─────────────────────────────────────────────────────────
Component CreateGalleryScreen(
    std::function<void()> on_switch_to_generate,
    std::function<void()> on_ui_refresh)
{
    struct State {
        std::vector<GalleryEntry> entries;
        int selected = 0;
        int scroll_offset = 0;
        bool sort_by_name = false;
        std::string message;
        std::string outputs_dir = "outputs";
        bool scanning = true;
    };
    auto st = std::make_shared<State>();

    // Background scan
    std::thread([st, on_ui_refresh] {
        // Try a few common locations for outputs
        std::vector<std::string> dirs = {"outputs", "../outputs", "../../outputs"};
        for (const auto& d : dirs) {
            if (fs::exists(d)) { st->outputs_dir = d; break; }
        }
        st->entries = scan_outputs(st->outputs_dir);
        st->scanning = false;
        on_ui_refresh();
    }).detach();

    // Open file with system viewer
    auto open_file = [st] {
        if (st->selected < 0 || st->selected >= (int)st->entries.size()) return;
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", st->entries[st->selected].path.c_str(),
                      nullptr, nullptr, SW_SHOW);
#endif
    };

    // Delete selected file
    auto delete_file = [st, on_ui_refresh] {
        if (st->selected < 0 || st->selected >= (int)st->entries.size()) return;
        std::error_code ec;
        fs::remove(st->entries[st->selected].path, ec);
        if (ec) {
            st->message = "删除失败: " + ec.message();
        } else {
            st->message = "已删除: " + st->entries[st->selected].filename;
            st->entries = scan_outputs(st->outputs_dir);
            if (st->selected >= (int)st->entries.size() && !st->entries.empty())
                st->selected = (int)st->entries.size() - 1;
        }
        on_ui_refresh();
    };

    // Refresh file list
    auto refresh = [st, on_ui_refresh] {
        st->entries = scan_outputs(st->outputs_dir);
        st->selected = 0;
        st->scroll_offset = 0;
        st->message = "已刷新";
        on_ui_refresh();
    };

    // Toggle sort
    auto toggle_sort = [st, on_ui_refresh] {
        st->sort_by_name = !st->sort_by_name;
        if (st->sort_by_name) {
            std::sort(st->entries.begin(), st->entries.end(),
                [](const GalleryEntry& a, const GalleryEntry& b) { return a.filename < b.filename; });
        } else {
            std::sort(st->entries.begin(), st->entries.end(),
                [](const GalleryEntry& a, const GalleryEntry& b) { return a.date > b.date; });
        }
        st->selected = 0;
        st->scroll_offset = 0;
        st->message = st->sort_by_name ? "按名称排序" : "按日期排序";
        on_ui_refresh();
    };

    // Move selection
    auto move_up = [st] {
        if (st->entries.empty()) return;
        if (st->selected > 0) st->selected--;
    };
    auto move_down = [st] {
        if (st->entries.empty()) return;
        if (st->selected < (int)st->entries.size() - 1) st->selected++;
    };

    auto container = Container::Vertical({});

    return Renderer(container, [st, on_switch_to_generate, open_file, delete_file,
                                refresh, toggle_sort, move_up, move_down] {
        Elements all;

        // ═══ Header ═══
        auto title_bar = hbox({
            text(" 🖼  画廊 ") | bold | color(Color::Cyan),
            separator(),
            text(" [1] 生成  [2] 画廊  [Enter] 打开  [Del] 删除  [R] 刷新  [S] 排序  [Q] 退出 ") | dim,
        });
        all.push_back(title_bar);
        all.push_back(separator());

        // Scanning indicator
        if (st->scanning) {
            all.push_back(text(" 正在扫描图片...") | dim | center);
            return vbox(all) | border;
        }

        // Empty state
        if (st->entries.empty()) {
            all.push_back(separatorEmpty());
            all.push_back(text("  outputs/ 目录中暂无生成图片") | dim | center);
            all.push_back(text("  先生成图片后再来查看") | dim | center);
            all.push_back(text("  按 [R] 刷新 ") | center);
            all.push_back(separatorEmpty());
            if (!st->message.empty()) {
                all.push_back(separator());
                all.push_back(text(" " + st->message) | color(Color::Yellow));
            }
            return vbox(all) | border;
        }

        // Summary
        uint64_t total_bytes = 0;
        for (const auto& e : st->entries) total_bytes += e.size_bytes;
        auto summary = hbox({
            text(" 共 " + std::to_string(st->entries.size()) + " 张  |  " +
                 fmt_size(total_bytes) + "  |  " +
                 st->outputs_dir + " ") | dim,
        });
        all.push_back(summary);
        all.push_back(separator());
        all.push_back(text("")); // spacing

        // Column header
        auto header = hbox({
            text("  文件名") | bold | color(Color::Cyan) | size(WIDTH, EQUAL, 36),
            separator(),
            text("  日期") | bold | color(Color::Cyan) | size(WIDTH, EQUAL, 13),
            separator(),
            text("  大小") | bold | color(Color::Cyan) | size(WIDTH, EQUAL, 10),
            separator(),
            text("  尺寸") | bold | color(Color::Cyan) | size(WIDTH, EQUAL, 11),
        });
        all.push_back(header);
        all.push_back(separator());

        // File list (render visible portion)
        const int MAX_VISIBLE = 14;
        int start = st->scroll_offset;
        int end = (std::min)(start + MAX_VISIBLE, (int)st->entries.size());

        // Adjust scroll so selected is visible
        if (st->selected < start) start = st->selected;
        if (st->selected >= end) { start = st->selected - MAX_VISIBLE + 1; end = st->selected + 1; }
        if (start < 0) start = 0;
        st->scroll_offset = start;

        for (int i = start; i < end; i++) {
            const auto& e = st->entries[i];
            bool sel = (i == st->selected);

            auto dims_str = (e.width > 0 && e.height > 0)
                ? std::to_string(e.width) + "×" + std::to_string(e.height)
                : "?";

            auto row = hbox({
                text(sel ? "▶ " : "  "),
                text(" " + e.filename + " ")
                    | (sel ? color(Color::Yellow) | bold : dim)
                    | size(WIDTH, EQUAL, 34),
                separator(),
                text(" " + e.date + " ") | dim | size(WIDTH, EQUAL, 13),
                separator(),
                text(" " + fmt_size(e.size_bytes) + " ") | size(WIDTH, EQUAL, 10),
                separator(),
                text(" " + dims_str + " ") | size(WIDTH, EQUAL, 11),
            });

            all.push_back(row);
        }

        // Scroll indicator
        if ((int)st->entries.size() > MAX_VISIBLE) {
            all.push_back(separator());
            auto scroll_bar = hbox({
                text(" 显示 " + std::to_string(start + 1) + "-" + std::to_string(end)
                     + " / " + std::to_string(st->entries.size()) + "  ") | dim,
                gauge((float)st->selected / (float)(st->entries.size() - 1)) | flex,
            });
            all.push_back(scroll_bar);
        }

        // Message / status
        if (!st->message.empty()) {
            all.push_back(separator());
            all.push_back(text(" ℹ " + st->message) | color(Color::Yellow));
        }

        return vbox(all) | border;
    }) | CatchEvent([st, move_up, move_down, open_file, delete_file,
                     refresh, toggle_sort, on_switch_to_generate](Event e) {
        if (e == Event::ArrowUp || e == Event::Character('k')) { move_up(); return true; }
        if (e == Event::ArrowDown || e == Event::Character('j')) { move_down(); return true; }
        if (e == Event::PageUp) {
            for (int i = 0; i < 10 && st->selected > 0; i++) st->selected--;
            return true;
        }
        if (e == Event::PageDown) {
            for (int i = 0; i < 10 && st->selected < (int)st->entries.size() - 1; i++) st->selected++;
            return true;
        }
        if (e == Event::Home) { st->selected = 0; st->scroll_offset = 0; return true; }
        if (e == Event::End) {
            st->selected = st->entries.empty() ? 0 : (int)st->entries.size() - 1;
            return true;
        }
        if (e == Event::Return) { open_file(); return true; }
        if (e == Event::Character('r') || e == Event::Character('R')) { refresh(); return true; }
        if (e == Event::Character('s') || e == Event::Character('S')) { toggle_sort(); return true; }
        if (e == Event::Character('1')) { on_switch_to_generate(); return true; }
        if (e == Event::Character('d') || e == Event::Character('D') || e == Event::Delete) {
            delete_file(); return true;
        }
        return false;
    });
}
