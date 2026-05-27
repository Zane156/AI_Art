#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "backend_ipc.hpp"
#include "generate_screen.hpp"
#include "gallery_screen.hpp"
#include "agent_panel.hpp"
#include <memory>
#include <string>
#include <thread>
#include <chrono>

using namespace ftxui;

// ── Simple tab component: keeps both children in the tree, renders only active ──
class TabSwitcher : public ComponentBase {
    Component a_;
    Component b_;
    int* idx_;
public:
    TabSwitcher(Component a, Component b, int* idx)
        : a_(std::move(a)), b_(std::move(b)), idx_(idx) {
        Add(a_);
        Add(b_);
    }

    Element Render() override {
        if (*idx_ == 0) return a_->Render();
        return b_->Render();
    }

    bool Focusable() const override {
        if (*idx_ == 0) return a_->Focusable();
        return b_->Focusable();
    }

    bool OnEvent(Event event) override {
        if (*idx_ == 0) return a_->OnEvent(event);
        return b_->OnEvent(event);
    }

    Component ActiveChild() override {
        if (*idx_ == 0) return a_;
        return b_;
    }

    void SetActiveChild(ComponentBase* child) override {
        if (child == a_.get()) *idx_ = 0;
        else if (child == b_.get()) *idx_ = 1;
    }
};

int main() {
    auto screen = ScreenInteractive::Fullscreen();
    auto backend = std::make_shared<BackendIPC>();

    backend->LoadApiKey();

    std::thread bg([backend] {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        backend->Start();
    });
    bg.detach();

    auto cs = std::make_shared<std::string>("generate");
    auto st = std::make_shared<std::string>("正在启动后端...");
    auto set_status = [st](const std::string& s) { *st = s; };

    auto tab_idx = std::make_shared<int>(0);

    auto gen     = CreateGenerateScreen(backend, nullptr, set_status, [&]{ screen.RequestAnimationFrame(); });
    auto gallery = CreateGalleryScreen([&]{ *tab_idx = 0; *cs = "generate"; }, [&]{ screen.RequestAnimationFrame(); });
    auto agent   = CreateAgentPanel(backend,
        [cs]{ return *cs; }, [st]{ return *st; },
        [&]{ screen.RequestAnimationFrame(); });

    // Custom TabSwitcher: both gen and gallery in component tree
    auto tab_body = Make<TabSwitcher>(gen, gallery, tab_idx.get());

    auto layout = Container::Horizontal({
        tab_body | flex,
        agent | size(WIDTH, EQUAL, 44),
    });

    layout = CatchEvent(layout, [&](Event e) {
        if (e == Event::Character('q') || e == Event::Escape) {
            screen.ExitLoopClosure()();
            return true;
        }
        if (e == Event::Tab) { *tab_idx = (*tab_idx == 0) ? 1 : 0; *cs = (*tab_idx == 0) ? "generate" : "gallery"; return true; }
        return false;
    });

    screen.Loop(layout);
    backend->Stop();
    return 0;
}
