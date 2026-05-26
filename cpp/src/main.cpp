#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "backend_ipc.hpp"
#include "generate_screen.hpp"
#include "agent_panel.hpp"
#include <memory>
#include <string>
#include <thread>
#include <chrono>

using namespace ftxui;

int main() {
    auto screen = ScreenInteractive::Fullscreen();
    auto backend = std::make_shared<BackendIPC>();

    // Load API key from file
    backend->LoadApiKey();

    // Start backend in background (after screen is ready)
    std::thread bg([backend] {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        backend->Start();
    });
    bg.detach();

    auto cs = std::make_shared<std::string>("generate");
    auto st = std::make_shared<std::string>("正在启动后端...");
    auto set_status = [st](const std::string& s) { *st = s; };
    auto gen = CreateGenerateScreen(backend, nullptr, set_status, [&]{ screen.RequestAnimationFrame(); });
    auto agent = CreateAgentPanel(backend,
        [cs]{ return *cs; }, [st]{ return *st; },
        [&]{ screen.RequestAnimationFrame(); });

    auto layout = Container::Horizontal({
        gen | flex,
        agent | size(WIDTH, EQUAL, 44),
    });

    layout = CatchEvent(layout, [&](Event e) {
        if (e == Event::Character('q') || e == Event::Escape) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    screen.Loop(layout);
    backend->Stop();
    return 0;
}
