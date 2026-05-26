cmake_minimum_required(VERSION 3.20)
project(AITermUI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /EHsc /W3 /wd4267)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# 优先使用本地已有的 FTXUI 源码（避免在国内网络下 git clone 超时）
if(NOT FETCHCONTENT_SOURCE_DIR_FTXUI)
    set(_ftxui_candidates
        "${CMAKE_SOURCE_DIR}/../PixArt-FTXUI/build/_deps/ftxui-src"
        "$ENV{FTXUI_SOURCE_DIR}"
    )
    foreach(_path ${_ftxui_candidates})
        if(EXISTS "${_path}/CMakeLists.txt")
            set(FETCHCONTENT_SOURCE_DIR_FTXUI "${_path}")
            message(STATUS "使用本地 FTXUI: ${_path}")
            break()
        endif()
    endforeach()
endif()

FetchContent_Declare(ftxui
    GIT_REPOSITORY  https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG         v5.0.0
    GIT_SHALLOW     TRUE
)
FetchContent_MakeAvailable(ftxui)

set(SOURCES
    cpp/src/main.cpp cpp/src/generate_screen.cpp cpp/src/train_screen.cpp
    cpp/src/agent_panel.cpp cpp/src/backend_ipc.cpp
)
add_executable(aitermui ${SOURCES})
target_include_directories(aitermui PRIVATE cpp/include ${ftxui_SOURCE_DIR}/include)
target_link_libraries(aitermui PRIVATE ftxui::component ftxui::screen ftxui::dom)
file(COPY python/ DESTINATION ${CMAKE_BINARY_DIR}/python/)

