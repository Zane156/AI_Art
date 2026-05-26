@echo off
if exist "build\Release\aitermui.exe" (
    start "" "build\Release\aitermui.exe"
) else (
    echo 未找到编译产物，请先编译：
    echo   mkdir build
    echo   cd build
    echo   cmake .. -G "Visual Studio 17 2022" -A x64
    echo   cmake --build . --config Release
    pause
)