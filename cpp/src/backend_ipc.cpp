#include "backend_ipc.hpp"
#include "json_util.hpp"
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <memory>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

// RAII cleanup wrapper for WinHTTP handles
struct WinHttpHandle {
    HINTERNET handle;
    WinHttpHandle(HINTERNET h = nullptr) : handle(h) {}
    ~WinHttpHandle() { if (handle) WinHttpCloseHandle(handle); }
    operator HINTERNET() const { return handle; }
    HINTERNET* operator&() { return &handle; }
};

BackendIPC::BackendIPC() {}
BackendIPC::~BackendIPC() { Stop(); }

// ---- API Key ----

void BackendIPC::LoadApiKey() {
    // Try known locations for api.txt
    // Priority: project dir > AppData > user Desktop > current dir
    
    // 1. Get project directory (exe parent/parent/parent)
    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    std::string proj_dir = exe_path;
    auto p = proj_dir.rfind('\\');
    if (p != std::string::npos) proj_dir = proj_dir.substr(0, p);  // Release
    p = proj_dir.rfind('\\');
    if (p != std::string::npos) proj_dir = proj_dir.substr(0, p);  // build
    p = proj_dir.rfind('\\');
    if (p != std::string::npos) proj_dir = proj_dir.substr(0, p);  // project root
    
    std::string proj_key = proj_dir + "\\api.txt";
    
    // 2. AppData
    char appdata[MAX_PATH] = {};
    GetEnvironmentVariableA("APPDATA", appdata, MAX_PATH);
    std::string app_key = std::string(appdata) + "\\AITermUI\\api.txt";
    
    // 3. User profile
    char userprofile[MAX_PATH] = {};
    GetEnvironmentVariableA("USERPROFILE", userprofile, MAX_PATH);
    
    const char* paths[] = {
        proj_key.c_str(),     // Project root: PixArt-FTXUI/api.txt
        app_key.c_str(),      // AppData: %APPDATA%/AITermUI/api.txt
        "api.txt",            // Current directory
        nullptr
    };
    for (int i = 0; paths[i]; i++) {
        std::string key = LoadApiKeyFromFile(paths[i]);
        if (!key.empty()) {
            api_key_ = key;
            return;
        }
    }
}

std::string BackendIPC::LoadApiKeyFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    std::string line;
    // Read all lines, find the first valid API key
    // Valid keys start with "sk-" (DeepSeek/OpenAI/SiliconFlow/etc.)
    // Skip provider name lines (e.g., "deepseek1", "qwen:", etc.)
    while (std::getline(f, line)) {
        // Trim whitespace
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t'))
            line.erase(0, 1);
        if (line.empty()) continue;
        // Strip UTF-8 BOM if present (common in Windows editors)
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF && (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            line.erase(0, 3);
        // Skip obvious provider name lines (short, non-key patterns)
        if (line.size() < 20) continue;  // Keys are 30+ chars
        if (line.find("sk-") == 0 || line.find("sk-") != std::string::npos) {
            return line;  // Found a valid key
        }
    }
    return "";  // No valid key found
}

// ---- Start / Stop ----

bool BackendIPC::Start(const std::string& python_exe) {
    if (running_) return true;

    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    std::string project_root = exe_path;
    auto p = project_root.rfind('\\');
    if (p != std::string::npos) project_root = project_root.substr(0, p);
    p = project_root.rfind('\\');
    if (p != std::string::npos) project_root = project_root.substr(0, p);
    p = project_root.rfind('\\');
    if (p != std::string::npos) project_root = project_root.substr(0, p);

    std::string script_path = project_root + "\\python\\server.py";

    // Find Python
    std::string py = python_exe;
    if (py == "python" || py == "python3") {
        bool found = false;
        {
            STARTUPINFOA si2 = {sizeof(si2)};
            PROCESS_INFORMATION pi2;
            std::string ver = py + " --version";
            char* buf = new char[ver.size() + 1];
            memcpy(buf, ver.c_str(), ver.size() + 1);
            if (CreateProcessA(nullptr, buf, nullptr, nullptr, FALSE,
                               CREATE_NO_WINDOW, nullptr, nullptr, &si2, &pi2)) {
                WaitForSingleObject(pi2.hProcess, 2000);
                DWORD ec = 255;
                GetExitCodeProcess(pi2.hProcess, &ec);
                CloseHandle(pi2.hThread);
                CloseHandle(pi2.hProcess);
                found = (ec == 0);
            }
            delete[] buf;
        }
        if (!found) {
            py.clear(); // reset & search properly
            // 1. Try common names
            const char* fallback[] = {"python.exe", "python3.exe", nullptr};
            for (int i = 0; fallback[i]; i++) {
                if (GetFileAttributesA(fallback[i]) != INVALID_FILE_ATTRIBUTES) {
                    py = fallback[i]; break;
                }
            }
            // 2. Try LOCALAPPDATA paths
            if (py.empty()) {
                char localAppData[MAX_PATH];
                if (GetEnvironmentVariableA("LOCALAPPDATA", localAppData, MAX_PATH)) {
                    std::string base(localAppData);
                    const char* vers[] = {"Python311", "Python312", "Python310", nullptr};
                    for (int i = 0; vers[i] && py.empty(); i++) {
                        std::string cand = base + "\\Programs\\Python\\" + vers[i] + "\\python.exe";
                        if (GetFileAttributesA(cand.c_str()) != INVALID_FILE_ATTRIBUTES)
                            py = cand;
                    }
                }
            }
            // 3. Last resort: trust PATH
            if (py.empty()) py = "python.exe";
        }
    }

    std::string cmd = "\"" + py + "\" \"" + script_path + "\" --parent-pid=" + std::to_string(GetCurrentProcessId());
    char* cmdBuf = new char[cmd.size() + 1];
    memcpy(cmdBuf, cmd.c_str(), cmd.size() + 1);

    STARTUPINFOA si = {sizeof(si)};
    PROCESS_INFORMATION pi;
    running_ = false;
    if (CreateProcessA(nullptr, cmdBuf,
                       nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr,
                       project_root.c_str(),
                       &si, &pi))
    {
        CloseHandle(pi.hThread);

        // Verify backend actually responds AND has correct project_root
        // (port may be taken by another backend instance with different work dir)
        bool verified = false;
        for (int i = 0; i < 50; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::string r = HttpGet("/sys_info");
            if (!r.empty() && r.find("gpu_name") != std::string::npos) {
                // Verify the backend's project_root matches ours
                std::string br = JsonUnescape(JsonGet(r, "project_root"));
                auto norm = [](std::string& s) { for (auto& c : s) if (c == '\\') c = '/'; };
                std::string nr = br, np = project_root;
                norm(nr); norm(np);
                if (nr == np) {
                    verified = true;
                    break;
                }
            }
        }

        if (verified) {
            running_ = true;
            {
                std::lock_guard<std::mutex> lk(proc_mutex_);
                server_process_ = pi.hProcess;
            }
            server_thread_ = std::thread([this, cmdBuf]() {
                HANDLE h = nullptr;
                {
                    std::lock_guard<std::mutex> lk(proc_mutex_);
                    h = server_process_;
                }
                if (h) {
                    WaitForSingleObject(h, INFINITE);
                    CloseHandle(h);
                    std::lock_guard<std::mutex> lk(proc_mutex_);
                    if (server_process_ == h) server_process_ = nullptr;
                }
                delete[] cmdBuf;
            });
            server_thread_.detach();
        } else {
            // Backend didn't start - port likely taken, kill the failed process
            TerminateProcess(pi.hProcess, 1);
            CloseHandle(pi.hProcess);
            delete[] cmdBuf;
        }
    } else {
        delete[] cmdBuf;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    return running_;
}

void BackendIPC::Stop() {
    if (!running_) return;
    running_ = false;

    // Step 1: Graceful HTTP shutdown
    HttpGet("/shutdown");

    // Step 2: Wait a bit, then force-kill if still alive
    ForceKillBackend();
}

void BackendIPC::ForceKillBackend() {
    HANDLE h = nullptr;
    {
        std::lock_guard<std::mutex> lk(proc_mutex_);
        h = server_process_;
    }
    if (!h) return;

    // Wait up to 3 seconds for graceful exit
    DWORD waitResult = WaitForSingleObject(h, 3000);
    if (waitResult == WAIT_TIMEOUT) {
        // Still running -- force terminate
        TerminateProcess(h, 1);
        CloseHandle(h);
    }
    std::lock_guard<std::mutex> lk(proc_mutex_);
    server_process_ = nullptr;
}

// ---- URL Parsing + WinHTTP ----

struct ParsedUrl {
    std::string host;
    int port;
    std::string path;
};

static ParsedUrl ParseUrl(const std::string& url) {
    ParsedUrl p;
    p.port = 80;
    size_t s = url.find("://");
    size_t start = (s == std::string::npos) ? 0 : s + 3;
    size_t end = url.find('/', start);
    if (end == std::string::npos) end = url.size();
    std::string hp = url.substr(start, end - start);
    size_t colon = hp.find(':');
    if (colon != std::string::npos) {
        p.host = hp.substr(0, colon);
        p.port = std::stoi(hp.substr(colon + 1));
    } else {
        p.host = hp;
    }
    p.path = (end < url.size()) ? url.substr(end) : "/";
    return p;
}

static std::string DoHttp(const std::string& url,
                           const std::string& method,
                           const std::string& body = "") {
    std::string result;
    auto p = ParseUrl(url);

    WinHttpHandle hSession(WinHttpOpen(L"AITermUI/1.0",
                                       WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                       nullptr, nullptr, 0));
    if (!hSession) return "";

    std::wstring whost(p.host.begin(), p.host.end());
    WinHttpHandle hConnect(WinHttpConnect(hSession, whost.c_str(), (INTERNET_PORT)p.port, 0));
    if (!hConnect) return "";  // hSession auto-closed via RAII

    std::wstring wpath(p.path.begin(), p.path.end());
    LPCWSTR mtd = (method == "GET") ? L"GET" : L"POST";
    WinHttpHandle hRequest(WinHttpOpenRequest(hConnect, mtd, wpath.c_str(),
                                               nullptr, nullptr, nullptr, 0));
    if (!hRequest) return "";  // hConnect+hSession auto-closed via RAII

    if (method == "POST" && !body.empty()) {
        std::wstring hdr = L"Content-Type: application/json; charset=utf-8";
        WinHttpAddRequestHeaders(hRequest, hdr.c_str(), (DWORD)hdr.size(),
                                 WINHTTP_ADDREQ_FLAG_ADD);
    }

    LPVOID pData = nullptr;
    DWORD dataLen = 0;
    std::string bodyCopy;
    if (method == "POST" && !body.empty()) {
        bodyCopy = body;
        pData = (LPVOID)bodyCopy.data();
        dataLen = (DWORD)bodyCopy.size();
    }

    try {
        if (WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                               pData, dataLen, dataLen, 0))
        {
            WinHttpReceiveResponse(hRequest, nullptr);
            DWORD avail = 0;
            while (WinHttpQueryDataAvailable(hRequest, &avail) && avail > 0) {
                std::vector<char> buf(avail + 1, 0);
                DWORD read = 0;
                if (WinHttpReadData(hRequest, buf.data(), avail, &read))
                    result.append(buf.data(), read);
            }
        }
    } catch (...) {
        // RAII will close all handles
    }
    // hRequest, hConnect, hSession auto-closed by WinHttpHandle destructors
    return result;
}

std::string BackendIPC::HttpPost(const std::string& endpoint, const std::string& body) {
    return DoHttp(backend_url_ + endpoint, "POST", body);
}

std::string BackendIPC::HttpGet(const std::string& endpoint) {
    return DoHttp(backend_url_ + endpoint, "GET");
}

// ---- API calls ----

std::string BackendIPC::Generate(const std::string& json_params) {
    return HttpPost("/generate", json_params);
}

std::string BackendIPC::TrainLoRA(const std::string& json_params) {
    return HttpPost("/train_lora", json_params);
}

std::string BackendIPC::ChatWithAgent(const std::string& json_body) {
    return HttpPost("/agent_chat", json_body);
}


std::string BackendIPC::ScanModels() {
    return HttpGet("/scan_models");
}

std::string BackendIPC::GetSysInfo() {
    return HttpGet("/sys_info");
}

std::string BackendIPC::CheckModelStatus(const std::string& model_name) {
    return HttpGet("/model_status?model=" + model_name);
}


// ---- Progress tracking ----

std::string BackendIPC::CheckTaskStatus(const std::string& task_id) {
    return HttpGet("/task_status?id=" + task_id);
}

std::string BackendIPC::GetTaskResult(const std::string& task_id) {
    return HttpGet("/task_result?id=" + task_id);
}

// ---- Download (extended) ----

std::string BackendIPC::DownloadModel(const std::string& model_name,
                                       const std::string& repo_id,
                                       const std::string& target_dir) {
    std::ostringstream js;
    js << "{";
    JsonPair(js, "model", model_name); js << ",";
    JsonPair(js, "repo", repo_id); js << ",";
    JsonPair(js, "target_dir", target_dir);
    js << "}";
    return HttpPost("/download_model", js.str());
}