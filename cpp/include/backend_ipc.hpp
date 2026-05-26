#pragma once
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

class BackendIPC {
public:
    BackendIPC();
    ~BackendIPC();

    bool Start(const std::string& python_exe = "python");
    void Stop();
    bool IsRunning() const { return running_; }

    std::string Generate(const std::string& json_params);
    std::string TrainLoRA(const std::string& json_params);
    std::string ChatWithAgent(const std::string& json_body);
    std::string DownloadModel(const std::string& model_name, const std::string& repo_id, const std::string& target_dir);
    std::string ScanModels();
    std::string GetSysInfo();
    std::string CheckModelStatus(const std::string& model_name);

    // Progress tracking
    std::string CheckTaskStatus(const std::string& task_id);
    std::string GetTaskResult(const std::string& task_id);

    // API Key
    void LoadApiKey();
    void SetApiKey(const std::string& key) { api_key_ = key; }
    std::string GetApiKey() const { return api_key_; }
    std::string LoadApiKeyFromFile(const std::string& path);

    // HTTP helpers (public for screens that need direct API access)
    std::string HttpPost(const std::string& endpoint, const std::string& body);
    std::string HttpGet(const std::string& endpoint);

private:
    void ForceKillBackend();

    std::string backend_url_ = "http://127.0.0.1:17860";
    std::string api_key_;
    std::atomic<bool> running_{false};
    std::thread server_thread_;
    HANDLE server_process_ = nullptr;
    std::mutex proc_mutex_;
};
