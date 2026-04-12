#include "SurvivalModel.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <memory>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include "Defs.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static const std::string PYTHON_EXE =
    "C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe";

static const std::string SCRIPTS_DIR =
    "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/";

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
SurvivalModel::SurvivalModel(const std::string& model_dir,
                             const std::string& ml_model)
    : model_dir_(model_dir), ml_model_(ml_model)
{
    // Ensure model directory exists
    std::filesystem::create_directories(model_dir_);
}

SurvivalModel::~SurvivalModel()
{
    stop_python_process();
}

// ---------------------------------------------------------------------------
// Persistent Python process helpers
// ---------------------------------------------------------------------------

bool SurvivalModel::start_python_process(const std::string& script_path)
{
    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE stdin_read  = INVALID_HANDLE_VALUE;
    HANDLE stdin_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) {
        std::cerr << "[ML ERROR] CreatePipe (stdin) failed: " << GetLastError() << "\n";
        return false;
    }

    HANDLE stdout_read  = INVALID_HANDLE_VALUE;
    HANDLE stdout_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        std::cerr << "[ML ERROR] CreatePipe (stdout) failed: " << GetLastError() << "\n";
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return false;
    }

    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    // Pass the per-island model directory to the server script
    std::string cmd = "\"" + PYTHON_EXE + "\" \"" + script_path + "\""
                      + " --model_dir \"" + model_dir_ + "\"";

    STARTUPINFOA si{};
    si.cb          = sizeof(STARTUPINFOA);
    si.hStdInput   = stdin_read;
    si.hStdOutput  = stdout_write;
    si.hStdError   = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags     = STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(
        nullptr,
        const_cast<char*>(cmd.c_str()),
        nullptr, nullptr,
        TRUE, 0, nullptr, nullptr,
        &si, &pi
    );

    CloseHandle(stdin_read);
    CloseHandle(stdout_write);

    if (!ok) {
        std::cerr << "[ML ERROR] CreateProcess failed: " << GetLastError() << "\n";
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        return false;
    }

    CloseHandle(pi.hThread);

    proc_stdin_write_ = static_cast<void*>(stdin_write);
    proc_stdout_read_ = static_cast<void*>(stdout_read);
    proc_handle_      = static_cast<void*>(pi.hProcess);
    proc_running_     = true;

    std::cout << "[ML] Python prediction server started (pid " << pi.dwProcessId
              << ", model_dir=" << model_dir_ << ")\n";
    return true;
}

double SurvivalModel::predict_via_process(const std::string& json_line)
{
    HANDLE hw = static_cast<HANDLE>(proc_stdin_write_);
    HANDLE hr = static_cast<HANDLE>(proc_stdout_read_);

    std::string msg = json_line + "\n";
    DWORD written = 0;
    if (!WriteFile(hw, msg.c_str(), static_cast<DWORD>(msg.size()), &written, nullptr)) {
        std::cerr << "[ML ERROR] WriteFile to Python process failed: " << GetLastError() << "\n";
        proc_running_ = false;
        return 0.0;
    }

    std::string result;
    char c = 0;
    DWORD nread = 0;
    while (ReadFile(hr, &c, 1, &nread, nullptr) && nread == 1) {
        if (c == '\n') break;
        if (c != '\r') result += c;
    }

    try {
        return std::stod(result);
    } catch (...) {
        std::cerr << "[ML ERROR] Bad score from Python server: '" << result << "'\n";
        return 0.0;
    }
}

void SurvivalModel::stop_python_process()
{
    if (!proc_running_) return;

    HANDLE hw = static_cast<HANDLE>(proc_stdin_write_);
    const char exit_msg[] = "EXIT\n";
    DWORD written = 0;
    WriteFile(hw, exit_msg, sizeof(exit_msg) - 1, &written, nullptr);

    HANDLE hp = static_cast<HANDLE>(proc_handle_);
    if (WaitForSingleObject(hp, 3000) != WAIT_OBJECT_0) {
        TerminateProcess(hp, 1);
    }

    CloseHandle(hw);
    CloseHandle(static_cast<HANDLE>(proc_stdout_read_));
    CloseHandle(hp);

    proc_stdin_write_ = nullptr;
    proc_stdout_read_ = nullptr;
    proc_handle_      = nullptr;
    proc_running_     = false;
}

// ---------------------------------------------------------------------------
// Public: predict_survival_score  (RSF / GBSA)
// ---------------------------------------------------------------------------
double SurvivalModel::predict_survival_score(const std::string& json_features)
{
    std::string script;
    if (ml_model_ == "RSF") {
        script = SCRIPTS_DIR + "predict_rsf.py";
    } else if (ml_model_ == "GBSA") {
        script = SCRIPTS_DIR + "predict_gbsa.py";
    } else {
        std::cerr << "[ML ERROR] Unknown ml_model: " << ml_model_ << "\n";
        return 0.0;
    }

    if (!proc_running_) {
        if (!start_python_process(script)) return 0.0;
    }

    double score = predict_via_process(json_features);

    if (!proc_running_) {
        std::cerr << "[ML] Python server died — restarting...\n";
        if (start_python_process(script)) {
            score = predict_via_process(json_features);
        }
    }

    return score;
}

// ---------------------------------------------------------------------------
// Public: reset_cox_cache
// ---------------------------------------------------------------------------
void SurvivalModel::reset_cox_cache()
{
    cox_beta.clear();
    cox_norm.clear();
    cox_loaded      = false;
    cox_norm_loaded = false;
    stop_python_process();
}

// ---------------------------------------------------------------------------
// Public: invalidate  — force-restart the Python server on next prediction
// ---------------------------------------------------------------------------
void SurvivalModel::invalidate()
{
    stop_python_process();
    // Cox cache also needs to be refreshed so new coefficients are read
    cox_beta.clear();
    cox_norm.clear();
    cox_loaded      = false;
    cox_norm_loaded = false;
}

// ---------------------------------------------------------------------------
// Cox native prediction
// ---------------------------------------------------------------------------
double SurvivalModel::predict_cox_score(
    const std::map<std::string, double>& feats)
{
    if (!cox_loaded) {
        load_cox_coeffs();
        if (!cox_loaded) return 0.0;
    }

    if (!cox_norm_loaded) {
        load_cox_norm();
        if (!cox_norm_loaded) return 0.0;
    }

    double linear = 0.0;
    for (const auto& kv : cox_beta) {
        auto it = feats.find(kv.first);
        if (it == feats.end()) continue;

        double x = it->second;
        auto ns = cox_norm.find(kv.first);
        if (ns != cox_norm.end()) {
            double mean = ns->second.mean;
            double stdv = ns->second.std;
            if (stdv > 1e-12) x = (x - mean) / stdv;
        }
        linear += kv.second * x;
    }
    return std::exp(linear);
}

void SurvivalModel::load_cox_coeffs()
{
    if (cox_loaded) return;
    std::string path = model_dir_ + "cox_coeffs.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open Cox coeffs: " << path << "\n";
        return;
    }

    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (true) {
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        size_t colon = json.find(':', key_end);
        if (colon == std::string::npos) break;
        size_t value_end = json.find_first_of(",}", colon + 1);
        if (value_end == std::string::npos) value_end = json.size();

        std::string value_str = json.substr(colon + 1, value_end - (colon + 1));
        value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
            [](unsigned char c) { return std::isspace(c); }), value_str.end());

        try { cox_beta[key] = std::stod(value_str); }
        catch (...) {
            std::cerr << "[ML ERROR] Failed to parse Cox beta for " << key << "\n";
        }
        pos = value_end;
    }
    cox_loaded = true;
}

void SurvivalModel::load_cox_norm()
{
    if (cox_norm_loaded) return;
    std::string path = model_dir_ + "cox_norm.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open Cox norm: " << path << "\n";
        return;
    }

    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    size_t pos = 0;
    while (true) {
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = json.substr(key_start + 1, key_end - key_start - 1);

        size_t mean_pos   = json.find("\"mean\"", key_end);
        if (mean_pos == std::string::npos) break;
        size_t mean_colon = json.find(':', mean_pos);
        size_t mean_end   = json.find_first_of(",}", mean_colon + 1);
        std::string mean_str = json.substr(mean_colon + 1, mean_end - (mean_colon + 1));

        size_t std_pos   = json.find("\"std\"", mean_end);
        if (std_pos == std::string::npos) break;
        size_t std_colon = json.find(':', std_pos);
        size_t std_end   = json.find_first_of(",}", std_colon + 1);
        std::string std_str = json.substr(std_colon + 1, std_end - (std_colon + 1));

        mean_str.erase(remove_if(mean_str.begin(), mean_str.end(), ::isspace), mean_str.end());
        std_str.erase(remove_if(std_str.begin(), std_str.end(), ::isspace), std_str.end());

        try { cox_norm[key] = { std::stod(mean_str), std::stod(std_str) }; }
        catch (...) { std::cerr << "[ML ERROR] Failed to parse norm for " << key << "\n"; }

        pos = std_end;
    }
    cox_norm_loaded = true;
}

// ---------------------------------------------------------------------------
// Threshold loaders  (all read from model_dir_)
// ---------------------------------------------------------------------------

// Helper: read entire file and extract "threshold" value from JSON.
static double read_threshold_from_file(const std::string& path, double fallback)
{
    std::ifstream f(path);
    if (!f.is_open()) return fallback;

    // Read whole file (threshold may be on any line)
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    size_t pos = json.find("\"threshold\"");
    if (pos == std::string::npos) return fallback;

    size_t colon   = json.find(':', pos);
    if (colon == std::string::npos) return fallback;
    size_t val_end = json.find_first_of(",}", colon + 1);
    if (val_end == std::string::npos) val_end = json.size();

    std::string value = json.substr(colon + 1, val_end - (colon + 1));
    value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());

    try { return std::stod(value); }
    catch (...) { return fallback; }
}

double SurvivalModel::load_cox_threshold()
{
    std::string path = model_dir_ + "cox_meta.json";
    double val = read_threshold_from_file(path, ML_THRESHOLD);
    if (val == ML_THRESHOLD)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_rsf_threshold()
{
    std::string path = model_dir_ + "rsf_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_gbsa_threshold()
{
    std::string path = model_dir_ + "gbsa_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}
