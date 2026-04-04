#include "SurvivalModel.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <memory>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "Defs.hpp"

// Windows API for persistent child process management
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
SurvivalModel::~SurvivalModel()
{
    stop_python_process();
}

// ---------------------------------------------------------------------------
// Persistent Python process helpers
// ---------------------------------------------------------------------------

bool SurvivalModel::start_python_process(const std::string& script_path)
{
    // Security attributes: make pipe handles inheritable by the child
    SECURITY_ATTRIBUTES sa{};
    sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle       = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    // Pipe for parent→child (child's stdin)
    HANDLE stdin_read  = INVALID_HANDLE_VALUE;
    HANDLE stdin_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) {
        std::cerr << "[ML ERROR] CreatePipe (stdin) failed: " << GetLastError() << "\n";
        return false;
    }

    // Pipe for child→parent (child's stdout)
    HANDLE stdout_read  = INVALID_HANDLE_VALUE;
    HANDLE stdout_write = INVALID_HANDLE_VALUE;
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        std::cerr << "[ML ERROR] CreatePipe (stdout) failed: " << GetLastError() << "\n";
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return false;
    }

    // Parent keeps stdin_write and stdout_read — make them non-inheritable so
    // the child does not accidentally inherit the wrong ends
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    // Build command: python.exe "<script_path>"
    std::string python_exec =
        "C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe";
    std::string cmd = "\"" + python_exec + "\" \"" + script_path + "\"";

    STARTUPINFOA si{};
    si.cb          = sizeof(STARTUPINFOA);
    si.hStdInput   = stdin_read;
    si.hStdOutput  = stdout_write;
    si.hStdError   = GetStdHandle(STD_ERROR_HANDLE); // let stderr pass through
    si.dwFlags     = STARTF_USESTDHANDLES;

    PROCESS_INFORMATION pi{};
    BOOL ok = CreateProcessA(
        nullptr,
        const_cast<char*>(cmd.c_str()),
        nullptr, nullptr,
        TRUE,   // inherit handles
        0, nullptr, nullptr,
        &si, &pi
    );

    // Close the child's ends on the parent side — child has its own copies
    CloseHandle(stdin_read);
    CloseHandle(stdout_write);

    if (!ok) {
        std::cerr << "[ML ERROR] CreateProcess failed: " << GetLastError() << "\n";
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        return false;
    }

    CloseHandle(pi.hThread); // not needed

    proc_stdin_write_ = static_cast<void*>(stdin_write);
    proc_stdout_read_ = static_cast<void*>(stdout_read);
    proc_handle_      = static_cast<void*>(pi.hProcess);
    proc_running_     = true;

    std::cout << "[ML] Python prediction server started (pid " << pi.dwProcessId << ")\n";
    return true;
}

double SurvivalModel::predict_via_process(const std::string& json_line)
{
    HANDLE hw = static_cast<HANDLE>(proc_stdin_write_);
    HANDLE hr = static_cast<HANDLE>(proc_stdout_read_);

    // Send one JSON line to the server
    std::string msg = json_line + "\n";
    DWORD written = 0;
    if (!WriteFile(hw, msg.c_str(), static_cast<DWORD>(msg.size()), &written, nullptr)) {
        std::cerr << "[ML ERROR] WriteFile to Python process failed: " << GetLastError() << "\n";
        proc_running_ = false;
        return 0.0;
    }

    // Read one response line (the float score followed by '\n')
    std::string result;
    char c = 0;
    DWORD nread = 0;
    while (ReadFile(hr, &c, 1, &nread, nullptr) && nread == 1) {
        if (c == '\n') break;
        if (c != '\r') result += c; // strip Windows CR
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

    // Ask the server to exit gracefully
    HANDLE hw = static_cast<HANDLE>(proc_stdin_write_);
    const char exit_msg[] = "EXIT\n";
    DWORD written = 0;
    WriteFile(hw, exit_msg, sizeof(exit_msg) - 1, &written, nullptr);

    // Wait up to 3 s, then kill
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
    // Determine which script to use
    std::string script;
    if (ML_MODEL == "RSF") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/predict_rsf.py";
    } else if (ML_MODEL == "GBSA") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/predict_gbsa.py";
    } else {
        std::cerr << "[ML ERROR] Unknown ML_MODEL in Defs.hpp\n";
        return 0.0;
    }

    // Lazy-start the persistent server on the first call
    if (!proc_running_) {
        if (!start_python_process(script)) {
            return 0.0;
        }
    }

    double score = predict_via_process(json_features);

    // If the process died mid-run, try restarting once
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
// Also stops the Python server so a fresh one is started after retraining.
// ---------------------------------------------------------------------------
void SurvivalModel::reset_cox_cache()
{
    cox_beta.clear();
    cox_norm.clear();
    cox_loaded      = false;
    cox_norm_loaded = false;

    // Stop the prediction server — the model pkl on disk just changed
    stop_python_process();
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

        // lifelines normalization: (x - mean) / std
        auto ns = cox_norm.find(kv.first);
        if (ns != cox_norm.end()) {
            double mean = ns->second.mean;
            double stdv = ns->second.std;
            if (stdv > 1e-12) {
                x = (x - mean) / stdv;
            }
        }

        linear += kv.second * x;
    }

    return std::exp(linear); // predict_partial_hazard
}

void SurvivalModel::load_cox_coeffs()
{
    if (cox_loaded) return;
    std::string path = "C:/Users/Demir/researchproject/MA-CETSP/ml/models/cox_coeffs.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open Cox coeffs file: " << path << "\n";
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
            [](unsigned char c) { return std::isspace(c); }),
            value_str.end());

        try {
            cox_beta[key] = std::stod(value_str);
        } catch (...) {
            std::cerr << "[ML ERROR] Failed to parse Cox beta for key " << key
                      << " with value '" << value_str << "'\n";
        }

        pos = value_end;
    }

    cox_loaded = true;
}

void SurvivalModel::load_cox_norm()
{
    if (cox_norm_loaded) return;

    std::string path =
        "C:/Users/Demir/researchproject/MA-CETSP/ml/models/cox_norm.json";

    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open Cox norm file: " << path << "\n";
        return;
    }

    std::string json(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );

    size_t pos = 0;
    while (true) {
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string key = json.substr(key_start + 1,
                                       key_end - key_start - 1);

        // mean
        size_t mean_pos   = json.find("\"mean\"", key_end);
        if (mean_pos == std::string::npos) break;
        size_t mean_colon = json.find(':', mean_pos);
        size_t mean_end   = json.find_first_of(",}", mean_colon + 1);
        std::string mean_str =
            json.substr(mean_colon + 1, mean_end - (mean_colon + 1));

        // std
        size_t std_pos   = json.find("\"std\"", mean_end);
        if (std_pos == std::string::npos) break;
        size_t std_colon = json.find(':', std_pos);
        size_t std_end   = json.find_first_of(",}", std_colon + 1);
        std::string std_str =
            json.substr(std_colon + 1, std_end - (std_colon + 1));

        mean_str.erase(remove_if(mean_str.begin(), mean_str.end(), ::isspace),
                       mean_str.end());
        std_str.erase(remove_if(std_str.begin(), std_str.end(), ::isspace),
                      std_str.end());

        try {
            cox_norm[key] = { std::stod(mean_str), std::stod(std_str) };
        } catch (...) {
            std::cerr << "[ML ERROR] Failed to parse norm for " << key << "\n";
        }

        pos = std_end;
    }

    cox_norm_loaded = true;
}

// ---------------------------------------------------------------------------
// Threshold loaders
// ---------------------------------------------------------------------------

double SurvivalModel::load_cox_threshold()
{
    std::string path = "C:/Users/Demir/researchproject/MA-CETSP/ml/models/cox_meta.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open Cox threshold file: " << path << "\n";
        return ML_THRESHOLD; // fallback to Defs.hpp value
    }

    std::string json;
    std::getline(f, json);

    size_t pos = json.find("\"threshold\"");
    if (pos == std::string::npos) {
        std::cerr << "[ML ERROR] 'threshold' key not found in cox_meta.json\n";
        return ML_THRESHOLD;
    }

    size_t colon   = json.find(':', pos);
    if (colon == std::string::npos) return ML_THRESHOLD;

    size_t val_end = json.find_first_of(",}", colon + 1);
    if (val_end == std::string::npos) val_end = json.size();

    std::string value = json.substr(colon + 1, val_end - (colon + 1));
    value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());

    try {
        return std::stod(value);
    } catch (...) {
        std::cerr << "[ML ERROR] Failed to parse Cox threshold: " << value << "\n";
        return ML_THRESHOLD;
    }
}

double SurvivalModel::load_rsf_threshold()
{
    std::string path = "C:/Users/Demir/researchproject/MA-CETSP/ml/models/rsf_meta.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open RSF threshold file: " << path << "\n";
        return 1e9;
    }

    std::string json;
    std::getline(f, json);

    size_t pos = json.find(":");
    if (pos == std::string::npos) {
        std::cerr << "[ML ERROR] Invalid RSF threshold JSON\n";
        return 1e9;
    }

    std::string value = json.substr(pos + 1);
    value.erase(std::remove(value.begin(), value.end(), '}'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    return std::stod(value);
}

double SurvivalModel::load_gbsa_threshold()
{
    std::string path = "C:/Users/Demir/researchproject/MA-CETSP/ml/models/gbsa_meta.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open GBSA threshold file: " << path << "\n";
        return 1e9;
    }

    std::string json;
    std::getline(f, json);

    size_t pos = json.find(":");
    if (pos == std::string::npos) {
        std::cerr << "[ML ERROR] Invalid GBSA threshold JSON\n";
        return 1e9;
    }

    std::string value = json.substr(pos + 1);
    value.erase(std::remove(value.begin(), value.end(), '}'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    return std::stod(value);
}
