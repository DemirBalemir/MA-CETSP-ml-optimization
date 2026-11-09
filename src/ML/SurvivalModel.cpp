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
#include <regex>
#include <set>
#include <stdexcept>
#include <locale>

#include "Defs.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Paths are no longer static constants — they come from Parameters (python_exe_, scripts_dir_)
// so the binary runs on any machine without recompilation.

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
SurvivalModel::SurvivalModel(const std::string& model_dir,
                             const std::string& ml_model,
                             const std::string& python_exe,
                             const std::string& scripts_dir)
    : model_dir_(model_dir), ml_model_(ml_model),
      python_exe_(python_exe), scripts_dir_(scripts_dir)
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
    std::string cmd = "\"" + python_exe_ + "\" \"" + script_path + "\""
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
        script = scripts_dir_ + "predict_rsf.py";
    } else if (ml_model_ == "GBSA") {
        script = scripts_dir_ + "predict_gbsa.py";
    } else if (ml_model_ == "DEEPSURV") {
        script = scripts_dir_ + "predict_deepsurv.py";
    } else if (ml_model_ == "SSVM") {
        script = scripts_dir_ + "predict_ssvm.py";
    } else if (ml_model_ == "WEIBULLAFT") {
        script = scripts_dir_ + "predict_weibullaft.py";
    } else if (ml_model_ == "KNN") {
        script = scripts_dir_ + "predict_knn.py";
    } else if (ml_model_ == "ELASTICNET") {
        script = scripts_dir_ + "predict_elasticnet.py";
    } else if (ml_model_ == "MTLR") {
        script = scripts_dir_ + "predict_mtlr.py";
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
namespace {
// Deliberately limited to the Cox producer's object/numeric JSON contract.
// No external dependency; reject unsupported values instead of partially loading.
struct CoxJson {
    bool object = false;
    double number = 0;
    std::map<std::string, CoxJson> fields;
};
class CoxJsonReader {
    std::string text;
    size_t pos = 0;
    [[noreturn]] void fail() const { throw std::runtime_error("[ML ERROR] Invalid Cox JSON at byte " + std::to_string(pos)); }
    void space() { while (pos < text.size() && (text[pos]==' ' || text[pos]=='\n' || text[pos]=='\r' || text[pos]=='\t')) ++pos; }
    bool take(char c) { space(); if (pos < text.size() && text[pos]==c) { ++pos; return true; } return false; }
    std::string key() {
        if (!take('"')) fail();
        std::string out;
        while (pos < text.size()) {
            unsigned char c = text[pos++];
            if (c == '"') return out;
            if (c < 32) fail();
            if (c == '\\') {
                if (pos == text.size()) fail();
                c = text[pos++];
                if (c == 'u') {
                    unsigned int code = 0;
                    for (int i=0;i<4;++i) {
                        if (pos == text.size()) fail();
                        char h=text[pos++];
                        int digit = h>='0' && h<='9' ? h-'0' : h>='a' && h<='f' ? h-'a'+10 : h>='A' && h<='F' ? h-'A'+10 : -1;
                        if (digit<0) fail();
                        code=code*16+digit;
                    }
                    // All supported feature/statistic names are ASCII.
                    if (code>127) fail();
                    c=static_cast<unsigned char>(code);
                } else if (c=='b') c='\b';
                else if (c=='f') c='\f';
                else if (c=='n') c='\n';
                else if (c=='r') c='\r';
                else if (c=='t') c='\t';
                else if (c!='"' && c!='\\' && c!='/') fail();
            }
            out+=static_cast<char>(c);
        }
        fail();
    }
    CoxJson value(int depth) {
        space();
        if (depth>2) fail();
        CoxJson node;
        if (take('{')) {
            node.object=true;
            if (take('}')) return node;
            do {
                auto name=key();
                if (!take(':')) fail();
                auto child=value(depth+1);
                if (!node.fields.emplace(name,std::move(child)).second) fail();
                if (take('}')) return node;
            } while (take(','));
            fail();
        }
        static const std::regex number(R"(-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)");
        std::smatch match;
        std::string rest=text.substr(pos);
        if (!std::regex_search(rest,match,number,std::regex_constants::match_continuous)) fail();
        std::istringstream input(match.str()); input.imbue(std::locale::classic());
        input >> node.number;
        if (input.fail() || !std::isfinite(node.number)) fail();
        pos+=match.length();
        return node;
    }
public:
    explicit CoxJsonReader(const std::string& path) {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("[ML ERROR] Missing Cox model: " + path);
        text.assign(std::istreambuf_iterator<char>(file),{});
    }
    CoxJson read() {
        auto node=value(0); space();
        if (pos!=text.size() || !node.object) fail();
        return node;
    }
};
const std::set<std::string> cox_features = {
    "avg_edge_length", "var_edge_length", "bbox_width", "bbox_height", "bbox_area",
    "centroid_x", "centroid_y", "centroid_dist_sum", "angle_variance", "pre_vnd_cost"
};
}

double SurvivalModel::predict_cox_score(const std::map<std::string, double>& feats)
{
    if (!cox_loaded) load_cox_coeffs();
    if (!cox_norm_loaded) load_cox_norm();
    double linear = 0.0;
    for (const auto& kv : cox_beta) {
        auto it = feats.find(kv.first);
        if (it == feats.end() || !std::isfinite(it->second))
            throw std::runtime_error("[ML ERROR] Missing/nonfinite Cox feature: " + kv.first);
        const auto& ns = cox_norm.at(kv.first);
        // lifelines params_ are already in original covariate units. Its
        // predict_partial_hazard centers X but does NOT divide by std again.
        linear += kv.second * (it->second - ns.mean);
    }
    double score=std::exp(linear);
    if (!std::isfinite(score)) throw std::runtime_error("[ML ERROR] Nonfinite Cox score");
    return score;
}

void SurvivalModel::load_cox_coeffs()
{
    if (cox_loaded) return;
    auto root=CoxJsonReader(model_dir_ + "cox_coeffs.json").read();
    std::map<std::string,double> loaded;
    for (const auto& item : root.fields) {
        if (!cox_features.count(item.first) || item.second.object)
            throw std::runtime_error("[ML ERROR] Invalid Cox coefficient: " + item.first);
        loaded.emplace(item.first,item.second.number);
    }
    if (loaded.empty()) throw std::runtime_error("[ML ERROR] Empty Cox coefficients");
    cox_beta=std::move(loaded);
    cox_loaded=true;
}

void SurvivalModel::load_cox_norm()
{
    if (cox_norm_loaded) return;
    if (!cox_loaded) load_cox_coeffs();
    auto root=CoxJsonReader(model_dir_ + "cox_norm.json").read();
    std::map<std::string,CoxNormStat> loaded;
    if (root.fields.size()!=cox_beta.size())
        throw std::runtime_error("[ML ERROR] Cox normalization feature mismatch");
    for (const auto& item : root.fields) {
        const auto& stats=item.second;
        if (!cox_beta.count(item.first) || !stats.object || stats.fields.size()!=2 ||
            !stats.fields.count("mean") || !stats.fields.count("std") ||
            stats.fields.at("mean").object || stats.fields.at("std").object ||
            stats.fields.at("std").number<=1e-12)
            throw std::runtime_error("[ML ERROR] Invalid Cox normalization: " + item.first);
        loaded.emplace(item.first,CoxNormStat{stats.fields.at("mean").number,stats.fields.at("std").number});
    }
    cox_norm=std::move(loaded);
    cox_norm_loaded=true;
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

double SurvivalModel::load_deepsurv_threshold()
{
    std::string path = model_dir_ + "deepsurv_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_ssvm_threshold()
{
    std::string path = model_dir_ + "ssvm_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_weibullaft_threshold()
{
    std::string path = model_dir_ + "weibullaft_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_knn_threshold()
{
    std::string path = model_dir_ + "knn_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_elasticnet_threshold()
{
    std::string path = model_dir_ + "elasticnet_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}

double SurvivalModel::load_mtlr_threshold()
{
    std::string path = model_dir_ + "mtlr_meta.json";
    double val = read_threshold_from_file(path, 1e9);
    if (val >= 1e9)
        std::cerr << "[ML ERROR] Could not read threshold from: " << path << "\n";
    return val;
}
