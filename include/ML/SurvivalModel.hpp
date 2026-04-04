#pragma once

#include <map>
#include <string>
#include <vector>
#include <limits>


class SurvivalModel {
public:
    SurvivalModel() = default;
    ~SurvivalModel();

    double predict_survival_score(const std::string& json_features);

    double predict_cox_score(const std::map<std::string, double>& feats);

    double load_cox_threshold();
    double load_rsf_threshold();
    double load_gbsa_threshold();

    // Resets Cox coefficient cache and stops any running Python server process.
    // Call this after retraining so fresh models are loaded on the next prediction.
    void reset_cox_cache();

private:

    // ---- Cox native state ----
    struct CoxNormStat { double mean; double std; };
    std::map<std::string, double> cox_beta;
    std::map<std::string, CoxNormStat> cox_norm;
    bool cox_loaded      = false;
    bool cox_norm_loaded = false;

    void load_cox_coeffs();
    void load_cox_norm();

    // ---- Persistent Python process (RSF / GBSA) ----
    // Stored as void* so <windows.h> is not needed in this header.
    void* proc_stdin_write_  = nullptr;   // HANDLE – parent writes features here
    void* proc_stdout_read_  = nullptr;   // HANDLE – parent reads scores from here
    void* proc_handle_       = nullptr;   // HANDLE – child process handle
    bool  proc_running_      = false;

    bool   start_python_process(const std::string& script_path);
    double predict_via_process(const std::string& json_line);
    void   stop_python_process();
};
