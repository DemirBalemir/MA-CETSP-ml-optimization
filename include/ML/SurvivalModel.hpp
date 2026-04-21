#pragma once

#include <map>
#include <string>
#include <vector>
#include <limits>


class SurvivalModel {
public:
    // model_dir : path to the island's model directory (e.g. "../../ml/models/island_0/")
    // ml_model  : "COX", "RSF", "GBSA", "DEEPSURV", or "SSVM"
    explicit SurvivalModel(const std::string& model_dir, const std::string& ml_model);
    ~SurvivalModel();

    double predict_survival_score(const std::string& json_features);
    double predict_cox_score(const std::map<std::string, double>& feats);

    double load_cox_threshold();
    double load_rsf_threshold();
    double load_gbsa_threshold();
    double load_deepsurv_threshold();
    double load_ssvm_threshold();

    // Resets Cox coefficient cache and stops any running Python server process.
    // Call this after retraining so fresh models are loaded on the next prediction.
    void reset_cox_cache();

    // Stops the Python prediction server so it is restarted fresh on the
    // next prediction call (picks up newly trained model files).
    void invalidate();

private:
    std::string model_dir_;   // per-island model directory
    std::string ml_model_;    // runtime model type

    // ---- Cox native state ----
    struct CoxNormStat { double mean; double std; };
    std::map<std::string, double>      cox_beta;
    std::map<std::string, CoxNormStat> cox_norm;
    bool cox_loaded      = false;
    bool cox_norm_loaded = false;

    void load_cox_coeffs();
    void load_cox_norm();

    // ---- Persistent Python process (RSF / GBSA) ----
    // Stored as void* so <windows.h> is not needed in this header.
    void* proc_stdin_write_  = nullptr;
    void* proc_stdout_read_  = nullptr;
    void* proc_handle_       = nullptr;
    bool  proc_running_      = false;

    bool   start_python_process(const std::string& script_path);
    double predict_via_process(const std::string& json_line);
    void   stop_python_process();
};
