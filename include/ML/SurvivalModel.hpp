#pragma once

#include <map>
#include <string>
#include <vector>
#include <limits>



class SurvivalModel {
public:
    SurvivalModel() = default;

    
// ===== EXACT SAME FUNCTION NAMES =====
    double predict_survival_score(const std::string& json_features);

    double predict_cox_score(const std::map<std::string, double>& feats);

    double load_cox_threshold();
    double load_rsf_threshold();
    double load_gbsa_threshold();
    void reset_cox_cache();




private:

    struct CoxNormStat {
        double mean;
        double std;
    };
    // ===== EXACT SAME STATE =====
    std::map<std::string, double> cox_beta;
    std::map<std::string, struct CoxNormStat> cox_norm;

    bool cox_loaded = false;
    bool cox_norm_loaded = false;
    

    

    // ===== EXACT SAME HELPERS =====
    void load_cox_coeffs();
    void load_cox_norm();
};
