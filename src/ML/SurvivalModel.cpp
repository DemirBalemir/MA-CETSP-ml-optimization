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

double SurvivalModel::predict_survival_score(
    const std::string& json_features)
{
    std::string temp_json = "C:/Users/Demir/researchproject/MA-CETSP/temp_features.json";
    {
        std::ofstream f(temp_json);
        f << json_features;
    }

    std::string python_exec =
        "C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe";

    std::string script;

    std::cout << "[DEBUG] ML_MODEL = '" << ML_MODEL << "'\n";


    if (ML_MODEL == "RSF") {
        script = "\"C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/predict_rsf.py\"";
    }
    else if (ML_MODEL == "GBSA") {
        script = "\"C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/predict_gbsa.py\"";
    }
    else {
        std::cerr << "[ML ERROR] Unknown ML_MODEL in Defs.hpp\n";
        return 0.0;
    }

    std::string json_arg = "\"" + temp_json + "\"";
    std::string cmd = python_exec + " " + script + " " + json_arg;

    std::array<char, 256> buffer{};
    std::string result;

    std::unique_ptr<FILE, decltype(&_pclose)>
        pipe(_popen(cmd.c_str(), "r"), _pclose);

    if (!pipe) {
        std::cerr << "[ML ERROR] popen failed\n";
        return 0.0;
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get())) {
        result += buffer.data();
    }

    std::remove(temp_json.c_str());

    try {
        return std::stod(result);
    }
    catch (...) {
        std::cerr << "[ML ERROR] Invalid Python output: " << result << "\n";
        return 0.0;
    }
}

void SurvivalModel::reset_cox_cache() {
    cox_beta.clear();
    cox_norm.clear();
    cox_loaded = false;
    cox_norm_loaded = false;
}

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

        // lifelines normalization
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

    double eta = linear;
    double score = std::exp(eta);   // predict_partial_hazard
    return score;
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

        // whitespace temizle
        value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
            [](unsigned char c) { return std::isspace(c); }),
            value_str.end());

        try {
            double beta = std::stod(value_str);
            cox_beta[key] = beta;
        }
        catch (...) {
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
        // feature name
        size_t key_start = json.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;

        std::string key = json.substr(key_start + 1,
            key_end - key_start - 1);

        // mean
        size_t mean_pos = json.find("\"mean\"", key_end);
        if (mean_pos == std::string::npos) break;
        size_t mean_colon = json.find(':', mean_pos);
        size_t mean_end = json.find_first_of(",}", mean_colon + 1);
        std::string mean_str =
            json.substr(mean_colon + 1, mean_end - (mean_colon + 1));

        // std
        size_t std_pos = json.find("\"std\"", mean_end);
        if (std_pos == std::string::npos) break;
        size_t std_colon = json.find(':', std_pos);
        size_t std_end = json.find_first_of(",}", std_colon + 1);
        std::string std_str =
            json.substr(std_colon + 1, std_end - (std_colon + 1));

        // cleanup
        mean_str.erase(remove_if(mean_str.begin(), mean_str.end(), ::isspace),
            mean_str.end());
        std_str.erase(remove_if(std_str.begin(), std_str.end(), ::isspace),
            std_str.end());

        try {
            cox_norm[key] = {
                std::stod(mean_str),
                std::stod(std_str)
            };
        }
        catch (...) {
            std::cerr << "[ML ERROR] Failed to parse norm for " << key << "\n";
        }

        pos = std_end;
    }

    cox_norm_loaded = true;
}


double SurvivalModel::load_rsf_threshold()
{
    std::string path = "C:/Users/Demir/researchproject/MA-CETSP/ml/models/rsf_meta.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ML ERROR] Could not open threshold file: " << path << "\n";
        return 1e9; // dev threshold
    }

    std::string json;
    std::getline(f, json);

    size_t pos = json.find(":");
    if (pos == std::string::npos) {
        std::cerr << "[ML ERROR] Invalid threshold JSON\n";
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
        std::cerr << "[ML ERROR] Could not open threshold file: " << path << "\n";
        return 1e9;
    }

    std::string json;
    std::getline(f, json);

    size_t pos = json.find(":");
    if (pos == std::string::npos) {
        std::cerr << "[ML ERROR] Invalid threshold JSON\n";
        return 1e9;
    }

    std::string value = json.substr(pos + 1);
    value.erase(std::remove(value.begin(), value.end(), '}'), value.end());
    value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

    return std::stod(value);
}

