#include "ML/SurvivalModel.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <sstream>

namespace fs = std::filesystem;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void rejects(const std::function<void()>& action) {
    try { action(); } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find("[ML ERROR]")!=std::string::npos,"missing explicit diagnostic"); return;
    }
    throw std::runtime_error("invalid model silently accepted");
}
int main(int argc, char** argv) {
    // Exercise the production predictor against a real Python-exported model.
    if (argc == 4 && std::string(argv[1]) == "--parity") {
        try {
            SurvivalModel model(std::string(argv[2])+"/","COX","unused","unused");
            std::ifstream in(argv[3]); require(bool(in),"missing parity CSV");
            std::string line, field; std::getline(in,line);
            std::vector<std::string> names; std::istringstream header(line);
            while(std::getline(header,field,',')) names.push_back(field);
            require(names.back()=="expected_hazard","missing reference hazard column");
            int count=0;
            while(std::getline(in,line)) {
                std::istringstream row(line);std::map<std::string,double> features;double expected=0;
                for(size_t i=0;i<names.size();++i) {
                    require(bool(std::getline(row,field,',')),"missing parity field");
                    double value=std::stod(field);
                    if(i+1==names.size()) expected=value; else features[names[i]]=value;
                }
                double actual=model.predict_cox_score(features);
                require(std::abs(actual-expected)<=1e-8*std::max(1.0,std::abs(expected)),"Python/native hazard mismatch");
                ++count;
            }
            require(count>0,"empty parity dataset");std::cout<<"Python/native parity rows="<<count<<" passed\n";
            return 0;
        } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
    }
    auto dir=fs::temp_directory_path()/("cetsp-cox-test-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p,ec); } } cleanup{dir};
    try {
        auto write=[&](const char* name,const std::string& data){std::ofstream out(dir/name);out<<data;};
        const std::string coeff=R"({"pre_vnd_cost":0.5,"bbox_width":-0.25})";
        const std::string norm=R"({"pre_vnd_cost":{"mean":2,"std":4},"bbox_width":{"mean":10,"std":2}})";
        const std::map<std::string,double> feats={{"pre_vnd_cost",10},{"bbox_width",14}};
        auto predict=[&](){SurvivalModel model(dir.string()+"/","COX","unused","unused");return model.predict_cox_score(feats);};
        write("cox_coeffs.json",coeff);write("cox_norm.json",norm);
        require(std::abs(predict()-std::exp(3.0))<1e-12,"known multi-feature score mismatch");
        write("cox_norm.json",R"({"bbox_width":{"std":2,"mean":10},"pre_vnd_cost":{"std":4,"mean":2}})");
        write("cox_coeffs.json",R"({"bbox_width":-2.5e-1,"pre_vnd_cost":5e-1})");
        require(std::abs(predict()-std::exp(3.0))<1e-12,"field order changes score");
        write("cox_norm.json",R"({"pre_vnd_cost":{"std":4,"m\u0065an":2},"bbox_width":{"mean":10,"std":2}})");
        require(std::abs(predict()-std::exp(3.0))<1e-12,"escaped JSON key mismatch");
        // Supported low-variance removal leaves a nonempty feature subset.
        write("cox_coeffs.json",R"({"pre_vnd_cost":0.5})");
        write("cox_norm.json",R"({"pre_vnd_cost":{"std":4,"mean":2}})");
        require(std::abs(predict()-std::exp(4.0))<1e-12,"feature subset rejected");
        write("cox_coeffs.json",coeff);
        for (const auto& bad : {std::string("broken"),std::string("{}"),
            std::string(R"({"pre_vnd_cost":{"mean":2},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":2,"std":0},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":NaN,"std":4},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"std":1e999,"mean":2},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":2,"mean":3,"std":4},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":2,"std":-1},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":2,"std":1e-13},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":"2","std":4},"bbox_width":{"mean":10,"std":2}})"),
            std::string(R"({"pre_vnd_cost":{"mean":2,"std":4},"unknown":{"mean":10,"std":2}})"),
            norm+"trailing"}) {
            write("cox_norm.json",bad); rejects([&](){predict();});
        }
        fs::remove(dir/"cox_norm.json");rejects([&](){predict();});
        write("cox_norm.json",norm);
        for (const auto& bad : {"{}","[]","{\"pre_vnd_cost\":Inf}","{\"pre_vnd_cost\":\"1\"}","{\"pre_vnd_cost\":1garbage}","{\"unknown\":1}"}) {
            write("cox_coeffs.json",bad);rejects([&](){predict();});
        }
        fs::remove(dir/"cox_coeffs.json"); rejects([&](){predict();});
        write("cox_coeffs.json",coeff);
        SurvivalModel cached(dir.string()+"/","COX","unused","unused");
        require(std::abs(cached.predict_cox_score(feats)-std::exp(3.0))<1e-12,"cache load mismatch");
        write("cox_norm.json","{}"); cached.reset_cox_cache();
        rejects([&](){cached.predict_cox_score(feats);});
        write("cox_norm.json",norm);
        require(std::abs(cached.predict_cox_score(feats)-std::exp(3.0))<1e-12,"failed load poisoned cache");
        std::cout<<"Cox production reader: known score, order independence, subsets and invalid models passed\n";
        return 0;
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
