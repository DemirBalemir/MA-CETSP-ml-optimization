#include "ML/CoxLifecycle.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace fs=std::filesystem;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void rejects(const std::function<void()>& action, const std::string& expected) {
    try { action(); } catch (const std::exception& e) {
        require(std::string(e.what()).find(expected)!=std::string::npos, std::string("unexpected error: ")+e.what());
        return;
    }
    throw std::runtime_error("expected explicit error");
}
int main() {
    auto dir=fs::temp_directory_path()/("cetsp-cox-lifecycle-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    struct Cleanup {fs::path path; ~Cleanup(){std::error_code ec;fs::remove_all(path,ec);}} cleanup{dir};
    try {
        SurvivalModel model(dir.string()+"/", "COX", "unused", "unused");
        List first, second, initial, best;
        Node depot(0,0,0), other(1,14,0);
        initial.add(&depot); initial.add(&other); initial.setValue(10); // No pre-VND snapshot: production fallback path.
        first.pre_vnd_coords={{0,0},{14,0}}; first.pre_vnd_value=10;
        second.pre_vnd_coords=first.pre_vnd_coords; second.pre_vnd_value=6;
        first.cox_lp=-777; second.cox_lp=-888;
        std::vector<List*> members={&first,&second,&initial};
        int completed=-1;
        auto fill=[&](){CoxLifecycle::backfill(true,"COX",completed,&model,members);};
        auto finish=[&](bool success,int iteration){CoxLifecycle::complete_training(success,iteration,true,"COX",completed,&model,members,&best);};
        auto check_scores=[&](double a,double b){
            require(first.has_cox_lp && second.has_cox_lp && initial.has_cox_lp,"successful scores not cached");
            require(std::abs(initial.cox_lp-a)<1e-12,"initial-population fallback features/score mismatch");
            require(std::abs(first.cox_lp-a)<1e-12 && std::abs(second.cox_lp-b)<1e-12,"wrong model generation/feature scores");
        };
        auto write=[&](const char* name,const std::string& text){std::ofstream out(dir/name);out<<text;};

        // The actual init/end-of-iteration helpers must not touch an empty model.
        CoxLifecycle::cache_score(true,"COX",completed,&model,first);
        fill(); fill();
        require(fs::is_empty(dir),"pretraining created model files");
        require(!first.has_cox_lp && !second.has_cox_lp && !initial.has_cox_lp && first.cox_lp==-777 && second.cox_lp==-888,"pretraining published fake scores");
        require(!CoxLifecycle::ready(true,"COX",completed),"pretraining filter active");
        rejects([&](){CoxLifecycle::predict(true,"COX",completed,&model,{});},"requires successful training");

        // Failed training: production completion path clears stale flags but never reads model.
        first.has_cox_lp=true;best.has_cox_lp=true;
        finish(false,40);
        fill(); fill();
        require(completed==-1 && !CoxLifecycle::ready(true,"COX",completed),"failed training activated filter");
        require(!first.has_cox_lp && !second.has_cox_lp && !best.has_cox_lp,"failed training retained stale flags");
        require(first.cox_lp==-777 && second.cox_lp==-888 && fs::is_empty(dir),"failed training attempted score/model work");

        write("cox_coeffs.json",R"({"pre_vnd_cost":0.5,"bbox_width":-0.25})");
        write("cox_norm.json",R"({"pre_vnd_cost":{"std":4,"mean":2},"bbox_width":{"std":2,"mean":10}})");
        // Seed a bogus old score: successful transition must ignore it and backfill anew.
        first.has_cox_lp=true;first.cox_lp=999; best.has_cox_lp=true;
        finish(true,50);
        require(completed==50 && CoxLifecycle::ready(true,"COX",completed),"successful filter inactive");
        check_scores(std::exp(3.0),std::exp(1.0));
        require(!best.has_cox_lp,"detached best retained old score");
        fill();check_scores(std::exp(3.0),std::exp(1.0));

        // The real SurvivalModel cache is now loaded. Replacing the model and
        // completing training again must invalidate BOTH model and member caches.
        write("cox_coeffs.json",R"({"pre_vnd_cost":0.25,"bbox_width":-0.25})");
        finish(true,60);
        require(completed==60,"replacement completion not recorded");
        check_scores(std::exp(1.0),1.0);

        // Missing/bad model AFTER success: exercise end-of-iteration backfill
        // with a cache-miss solution and a reloaded model, not a test-side guard.
        model.invalidate(); first.has_cox_lp=false;
        fs::remove(dir/"cox_norm.json");
        rejects(fill,"Missing Cox model");
        require(!first.has_cox_lp,"missing model published a score");
        write("cox_norm.json","{}");
        rejects(fill,"normalization feature mismatch");
        require(!first.has_cox_lp,"invalid model published a score");
        rejects([&](){finish(true,70);},"normalization feature mismatch");
        require(completed==-1 && !first.has_cox_lp && !second.has_cox_lp,"bad successful trainer activated stale scores");

        // A previously loaded generation must also be discarded on training failure.
        write("cox_norm.json",R"({"pre_vnd_cost":{"mean":2,"std":4},"bbox_width":{"mean":10,"std":2}})");
        finish(true,80);check_scores(std::exp(1.0),1.0);
        fs::remove(dir/"cox_coeffs.json");fs::remove(dir/"cox_norm.json");
        finish(false,90);fill();
        require(completed==-1 && !first.has_cox_lp && !second.has_cox_lp,"failure reused prior success");
        std::cout<<"Production Cox lifecycle: pretraining, failure, success, model/cache replacement and post-success model errors passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
