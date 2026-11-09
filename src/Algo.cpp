/**
 * Algo.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Algo.hpp"
#include <filesystem>
#include <fstream>


Algo::Algo(Parameters* params) : population(params), data(params) {
    this->params            = params;
    this->random            = new Random(params->seed);
    this->iteration         = params->iteration;
    this->patience_threshold = params->patience;
    this->instance_index    = params->instance_index;
    this->timestamp         = params->timestamp;
    this->island_prefix_    = "[Island " + std::to_string(params->island_id) + "] ";
}

Algo::~Algo() {
    delete random;
}

void Algo::run() {
    Centers centers = data.getData();
    population.setContext(centers, random, timestamp);
    population.data = &data;

    auto start_run = std::chrono::high_resolution_clock::now();

    int  iter     = 0;
    int processed_iter = 0;
    int  best_iter = 0;
    int  patience = 0;
    bool improved  = false;
    bool ml_trained = false;
    bool ml_training_attempted = false;  // set true on first trigger, prevents re-firing
    std::chrono::duration<double> best_running_time{};

    List* best_solution = population.initPopulation();
    best_value_         = best_solution->getValue();
    double best_solution_value = best_value_;

    while (iter++ < iteration) {
        std::cout << island_prefix_ << "Iteration " << iter << "\n";

        auto start_iter = std::chrono::high_resolution_clock::now();

        population.current_iter = iter;
        const int reject_before = population.ml_reject_count;
        best_solution = population.nextPopulation(patience);
        if (params->filter_mode != "legacy") ++processed_iter;
        // An offspring filtered out by the ML model never reached VND, so it is
        // NOT evidence that the search has converged — it is just the filter doing
        // its job.  Such iterations are treated as NEUTRAL for early stopping
        // (they neither reset nor advance patience), so an aggressive filter can
        // no longer trigger a premature stop.  Patience thus measures genuine
        // search stagnation (VND'd offspring that failed to improve) only.
        const bool ml_rejected = population.ml_reject_count > reject_before;
        if (params->filter_mode != "legacy")
            std::cout << island_prefix_ << "[FILTER] iter=" << iter
                      << " rejected=" << population.last_filter_decision.rejected
                      << " reached_vnd=" << population.last_filter_decision.reached_vnd
                      << " rejects_total=" << population.ml_reject_count << "\n";

        improved           = best_solution->getValue() - best_solution_value < -DELTA;
        best_solution_value = best_solution->getValue();
        best_value_         = best_solution_value;

        auto end_iter = std::chrono::high_resolution_clock::now();

        // ==== ADAPTIVE ML TRAINING TRIGGER ====
        //
        // Trigger logic (fires at most once per run):
        //
        //   The training trigger is DECOUPLED from the stagnation/patience signal.
        //   `patience` is the shared early-stop + adaptive-mutation counter
        //   (see Population::nextPopulation) and must advance IDENTICALLY for the
        //   baseline island and every ML island — so training never reads or
        //   resets it.  Instead training fires on a fixed schedule:
        //
        //     Primary  — fires as soon as BOTH floors are met:
        //       (a) iter >= ML_TRAIN_FRAC_MIN * budget    → budget floor (guarded)
        //       (b) data.getEventCount() >= ML_MIN_EVENTS → enough death-signal
        //
        //     Hard fallback — overrides (b) at ML_TRAIN_FRAC_HARD * budget so that
        //       a stable population (few deaths) is never postponed indefinitely.
        //
        //   Firing early (right at the budget floor, while the solution log still
        //   holds a diverse mix of good/censored and bad/short-lived offspring)
        //   keeps the training set diverse AND leaves maximum runway for the
        //   filter to act.  Budget fractions scale with params->iteration so the
        //   trigger works for any -r value (200, 1000, 5000, …).
        {
        const int ml_min_iter  = static_cast<int>(params->iteration * params->train_frac_min);
        const int ml_hard_iter = static_cast<int>(params->iteration * params->train_frac_hard);

        if (params->ml_enable && !ml_trained && !ml_training_attempted
                && iter >= ml_min_iter) {

            int  n_events    = data.getEventCount();
            bool enough_data = n_events >= params->min_events;
            bool hard_ceil   = iter >= ml_hard_iter;

            bool primary  = enough_data;          // budget floor already guarded above
            bool fallback = hard_ceil;            // fires even without enough events

            if (primary || fallback) {
                if (!enough_data) {
                    std::cout << island_prefix_
                              << "[ML WARN] Hard-ceiling fallback: only " << n_events
                              << " events logged (< " << params->min_events
                              << "). Model quality may be limited.\n";
                }
                ml_training_attempted = true;  // never re-enter this block

                std::cout << island_prefix_
                          << "[ML] Training triggered at iter " << iter
                          << " (events=" << n_events
                          << ", patience=" << patience
                          << ", enough_data=" << enough_data
                          << ", hard_ceil=" << hard_ceil << ")\n";

                // Log surviving population members (censored).
                // Mark them so populationManagement won't log them again on death.
                for (List* s : population.population) {
                    if (!s->was_inserted) continue;
                    if (s->birth_iter < 0) continue;
                    s->death_iter        = iter;
                    s->censored          = true;
                    s->already_logged    = true;  // prevent double-logging on actual death
                    s->final_fitness     = s->getValue();
                    data.writeSolutionLog(s);
                }

                bool train_ok = run_ml_training();

                population.complete_ml_training(train_ok, iter);

                if (train_ok) {
                    ml_trained = true;
                    learned_filter_active_ = true;
                } else {
                    std::cerr << island_prefix_
                              << "[ML] Training failed — ML filter disabled for this run.\n";
                }

                // NOTE: patience is deliberately NOT reset here.  It is the shared
                // early-stop + adaptive-mutation counter (Population::nextPopulation)
                // and must advance identically for the baseline and every ML island.
                // Resetting it gave ML islands an unfair "second wind" (lower
                // mutation + extended search), which confounded the speed comparison.
            }
        }
        } // end ML training trigger scope

        if (improved) {
            best_iter         = iter;
            patience          = 0;
            best_running_time = end_iter - start_run;
            if (LOG) data.write(best_solution, iter,
                std::to_string(best_running_time.count()));
        } else if (!ml_rejected) {
            ++patience;          // genuine stagnation: VND ran but did not improve
        }
        // (else: ML-rejected iteration — neutral, patience unchanged)

        std::cout << island_prefix_
                  << "[LOG] iter=" << iter
                  << " best=" << best_solution->getValue()
                  << " patience=" << patience
                  << " ml=" << (params->filter_mode == "random" ? "random" :
                              !params->ml_enable ? "none" : ml_trained ? params->ml_model : "pending")
                  << " iter_t=" << std::chrono::duration<double>(end_iter - start_iter).count()
                  << " total_t=" << std::chrono::duration<double>(end_iter - start_run).count()
                  << "\n";

        // ==== EARLY STOPPING ====
        //
        // Fires when patience >= patience_threshold consecutive iterations
        // produce no improvement (patience_threshold set in Parameters.cpp).
        //
        // Design (model-independent stopping):
        //   `patience` advances purely on search stagnation and is NEVER touched
        //   by the ML training trigger.  Every island — the baseline and all 9 ML
        //   models — therefore stops on one and the same rule, so a speed/quality
        //   comparison between them is apples-to-apples.  If ML helps, it shows up
        //   as reaching better cost (or the same cost sooner) before the identical
        //   stagnation stop — not as an artificially extended run.
        //
        //   This also makes the budget genuinely adaptive: a fast-converging
        //   instance (e.g. car_door_25 from LA-CETSP, ~200 iters regardless of
        //   budget) stops early instead of wasting a 5000-iter budget, while a
        //   hard instance (e.g. bonus1000) keeps improving and runs to the cap.
        //
        // Motivation / reference:
        //   LA-CETSP (Balemir et al.) used a fixed 500-iter budget for all
        //   instances, which over-allocated time to easy instances and under-
        //   allocated to hard ones.  Patience-based stopping lets each island
        //   terminate as soon as its model+search combination converges,
        //   making wall-clock time a fairer reflection of true difficulty.
        if (patience >= patience_threshold) {
            std::cout << island_prefix_
                      << "[STOP] no improvement for " << patience_threshold
                      << " iterations (model="
                      << (ml_trained ? params->ml_model : "none") << ")"
                      << "\n";
            break;
        }

        const auto guard_time = params->filter_mode == "legacy" ? end_iter :
            std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(guard_time - start_run).count() > params->max_time) {
            std::cout << island_prefix_
                      << "[STOP] time limit reached (" << params->max_time << "s)\n";
            break;
        }
    }

    auto end_run = std::chrono::high_resolution_clock::now();
    total_time_  = std::chrono::duration<double>(end_run - start_run).count();

    if (!LOG) data.write(best_solution, best_iter,
                         std::to_string(best_running_time.count()));

    if (params->filter_mode != "legacy") {
        for (List* s : population.population)
            data.recordLifetime(s->candidate_id, s->birth_iter, processed_iter, true);
        data.verifyResearch();
        const auto& recorded = data.researchCounts();
        if (population.generated_count != population.ml_reject_count + population.vnd_evaluated_count ||
            population.vnd_evaluated_count != population.admitted_count + population.not_admitted_count ||
            population.generated_count != processed_iter ||
            recorded.generated != population.generated_count ||
            recorded.filter_rejected != population.ml_reject_count ||
            recorded.vnd_evaluated != population.vnd_evaluated_count ||
            recorded.admitted != population.admitted_count ||
            recorded.not_admitted != population.not_admitted_count)
            throw std::logic_error("candidate accounting invariant failed");
        total_time_ = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start_run).count();
    }

    std::cout << island_prefix_
              << "[SUMMARY] instance=" << FILENAMES[instance_index]
              << " model=" << (!params->ml_enable ? (params->filter_mode == "random" ? "RANDOM" : "BASELINE") :
                               (params->filter_mode == "legacy" || learned_filter_active_) ? params->ml_model : "LEARNED_INACTIVE")
              << " best=" << best_solution->getValue()
              << " best_iter=" << best_iter
              << " effective_iter=" << (params->filter_mode == "legacy" ? iter : processed_iter)
              << " total_t=" << total_time_
              << " ml_rejects=" << population.ml_reject_count
              << "\n";
    if (params->filter_mode != "legacy") {
        std::cout << island_prefix_ << "[CANDIDATES] generated=" << population.generated_count
                  << " filter_rejected=" << population.ml_reject_count
                  << " vnd_evaluated=" << population.vnd_evaluated_count
                  << " admitted=" << population.admitted_count
                  << " not_admitted=" << population.not_admitted_count
                  << " admitted/generated=";
        if (population.generated_count) std::cout << double(population.admitted_count)/population.generated_count;
        else std::cout << "undefined";
        std::cout << " admitted/vnd_evaluated=";
        if (population.vnd_evaluated_count) std::cout << double(population.admitted_count)/population.vnd_evaluated_count;
        else std::cout << "undefined";
        std::cout << "\n";
    }
}

bool Algo::run_ml_training() {
    std::cout << island_prefix_ << "[ML] Starting training (" << params->ml_model << ")...\n";

    const std::string& sd = params->scripts_dir;  // e.g. "C:/…/ml/scripts/"
    std::string script;
    if      (params->ml_model == "COX")        script = sd + "train_cox.py";
    else if (params->ml_model == "RSF")        script = sd + "train_rsf.py";
    else if (params->ml_model == "GBSA")       script = sd + "train_gbsa.py";
    else if (params->ml_model == "DEEPSURV")   script = sd + "train_deepsurv.py";
    else if (params->ml_model == "SSVM")       script = sd + "train_ssvm.py";
    else if (params->ml_model == "WEIBULLAFT") script = sd + "train_weibullaft.py";
    else if (params->ml_model == "KNN")        script = sd + "train_knn.py";
    else if (params->ml_model == "ELASTICNET") script = sd + "train_elasticnet.py";
    else if (params->ml_model == "MTLR")       script = sd + "train_mtlr.py";
    else {
        std::cerr << island_prefix_ << "[ML ERROR] Unknown ml_model: "
                  << params->ml_model << "\n";
        return false;
    }

    // Build absolute path to this island's log directory.
    // We pass the island-level dir (not the run-level) so the training script
    // accumulates data from ALL historical runs for this island.
    namespace fs = std::filesystem;
    fs::path log_dir = fs::weakly_canonical(
        (params->results_dir.empty() ? fs::current_path() / LOCAL_RES_DIR : fs::path(params->results_dir)) / "ml_logs" /
        FILENAMES[params->instance_index] /
        ("island_" + std::to_string(params->island_id))
    );
    if (params->filter_mode != "legacy") {
        const fs::path this_run = log_dir / ("run-" + params->timestamp);
        bool has_training_rows = false;
        if (fs::exists(this_run))
            for (const auto& file : fs::directory_iterator(this_run))
                if (file.path().extension() == ".json") { has_training_rows = true; break; }
        if (!has_training_rows) {
            std::cerr << island_prefix_ << "[ML ERROR] No current-run training rows; refusing historical fallback\n";
            return false;
        }
    }

    // Normalise helpers
    auto to_fwd = [](std::string s) {
        std::replace(s.begin(), s.end(), '\\', '/'); return s; };
    auto to_bwd = [](std::string s) {
        std::replace(s.begin(), s.end(), '/', '\\'); return s; };

    std::string model_dir_fwd = to_fwd(params->model_dir);
    std::string log_dir_fwd   = to_fwd(log_dir.string());
    std::string training_log  = model_dir_fwd + "training_log.txt";

    // Write a .bat file so cmd.exe quoting issues are avoided entirely.
    // The batch file is placed in the island's model dir (no spaces in path).
    std::string bat_path = to_bwd(model_dir_fwd) + "train.bat";
    {
        std::string py_bwd     = to_bwd(params->python_exe);
        std::string script_bwd = to_bwd(script);

        std::ofstream bat(bat_path);
        bat << "@echo off\r\n";
        bat << "\"" << py_bwd << "\""
            << " \"" << script_bwd << "\""
            << " --model_dir \"" << model_dir_fwd << "\""
            << " --log_dir \""   << log_dir_fwd   << "\""
            << " --logfile \""   << training_log  << "\"";
        if (params->filter_mode != "legacy" && params->ml_model == "COX")
            bat << " --dataset_dir \"" << model_dir_fwd << "\"";
        if (params->filter_mode != "legacy") bat << " --strict_log_dir";
        bat << "\r\n";
    }

    std::cout << island_prefix_ << "[ML] log_dir="   << log_dir_fwd   << "\n";
    std::cout << island_prefix_ << "[ML] model_dir=" << model_dir_fwd << "\n";

    // Execute the batch file directly (path has no spaces, no outer quotes needed)
    int result = system(bat_path.c_str());

    // Print the training log (written by Python itself) after the process exits.
    std::cout << island_prefix_ << "[ML] ---- training output ----\n";
    std::ifstream log_in(training_log);
    if (log_in) {
        std::string line;
        while (std::getline(log_in, line))
            std::cout << island_prefix_ << "  " << line << "\n";
    } else {
        std::cout << island_prefix_ << "  (no log file created)\n";
    }
    std::cout << island_prefix_ << "[ML] ---- end training output ----\n";

    if (result != 0) {
        std::cerr << island_prefix_ << "[ML ERROR] Training failed (exit=" << result << ")\n";
        return false;
    }
    std::cout << island_prefix_ << "[ML] Training completed successfully.\n";
    return true;
}
