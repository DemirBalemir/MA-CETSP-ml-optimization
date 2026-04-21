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
        best_solution = population.nextPopulation(patience);

        improved           = best_solution->getValue() - best_solution_value < -DELTA;
        best_solution_value = best_solution->getValue();
        best_value_         = best_solution_value;

        auto end_iter = std::chrono::high_resolution_clock::now();

        // ==== ADAPTIVE ML TRAINING TRIGGER ====
        // Fires at most once: on first stagnation/ceiling condition.
        // If training fails, the model just stays disabled — no re-triggering.
        if (params->ml_enable && !ml_trained && !ml_training_attempted
                && iter >= ML_MIN_TRAINING_ITER) {
            bool stagnation = patience >= static_cast<int>(patience_threshold * ML_PATIENCE_FRACTION);
            bool ceiling    = iter >= ML_MAX_TRAINING_ITER;

            if (stagnation || ceiling) {
                ml_training_attempted = true;  // never re-enter this block

                std::cout << island_prefix_
                          << "[ML] Training triggered at iter " << iter
                          << " (patience=" << patience
                          << ", stagnation=" << stagnation
                          << ", ceiling=" << ceiling << ")\n";

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

                if (train_ok) {
                    ml_trained = true;
                    population.training_completed_at = iter;
                } else {
                    std::cerr << island_prefix_
                              << "[ML] Training failed — ML filter disabled for this run.\n";
                }

                if (population.ml_model) {
                    // invalidate() clears Cox cache AND kills the Python server
                    // so the next prediction call restarts it with the fresh model
                    population.ml_model->invalidate();
                }

                // Back-fill Cox scores for current population
                if (params->ml_model == "COX") {
                    for (List* s : population.population) {
                        if (s->has_cox_lp) continue;
                        std::vector<std::pair<double,double>> coords = s->pre_vnd_coords;
                        if (coords.empty()) {
                            Node* p = s->head();
                            for (int i = 0; i < s->size(); ++i) {
                                coords.push_back({p->x, p->y});
                                p = p->next;
                            }
                        }
                        auto feats = GeometryFeatures::extract(coords);
                        double cost = (s->pre_vnd_value >= 0)
                                      ? s->pre_vnd_value : s->getValue();
                        feats["pre_vnd_cost"] = cost;
                        s->cox_lp     = population.predict_cox_score(feats);
                        s->has_cox_lp = true;
                    }
                }

                patience = 0;
            }
        }

        if (improved) {
            best_iter         = iter;
            patience          = 0;
            best_running_time = end_iter - start_run;
            if (LOG) data.write(best_solution, iter,
                std::to_string(best_running_time.count()));
        } else {
            ++patience;
        }

        std::cout << island_prefix_
                  << "[LOG] iter=" << iter
                  << " best=" << best_solution->getValue()
                  << " patience=" << patience
                  << " ml=" << (ml_trained ? params->ml_model : "pending")
                  << " iter_t=" << std::chrono::duration<double>(end_iter - start_iter).count()
                  << " total_t=" << std::chrono::duration<double>(end_iter - start_run).count()
                  << "\n";

        // Early stopping only active AFTER training has fired
        if (ml_trained && patience >= patience_threshold) {
            std::cout << island_prefix_
                      << "[STOP] no improvement for " << patience_threshold
                      << " iterations (post-training)\n";
            break;
        }

        if (std::chrono::duration<double>(end_iter - start_run).count() > params->max_time) {
            std::cout << island_prefix_
                      << "[STOP] time limit reached (" << params->max_time << "s)\n";
            break;
        }
    }

    auto end_run = std::chrono::high_resolution_clock::now();
    total_time_  = std::chrono::duration<double>(end_run - start_run).count();

    if (!LOG) data.write(best_solution, best_iter,
                         std::to_string(best_running_time.count()));

    std::cout << island_prefix_
              << "[SUMMARY] instance=" << FILENAMES[instance_index]
              << " model=" << params->ml_model
              << " best=" << best_solution->getValue()
              << " best_iter=" << best_iter
              << " total_t=" << total_time_
              << " ml_rejects=" << population.ml_reject_count
              << "\n";
}

bool Algo::run_ml_training() {
    std::cout << island_prefix_ << "[ML] Starting training (" << params->ml_model << ")...\n";

    std::string python_exec =
        "\"C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe\"";

    std::string script;
    if (params->ml_model == "COX") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_cox.py";
    } else if (params->ml_model == "RSF") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_rsf.py";
    } else if (params->ml_model == "GBSA") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_gbsa.py";
    } else if (params->ml_model == "DEEPSURV") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_deepsurv.py";
    } else if (params->ml_model == "SSVM") {
        script = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_ssvm.py";
    } else {
        std::cerr << island_prefix_ << "[ML ERROR] Unknown ml_model: "
                  << params->ml_model << "\n";
        return false;
    }

    // Build absolute path to this island's log directory.
    // We pass the island-level dir (not the run-level) so the training script
    // accumulates data from ALL historical runs for this island.
    namespace fs = std::filesystem;
    fs::path log_dir = fs::weakly_canonical(
        fs::current_path() / LOCAL_RES_DIR / "ml_logs" /
        FILENAMES[params->instance_index] /
        ("island_" + std::to_string(params->island_id))
    );

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
        std::string py_bwd     = to_bwd(
            "C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe");
        std::string script_bwd = to_bwd(script);

        std::ofstream bat(bat_path);
        bat << "@echo off\r\n";
        bat << "\"" << py_bwd << "\""
            << " \"" << script_bwd << "\""
            << " --model_dir \"" << model_dir_fwd << "\""
            << " --log_dir \""   << log_dir_fwd   << "\""
            << " --logfile \""   << training_log  << "\"\r\n";
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
