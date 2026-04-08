/**
 * Algo.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Algo.hpp"
#include <filesystem>


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
        if (params->ml_enable && !ml_trained && iter >= ML_MIN_TRAINING_ITER) {
            bool stagnation = patience >= static_cast<int>(patience_threshold * ML_PATIENCE_FRACTION);
            bool ceiling    = iter >= ML_MAX_TRAINING_ITER;

            if (stagnation || ceiling) {
                std::cout << island_prefix_
                          << "[ML] Training triggered at iter " << iter
                          << " (patience=" << patience
                          << ", stagnation=" << stagnation
                          << ", ceiling=" << ceiling << ")\n";

                // Log surviving population members (censored)
                for (List* s : population.population) {
                    if (!s->was_inserted) continue;
                    if (s->birth_iter < 0) continue;
                    s->death_iter    = iter;
                    s->censored      = true;
                    s->final_fitness = s->getValue();
                    data.writeSolutionLog(s);
                }

                run_ml_training();

                ml_trained = true;
                population.training_completed_at = iter;

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

    if (!LOG) data.write(best_solution, best_iter,
                         std::to_string(best_running_time.count()));

    std::cout << island_prefix_
              << "[SUMMARY] instance=" << FILENAMES[instance_index]
              << " model=" << params->ml_model
              << " best=" << best_solution->getValue()
              << " best_iter=" << best_iter
              << " total_t=" << std::chrono::duration<double>(end_run - start_run).count()
              << " ml_rejects=" << population.ml_reject_count
              << "\n";
}

void Algo::run_ml_training() {
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
    } else {
        std::cerr << island_prefix_ << "[ML ERROR] Unknown ml_model: "
                  << params->ml_model << "\n";
        return;
    }

    // Build absolute path to this island's log directory so the training script
    // reads only THIS island's solutions (avoids parallel-island file races).
    namespace fs = std::filesystem;
    std::string log_dir = fs::weakly_canonical(
        fs::current_path() / LOCAL_RES_DIR / "ml_logs" /
        (FILENAMES[params->instance_index] + "-" + params->timestamp)
    ).string();

    // Pass per-island model dir (absolute) and log dir so islands are fully isolated.
    std::string cmd = python_exec
                    + " \"" + script + "\""
                    + " --model_dir \"" + params->model_dir + "\""
                    + " --log_dir \"" + log_dir + "\"";

    std::cout << island_prefix_ << "[ML] cmd: " << cmd << "\n";

    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << island_prefix_ << "[ML ERROR] Training failed (exit=" << result
                  << ") — check Python output above for details\n";
    } else {
        std::cout << island_prefix_ << "[ML] Training completed successfully.\n";
    }
}
