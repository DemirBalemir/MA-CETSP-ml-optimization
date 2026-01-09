/**
 * Algo.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Algo.hpp"


Algo::Algo(Parameters* params) : population(params), data(params) {
    this->params = params;
    this->random = new Random(params->seed);
    this->iteration = params->iteration;
    this->patience_threshold = params->patience;
    this->instance_index = params->instance_index;
    this->timestamp = params->timestamp;
}

Algo::~Algo() {
    delete random;
}

void Algo::run() {
    // read data;
    Centers centers = data.getData();
    // set context
    population.setContext(centers, random, timestamp);
    population.data = &data;


    auto start_run = std::chrono::high_resolution_clock::now();

    int iter = 0;
    int best_iter = 0;
    int patience = 0;
    bool improved = false;
    std::chrono::duration<double> best_running_time;
    // init population
    List* best_solution = population.initPopulation();
    double best_solution_value = best_solution->getValue();

    // iteration
    while (iter++ < iteration) {
        std::cout << "\nIteration " << iter << " : " << std::endl;

        auto start_iter = std::chrono::high_resolution_clock::now();

        population.current_iter = iter;

        // === produce next gen (birth/death happens here)
        best_solution = population.nextPopulation(patience);

        improved = best_solution->getValue() - best_solution_value < -DELTA;
        best_solution_value = best_solution->getValue();

        auto end_iter = std::chrono::high_resolution_clock::now();


        // ==== AT EXACT ITERATION TRAINING_TIME: LOG SURVIVORS ====
        if (ML_ENABLE && iter == TRAINING_TIME) {

            for (List* s : population.population) {

                if (!s->was_inserted) continue;
                if (s->birth_iter < 0) continue;
                if (s->birth_iter > TRAINING_TIME) continue;

                s->death_iter = TRAINING_TIME;
                s->censored = true;
                s->final_fitness = s->getValue();

                data.writeSolutionLog(s);
            }
            run_ml_training();

            if (population.ml_model) {
                population.ml_model->reset_cox_cache();
            }


            if (ML_MODEL == "COX") {
                for (List* s : population.population) {

                    if (s->has_cox_lp) continue;

                    std::vector<std::pair<double, double>> coords = s->pre_vnd_coords;
                    if (coords.empty()) {
                        Node* p = s->head();
                        for (int i = 0; i < s->size(); ++i) {
                            coords.push_back({ p->x, p->y });
                            p = p->next;
                        }
                    }

                    auto feats = GeometryFeatures::extract(coords);


                    double cost = (s->pre_vnd_value >= 0)
                        ? s->pre_vnd_value
                        : s->getValue();

                    feats["pre_vnd_cost"] = cost;

                    s->cox_lp = population.predict_cox_score(feats);
                    s->has_cox_lp = true;
                }
            }




        }


        if (improved) {
            best_iter = iter;
            patience = 0;
            best_running_time = end_iter - start_run;
            if (LOG) data.write(best_solution, iter,
                std::to_string(best_running_time.count()));
        }
        else {
            ++patience;
        }

        std::cout << "[LOG] iter: " << iter
            << " best_iter: " << best_iter
            << " best_value: " << best_solution->getValue()
            << " iter_time: " << std::chrono::duration<double>(end_iter - start_iter).count()
            << " total_time: " << std::chrono::duration<double>(end_iter - start_run).count()
            << std::endl;

        if (patience >= patience_threshold) {
            std::cout << std::endl << "[STOP] best solution hasn't been improved since " << patience_threshold << " iterations" << std::endl;
            break;
        }

        if (std::chrono::duration<double>(end_iter - start_run).count() > params->max_time) {
            std::cout << std::endl << "[STOP] running time exceeds " << params->max_time << " seconds" << std::endl;
            break;
        }
    }

    auto end_run = std::chrono::high_resolution_clock::now();

    if (!LOG)  data.write(best_solution, best_iter, std::to_string(best_running_time.count()));

    std::cout << std::endl
        << "[SUMMARY] instance: " << FILENAMES[instance_index]
        << " best_value: " << best_solution->getValue()
        << " best_time: " << std::to_string(best_running_time.count())
        << " total_time: " << std::chrono::duration<double>(end_run - start_run).count()
        << " result_file: " << data.getResultFilename()
        << std::endl;

    std::cout << "[ML] Total offspring rejected before VND: "
        << population.ml_reject_count << std::endl;

}
void Algo::run_ml_training() {
    std::cout << "\n[ML] Starting machine learning training phase..." << std::endl;

    std::string python_exec =
        "\"C:/Users/Demir/AppData/Local/Programs/Python/Python310/python.exe\"";

    std::string script_path;

    if (ML_MODEL == "COX") {
        script_path = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_cox.py";
    }
    else if (ML_MODEL == "RSF") {
        script_path = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_rsf.py";
    }
    else if (ML_MODEL == "GBSA") {
        script_path = "C:/Users/Demir/researchproject/MA-CETSP/ml/scripts/train_gbsa.py";
    }
    else {
        std::cerr << "[ML ERROR] Unknown ML_MODEL in Defs.hpp: " << ML_MODEL << std::endl;
        return;
    }

    std::string cmd = python_exec + " " + script_path;

    std::cout << "[ML] Running command: " << cmd << std::endl;

    int result = system(cmd.c_str());

    if (result != 0) {
        std::cerr << "[ML ERROR] Training script failed. Exit code: "
            << result << std::endl;
    }
    else {
        std::cout << "[ML] Training completed successfully.\n" << std::endl;
    }
}