/**
 * Population.cpp
 * created on : March 03 2022
 * author : Z.LEI
 **/

#include "Genetic/Population.hpp"
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <cmath>
#include <map>
#include <sstream>

Population::Population(Parameters* params)
    : neighbor(params->neighbor_size),
    ls(params),
    random(nullptr),          // ✅ ADD THIS
    best_solution(nullptr),
    crossover(nullptr),
    ml_model(new SurvivalModel())   // ✅ ADD THIS

{
    this->initialization = params->init;
    this->selection = params->select;
    this->crossover_type = params->crossover;
    this->population_size = params->population_size;
    this->fit_beta = params->fit_beta;
    this->dist_th = params->dist_th;
}


Population::~Population() {
    for (int i = 0; i < population.size(); ++i) {
        delete population[i];
    }
    delete best_solution;
    delete crossover;
    delete ml_model;   // ✅ ADD THIS

}

void Population::setContext(Centers& centers, Random* random, std::string timestamp) {
    this->random = random;
    this->centers = centers;
    neighbor.setContext(centers.size());
    ls.setContext(random, centers, &neighbor, timestamp);
    kmeans.setContext(random, centers);
    crossover = CrossoverFactory::createCrossover(crossover_type);
    crossover->setContext(random);
}

List* Population::randomSolution() {
    std::vector<int> ids(centers.size() - 1);
    std::iota(ids.begin(), ids.end(), 1);
    random->permutation(ids);
    List* solution = new List();
    Node* head = new Node(0, centers[0][0], centers[0][1]);
    solution->add(head);
    for (int i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        double theta = random->randomDoubleDistr(0, 2 * PI);
        double r = random->randomDoubleDistr(0, 1);
        double x = centers[id][0] + r * centers[id][2] * cos(theta);
        double y = centers[id][1] + r * centers[id][2] * sin(theta);
        if (pow(x - centers[id][0], 2) + pow(y - centers[id][1], 2) > pow(centers[id][2], 2)) {
            std::cout << "ERROR : init point out of circle" << std::endl;
        }
        Node* node = new Node(id, x, y);
        solution->add(node);
    }
    return solution;
}

List* Population::kmeansSolution() {
    std::vector<std::vector<int>> groups = kmeans.getGroups();
    // random.permutation(groups);
    List* solution = new List();
    for (int i = 0; i < groups.size(); ++i) {
        random->permutation(groups[i]);
        for (int j = 0; j < groups[i].size(); ++j) {
            int id = groups[i][j];
            double x = centers[id][0], y = centers[id][1];
            if (id != 0) {
                double theta = random->randomDoubleDistr(0, 2 * PI);
                double r = random->randomDoubleDistr(0, 1);
                x = centers[id][0] + r * centers[id][2] * cos(theta);
                y = centers[id][1] + r * centers[id][2] * sin(theta);
                if (pow(x - centers[id][0], 2) + pow(y - centers[id][1], 2) > pow(centers[id][2], 2)) {
                    std::cout << "ERROR : init point out of circle" << std::endl;
                }
            }
            Node* node = new Node(id, x, y);
            solution->add(node);
        }
    }
    Node* p = solution->head();
    while (p->id != 0) {
        p = p->next;
    }
    solution->setHead(p);
    return solution;
}

List* Population::initSolution() {
    List* solution = nullptr;
    if (initialization == "RANDOM") {
        solution = randomSolution();
    }
    else if (initialization == "KMEANS") {
        solution = kmeansSolution();
    }
    else {
        std::cerr << "[ERROR] invalid initialization method" << std::endl;
    }
    solution = ls.initSolOpt(solution);
    return solution;
}

List* Population::initPopulation() {
    auto start = std::chrono::high_resolution_clock::now();

    if (initialization == "KMEANS") {
        auto start_kmeans = std::chrono::high_resolution_clock::now();
        kmeans.run();
        auto end_kmeans = std::chrono::high_resolution_clock::now();
        std::cout << "kmeans time : " << std::chrono::duration <double>(end_kmeans - start_kmeans).count() << " s" << std::endl;
    }

    List* best = nullptr;

    for (int i = 0; i < population_size; ++i) {
        List* solution = initSolution();
        if (!solution) continue;

        insertSolution(solution);
        if (ML_ENABLE && ML_MODEL == "COX" && !solution->has_cox_lp) {
            auto coords = solution->pre_vnd_coords;
            if (coords.empty()) {
                Node* p = solution->head();
                for (int i = 0; i < solution->size(); ++i) {
                    coords.push_back({ p->x, p->y });
                    p = p->next;
                }
            }

            auto feats = GeometryFeatures::extract(coords);
            feats["pre_vnd_cost"] = solution->getValue();

            solution->cox_lp = predict_cox_score(feats);
            solution->has_cox_lp = true;
        }


        if (!best || solution->getValue() < best->getValue()) {
            best = solution;
        }
    }

    if (!best) {
        throw std::runtime_error("Population initialization failed: no valid solutions");
    }

    best_solution = new List(*best);

    std::sort(population.begin(), population.end(), [](List* s1, List* s2) {
        return s1->getValue() < s2->getValue();
        });

    auto end = std::chrono::high_resolution_clock::now();
    if (LOG) {
        std::cout << initialization << " init time : " << std::chrono::duration <double>(end - start).count() << " s" << std::endl;
    }
    neighbor.updateNeighbors();
    return best_solution;
}

std::pair<List*, List*> Population::chooseParent() {
    int size = population.size();
    int i = -1, j = -1;
    if (selection == "RANDOM") {
        while (i == j) {
            i = random->randomInt(size);
            j = random->randomInt(size);
        }
    }
    else if (selection == "ROULETTE") {
        int sum_portion = (1 + size) * size / 2;
        int lucky;
        while (i == j) {
            lucky = random->randomInt(sum_portion) + 1;
            i = size - int(sqrt(2 * lucky));
            lucky = random->randomInt(sum_portion) + 1;
            j = size - int(sqrt(2 * lucky));
        }
        if (i > j) {
            i = i ^ j;
            j = i ^ j;
            i = i ^ j;
        }
    }
    if (LOG) {
        std::cout << "parents indices : " << i << " " << j << std::endl;
    }
    return { population[i], population[j] };
}

List* Population::nextPopulation(int patience) {
    std::pair<List*, List*> parents;
    double dist1 = 0, dist2 = 0;
    List* offspring = nullptr;
    int try_times = 5;

    // ================= CREATE OFFSPRING =================
    while (dist1 == 0 || dist2 == 0) {
        if (try_times-- <= 0) {
            randomSwap(offspring);
        }
        else {
            parents = chooseParent();
            offspring = crossover->run(parents.first, parents.second);
        }
        dist1 = Distance::run(offspring, parents.first);
        dist2 = Distance::run(offspring, parents.second);
        if (LOG) {
            std::cout << "dists between offspring and parents :" << dist1 << " " << dist2 << std::endl;
        }
    }

    // Mutation
    if (random->randomInt(1000) < patience) {
        randomSwap(offspring);
    }

    // ================= PRE-VND COST =================
    offspring->evaluate();
    double pre_cost = offspring->getValue();

    // ================= PRE-VND RAW COORDS =================
    std::vector<std::pair<double, double>> raw_coords;
    Node* p_raw = offspring->head();
    for (int k = 0; k < offspring->size(); ++k) {
        raw_coords.push_back({ p_raw->x, p_raw->y });
        p_raw = p_raw->next;
    }
    // ================= ML FILTER =================
    if (ML_ENABLE && current_iter > TRAINING_TIME) {

        // ---- feature extraction ----
        auto feats = GeometryFeatures::extract(raw_coords);
        feats["pre_vnd_cost"] = pre_cost;

        if (ML_MODEL == "COX") {

            double threshold = ML_THRESHOLD;

            // IMPORTANT: this must be exp(beta^T x)
            double score = predict_cox_score(feats);

            offspring->cox_lp = score;
            offspring->has_cox_lp = true;

            if (LOG) {
                std::cout << "[ML-COX] score=" << score
                    << " threshold=" << threshold << "\n";
            }

            // === EXACT OLD BEHAVIOR ===
            if (score > threshold) {
                ml_reject_count++;
                if (LOG) {
                    std::cout << "[ML] Offspring rejected before VND (COX). "
                        << "score=" << score << "\n";
                }
                return best_solution;
            }
        }
        else if (ML_MODEL == "RSF" || ML_MODEL == "GBSA") {

            double threshold = 0.0;
            if (ML_MODEL == "RSF") {
                threshold = ml_model->load_rsf_threshold();
            }
            else { // ML_MODEL == "GBSA"
                threshold = ml_model->load_gbsa_threshold();
            }

            // ---- convert to JSON ----
            std::string json = GeometryFeatures::to_json(feats);

            // ---- ML predict via Python ----
            double score = ml_model->predict_survival_score(json);

            if (LOG) {
                std::cout << "[ML-" << ML_MODEL << "] score=" << score
                    << " threshold=" << threshold << "\n";
            }

            // ---- reject low-survival offspring ----
            if (score > threshold) {
                ml_reject_count++;
                if (LOG) {
                    std::cout << "[ML] Offspring rejected before VND ("
                        << ML_MODEL << "). Score=" << score << "\n";
                }
                return best_solution;
            }
        }
        else {
            std::cerr << "[ML ERROR] Unknown ML_MODEL = '" << ML_MODEL << "'\n";
        }
    }



    // ================= VND IMPROVEMENT =================
    offspring = ls.VND(offspring);

    // ================= POST-VND COST =================
    offspring->evaluate();
    double post_cost = offspring->getValue();

    // ================= SAVE TO OFFSPRING OBJECT =================
    offspring->pre_vnd_value = pre_cost;
    offspring->post_vnd_value = post_cost;
    offspring->pre_vnd_coords = raw_coords;
    offspring->birth_iter = current_iter;
    offspring->instance_index = data->instance_index;

    // ================= INSERT & SURVIVAL MGMT =================
    insertSolution(offspring);
    populationManagement();

    offspring->post_vnd_fitness_at_birth = offspring->getFitness();
    if (ML_ENABLE && ML_MODEL == "COX") {
        for (List* s : population) {

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
            feats["pre_vnd_cost"] =
                (s->pre_vnd_value >= 0) ? s->pre_vnd_value : s->getValue();

            s->cox_lp = predict_cox_score(feats);
            s->has_cox_lp = true;
        }
    }

    // ================= LOGGING =================
    if (LOG) {
        std::cout << "standard population size : " << population_size
            << ", current population size : " << population.size() << std::endl;
        std::cout << "population distances : " << std::endl;
        for (auto& p : population) {
            std::cout << p->getDistance() << " ";
        }
        std::cout << std::endl;
        std::cout << "population costs : " << std::endl;
        for (auto& p : population) {
            std::cout << p->getValue() << " ";
        }
        std::cout << std::endl;
    }

    return best_solution;
}


bool Population::insertSolution(List* s) {
    double distance_threshold = dist_th;
    double min_dist = INT_MAX;
    std::vector<double> distances(population.size(), 0);
    for (int i = 0; i < population.size(); ++i) {
        double dist = Distance::run(s, population[i]);
        min_dist = std::min(min_dist, dist);
        distances[i] = std::min(population[i]->getDistance(), dist);
    }
    s->setDistance(min_dist);
    if ((min_dist > 0 && best_solution && s->getValue() < best_solution->getValue()) || min_dist > distance_threshold) {
        s->was_inserted = true;

        for (int i = 0; i < population.size(); ++i) {
            population[i]->setDistance(distances[i]);
        }
        population.emplace_back(s);
        neighbor.updateCentroids(s);

        return true;
    }
    else {
        s->was_inserted = false;

        return false;
    }
}

void Population::updateDistances() {
    std::vector<double> distances(population.size(), INT_MAX);
    for (int i = 0; i < population.size(); ++i) {
        for (int j = i + 1; j < population.size(); ++j) {
            double dist = Distance::run(population[i], population[j]);
            distances[i] = std::min(distances[i], dist);
            distances[j] = std::min(distances[j], dist);
        }
    }
    for (int i = 0; i < population.size(); ++i) {
        population[i]->setDistance(distances[i]);
    }
}

void Population::populationManagement() {
    std::unordered_map<List*, std::vector<double>> rank;

    // value rank
    std::sort(population.begin(), population.end(), [](List* s1, List* s2) {
        return s1->getValue() < s2->getValue();
        });

    for (int i = 0; i < population.size(); ++i) {
        rank[population[i]].emplace_back(100.0 * i / (population.size() - 1));
    }

    // distance rank
    std::sort(population.begin(), population.end(), [](List* s1, List* s2) {
        return s1->getDistance() > s2->getDistance();
        });

    for (int i = 0; i < population.size(); ++i) {
        rank[population[i]].emplace_back(100.0 * i / (population.size() - 1));
    }

    for (int i = 0; i < population.size(); ++i) {
        double alpha = 1, beta = fit_beta;
        double fitness = alpha * rank[population[i]][0] + beta * rank[population[i]][1];
        population[i]->setFitness(fitness);
    }

    std::sort(population.begin(), population.end(), [](List* s1, List* s2) {
        return s1->getFitness() < s2->getFitness();
        });

    if (population.size() >= 1.5 * population_size) {

        // ========== LOGGING: DEATH OF REMOVED SOLUTIONS ==========
        // Solutions that will be deleted are from index population_size to old_size-1
        int old_size = population.size();
        for (int i = population_size; i < old_size; ++i) {
            List* dying = population[i];


            if (!dying->was_inserted)
                continue;
            if (dying->birth_iter < 0)
                continue;
            // Assign death iteration
            dying->death_iter = current_iter;

            dying->censored = false;


            // Log final objective value
            dying->final_fitness = dying->getValue();

            // Call Data logger
            if (current_iter <= TRAINING_TIME) {
                data->writeSolutionLog(dying);
            }
        }

        // Actually delete them
        population.resize(population_size);

        // The rest stays the same
        neighbor.updateNeighbors();
        updateDistances();
    }

    List* best = *std::min_element(population.begin(), population.end(), [](List* s1, List* s2) {
        return s1->getValue() < s2->getValue();
        });

    if (best->getValue() < best_solution->getValue()) {
        delete best_solution;
        best_solution = new List(*best);
    }

}

void Population::randomSwap(List* s) {
    int steps = 5;
    while (steps-- > 0) {
        int i = random->randomInt(s->size() - 1) + 1;
        int j = random->randomInt(s->size() - 1) + 1;
        while (i == j && abs(i - j) == 1) {
            i = random->randomInt(s->size() - 1) + 1;
            j = random->randomInt(s->size() - 1) + 1;
        }
        Node* p1 = s->head();
        Node* p2 = s->head();
        while (i-- > 0) p1 = p1->next;
        while (j-- > 0) p2 = p2->next;
        Node::swap(p1, p2);
    }
}
double Population::predict_cox_score(
    const std::map<std::string, double>& feats
) {
    if (!ml_model) return 0.0;
    return ml_model->predict_cox_score(feats);
}

double Population::compute_relative_lp_threshold(double q) {
    std::vector<double> lps;

    for (List* s : population) {
        if (s->has_cox_lp) {
            lps.push_back(s->cox_lp);
        }
    }

    // warm-up safeguard
    size_t ref_size = population.size();

    if (lps.size() < ref_size / 2) {
        return std::numeric_limits<double>::infinity();
    }

    std::sort(lps.begin(), lps.end());

    size_t idx = static_cast<size_t>(q * lps.size());
    if (idx >= lps.size()) idx = lps.size() - 1;

    return lps[idx];
}