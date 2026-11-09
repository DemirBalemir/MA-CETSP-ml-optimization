/**
 * Population.hpp
 * created on : March 03 2022
 * author : Z.LEI
 **/

#ifndef CETSP_POPULATION_HPP
#define CETSP_POPULATION_HPP

#include "Defs.hpp"
#include "Utils/Random.hpp"
#include "Utils/Data.hpp"
#include "List.hpp"
#include "Geometry.hpp"
#include "LocalSearch/LocalSearch.hpp"
#include "Neighbor.hpp"
#include "Utils/Kmeans.hpp"
#include "Distance.hpp"
#include "Crossover/CrossoverFactory.hpp"
#include <vector>
#include <unordered_map>
#include <GeometryFeatures.hpp>
#include <SurvivalModel.hpp>
#include "Utils/PrunePolicy.hpp"
#include <memory>




class Population {
private:
    Random* random;
    LocalSearch ls;
    Crossover* crossover;
    Kmeans kmeans;
    Centers centers;
    Neighbor neighbor;
    List* best_solution;
    int population_size;
    std::string initialization;
    std::string selection;
    std::string crossover_type;
    double fit_beta;
    int dist_th;
    std::unordered_map<List*, std::vector<double>> solution_map;
    List* initSolution();
    List* randomSolution();
    List* kmeansSolution();
    std::pair<List*, List*> chooseParent();
    bool insertSolution(List* s);
    void updateDistances();
    void populationManagement();
    void randomSwap(List* s);


public:
    std::vector<List*> population;
    Population(Parameters* params);
    ~Population();
    void setContext(Centers& centers, Random* random, std::string timestamp);
    List* initPopulation();
    List* nextPopulation(int patience);
    int current_iter = -1;
    int ml_reject_count = 0;
    int generated_count = 0, vnd_evaluated_count = 0;
    int admitted_count = 0, not_admitted_count = 0;
    // Last offspring outcome for the candidate logger (task 2).
    struct FilterDecision { bool rejected = false; bool reached_vnd = false; };
    FilterDecision last_filter_decision;
    std::unique_ptr<PrunePolicy> prune_policy;
    std::string filter_mode_;
    int training_completed_at = -1;  // successful training only; -1 = pending or failed

    // rolling rejection-rate cap state
    int  ml_win_attempts_ = 0;   // ML-eligible offspring seen in current window
    int  ml_win_rejects_  = 0;   // rejections in current window
    bool ml_suspended_    = false; // true → filter suspended this window
    Data* data = nullptr;
    SurvivalModel* ml_model;

    // Runtime ML config (copied from params at construction — avoids Defs.hpp constants)
    std::string ml_model_name;
    bool        ml_enable_;

    void complete_ml_training(bool success, int iteration);
    void backfill_cox_scores();
    double compute_relative_lp_threshold(double q);
    double predict_cox_score(const std::map<std::string, double>& feats);





};

#endif //CETSP_POPULATION_HPP
