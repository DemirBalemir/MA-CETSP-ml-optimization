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
    int training_completed_at = -1;  // set by Algo after training fires; -1 = not yet trained

    // rolling rejection-rate cap state
    int  ml_win_attempts_ = 0;   // ML-eligible offspring seen in current window
    int  ml_win_rejects_  = 0;   // rejections in current window
    bool ml_suspended_    = false; // true → filter suspended this window
    Data* data = nullptr;
    SurvivalModel* ml_model;

    // Runtime ML config (copied from params at construction — avoids Defs.hpp constants)
    std::string ml_model_name;
    bool        ml_enable_;

    double compute_relative_lp_threshold(double q);
    double predict_cox_score(const std::map<std::string, double>& feats);





};

#endif //CETSP_POPULATION_HPP