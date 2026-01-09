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
    Data* data = nullptr;    // ADD THIS LINE 
    SurvivalModel* ml_model;

    double compute_relative_lp_threshold(double q);//keep
    double predict_cox_score(const std::map<std::string, double>& feats);





};

#endif //CETSP_POPULATION_HPP