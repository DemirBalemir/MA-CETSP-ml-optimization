/**
 * Parameters.hpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#ifndef CETSP_PARAMETERS_HPP
#define CETSP_PARAMETERS_HPP


#include <iostream>
#include <random>
#include <ctime>
#include "cmdline.h"
#include "Defs.hpp"

class Parameters {
public:
    int seed = 0;
    int random_num;
    std::string timestamp;
    std::string init;
    std::string select;
    std::string crossover;
    std::string improvement;
    std::string greed;
    std::string distance;
    int instance_index;
    int population_size;
    int iteration;
    int patience;
    double max_time;
    double fit_beta;
    int dist_th;
    int neighbor_size;

    // ---- island / ML fields ----
    int         n_islands  = N_ISLANDS;   // total number of parallel islands
    int         island_id  = 0;           // this island's index (set by parallel runner)
    std::string ml_model   = ML_MODEL;    // "COX", "RSF", or "GBSA"
    bool        ml_enable  = ML_ENABLE;   // enable/disable ML filter
    std::string model_dir;                // path to per-island model directory (derived)

    Parameters(int argc, char **argv);
    Parameters() = default;
    ~Parameters() = default;
    void print() const;

    // Create a copy configured for a specific island index.
    // Called by the parallel runner in main.cpp.
    Parameters make_island(int id, const std::string& model) const;
};


#endif //CETSP_PARAMETERS_HPP
