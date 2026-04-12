/**
 * Algo.hpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#ifndef CETSP_ALGO_HPP
#define CETSP_ALGO_HPP

#include "Defs.hpp"
#include "Utils/Parameters.hpp"
#include "Utils/Data.hpp"
#include "List.hpp"
#include "Genetic/Population.hpp"
#include <vector>
#include <chrono>
#include <climits>

class Algo {
private:
    Random* random;
    Parameters* params;
    Data data;
    Population population;
    int iteration;
    int patience_threshold;
    int instance_index;
    std::string timestamp;
    double best_value_ = std::numeric_limits<double>::max();
    std::string island_prefix_;   // "[Island N] " prepended to log lines

public:
    Algo(Parameters* params);
    ~Algo();
    void run();
    bool run_ml_training();  // returns true on success
    double get_best_value()   const { return best_value_; }
    int    get_reject_count() const { return population.ml_reject_count; }
};

#endif //CETSP_ALGO_HPP
