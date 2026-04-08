/**
 * Parameters.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Utils/cmdline.h"
#include "Utils/Parameters.hpp"
#include <filesystem>

Parameters::Parameters(int argc, char **argv) {
    cmdline::parser parser;
    parser.add<int>("random",       'g', "random number for LKH",          false, 0);
    parser.add<int>("instance",     'i', "instance index",                  false, INSTANCE_INDEX);
    parser.add<int>("seed",         's', "seed for random generator",       false, SEED);
    parser.add<std::string>("init",        '\0', "initialization",          false, INITIALIZATION);
    parser.add<std::string>("select",      '\0', "selection",               false, SELECTION);
    parser.add<std::string>("crossover",   '\0', "crossover",               false, CROSSOVER);
    parser.add<std::string>("improvement", '\0', "improvement",             false, IMPROVEMENT);
    parser.add<std::string>("greed",       '\0', "greed",                   false, GREEDY_ALGO);
    parser.add<std::string>("distance",    '\0', "distance",                false, DISTANCE);
    // parameters
    parser.add<int>   ("pop_size",    'p', "population size",              false, POPULATION_SIZE);
    parser.add<int>   ("iteration",   'r', "iteration",                    false, ITERATION);
    parser.add<double>("max_time",    't', "max running time",             false, MAX_TIME);
    parser.add<double>("fit_beta",    'b', "coefficient for fitness",      false, FIT_BETA);
    parser.add<int>   ("dist_th",     'd', "distance threshold",           false, DISTANCE_THRESHOLD);
    parser.add<int>   ("neighbor_size",'n',"neighbor size",                false, NEIGHBOR_SIZE);
    // island / ML parameters
    parser.add<int>        ("islands",   '\0', "number of parallel islands", false, N_ISLANDS);
    parser.add<std::string>("ml_model",  '\0', "ML model: COX, RSF, GBSA",   false, ML_MODEL);
    parser.add<bool>       ("ml_enable", '\0', "enable ML filtering",         false, ML_ENABLE);

    parser.parse_check(argc, argv);

    random_num      = parser.get<int>("random");
    instance_index  = parser.get<int>("instance");
    seed            = parser.get<int>("seed");
    init            = parser.get<std::string>("init");
    select          = parser.get<std::string>("select");
    crossover       = parser.get<std::string>("crossover");
    improvement     = parser.get<std::string>("improvement");
    greed           = parser.get<std::string>("greed");
    distance        = parser.get<std::string>("distance");
    population_size = parser.get<int>("pop_size");
    iteration       = parser.get<int>("iteration");
    patience        = iteration;
    max_time        = parser.get<double>("max_time");
    fit_beta        = parser.get<double>("fit_beta");
    dist_th         = parser.get<int>("dist_th");
    neighbor_size   = parser.get<int>("neighbor_size");
    n_islands       = parser.get<int>("islands");
    ml_model        = parser.get<std::string>("ml_model");
    ml_enable       = parser.get<bool>("ml_enable");

    // island_id stays 0 for single-island runs; set by make_island() otherwise
    island_id  = 0;
    timestamp  = std::to_string(std::time(nullptr));
    model_dir  = "../../ml/models/";  // default: same as before
}

Parameters Parameters::make_island(int id, const std::string& model) const {
    Parameters p = *this;
    p.island_id = id;
    p.ml_model  = model;
    p.seed      = this->seed + id;
    // unique timestamp per island so LKH tmp dirs and log dirs don't collide
    p.timestamp = this->timestamp + "_i" + std::to_string(id);
    // per-island model directory; island 0 in a single-island run keeps the default path
    if (n_islands > 1 || id > 0) {
        p.model_dir = "../../ml/models/island_" + std::to_string(id) + "/";
        // ensure the directory exists
        std::filesystem::create_directories(p.model_dir);
    }
    return p;
}

void Parameters::print() const {
    std::cout << "[Island " << island_id << "] "
              << "instance=" << instance_index
              << " seed=" << seed
              << " pop=" << population_size
              << " iter=" << iteration
              << " ml=" << (ml_enable ? ml_model : "OFF")
              << " model_dir=" << model_dir
              << " timestamp=" << timestamp
              << std::endl;
}
