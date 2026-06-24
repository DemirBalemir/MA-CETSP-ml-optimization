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
    parser.add<int>        ("islands",    '\0', "number of parallel islands",  false, N_ISLANDS);
    parser.add<std::string>("ml_model",   '\0', "ML model: COX, RSF, GBSA",    false, ML_MODEL);
    parser.add<bool>       ("ml_enable",  '\0', "enable ML filtering",          false, ML_ENABLE);
    // portable path overrides — set these to run on any machine
    parser.add<std::string>("python_exe", '\0', "path to python interpreter",   false, LOCAL_PYTHON_EXE);
    parser.add<std::string>("scripts_dir",'\0', "path to ml/scripts/ directory",false, LOCAL_SCRIPTS_DIR);
    parser.add<std::string>("lkh_exe",    '\0', "path to LKH executable",       false, LOCAL_LKH_EXE);
    parser.add<std::string>("lkh_tmp",    '\0', "path to LKH tmp directory",    false, LOCAL_LKH_TMP_ROOT);

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
    // Patience threshold: how many consecutive non-improving iterations are
    // tolerated before early stopping fires.
    //
    // Setting patience = iteration (the old behaviour) effectively disabled
    // early stopping — you would need ALL remaining iterations to be
    // non-improving, which never happens in practice.
    //
    // We set patience = iteration / 5 (20 % of budget) so that:
    //   • Fast-converging instances (e.g. car_door_25 in LA-CETSP, which
    //     saturates around iter ~200 even in a 5000-iter run) stop early
    //     instead of burning the remaining budget on fruitless iterations.
    //   • The ML training trigger (stagnation = patience >= 40 % of this
    //     threshold, i.e. 8 % of total budget) still fires well before the
    //     budget is exhausted, giving the model time to filter offspring.
    //
    // Reference: Balemir et al. (LA-CETSP presentation) noted that
    // "training horizon is fixed; tuning left for future work." This
    // patience-based scheme is our response to that open problem — the
    // training horizon now adapts to when the search actually stagnates
    // rather than being pegged to a fixed iteration count.
    patience        = std::max(iteration / 10, 50);  // 10% of budget; follows base MA-CETSP paper (Lei & Hao 2024)
    max_time        = parser.get<double>("max_time");
    fit_beta        = parser.get<double>("fit_beta");
    dist_th         = parser.get<int>("dist_th");
    neighbor_size   = parser.get<int>("neighbor_size");
    n_islands       = parser.get<int>("islands");
    ml_model        = parser.get<std::string>("ml_model");
    ml_enable       = parser.get<bool>("ml_enable");
    python_exe      = parser.get<std::string>("python_exe");
    scripts_dir     = parser.get<std::string>("scripts_dir");
    lkh_exe         = parser.get<std::string>("lkh_exe");
    lkh_tmp_root    = parser.get<std::string>("lkh_tmp");
    // Convert relative paths to absolute based on cwd at launch time (always build/Release/).
    // model_dir is already canonicalised below; do the same for scripts_dir and lkh_tmp_root
    // so Python subprocess calls and LKH file I/O work regardless of launch directory.
    namespace fs = std::filesystem;
    auto to_abs_dir = [](std::string p) -> std::string {
        if (p.empty()) return p;
        fs::path fp(p);
        if (fp.is_relative())
            fp = fs::weakly_canonical(fs::current_path() / fp);
        std::string s = fp.string();
        if (s.back() != '/' && s.back() != '\\') s += '/';
        return s;
    };
    auto to_abs_file = [](std::string p) -> std::string {
        if (p.empty()) return p;
        fs::path fp(p);
        if (fp.is_relative())
            fp = fs::weakly_canonical(fs::current_path() / fp);
        return fp.string();
    };
    scripts_dir  = to_abs_dir(scripts_dir);
    lkh_tmp_root = to_abs_dir(lkh_tmp_root);
    lkh_exe      = to_abs_file(lkh_exe);

    // island_id stays 0 for single-island runs; set by make_island() otherwise
    island_id  = 0;
    timestamp  = std::to_string(std::time(nullptr));
    // Use absolute path so training/prediction scripts always find the right directory
    // regardless of which working directory the exe is launched from.
    model_dir  = std::filesystem::weakly_canonical(
                     std::filesystem::current_path() / "../../ml/models/").string();
    if (!model_dir.empty() && model_dir.back() != '/' && model_dir.back() != '\\')
        model_dir += '/';
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
        namespace fs = std::filesystem;
        fs::path base = fs::weakly_canonical(
            fs::current_path() / "../../ml/models/" / ("island_" + std::to_string(id)));
        p.model_dir = base.string();
        if (!p.model_dir.empty() && p.model_dir.back() != '/' && p.model_dir.back() != '\\')
            p.model_dir += '/';
        // ensure the directory exists
        fs::create_directories(p.model_dir);
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
