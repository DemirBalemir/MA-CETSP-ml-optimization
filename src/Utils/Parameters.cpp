/**
 * Parameters.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "Utils/cmdline.h"
#include "Utils/Parameters.hpp"
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "Utils/PrunePolicy.hpp"

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
    parser.add<std::string>("filter_mode", '\0', "legacy, none, learned, random", false, "legacy");
    parser.add<double>("train_frac_min", '\0', "minimum training budget fraction", false, ML_TRAIN_FRAC_MIN);
    parser.add<double>("train_frac_hard", '\0', "fallback training budget fraction", false, ML_TRAIN_FRAC_HARD);
    parser.add<int>("min_events", '\0', "minimum death events for training", false, ML_MIN_EVENTS);
    parser.add<double>("random_reject_rate", '\0', "random rejection probability", false, 0.0);
    parser.add<int>("random_start_iter", '\0', "random filtering starts after this iteration", false, 0);
    parser.add<unsigned int>("prune_seed", '\0', "independent pruning RNG seed", false, 0u);
    parser.add<std::string>("prune_schedule", '\0', "CSV pruning schedule", false, "");
    parser.add<std::string>("output_root", '\0', "root for isolated explicit-mode runs", false, "experiment_runs");
    parser.add<std::string>("run_id", '\0', "unique run directory name", false, "");
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
    ml_model_explicit = parser.exist("ml_model");
    filter_mode = parser.get<std::string>("filter_mode");
    output_root = parser.get<std::string>("output_root");
    run_id = parser.get<std::string>("run_id");
    train_frac_min = parser.get<double>("train_frac_min");
    train_frac_hard = parser.get<double>("train_frac_hard");
    min_events = parser.get<int>("min_events");
    random_reject_rate = parser.get<double>("random_reject_rate");
    random_start_iter = parser.get<int>("random_start_iter");
    prune_seed = parser.get<unsigned int>("prune_seed");
    prune_schedule = parser.get<std::string>("prune_schedule");
    if (filter_mode != "legacy" && filter_mode != "none" && filter_mode != "learned" && filter_mode != "random")
        throw std::invalid_argument("invalid filter_mode");
    if (!std::isfinite(train_frac_min) || !std::isfinite(train_frac_hard) ||
        train_frac_min < 0 || train_frac_min > train_frac_hard || train_frac_hard > 1 || min_events < 1)
        throw std::invalid_argument("invalid training trigger parameters");
    if (iteration < 1 || n_islands < 1) throw std::invalid_argument("iteration and islands must be positive");
    if (!std::isfinite(random_reject_rate) || random_reject_rate < 0 || random_reject_rate > 1 || random_start_iter < 0)
        throw std::invalid_argument("invalid random pruning parameters");
    if (filter_mode != "legacy" && n_islands != 1) throw std::invalid_argument("explicit filter modes require --islands 1");
    if (filter_mode == "legacy" && (parser.exist("output_root") || parser.exist("run_id")))
        throw std::invalid_argument("--output_root and --run_id require an explicit filter mode");
    if (filter_mode != "legacy" && ml_model_explicit && filter_mode != "learned")
        throw std::invalid_argument("--ml_model only applies to --filter_mode learned");
    if (filter_mode == "learned") {
        const std::vector<std::string> supported = {"COX","RSF","GBSA","DEEPSURV","SSVM",
            "WEIBULLAFT","KNN","ELASTICNET","MTLR"};
        if (std::find(supported.begin(), supported.end(), ml_model) == supported.end())
            throw std::invalid_argument("unsupported --ml_model: " + ml_model +
                " (choose COX, RSF, GBSA, DEEPSURV, SSVM, WEIBULLAFT, KNN, ELASTICNET, MTLR)");
    }
    if (filter_mode != "legacy" && parser.exist("ml_enable")) throw std::invalid_argument("--ml_enable conflicts with explicit filter mode");
    if (filter_mode != "random" && (parser.exist("random_reject_rate") || parser.exist("random_start_iter") ||
        parser.exist("prune_seed") || parser.exist("prune_schedule")))
        throw std::invalid_argument("random pruning options require --filter_mode random");
    if (filter_mode == "random") {
        if (!parser.exist("prune_seed")) throw std::invalid_argument("random mode requires --prune_seed");
        if ((!prune_schedule.empty()) == parser.exist("random_reject_rate"))
            throw std::invalid_argument("provide exactly one of --prune_schedule or --random_reject_rate");
        if (!prune_schedule.empty() && parser.exist("random_start_iter"))
            throw std::invalid_argument("schedule conflicts with --random_start_iter");
        if (!prune_schedule.empty()) {
            prune_schedule = std::filesystem::absolute(prune_schedule).string();
            auto windows = PrunePolicy::read_schedule(prune_schedule, iteration, prune_schedule_hash);
            for (const auto& w : windows) prune_windows.emplace_back(w.first,w.last,w.probability);
        }
    }
    if (filter_mode == "none" || filter_mode == "random") ml_enable = false;
    if (filter_mode == "learned") ml_enable = true;
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
    if (filter_mode != "legacy") {
        if (output_root.empty()) throw std::invalid_argument("--output_root cannot be empty");
        fs::path root = fs::absolute(output_root).lexically_normal();
        output_root = root.string();
        fs::create_directories(root);
        if (!run_id.empty() && (run_id == "." || run_id == ".." ||
            run_id.find('/') != std::string::npos || run_id.find('\\') != std::string::npos ||
            run_id.find(':') != std::string::npos))
            throw std::invalid_argument("--run_id must be one directory name");
        bool created = false;
        if (!run_id.empty()) {
            created = fs::create_directory(root / run_id);
            if (!created) throw std::invalid_argument("run directory already exists: " + (root / run_id).string());
        } else {
            std::random_device rd;
            for (int attempt=0; attempt<100 && !created; ++attempt) {
                std::ostringstream name;
                name << "run-" << std::hex << std::chrono::high_resolution_clock::now().time_since_epoch().count()
                     << '-' << rd() << '-' << attempt;
                run_id = name.str();
                created = fs::create_directory(root / run_id);
            }
            if (!created) throw std::runtime_error("could not allocate unique run directory");
        }
        run_dir = (root / run_id).string();
        model_dir = (fs::path(run_dir) / "model").string() + "/";
        results_dir = (fs::path(run_dir) / "solutions").string() + "/";
        lkh_tmp_root = (fs::path(run_dir) / "lkh").string() + "/";
        fs::create_directories(model_dir);
        fs::create_directories(results_dir);
        fs::create_directories(lkh_tmp_root);
    }
}

Parameters Parameters::make_island(int id, const std::string& model) const {
    Parameters p = *this;
    p.island_id = id;
    p.ml_model  = model;
    p.seed      = this->seed + id;
    // unique timestamp per island so LKH tmp dirs and log dirs don't collide
    p.timestamp = this->timestamp + "_i" + std::to_string(id);
    // per-island model directory; island 0 in a single-island run keeps the default path
    if (filter_mode == "legacy" && (n_islands > 1 || id > 0)) {
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
    auto precise = [](double value) {
        std::ostringstream out; out << std::setprecision(17) << value; return out.str();
    };
    if (filter_mode != "legacy")
        std::cout << "[RUNTIME] python=" << std::quoted(python_exe)
                  << " scripts=" << std::quoted(scripts_dir)
                  << " lkh=" << std::quoted(lkh_exe) << std::endl;
    std::cout << "[Island " << island_id << "] "
              << std::setprecision(17)
              << "instance=" << instance_index
              << " seed=" << seed
              << " pop=" << population_size
              << " iter=" << iteration
              << " mode=" << filter_mode << " ml=" << (ml_enable ? ml_model : "OFF")
              << " prune_seed=" << (filter_mode == "random" ? std::to_string(prune_seed) : "NA")
              << " random_start_iter=" << (filter_mode == "random" && prune_schedule.empty() ? std::to_string(random_start_iter) : "NA")
              << " random_rate=" << (filter_mode == "random" && prune_schedule.empty() ? precise(random_reject_rate) : "NA")
              << " schedule=" << (prune_schedule.empty() ? "NA" : prune_schedule)
              << " schedule_fnv1a=" << (prune_schedule_hash.empty() ? "NA" : prune_schedule_hash)
              << " train_min=" << train_frac_min << " train_hard=" << train_frac_hard << " min_events=" << min_events
              << " fit_beta=" << fit_beta << " dist_th=" << dist_th << " neighbor_size=" << neighbor_size
              << " max_time=" << max_time << " init=" << init << " select=" << select
              << " crossover=" << crossover << " improvement=" << improvement << " greed=" << greed
              << " distance=" << distance << " random=" << random_num << " islands=" << n_islands
              << " model_dir=" << model_dir
              << " run_dir=" << (run_dir.empty() ? "NA" : run_dir)
              << " results_dir=" << (results_dir.empty() ? "legacy" : results_dir)
              << " lkh_tmp=" << lkh_tmp_root
              << " timestamp=" << timestamp
              << std::endl;
}
