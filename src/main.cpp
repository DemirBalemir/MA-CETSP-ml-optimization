#include "Utils/Parameters.hpp"
#include "Algo.hpp"
#include <thread>
#include <mutex>
#include <vector>
#include <limits>
#include <iostream>

// Models cycled across islands when n_islands > 1.
// Island 0 → COX, 1 → RSF, 2 → GBSA, 3 → DEEPSURV, 4 → SSVM,
//         5 → WEIBULLAFT, 6 → KNN, 7 → ELASTICNET, 8 → MTLR, 9 → BASELINE (no ML)
// "BASELINE" is a sentinel: the island runs with ml_enable=false so it serves
// as a built-in ablation control without a separate experiment.
static const std::vector<std::string> ISLAND_MODEL_CYCLE = {
    "COX", "RSF", "GBSA", "DEEPSURV", "SSVM",
    "WEIBULLAFT", "KNN", "ELASTICNET", "MTLR",
    "BASELINE"   // island 9: ML disabled — direct per-run ablation control
};

int main(int argc, char *argv[]) {
    std::cout << "Memetic Algo for CETSP\n\n";

    Parameters base(argc, argv);
    base.print();

    const int n = base.n_islands;

    if (n == 1) {
        // ---- single-island: identical to original behaviour ----
        Parameters p = base.make_island(0, base.ml_model);
        Algo algo(&p);
        algo.run();
        std::cout << "\n========== RUN SUMMARY ==========\n"
                  << "  Island 0"
                  << "  model="    << p.ml_model
                  << "  best="     << algo.get_best_value()
                  << "  time="     << algo.get_total_time() << "s"
                  << "  rejected=" << algo.get_reject_count() << "\n"
                  << "==================================\n";
        return 0;
    }

    // ---- multi-island: launch one thread per island ----
    std::vector<double>      best_values(n, std::numeric_limits<double>::max());
    std::vector<double>      total_times(n, 0.0);
    std::vector<int>         reject_counts(n, 0);
    std::vector<std::string> model_used(n);
    std::mutex               cout_mutex;

    auto run_island = [&](int id) {
        // Cycle through models unless the user fixed one with --ml_model
        std::string model = (base.ml_model == ML_MODEL)  // default not overridden
            ? ISLAND_MODEL_CYCLE[id % ISLAND_MODEL_CYCLE.size()]
            : base.ml_model;

        // "BASELINE" sentinel → ML disabled, use default model name for the slot
        const bool is_baseline = (model == "BASELINE");
        Parameters p = base.make_island(id, is_baseline ? "COX" : model);
        if (is_baseline) {
            p.ml_enable = false;
            p.ml_model  = "BASELINE";   // kept for logging; filter is off
        }

        {
            std::lock_guard<std::mutex> lk(cout_mutex);
            std::cout << "[Island " << id << "] starting — model=" << p.ml_model
                      << "  seed=" << p.seed
                      << (is_baseline ? "  [NO-ML baseline]" : "") << "\n";
        }

        Algo algo(&p);
        algo.run();

        best_values[id]   = algo.get_best_value();
        total_times[id]   = algo.get_total_time();
        reject_counts[id] = algo.get_reject_count();
        model_used[id]    = p.ml_model;
    };

    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int i = 0; i < n; ++i)
        threads.emplace_back(run_island, i);

    for (auto& t : threads) t.join();

    // ---- collect results ----
    // Note: islands run concurrently so total_times reflect wall-clock per island,
    // not additive sequential time. The wall-clock elapsed is ~max(total_times).
    std::cout << "\n========== PARALLEL RUN SUMMARY ==========\n";
    int    best_island  = 0;
    double best_val     = std::numeric_limits<double>::max();
    double max_walltime = 0.0;
    for (int i = 0; i < n; ++i) {
        std::cout << "  Island " << i
                  << "  model="    << model_used[i]
                  << "  best="     << best_values[i]
                  << "  time="     << total_times[i] << "s"
                  << "  rejected=" << reject_counts[i] << "\n";
        if (best_values[i] < best_val) {
            best_val    = best_values[i];
            best_island = i;
        }
        if (total_times[i] > max_walltime)
            max_walltime = total_times[i];
    }
    std::cout << "------------------------------------------\n"
              << "  GLOBAL BEST: island=" << best_island
              << "  model="    << model_used[best_island]
              << "  value="    << best_val
              << "  time="     << total_times[best_island] << "s"
              << "  rejected=" << reject_counts[best_island] << "\n"
              << "  Wall-clock (longest island): " << max_walltime << "s\n"
              << "==========================================\n";

    return 0;
}
