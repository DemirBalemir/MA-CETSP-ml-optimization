#include "Utils/Parameters.hpp"
#include "Algo.hpp"
#include <thread>
#include <mutex>
#include <vector>
#include <limits>
#include <iostream>

// Models cycled across islands when n_islands > 1.
// Island 0 → COX, 1 → RSF, 2 → GBSA, 3 → COX, ...
static const std::vector<std::string> ISLAND_MODEL_CYCLE = {"COX", "RSF", "GBSA"};

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
        return 0;
    }

    // ---- multi-island: launch one thread per island ----
    std::vector<double>      best_values(n, std::numeric_limits<double>::max());
    std::vector<std::string> model_used(n);
    std::mutex               cout_mutex;

    auto run_island = [&](int id) {
        // Cycle through models unless the user fixed one with --ml_model
        std::string model = (base.ml_model == ML_MODEL)  // default not overridden
            ? ISLAND_MODEL_CYCLE[id % ISLAND_MODEL_CYCLE.size()]
            : base.ml_model;

        Parameters p = base.make_island(id, model);

        {
            std::lock_guard<std::mutex> lk(cout_mutex);
            std::cout << "[Island " << id << "] starting — model=" << model
                      << "  seed=" << p.seed << "\n";
        }

        Algo algo(&p);
        algo.run();

        best_values[id] = algo.get_best_value();
        model_used[id]  = model;
    };

    std::vector<std::thread> threads;
    threads.reserve(n);
    for (int i = 0; i < n; ++i)
        threads.emplace_back(run_island, i);

    for (auto& t : threads) t.join();

    // ---- collect results ----
    std::cout << "\n========== PARALLEL RUN SUMMARY ==========\n";
    int    best_island = 0;
    double best_val    = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        std::cout << "  Island " << i
                  << "  model=" << model_used[i]
                  << "  best=" << best_values[i] << "\n";
        if (best_values[i] < best_val) {
            best_val    = best_values[i];
            best_island = i;
        }
    }
    std::cout << "------------------------------------------\n"
              << "  GLOBAL BEST: island=" << best_island
              << "  model=" << model_used[best_island]
              << "  value=" << best_val << "\n"
              << "==========================================\n";

    return 0;
}
