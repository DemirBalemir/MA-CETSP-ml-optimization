#include "MLRejectLogger.hpp"

#include <fstream>
#include <filesystem>

void log_ml_reject(
    int iter,
    const std::string& model,
    double score,
    double threshold,
    const std::string& out_dir
) {
    static std::ofstream out;
    static bool initialized = false;

    if (!initialized) {
        // create directory if it does not exist
        std::filesystem::create_directories(out_dir);

        // open CSV file
        out.open(out_dir + "/ml_reject_log.csv");

        // write header
        out << "iter,model,score,threshold\n";

        initialized = true;
    }

    // write log entry
    out << iter << ","
        << model << ","
        << score << ","
        << threshold << "\n";
}
