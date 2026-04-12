#pragma once

#include <string>

/**
 * Log ML-based pruning (offspring rejection) events.
 *
 * Each entry records:
 *  - iteration index
 *  - ML model name
 *  - predicted score
 *  - applied threshold
 *
 * Output format: CSV
 */
void log_ml_reject(
    int iter,
    const std::string& model,
    double score,
    double threshold,
    const std::string& out_dir
);
