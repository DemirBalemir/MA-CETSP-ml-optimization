#pragma once

#include "Genetic/List.hpp"
#include "ML/SurvivalModel.hpp"

// Uses the existing completion timestamp as the successful-training state.
// No model access or score-cache publication is allowed while it is negative.
namespace CoxLifecycle {
    bool ready(bool enabled, const std::string& model_name, int training_completed_at);
    double predict(bool enabled, const std::string& model_name, int training_completed_at,
                   SurvivalModel* model, const std::map<std::string, double>& features);
    void cache_score(bool enabled, const std::string& model_name, int training_completed_at,
                     SurvivalModel* model, List& solution);
    void backfill(bool enabled, const std::string& model_name, int training_completed_at,
                  SurvivalModel* model, const std::vector<List*>& members);
    // Called once after the training attempt; invalidates model and solution caches
    // before any successful-model backfill. Failure leaves filtering inactive.
    void complete_training(bool success, int iteration, bool enabled, const std::string& model_name,
                           int& training_completed_at, SurvivalModel* model,
                           const std::vector<List*>& members, List* best_solution);
}
