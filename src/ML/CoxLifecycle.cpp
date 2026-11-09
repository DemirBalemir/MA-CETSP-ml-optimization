#include "ML/CoxLifecycle.hpp"
#include "Features/GeometryFeatures.hpp"
#include <stdexcept>

namespace CoxLifecycle {
bool ready(bool enabled, const std::string& model_name, int training_completed_at) {
    return enabled && model_name == "COX" && training_completed_at >= 0;
}

double predict(bool enabled, const std::string& model_name, int training_completed_at,
               SurvivalModel* model, const std::map<std::string, double>& features) {
    if (!ready(enabled, model_name, training_completed_at))
        throw std::logic_error("[ML ERROR] Cox prediction requires successful training");
    if (!model) throw std::logic_error("[ML ERROR] Trained Cox model is unavailable");
    return model->predict_cox_score(features);
}

void cache_score(bool enabled, const std::string& model_name, int training_completed_at,
                 SurvivalModel* model, List& solution) {
    if (!ready(enabled, model_name, training_completed_at) || solution.has_cox_lp) return;
    auto coords = solution.pre_vnd_coords;
    if (coords.empty()) {
        Node* node = solution.head();
        for (int i=0; i<solution.size(); ++i) {
            coords.push_back({node->x, node->y});
            node = node->next;
        }
    }
    auto features = GeometryFeatures::extract(coords);
    features["pre_vnd_cost"] = solution.pre_vnd_value >= 0 ? solution.pre_vnd_value : solution.getValue();
    const double score = predict(enabled, model_name, training_completed_at, model, features);
    solution.cox_lp = score;
    solution.has_cox_lp = true; // Only publish a successfully computed score.
}

void backfill(bool enabled, const std::string& model_name, int training_completed_at,
              SurvivalModel* model, const std::vector<List*>& members) {
    if (!ready(enabled, model_name, training_completed_at)) return;
    for (List* solution : members)
        cache_score(enabled, model_name, training_completed_at, model, *solution);
}

void complete_training(bool success, int iteration, bool enabled, const std::string& model_name,
                       int& training_completed_at, SurvivalModel* model,
                       const std::vector<List*>& members, List* best_solution) {
    training_completed_at = -1;
    if (model) model->invalidate(); // Also invalidates non-Cox Python prediction servers.
    if (model_name == "COX") {
        for (List* solution : members) solution->has_cox_lp = false;
        if (best_solution) best_solution->has_cox_lp = false;
    }
    if (!success || !enabled) return;
    // Keep the timestamp unpublished if a successful trainer left an invalid model.
    backfill(enabled, model_name, iteration, model, members);
    training_completed_at = iteration;
}
}
