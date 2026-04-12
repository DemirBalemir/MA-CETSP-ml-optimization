#include "Defs.hpp"
#include "SurvivalModel.hpp"

static double cached_threshold = -1.0;

double get_ml_threshold() {
    static SurvivalModel sm;

    if (ML_MODEL == "COX") {
        return sm.load_cox_threshold();
    }
    return 2.0;
}
