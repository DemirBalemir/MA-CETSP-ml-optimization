/**
 * Defs.h"p"p
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#ifndef CETSP_DEFS_HPP
#define CETSP_DEFS_HPP

#include <vector>
#include <string>

typedef std::vector<std::vector<double>> Centers;
typedef std::vector<std::vector<int>> Neighbors;

const int INSTANCE_INDEX = 0;           // which instance file
const int ITERATION = 200;             // max iteration
const double MAX_TIME = 36000;          // max running time
const int POPULATION_SIZE = 20;         // population size
const int N_ISLANDS = 1;               // number of parallel islands (1 = single-island, original behaviour)
const int DISTANCE_THRESHOLD = 5;       // min distance in population
const double FIT_BETA = 0.96;           // fitness function distance coef
const int NEIGHBOR_SIZE = 50;           // neighbors size of a target
const double EPSILON = 1e-4;            // approximation in geometry and local search
const double DELTA = 1e-3;              // approximation in accepting best solution
const double PI = 3.14159265358979323846;

// ---- Adaptive ML training trigger ----
//
// MOTIVATION: survival analysis models require a sufficient number of observed
// *death events* (non-censored solutions evicted from the population) before
// the learned risk scores carry any discriminative signal.  Using wall-clock
// iterations as the sole trigger ignores how many deaths have actually occurred,
// which varies with population dynamics and instance difficulty.
//
// TRAINING FIRES when ALL three conditions hold:
//   1. iter  >= ML_TRAIN_FRAC_MIN * params->iteration   (budget floor: 20 %)
//   2. data.getEventCount() >= ML_MIN_EVENTS             (death-signal floor)
//   3. patience >= patience_threshold * ML_PATIENCE_FRACTION   (stagnation)
//        OR iter >= ML_TRAIN_FRAC_MAX * params->iteration      (budget ceiling)
//
// A hard fallback fires at ML_TRAIN_FRAC_HARD regardless of event count so
// that training is never skipped on stable populations with few evictions.
//
// REFERENCES (for EPV / sample-size justification):
//   - Peduzzi et al. (1995), J Clin Epidemiol 48(12):1503-1510.
//     "Events per variable" (EPV) rule: ≥10 events per predictor for Cox PH.
//   - Riley et al. (2019), Stat Med 38(7):1276-1296.
//     Minimum sample size for time-to-event prediction models.
//   - Jin (2005), Soft Comput 9(1):3-12.
//     Survey of fitness approximation in EAs: recommends a burn-in phase of
//     10-20 % of the evaluation budget before activating any surrogate.
//   - Lim et al. (2010), IEEE Trans Evol Comput 14(3):329-355.
//     Surrogate-assisted EAs: budget fraction for initial model building.
//   - Karafotias et al. (2015), IEEE Trans Evol Comput 19(2):167-187.
//     Parameter control / online learning in EAs: exploration vs exploitation.
//
// DESIGN CHOICE — equal conditions across models:
//   All island models (COX, RSF, GBSA, DeepSurv, …) share the same trigger so
//   every model trains on the same amount of data.  This ensures a fair
//   apples-to-apples comparison in the paper.  The EPV threshold (100 events,
//   ~10 features × EPV=10) is the most demanding model's lower bound; simpler
//   models (Cox, WeibullAFT) benefit from the extra data at no cost.

const int    ML_MIN_EVENTS        = 100;   // min death events before training (EPV ≥ 10 × n_features)
const double ML_TRAIN_FRAC_MIN    = 0.20;  // earliest training: 20 % of iteration budget consumed
const double ML_TRAIN_FRAC_MAX    = 0.25;  // soft ceiling: trigger if stagnation hasn't fired by 25 %
const double ML_TRAIN_FRAC_HARD   = 0.50;  // hard fallback: train unconditionally at 50 % (stable pop.)
const double ML_PATIENCE_FRACTION = 0.40;  // stagnation signal: 40 % of patience budget exhausted

// ---- Runtime rejection-rate cap ----
//
// After ML training fires, the rejection rate observed in production can
// exceed the rate calibrated on the validation set (covariate shift: early
// offspring differ from post-training offspring).  On some seeds this drift
// causes >40 % rejection, destroying population diversity and halting search.
//
// The cap works with a sliding window: every ML_ROLLING_WINDOW ML-eligible
// offspring we measure the fraction rejected.  If it exceeds
// ML_MAX_ROLLING_REJECT_RATE the filter is suspended for the next window,
// allowing diversity to recover.  The check repeats every window.
const int    ML_ROLLING_WINDOW          = 50;   // window length (# ML-eligible offspring)
const double ML_MAX_ROLLING_REJECT_RATE = 0.30; // suspend if window reject rate > 30 %

const std::string ML_MODEL = "MTLR"; // "COX", "RSF", "GBSA", "DEEPSURV", "SSVM", "WEIBULLAFT", "KNN", "ELASTICNET", "MTLR"
const bool ML_ENABLE = true;
const double ML_THRESHOLD = 3;



const std::string INITIALIZATION = "KMEANS";    // RANDOM, KMEANS
const std::string SELECTION = "RANDOM";         // RANDOM, ROULETTE
const std::string CROSSOVER = "KSX";            // KSX, GAX, EAX
const std::string IMPROVEMENT = "BEST";         // FIRST, BEST
const std::string GREEDY_ALGO = "SQUEEZE";      // SQUEEZE, SPARSE
const std::string DISTANCE = "EDIT";

const std::string ENV = "LOCAL";                 // LOCAL, SERVER
const bool LOG = true;                          // log more details
const int SEED = 0;                              // random seed

// server
const std::string SERVER_DATA_DIR = "";
const std::string SERVER_RES_DIR = "";
const std::string SERVER_LKH_EXE = "";
const std::string SERVER_LKH_TMP_ROOT = "";

// local
const std::string LOCAL_DATA_DIR    = "../../datasets/";
const std::string LOCAL_RES_DIR     = "../../solutions/";
const std::string LOCAL_LKH_EXE     = "../../external/LKH-2.0.11/LKH.exe";
const std::string LOCAL_LKH_TMP_ROOT= "../../external/LKH-2.0.11/tmp/";
const std::string LOCAL_PYTHON_EXE  = "python";
const std::string LOCAL_SCRIPTS_DIR = "../../ml/scripts/";

const std::string ML_MODEL_DIR =
(ENV == "LOCAL")
? "../../ml/models/"
    : SERVER_RES_DIR + "/ml/models/";

// instances
const std::vector<std::string> FILENAMES = {
    // varied overlap ratios instances
    "bonus1000",              // 0
    "bubbles1",               // 1
    "bubbles2",               // 2
    "bubbles3",               // 3
    "bubbles4",               // 4
    "bubbles5",               // 5
    "bubbles6",               // 6
    "bubbles7",               // 7
    "bubbles8",               // 8
    "bubbles9",               // 9
    "chaoSingleDep",          // 10
    "concentricCircles1",     // 11
    "concentricCircles2",     // 12
    "concentricCircles3",     // 13
    "concentricCircles4",     // 14
    "concentricCircles5",     // 15
    "rotatingDiamonds1",      // 16
    "rotatingDiamonds2",      // 17
    "rotatingDiamonds3",      // 18
    "rotatingDiamonds4",      // 19
    "rotatingDiamonds5",      // 20
    "team1_100",              // 21
    "team2_200",              // 22
    "team3_300",              // 23
    "team4_400",              // 24
    "team5_499",              // 25
    "team6_500",              // 26
    // instances with different overlap ratio 0.02
    "d493_or2",                   // 27
    "dsj1000_or2",                // 28
    "kroD100_or2",                // 29
    "lin318_or2",                 // 30
    "pcb442_or2",                 // 31
    "rat195_or2",                 // 32
    "rd400_or2",                  // 33
    // instances with different overlap ratio 0.1
    "d493_or10",                   // 34
    "dsj1000_or10",                // 35
    "kroD100_or10",                // 36
    "lin318_or10",                 // 37
    "pcb442_or10",                 // 38
    "rat195_or10",                 // 39
    "rd400_or10",                  // 40
    // instances with different overlap ratio 0.3
    "d493_or30",                   // 41
    "dsj1000_or30",                // 42
    "kroD100_or30",                // 43
    "lin318_or30",                 // 44
    "pcb442_or30",                 // 45
    "rat195_or30",                 // 46
    "rd400_or30",                  // 47
    // arbitrary radii instances
    "bonus1000rdmRad",        // 48
    "d493rdmRad",             // 49
    "dsj1000rdmRad",          // 50
    "kroD100rdmRad",          // 51
    "lin318rdmRad",           // 52
    "pcb442rdmRad",           // 53
    "rat195rdmRad",           // 54
    "rd400rdmRad",            // 55
    "team1_100rdmRad",        // 56
    "team2_200rdmRad",        // 57
    "team3_300rdmRad",        // 58
    "team4_400rdmRad",        // 59
    "team5_499rdmRad",        // 60
    "team6_500rdmRad",        // 61
    // real world instances
    "car_door_25",               // 62
    "car_door_30",               // 63
    "car_door_35",               // 64
    "car_door_40",               // 65
    "car_door_45",               // 66
    "car_door_50",               // 67
};

#endif //CETSP_DEFS_HPP
