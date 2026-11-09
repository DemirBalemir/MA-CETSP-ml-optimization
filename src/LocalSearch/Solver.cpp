/**
 * Solver.cpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#include "LocalSearch/Solver.hpp"
#include <stdexcept>
#include <cstdlib>
#include <memory>

Solver::Solver() {}
Solver::~Solver() {}

void Solver::setContext(Centers &centers) {
    this->centers = centers;
}

void Solver::solve(List* solution) {
    auto start = std::chrono::high_resolution_clock::now();
    // params
    const int n = centers.size();               // number of points
    std::vector<double> cx(n);                  // center x
    std::vector<double> cy(n);                  // center y
    std::vector<double> r(n);                   // radius

    Node *p = solution->head();
    for (int i = 0; i < n; ++i) {
        cx[i] = centers[p->id][0];
        cy[i] = centers[p->id][1];
        r[i] = centers[p->id][2];
        p = p->next;

    }

    try {
        // Create an environment
        GRBEnv env = GRBEnv(true);
        // env.set("LogFile", "solve.log");
        env.set(GRB_IntParam_LogToConsole, 0);
        env.start();

        // Create an empty model
        GRBModel model = GRBModel(env);

        std::unique_ptr<double[]> lbs(new double[n]);
        std::unique_ptr<double[]> ubs(new double[n]);
        std::unique_ptr<char[]> types(new char[n]);
        for (int i = 0; i < n; ++i) {
            lbs[i] = -GRB_INFINITY;
            ubs[i] = GRB_INFINITY;
            types[i] = GRB_CONTINUOUS;
        }

        // Create variables
        std::unique_ptr<GRBVar[]> dist(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> dx(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> dy(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> s(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> t(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> x(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));
        std::unique_ptr<GRBVar[]> y(model.addVars(lbs.get(), ubs.get(), nullptr, types.get(), nullptr, n));

        // Set objective
        GRBLinExpr obj = 0;
        for (int i = 0; i < n; ++i) {
            obj += dist[i];
        }
        model.setObjective(obj, GRB_MINIMIZE);

        for (int i = 0; i < n - 1; ++i) {
            model.addConstr(dx[i] == x[i] - x[i+1]);
            model.addConstr(dy[i] == y[i] - y[i+1]);
        }

        model.addConstr(dx[n-1] == x[n-1] - x[0]);
        model.addConstr(dy[n-1] == y[n-1] - y[0]);

        for (int i = 0; i < n; ++i) {
            model.addConstr(s[i] == cx[i] - x[i]);
            model.addConstr(t[i] == cy[i] - y[i]);
            model.addQConstr(dist[i] * dist[i] >= dx[i] * dx[i] + dy[i] * dy[i]);
            model.addQConstr(r[i] * r[i] >= s[i] * s[i] + t[i] * t[i]);
            model.addConstr(dist[i] >= 0);
        }

        // Optimize model
        model.optimize();
        const int status = model.get(GRB_IntAttr_Status);
        const int solutions = model.get(GRB_IntAttr_SolCount);
        std::cout << "[SOCP] status=" << status << " solutions=" << solutions << "\n";
        if (solutions == 0) {
            if (const char* diagnostic = std::getenv("MA_CETSP_SOCP_DIAGNOSTIC_FILE"))
                model.write(diagnostic);
            if (status == GRB_NUMERIC) {
                // Preserve the original no-solution behavior (unchanged input
                // candidate) explicitly. Never confuse a license exception with
                // this numerical fallback, and never read unavailable X values.
                std::cout << "[SOCP FALLBACK] numerical_no_solution; candidate_unchanged\n";
                return;
            }
            throw std::runtime_error("[SOCP ERROR] No solution available; status=" + std::to_string(status));
        }
        if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL)
            throw std::runtime_error("[SOCP ERROR] Unexpected solution status=" + std::to_string(status));
        Node* p = solution->head();
        for (int i = 0 ; i < n; ++i) {
            p->x = x[i].get(GRB_DoubleAttr_X);
            p->y = y[i].get(GRB_DoubleAttr_X);
            p = p->next;
        }

        solution->setValue(model.get(GRB_DoubleAttr_ObjVal));

        std::cout << "[SOCP SUCCESS] objective=" << solution->getValue() << "\n";

    } catch(GRBException e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
        throw std::runtime_error("[SOCP ERROR] Gurobi error " + std::to_string(e.getErrorCode()) + ": " + e.getMessage());
    } catch(...) {
        std::cout << "Exception during optimization" << std::endl;
        throw;
    }
    auto end = std::chrono::high_resolution_clock::now();
    if (LOG) std::cout << "socp solution : " << solution->getValue() << " time : " << std::chrono::duration<double> (end - start).count() << " s" << std::endl;
}
