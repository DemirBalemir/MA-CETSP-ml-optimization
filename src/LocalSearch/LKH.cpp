/**
 * LKH.cpp
 * created on : Jan 03 2023
 * author : Z.LEI
 **/

#include "LocalSearch/LKH.hpp"

#include <filesystem>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

LKH::LKH() {}

LKH::~LKH() {
    if (std::filesystem::exists(root_dir)) {
        try {
            std::filesystem::remove_all(root_dir);
        }
        catch (const std::exception& e) {
            std::cout << "[LKH ERROR] " << e.what() << std::endl;
        }
    }
}

void LKH::setContext(std::string timestamp, int random_value) {
    if (ENV == "LOCAL") {
        lkh_exe = LOCAL_LKH_EXE;
        root_dir = LOCAL_LKH_TMP_ROOT;
    }
    else if (ENV == "SERVER") {
        lkh_exe = SERVER_LKH_EXE;
        root_dir = SERVER_LKH_TMP_ROOT + "_" + timestamp + "_" + std::to_string(random_value) + "/";
    }

    params_file = root_dir + "params.par";
    problem_file = root_dir + "problem.tsp";
    tour_file = root_dir + "tour.txt";
    result_file = root_dir + "result.txt";
}

List* LKH::run(List* solution, bool adapted) {
    auto start = std::chrono::high_resolution_clock::now();
    ListAdapter la;

    if (adapted) {
        solution = la.real2Reduced(solution);
    }

    write(solution);

#ifdef _WIN32
    // Windows: ensure full path and correct environment
    std::filesystem::path exePath = std::filesystem::absolute(lkh_exe);
    std::filesystem::path paramPath = std::filesystem::absolute(params_file);

    std::string str = "\"" + exePath.string() + "\" \"" + paramPath.string() + "\"";
    std::string cmd = "C:\\Windows\\System32\\cmd.exe /C \"" + str + "\"";

    std::cout << "[DEBUG CMD] Executing: " << cmd << std::endl;
    int result = std::system(cmd.c_str());
#else
    std::string str = lkh_exe + " " + params_file;
    int result = std::system(str.c_str());
#endif

    // Handle stack overflow exit code case
    if (result == -1073740791) {
        std::cerr << "[LKH Warning] LKH likely executed successfully but returned stack code (-1073740791)" << std::endl;
        result = 0; // treat as success
    }

    if (result != 0 && result != 1) {
        std::cerr << "[LKH Error] LKH execution failed with code " << result << std::endl;
    }

    List* new_solution = read();

    if (adapted) {
        new_solution = la.reduced2Real(new_solution);
    }

    new_solution->evaluate();

    auto end = std::chrono::high_resolution_clock::now();
    if (LOG)
        std::cout << "LKH solution : " << new_solution->getValue()
        << " time : " << std::chrono::duration<double>(end - start).count()
        << " s" << std::endl;

    return new_solution;
}

void LKH::write(List* solution) {
    positions.resize(solution->size(), std::vector<double>{-1, -1});
    Node* p = solution->head();
    for (int i = 0; i < solution->size(); ++i) {
        positions[p->id][0] = p->x;
        positions[p->id][1] = p->y;
        p = p->next;
    }

    try {
        if (!std::filesystem::exists(root_dir)) {
            std::filesystem::create_directories(root_dir);
        }

        // Write params
        {
            std::ofstream params_out(params_file);
            if (params_out.is_open()) {
                auto normPath = [](std::string path) {
#ifdef _WIN32
                    std::replace(path.begin(), path.end(), '\\', '/'); // LKH needs forward slashes
#endif
                    return path;
                    };

                params_out << "PROBLEM_FILE = " << normPath(problem_file) << "\n";
                params_out << "TOUR_FILE = " << normPath(result_file) << "\n";
                params_out << "INITIAL_TOUR_FILE = " << normPath(tour_file) << "\n";
                params_out << "MOVE_TYPE = 5\n";
                params_out << "RUNS = 1\n";
                params_out << "TRACE_LEVEL = 0\n";
                params_out << "MAX_TRIALS = 10\n";
                params_out.close();
            }
            else {
                std::cout << "[LKH Error] Unable to open params file: " << params_file << std::endl;
            }
        }

        // Write problem
        {
            std::ofstream out(problem_file);
            if (out.is_open()) {
                out << "NAME : lhk\n";
                out << "TYPE : TSP\n";
                out << "DIMENSION : " << solution->size() << "\n";
                out << "EDGE_WEIGHT_TYPE : EUC_2D\n";
                out << "NODE_COORD_SECTION\n";
                for (int i = 0; i < positions.size(); ++i) {
                    out << i + 1 << " " << int(positions[i][0] * 1000) << " " << int(positions[i][1] * 1000) << "\n";
                }
                out << "EOF\n";
                out.close();
            }
            else {
                std::cout << "[LKH Error] Unable to open problem file: " << problem_file << std::endl;
            }
        }

        // Write tour
        {
            std::ofstream tf_out(tour_file);
            if (tf_out.is_open()) {
                p = solution->head();
                tf_out << "TOUR_SECTION\n";
                for (int i = 0; i < solution->size(); ++i) {
                    tf_out << p->id + 1 << "\n";
                    p = p->next;
                }
                tf_out << "-1\n";
                tf_out << "EOF\n";
                tf_out.close();
            }
            else {
                std::cout << "[LKH Error] Unable to open tour file: " << tour_file << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cout << "[LKH Error] " << e.what() << std::endl;
    }
}

List* LKH::read() {
    std::ifstream in(result_file);
    if (!in.is_open()) {
        std::cerr << "[LKH Error] Could not open result file: " << result_file << std::endl;
        return nullptr;
    }

    std::string data;
    List* solution = new List();

    while (std::getline(in, data) && data != "TOUR_SECTION") {
        // Skip until TOUR_SECTION
    }

    while (std::getline(in, data) && data != "-1") {
        std::istringstream iss(data);
        int id;
        if (iss >> id) {
            Node* node = new Node(id - 1, positions[id - 1][0], positions[id - 1][1]);
            solution->add(node);
        }
    }

    in.close();
    return solution;
}
