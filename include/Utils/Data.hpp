/**
 * Data.hpp
 * created on : Nov 30 2022
 * author : Z.LEI
 **/

#ifndef CETSP_DATA_HPP
#define CETSP_DATA_HPP

#include "Defs.hpp"
#include "Utils/Parameters.hpp"
#include "Genetic/List.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <ctime>
#include <filesystem>
#include <memory>
#include "Utils/ResearchLedger.hpp"

class Data {
private:
    Centers centers;                            // targets (x, y, r), centers[0] is depot
    std::string data_dir;                       // dir to read instance data
    std::string res_dir;                        // dir to write results
    std::string filename;                       // instance filename
    std::string timestamp;                      // timestamp
    std::string result_filename;                // result filename
    void reduceSize();                          // reduce the size of centers
public:
    Data(Parameters* params);
    ~Data() = default;
    void read();                                                        // read data
    void write(List* solution, int iter, std::string running_time);     // write the best solution when updating
    Centers getData();                                                  // get centers
    std::string getResultFilename();                                    // get result filename
    void writeSolutionLog(List* s);
    int instance_index;                         // instance index
    int island_id_ = 0;                         // island index (set in constructor)

    int getEventCount() const { return event_count_; }
    void recordCandidate(int id, int iter, const char* outcome);
    void recordGenerated(int id, int iter);
    void recordLifetime(int id, int birth, int death, bool censored);
    bool research_enabled() const { return research_enabled_; }
    void verifyResearch() const { if (research_) research_->verify(); }
    const ResearchLedger::Counts& researchCounts() const { return research_->counts; }

private:
    long long log_counter_ = 0;                 // per-instance counter (replaces static local)
    int       event_count_ = 0;                 // number of non-censored deaths logged so far
    bool research_enabled_ = false;
    std::unique_ptr<ResearchLedger> research_;
};

#endif //CETSP_DATA_HPP
