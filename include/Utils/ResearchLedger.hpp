#ifndef CETSP_RESEARCH_LEDGER_HPP
#define CETSP_RESEARCH_LEDGER_HPP

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

// Run-local offspring ledger. Initial population (id 0) is deliberately excluded.
class ResearchLedger {
public:
    struct Counts { int generated=0, filter_rejected=0, vnd_evaluated=0, admitted=0, not_admitted=0; } counts;
    explicit ResearchLedger(const std::filesystem::path& directory) : dir_(directory) {
        std::filesystem::create_directories(dir_);
        write("candidates.csv", "candidate_id,iteration,outcome\n", false);
        write("lifetimes.csv", "candidate_id,birth_iter,death_iter,survival_iters,censored\n", false);
    }
    void generated(int id, int iter) {
        if (id <= 0 || iter <= 0 || states_.count(id)) throw std::logic_error("duplicate/invalid offspring ID");
        states_[id] = {iter, false, false};
        ++counts.generated;
    }
    void outcome(int id, int iter, const std::string& value) {
        auto it = states_.find(id);
        if (it == states_.end() || it->second.birth != iter || it->second.resolved)
            throw std::logic_error("invalid candidate outcome");
        if (value == "filter_rejected") ++counts.filter_rejected;
        else if (value == "admitted") { ++counts.vnd_evaluated; ++counts.admitted; it->second.admitted=true; }
        else if (value == "not_admitted") { ++counts.vnd_evaluated; ++counts.not_admitted; }
        else throw std::logic_error("unknown candidate outcome");
        it->second.resolved = true;
        write("candidates.csv", std::to_string(id)+','+std::to_string(iter)+','+value+'\n', true);
    }
    void lifetime(int id, int birth, int death, bool censored) {
        if (id == 0) return;
        auto it=states_.find(id);
        if (it == states_.end() || !it->second.admitted || !it->second.resolved ||
            it->second.finalized || it->second.birth != birth || death < birth)
            throw std::logic_error("invalid/duplicate final lifetime");
        write("lifetimes.csv", std::to_string(id)+','+std::to_string(birth)+','+
            std::to_string(death)+','+std::to_string(death-birth)+','+
            (censored ? "1\n" : "0\n"), true);
        it->second.finalized = true;
    }
    void verify() const {
        if (counts.generated != counts.filter_rejected + counts.vnd_evaluated ||
            counts.vnd_evaluated != counts.admitted + counts.not_admitted)
            throw std::logic_error("candidate accounting invariant failed");
        for (const auto& item : states_)
            if (!item.second.resolved || (item.second.admitted != item.second.finalized))
                throw std::logic_error("missing candidate outcome or final lifetime");
    }
private:
    struct State { int birth; bool resolved, admitted, finalized=false; };
    std::filesystem::path dir_;
    std::map<int,State> states_;
    void write(const char* filename, const std::string& line, bool append) {
        std::ofstream out(dir_/filename, append ? std::ios::app : std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open research record");
        out << line;
        out.flush();
        if (!out) throw std::runtime_error("cannot write research record");
    }
};
#endif
