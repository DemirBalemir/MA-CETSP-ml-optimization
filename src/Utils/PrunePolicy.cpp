#include "Utils/PrunePolicy.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

PrunePolicy::PrunePolicy(uint32_t seed, double rate, int start)
    : rng_(seed), rate_(rate), start_(start) {
    if (!std::isfinite(rate) || rate < 0 || rate > 1 || start < 0)
        throw std::invalid_argument("invalid random pruning rate or start iteration");
}

std::vector<PrunePolicy::Window> PrunePolicy::read_schedule(
    const std::string& path, int budget, std::string& hash) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot open prune schedule: " + path);
    std::string bytes((std::istreambuf_iterator<char>(in)), {});
    uint64_t digest = 14695981039346656037ull;
    for (unsigned char c : bytes) { digest ^= c; digest *= 1099511628211ull; }
    std::ostringstream hs; hs << std::hex << digest; hash = hs.str(); // FNV-1a, file identity
    std::istringstream rows(bytes);
    std::string line;
    if (!std::getline(rows, line) || line != "start_iter,end_iter,reject_probability\r" &&
        line != "start_iter,end_iter,reject_probability")
        throw std::invalid_argument("invalid prune schedule header");
    std::vector<Window> result;
    int expected = 1;
    while (std::getline(rows, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream row(line);
        std::string a,b,c,extra;
        if (!std::getline(row,a,',') || !std::getline(row,b,',') ||
            !std::getline(row,c,',') || std::getline(row,extra,','))
            throw std::invalid_argument("invalid prune schedule row");
        Window w{}; size_t pos=0;
        try {
            w.first=std::stoi(a,&pos); if(pos!=a.size()) throw std::invalid_argument("field");
            w.last=std::stoi(b,&pos); if(pos!=b.size()) throw std::invalid_argument("field");
            w.probability=std::stod(c,&pos); if(pos!=c.size()) throw std::invalid_argument("field");
        } catch (...) { throw std::invalid_argument("invalid prune schedule value"); }
        if (w.first != expected || w.last < w.first || w.last > budget ||
            !std::isfinite(w.probability) || w.probability < 0 || w.probability > 1)
            throw std::invalid_argument("prune schedule has gap, overlap, range or probability error");
        expected = w.last + 1;
        result.push_back(w);
    }
    if (expected != budget + 1) throw std::invalid_argument("prune schedule does not cover budget");
    return result;
}

void PrunePolicy::set_schedule(std::vector<Window> windows) { windows_ = std::move(windows); }

bool PrunePolicy::reject(int iteration) {
    double p = rate_;
    if (!windows_.empty()) {
        for (const auto& w : windows_) if (iteration >= w.first && iteration <= w.last) {
            p = w.probability; break;
        }
    } else if (iteration <= start_) return false;
    if (p == 0) return false;
    if (p == 1) return true;
    return std::generate_canonical<double, 53>(rng_) < p;
}
