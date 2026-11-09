#ifndef CETSP_PRUNE_POLICY_HPP
#define CETSP_PRUNE_POLICY_HPP

#include <cstdint>
#include <random>
#include <string>
#include <vector>

class PrunePolicy {
public:
    struct Window { int first, last; double probability; };
    explicit PrunePolicy(uint32_t seed, double rate, int start);
    static std::vector<Window> read_schedule(const std::string& path, int budget,
                                             std::string& hash);
    void set_schedule(std::vector<Window> windows);
    bool reject(int iteration);
private:
    std::mt19937 rng_;
    double rate_;
    int start_;
    std::vector<Window> windows_;
};
#endif
