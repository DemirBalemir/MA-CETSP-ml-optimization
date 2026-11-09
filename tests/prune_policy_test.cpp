#include "Utils/PrunePolicy.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

int main() {
    std::srand(12345);
    const int first_search_draw = std::rand();
    PrunePolicy isolated(99, .5, 0);
    for (int i=1; i<=100; ++i) isolated.reject(i);
    const int second_search_draw = std::rand();
    std::srand(12345);
    assert(first_search_draw == std::rand());
    assert(second_search_draw == std::rand());
    PrunePolicy a(17, .37, 3), b(17, .37, 3);
    for (int i=1; i<=1000; ++i) assert(a.reject(i) == b.reject(i));
    PrunePolicy zero(4, 0, 2), one(4, 1, 2);
    for (int i=1; i<=10; ++i) { assert(!zero.reject(i)); assert(one.reject(i) == (i>2)); }
    const std::string path = "prune_test_schedule.csv";
    { std::ofstream out(path); out << "start_iter,end_iter,reject_probability\n1,2,0\n3,5,1\n"; }
    std::string hash;
    auto windows = PrunePolicy::read_schedule(path,5,hash);
    assert(!hash.empty());
    PrunePolicy scheduled(8,0,0); scheduled.set_schedule(windows);
    for (int i=1;i<=5;++i) assert(scheduled.reject(i)==(i>=3));
    auto invalid = [&](const std::string& body) {
        { std::ofstream out(path); out << "start_iter,end_iter,reject_probability\n" << body; }
        try { PrunePolicy::read_schedule(path,5,hash); return false; }
        catch (const std::invalid_argument&) { return true; }
    };
    assert(invalid("1,2,0\n4,5,1\n"));
    assert(invalid("1,3,0\n3,5,1\n"));
    assert(invalid("1,5,nan\n"));
    assert(invalid("1,5,1.1\n"));
    assert(invalid("1,4,0\n"));
    assert(invalid("2,1,0\n"));
    std::remove(path.c_str());
}
