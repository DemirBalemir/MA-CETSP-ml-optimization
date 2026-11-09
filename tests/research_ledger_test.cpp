#include "Utils/ResearchLedger.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <iterator>

int main() {
    const auto dir = std::filesystem::current_path() / "research-ledger-test-output";
    std::filesystem::create_directories(dir);
    ResearchLedger ledger(dir);
    // VND ran but insertion rejected: no final lifetime is allowed.
    ledger.generated(1,1);
    ledger.outcome(1,1,"not_admitted");
    assert(ledger.counts.generated==1 && ledger.counts.vnd_evaluated==1);
    assert(ledger.counts.not_admitted==1 && ledger.counts.admitted==0);
    // Admitted then evicted in its birth iteration: lifetime zero.
    ledger.generated(2,2); ledger.outcome(2,2,"admitted"); ledger.lifetime(2,2,2,false);
    // Training-time snapshot is separate and is not a final lifetime.
    const auto snapshot=dir/"training_snapshot.json";
    ledger.generated(3,3); ledger.outcome(3,3,"admitted");
    { std::ofstream out(snapshot); out << "{\"candidate_id\":3,\"death_iter\":5,\"censored\":true}\n"; }
    ledger.lifetime(3,3,7,false); // death after training at iteration 5
    ledger.generated(4,4); ledger.outcome(4,4,"admitted");
    bool missing_lifetime_rejected=false;
    try { ledger.verify(); } catch(const std::logic_error&) { missing_lifetime_rejected=true; }
    assert(missing_lifetime_rejected);
    ledger.lifetime(4,4,8,true);
    ledger.generated(5,5); ledger.outcome(5,5,"filter_rejected");
    ledger.verify();
    assert(ledger.counts.generated==5 && ledger.counts.filter_rejected==1);
    assert(ledger.counts.vnd_evaluated==4 && ledger.counts.admitted==3 && ledger.counts.not_admitted==1);
    auto read=[](const std::filesystem::path& p) { std::ifstream in(p); return std::string(std::istreambuf_iterator<char>(in),{}); };
    const auto candidates=read(dir/"candidates.csv"), lives=read(dir/"lifetimes.csv");
    assert(candidates.find("1,1,not_admitted\n")!=std::string::npos);
    assert(lives.find("1,1,")==std::string::npos);
    assert(lives.find("2,2,2,0,0\n")!=std::string::npos);
    assert(lives.find("3,3,7,4,0\n")!=std::string::npos);
    assert(lives.find("4,4,8,4,1\n")!=std::string::npos);
    assert(read(snapshot).find("\"death_iter\":5")!=std::string::npos);
    int final_rows=0; for(char c:lives) if(c=='\n') ++final_rows;
    assert(final_rows==4); // header + one row for each of three admissions
    bool duplicate_rejected=false;
    try { ledger.lifetime(3,3,8,true); } catch(const std::logic_error&) { duplicate_rejected=true; }
    assert(duplicate_rejected);
}
