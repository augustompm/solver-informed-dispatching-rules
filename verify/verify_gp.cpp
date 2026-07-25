// Gate: the GP loop (core/gp.hpp) against the golden/gp.txt fixtures.
// Populations are compared tree by tree as prefix strings, fitness values bit
// by bit; sampled durations come from the fixture so the gate is independent
// of the local math library.
//     verify_gp golden/gp.txt
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/gp.hpp"
#include "../engine/sim_host.hpp"

static std::string dhex(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)u);
    return buf;
}

static const std::vector<std::string> EXTRA_TERMS = {
    "REGRET", "NSROB", "MAXN", "REACH2", "NSTW", "FRAGCNT", "FRAGSCORE", "PBUST",
    "REACH1", "REACH3", "REACH5", "SAT", "VFRAC", "NFEAS", "HARVEST", "BOLSAO",
    "ATRISK", "REACH2ROB"};

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "golden/gp.txt");
    if (!f) { std::fprintf(stderr, "gp golden not found\n"); return 2; }
    SimData sd(load_instance("c101"), 3, 0.2);
    std::vector<double> zp = zp_of(sd);
    EvolutionConfig cfg;

    long total = 0, bad = 0;
    std::string line;
    TreeConfig tc;
    std::vector<int64_t> mask;
    Mt19937* rng = nullptr;
    std::vector<NodeP> pop;
    std::vector<double> fits;
    size_t pop_i = 0;
    int expect_gen = 0;

    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.rfind("# case ", 0) == 0) {
            std::string kind = line.substr(line.find_last_of(' ') + 1);
            tc = TreeConfig{};
            tc.terminals = NS_TERMINALS;
            tc.functions = FUNCTIONS;
            if (kind == "ext29rmod") {
                for (auto& t : EXTRA_TERMS) tc.terminals.push_back(t);
                tc.functions = FUNCS_RMOD;
            }
            mask = term_mask_from(tc.terminals);
            delete rng;
            rng = new Mt19937(0);
            pop = ramped_half_and_half(tc, *rng, cfg.population_size);
            pop_i = 0;
            expect_gen = 0;
        } else if (line.rfind("p", 0) == 0 && line[2] == ' ') {
            std::string want = line.substr(3);
            total++;
            if (pop_i >= pop.size() || to_prefix(pop[pop_i]) != want) {
                if (bad < 5) std::printf("MISMATCH pop tree %zu:\n  got  %s\n  want %s\n", pop_i,
                                         pop_i < pop.size() ? to_prefix(pop[pop_i]).c_str() : "(none)", want.c_str());
                bad++;
            }
            pop_i++;
        } else if (line.rfind("sampled ", 0) == 0) {
            std::istringstream ss(line.substr(8));
            int gen; ss >> gen;
            std::vector<double> sampled;
            std::string h;
            while (ss >> h) {
                uint64_t u = std::stoull(h, nullptr, 16);
                double x; std::memcpy(&x, &u, 8);
                sampled.push_back(x);
            }
            fits.assign(pop.size(), 0.0);
            for (size_t p = 0; p < pop.size(); p++) {
                Compiled c = compile_rmod(pop[p]);
                std::vector<double> out = sim_batch(sd, c, sampled, 1, mask, zp);
                fits[p] = out[0];   // mean over 1 sample
            }
            (void)expect_gen;
            expect_gen = gen;
        } else if (line.rfind("fits ", 0) == 0) {
            std::istringstream ss(line.substr(5));
            int gen; ss >> gen;
            std::string h;
            size_t idx = 0;
            while (ss >> h) {
                total++;
                if (idx >= fits.size() || dhex(fits[idx]) != h) {
                    if (bad < 5) std::printf("MISMATCH fit gen %d idx %zu\n", gen, idx);
                    bad++;
                }
                idx++;
            }
            if (gen < 2) {   // GENS=3: steps after gens 0 and 1
                pop = gp_step(pop, fits, *rng, tc, cfg);
                pop_i = 0;
            }
        }
    }
    delete rng;
    std::printf("verify_gp: %ld items compared, %ld mismatches -> %s\n",
                total, bad, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
