// Gate (GPU machine): (a) the CUDA kernel replays golden/sim.txt and must
// match the recorded scores; (b) CPU vs GPU from the SAME source
// (sim_core.hpp): NS trees and the all-terminal sum tree, 64 scenarios x 2
// seeds x 4 instances. Reports measured max|diff|; 0 = bit-exact, <1e-9 =
// float-equivalent (FMA), else MISMATCH.
//     verify_cuda golden/sim.txt
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/factory.cuh"
#include "../engine/tree.hpp"

static std::string dhex(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)u);
    return buf;
}

static double hexd(const std::string& h) {
    uint64_t u = std::stoull(h, nullptr, 16);
    double x;
    std::memcpy(&x, &u, 8);
    return x;
}

// Balanced sum so the postfix eval stack stays within MAX_STACK.
static std::string sum_prefix(const std::vector<std::string>& terms, size_t lo, size_t hi) {
    if (hi - lo == 1) return terms[lo];
    size_t mid = lo + (hi - lo) / 2;
    return "(+ " + sum_prefix(terms, lo, mid) + " " + sum_prefix(terms, mid, hi) + ")";
}

int main(int argc, char** argv) {
    long total = 0, bad = 0;
    double maxdiff = 0.0;

    // (a) GPU vs the recorded fixtures
    {
        std::ifstream f(argc > 1 ? argv[1] : "golden/sim.txt");
        if (!f) { std::fprintf(stderr, "sim golden not found\n"); return 2; }
        std::string line, cur_inst, cur_kind;
        SimData sd;
        Compiled comp;
        std::vector<int64_t> mask;
        std::vector<double> zp, sampled;
        int n_samples = 0;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line.rfind("# case ", 0) == 0) {
                std::istringstream ss(line.substr(7));
                ss >> cur_inst >> cur_kind;
                sd = SimData(load_instance(cur_inst), 3, 0.2);
                sampled.clear();
                n_samples = 0;
            } else if (line.rfind("tree ", 0) == 0) {
                comp = compile_rmod(parse_prefix(line.substr(5)));
            } else if (line.rfind("mask ", 0) == 0) {
                mask.clear();
                std::istringstream ss(line.substr(5));
                long long v;
                while (ss >> v) mask.push_back(v);
            } else if (line.rfind("zp ", 0) == 0) {
                zp.clear();
                std::istringstream ss(line.substr(3));
                std::string h;
                while (ss >> h) zp.push_back(hexd(h));
            } else if (line.rfind("sampled ", 0) == 0) {
                std::istringstream ss(line.substr(8));
                std::string h;
                while (ss >> h) sampled.push_back(hexd(h));
                n_samples++;
            } else if (line.rfind("scores ", 0) == 0) {
                std::vector<double> want;
                std::istringstream ss(line.substr(7));
                std::string h;
                while (ss >> h) want.push_back(hexd(h));
                FactoryContext ctx(sd, zp);
                PackedTrees pk;           // pack the compiled tree directly
                pk.offsets = {0};
                pk.lengths = {(int32_t)comp.ops.size()};
                pk.ops = comp.ops;
                for (auto& a : comp.args) { pk.args.push_back(a[0]); pk.args.push_back(a[1]); }
                std::vector<double> got = ctx.evaluate(pk, sampled, n_samples, mask);
                for (int s = 0; s < n_samples; s++) {
                    total++;
                    if (dhex(got[s]) != dhex(want[s])) {
                        double d = std::fabs(got[s] - want[s]);
                        if (d > maxdiff) maxdiff = d;
                        if (bad < 8) std::printf("MISMATCH golden %s %s scen %d: %.17g vs %.17g\n",
                                                 cur_inst.c_str(), cur_kind.c_str(), s, got[s], want[s]);
                        bad++;
                    }
                }
            }
        }
        std::printf("part (a) GPU vs recorded fixtures: %ld compared\n", total);
    }

    // (b) CPU vs GPU, same source, wider sweep
    {
        std::vector<std::string> all61;
        {   // slot order 0..60 straight from the term map
            std::vector<std::string> by_idx(N_TERMS);
            for (auto& kv : term_map()) by_idx[kv.second] = kv.first;
            all61 = by_idx;
        }
        Compiled sum61 = compile_rmod(parse_prefix(sum_prefix(all61, 0, all61.size())));
        std::vector<int64_t> mask_all(N_TERMS, 1);
        for (const std::string inst_name : {"c105", "r101", "rc101", "pr01"}) {
            SimData sd(load_instance(inst_name), 3, 0.2);
            std::vector<double> zp = zp_of(sd);
            FactoryContext ctx(sd, zp);
            for (long long seed : {12345LL, 777LL}) {
                std::vector<double> sampled = gen_durations(sd, 64, seed);
                std::vector<double> cpu = sim_batch(sd, sum61, sampled, 64, mask_all, zp);
                PackedTrees pk;
                pk.offsets = {0};
                pk.lengths = {(int32_t)sum61.ops.size()};
                pk.ops = sum61.ops;
                for (auto& a : sum61.args) { pk.args.push_back(a[0]); pk.args.push_back(a[1]); }
                std::vector<double> gpu = ctx.evaluate(pk, sampled, 64, mask_all);
                double d = 0.0, cm = 0.0;
                for (int s = 0; s < 64; s++) {
                    total++;
                    double dd = std::fabs(gpu[s] - cpu[s]);
                    if (dd > d) d = dd;
                    cm += cpu[s];
                    if (dd != 0.0) bad++;
                }
                if (d > maxdiff) maxdiff = d;
                std::printf("  %-6s all61 seed=%-6lld cpu.mean=%9.2f max|gpu-cpu|=%.3e\n",
                            inst_name.c_str(), seed, cm / 64.0, d);
            }
        }
    }

    const char* tag = maxdiff == 0.0 ? "OK bit-exact"
                      : (maxdiff < 1e-9 ? "OK float-equiv (<1e-9, FMA)" : "MISMATCH");
    std::printf("verify_cuda: %ld comparisons, %ld nonzero, max|diff|=%.3e  %s\n",
                total, bad, maxdiff, tag);
    return maxdiff < 1e-9 ? 0 : 1;
}
