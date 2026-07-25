// Gate: the simulator (sim_core/sim_host) against the golden/sim.txt fixtures.
// The sampled durations come FROM the fixture file, so this compares the pure
// deterministic simulator; golden/durations.txt gates the duration sampling
// (generate the fixture with the same C library the gate links against).
//     verify_sim golden/sim.txt golden/durations.txt
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/sim_host.hpp"
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

static std::string rstrip_cr(std::string s) {
    if (!s.empty() && s.back() == '\r') s.pop_back();
    return s;
}

int main(int argc, char** argv) {
    long total = 0, bad = 0;
    double maxdiff = 0.0;

    // part 1: simulator vs golden/sim.txt
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
            line = rstrip_cr(line);
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
                // cross-check our own zstats against the golden zp
                std::vector<double> zpc = zp_of(sd);
                for (int k = 0; k < 10; k++) {
                    total++;
                    if (dhex(zpc[k]) != dhex(zp[k])) {
                        double d = std::fabs(zpc[k] - zp[k]);
                        if (d > maxdiff) maxdiff = d;
                        if (bad < 8) std::printf("MISMATCH zstats %s[%d]: %.17g vs %.17g\n",
                                                 cur_inst.c_str(), k, zpc[k], zp[k]);
                        bad++;
                    }
                }
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
                std::vector<double> got = sim_batch(sd, comp, sampled, n_samples, mask, zp);
                for (int s = 0; s < n_samples; s++) {
                    total++;
                    if (dhex(got[s]) != dhex(want[s])) {
                        double d = std::fabs(got[s] - want[s]);
                        if (d > maxdiff) maxdiff = d;
                        if (bad < 8) std::printf("MISMATCH sim %s %s scen %d: %.17g vs %.17g\n",
                                                 cur_inst.c_str(), cur_kind.c_str(), s, got[s], want[s]);
                        bad++;
                    }
                }
            }
        }
    }

    // part 2: duration sampling vs golden/durations.txt (same-libm)
    {
        std::ifstream f(argc > 2 ? argv[2] : "golden/durations.txt");
        if (!f) { std::fprintf(stderr, "durations golden not found\n"); return 2; }
        std::string line;
        SimData sd;
        long long base_seed = 0;
        int s_idx = 0;
        while (std::getline(f, line)) {
            line = rstrip_cr(line);
            if (line.empty()) continue;
            if (line.rfind("# inst ", 0) == 0) {
                std::istringstream ss(line.substr(7));
                std::string nm, w;
                ss >> nm >> w >> base_seed;
                sd = SimData(load_instance(nm), 3, 0.2);
                s_idx = 0;
            } else if (line.rfind("dur ", 0) == 0) {
                Mt19937 rng((uint64_t)(base_seed + s_idx));
                std::istringstream ss(line.substr(4));
                std::string h;
                for (int i = 0; i < sd.n; i++) {
                    ss >> h;
                    double got = sample_duration(sd.durations[i], 0.2, rng);
                    total++;
                    if (dhex(got) != h) {
                        double d = std::fabs(got - hexd(h));
                        if (d > maxdiff) maxdiff = d;
                        if (bad < 8) std::printf("MISMATCH dur seed %lld poi %d\n", base_seed + s_idx, i);
                        bad++;
                    }
                }
                s_idx++;
            }
        }
    }

    std::printf("verify_sim: %ld values compared, %ld mismatches, max|diff|=%.3g -> %s\n",
                total, bad, maxdiff, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
