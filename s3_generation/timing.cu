// Wall-clock timing of one full training run (30 seeds, the faithful
// protocol), with or without extra terminals (no extras = the NS-GP baseline
// vocabulary), no trace output.
//     timing c101 --seeds 30
//     timing c101 --terms NSTW,MAXN,NFEAS,REACH5 --seeds 30
#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/factory.cuh"
#include "../engine/gp.hpp"

static constexpr long long TEST_SEED = 1000000;
static constexpr int N_TEST = 500;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string inst_name, terms_arg;
    int n_seeds = 30;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--terms") terms_arg = argv[++i];
        else if (a == "--seeds") n_seeds = std::atoi(argv[++i]);
        else inst_name = a;
    }
    if (inst_name.empty()) { std::fprintf(stderr, "usage: timing <inst> [--terms A,B] [--seeds N]\n"); return 2; }
    std::vector<std::string> extra;
    if (!terms_arg.empty()) {
        std::istringstream ss(terms_arg);
        std::string w;
        while (std::getline(ss, w, ',')) {
            size_t a = w.find_first_not_of(' ');
            size_t b = w.find_last_not_of(' ');
            if (a != std::string::npos) extra.push_back(w.substr(a, b - a + 1));
        }
    }
    std::vector<std::string> vocab = NS_TERMINALS;
    for (auto& t : extra) vocab.push_back(t);

    SimData sd(load_instance(inst_name), 3, 0.2);
    EvolutionConfig cfg;
    FactoryContext ctx(sd, zp_of(sd));
    std::vector<int64_t> tmask = term_mask_from(vocab);
    TreeConfig tc;
    tc.terminals = vocab;
    tc.functions = FUNCTIONS;
    int ps = cfg.population_size;
    std::vector<Mt19937> rngs;
    for (int s = 0; s < n_seeds; s++) rngs.emplace_back((uint64_t)s);
    std::vector<std::vector<NodeP>> pops(n_seeds);
    for (int s = 0; s < n_seeds; s++) pops[s] = ramped_half_and_half(tc, rngs[s], ps);

    auto gfit = [&](std::vector<long long> gseeds) {
        std::vector<NodeP> allt;
        for (auto& pop : pops)
            for (auto& t : pop) allt.push_back(t);
        std::vector<double> grouped((size_t)n_seeds * cfg.train_samples * sd.n);
        for (int g = 0; g < n_seeds; g++) {
            std::vector<double> one = gen_durations(sd, cfg.train_samples, gseeds[g]);
            std::copy(one.begin(), one.end(), grouped.begin() + (size_t)g * cfg.train_samples * sd.n);
        }
        PackedTrees pk = pack_trees(allt);
        std::vector<double> out = ctx.evaluate_grouped(pk, grouped, n_seeds, cfg.train_samples, ps, tmask);
        std::vector<std::vector<double>> fits(n_seeds, std::vector<double>(ps));
        for (int s = 0; s < n_seeds; s++)
            for (int p = 0; p < ps; p++) {
                double acc = 0.0;
                for (int t = 0; t < cfg.train_samples; t++)
                    acc += out[((size_t)s * ps + p) * cfg.train_samples + t];
                fits[s][p] = acc / cfg.train_samples;
            }
        return fits;
    };

    double t0 = now_s();
    for (int gen = 0; gen < cfg.generations; gen++) {
        std::vector<long long> gseeds(n_seeds);
        for (int s = 0; s < n_seeds; s++) gseeds[s] = (long long)s * 10000 + gen;
        auto f = gfit(gseeds);
        if (gen < cfg.generations - 1)
            for (int s = 0; s < n_seeds; s++) pops[s] = gp_step(pops[s], f[s], rngs[s], tc, cfg);
    }
    std::vector<long long> fseeds(n_seeds);
    for (int s = 0; s < n_seeds; s++) fseeds[s] = (long long)s * 10000 + cfg.generations;
    auto ff = gfit(fseeds);
    double train_s = now_s() - t0;

    double t1 = now_s();
    std::vector<double> test = gen_durations(sd, N_TEST, TEST_SEED);
    std::vector<NodeP> finals;
    for (int s = 0; s < n_seeds; s++) {
        int b = 0;
        for (int p = 1; p < ps; p++)
            if (ff[s][p] > ff[s][b]) b = p;
        finals.push_back(pops[s][b]);
    }
    PackedTrees pk = pack_trees(finals);
    std::vector<double> out = ctx.evaluate(pk, test, N_TEST, tmask);
    (void)out;
    double test_s = now_s() - t1;

    std::string label;
    for (size_t i = 0; i < extra.size(); i++) label += (i ? "+" : "") + extra[i];
    if (label.empty()) label = "NSGP-baseline-11";
    std::printf("TIMING %s [%s] vocab=%zu seeds=%d | train %.0fs | heldout-eval %.1fs\n",
                inst_name.c_str(), label.c_str(), vocab.size(), n_seeds, train_s, test_s);
    return 0;
}
