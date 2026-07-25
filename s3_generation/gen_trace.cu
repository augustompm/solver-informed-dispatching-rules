// Per-generation trace of the faithful GP, for manual reading: per seed and
// per generation, that generation's best individual (train fitness, size, the
// extra terminals adopted); at the end the canonical final selection and the
// ruler mean-30. Writes results/trace_<inst>_<tag>.json.
//     gen_trace r101 --terms FRAGSCORE,PBUST,NSROB,REACH2ROB --tag T1 --seeds 30
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "../engine/baseline_csv.hpp"
#include "../engine/factory.cuh"
#include "../engine/gp.hpp"
#include "../engine/jwrite.hpp"

static constexpr long long TEST_SEED = 1000000;
static constexpr int N_TEST = 500;
static const std::set<std::string> OPS = {"+", "-", "*", "/", "min", "max", "rmod"};

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static bool is_num(const std::string& t) {
    std::string u = t;
    u.erase(std::remove(u.begin(), u.end(), '.'), u.end());
    return !u.empty() && std::all_of(u.begin(), u.end(), [](char c) { return c >= '0' && c <= '9'; });
}

static std::vector<std::string> extras_of(const NodeP& rule, const std::set<std::string>& base) {
    std::string t;
    for (char c : to_prefix(rule)) t += (c == '(' || c == ')') ? ' ' : c;
    std::istringstream ss(t);
    std::set<std::string> toks;
    std::string w;
    while (ss >> w) toks.insert(w);
    std::vector<std::string> out;
    for (auto& x : toks)
        if (!OPS.count(x) && !base.count(x) && !is_num(x)) out.push_back(x);
    return out;
}

int main(int argc, char** argv) {
    std::string inst_name, terms_arg, tag;
    int n_seeds = 30;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--terms") terms_arg = argv[++i];
        else if (a == "--tag") tag = argv[++i];
        else if (a == "--seeds") n_seeds = std::atoi(argv[++i]);
        else inst_name = a;
    }
    if (inst_name.empty() || terms_arg.empty() || tag.empty()) {
        std::fprintf(stderr, "usage: gen_trace <inst> --terms A,B --tag T [--seeds 30]\n");
        return 2;
    }
    std::vector<std::string> extra;
    {
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
    for (auto& t : vocab)
        if (term_index(t) < 0) { std::fprintf(stderr, "unknown terminal(s) ['%s']\n", t.c_str()); return 2; }
    std::set<std::string> base(NS_TERMINALS.begin(), NS_TERMINALS.end());

    SimData sd(load_instance(inst_name), 3, 0.2);
    EvolutionConfig cfg;
    FactoryContext ctx(sd, zp_of(sd));
    std::vector<int64_t> tmask = term_mask_from(vocab);
    double km = nsgp_mean_30(inst_name);
    std::vector<double> test = gen_durations(sd, N_TEST, TEST_SEED);
    int ps = cfg.population_size;
    TreeConfig tc;
    tc.terminals = vocab;
    tc.functions = FUNCTIONS;
    std::vector<Mt19937> rngs;
    for (int s = 0; s < n_seeds; s++) rngs.emplace_back((uint64_t)s);
    std::vector<std::vector<NodeP>> pops(n_seeds);
    for (int s = 0; s < n_seeds; s++) pops[s] = ramped_half_and_half(tc, rngs[s], ps);
    std::vector<std::vector<std::string>> trace(n_seeds);

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
        for (int s = 0; s < n_seeds; s++) {
            int b = 0;
            for (int p = 1; p < ps; p++)
                if (f[s][p] > f[s][b]) b = p;
            NodeP rule = pops[s][b];
            trace[s].push_back("{" + jstr("g") + ": " + jint(gen) + ", " +
                               jstr("fit") + ": " + jnum(round_half_even(f[s][b], 1)) + ", " +
                               jstr("sz") + ": " + jint(tree_size(rule)) + ", " +
                               jstr("ex") + ": " + jstrlist(extras_of(rule, base)) + "}");
        }
        if (gen < cfg.generations - 1)
            for (int s = 0; s < n_seeds; s++) pops[s] = gp_step(pops[s], f[s], rngs[s], tc, cfg);
    }
    std::vector<long long> fseeds(n_seeds);
    for (int s = 0; s < n_seeds; s++) fseeds[s] = (long long)s * 10000 + cfg.generations;
    auto ff = gfit(fseeds);
    std::vector<double> gains;
    std::vector<std::string> finals;
    for (int s = 0; s < n_seeds; s++) {
        int b = 0;
        for (int p = 1; p < ps; p++)
            if (ff[s][p] > ff[s][b]) b = p;
        NodeP rule = pops[s][b];
        PackedTrees one = pack_trees({rule});
        std::vector<double> ts = ctx.evaluate(one, test, N_TEST, tmask);
        double got = 0.0;
        for (int t = 0; t < N_TEST; t++) got += ts[t];
        got /= N_TEST;
        double g = 100.0 * (got - km) / km;
        gains.push_back(g);
        finals.push_back("{" + jstr("seed") + ": " + jint(s) + ", " +
                         jstr("gain") + ": " + jnum(round_half_even(g, 2)) + ", " +
                         jstr("sz") + ": " + jint(tree_size(rule)) + ", " +
                         jstr("ex") + ": " + jstrlist(extras_of(rule, base)) + ", " +
                         jstr("rule") + ": " + jstr(to_prefix(rule)) + "}");
    }
    double mean_g = 0.0, best_g = gains[0];
    int pos = 0;
    for (double g : gains) { mean_g += g; if (g > 0) pos++; if (g > best_g) best_g = g; }
    mean_g /= gains.size();
    double var = 0.0;
    for (double g : gains) var += (g - mean_g) * (g - mean_g);
    double std_g = gains.size() > 1 ? std::sqrt(var / (gains.size() - 1)) : 0.0;

    mkdir("results", 0755);
    std::ofstream o("results/trace_" + inst_name + "_" + tag + ".json");
    std::vector<std::string> tr_parts;
    for (auto& tl : trace) tr_parts.push_back(jlist(tl));
    std::string extra_join;
    for (size_t i = 0; i < extra.size(); i++) extra_join += (i ? "," : "") + extra[i];
    o << "{" << jstr("instance") << ": " << jstr(inst_name) << ", "
      << jstr("vocab_extra") << ": " << jstrlist(extra) << ", "
      << jstr("mean_gain") << ": " << jnum(round_half_even(mean_g, 2)) << ", "
      << jstr("std") << ": " << jnum(round_half_even(std_g, 2)) << ", "
      << jstr("seeds_positive") << ": " << jstr(std::to_string(pos) + "/" + std::to_string(gains.size())) << ", "
      << jstr("best_seed_gain") << ": " << jnum(round_half_even(best_g, 2)) << ", "
      << jstr("finals") << ": " << jlist(finals) << ", "
      << jstr("trace") << ": " << jlist(tr_parts) << "}";
    o.close();
    std::printf("  %s 11+[%s] | RULER mean-over-%zu = %+.2f%% (std %.2f, %d/%zu pos) | best %+.2f  (%.0fs)\n",
                inst_name.c_str(), extra_join.c_str(), gains.size(), mean_g, std_g, pos,
                gains.size(), best_g, now_s() - t0);
    std::printf("TRACE_DONE\n");
    return 0;
}
