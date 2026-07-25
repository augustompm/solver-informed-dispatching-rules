// The faithful generation engine: the Mei/Jackson NS-GP loop over an extended
// terminal set, population fitness on the GPU factory. Ruler protocol: mean of
// the 30 per-seed test gains vs the NS-GP mean-30 baseline; best of the final
// population on ONE final train sample; the 500-scenario held-out read once.
//     gp_engine r101 --seeds 30 [--terms A,B] [--vocab ...] [--rmod] [--seq] [--tag t]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "factory.cuh"
#include "gp.hpp"

// EXTRA (the 18-terminal menu) comes from vocab.hpp.
#include "baseline_csv.hpp"
static constexpr long long TEST_SEED = 1000000;
static constexpr int N_TEST = 500;

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<std::string> tokens_of(const std::string& pref) {
    std::string t;
    for (char c : pref) t += (c == '(' || c == ')') ? ' ' : c;
    std::istringstream ss(t);
    std::vector<std::string> out;
    std::string w;
    while (ss >> w) out.push_back(w);
    return out;
}

static const std::set<std::string> OPS_SET = {"+", "-", "*", "/", "min", "max", "rmod"};

static std::vector<std::string> beyond_baseline(const std::string& pref) {
    std::set<std::string> toks;
    for (auto& w : tokens_of(pref)) toks.insert(w);
    std::vector<std::string> out;
    for (auto& t : toks)
        if (std::find(EXTRA.begin(), EXTRA.end(), t) != EXTRA.end() || t == "WLOSS" || t == "rmod")
            out.push_back(t);
    std::sort(out.begin(), out.end());
    return out;
}

static std::vector<std::string> terminals_used(const std::string& pref) {
    std::set<std::string> out;
    for (auto& w : tokens_of(pref))
        if (!OPS_SET.count(w)) out.insert(w);
    return {out.begin(), out.end()};
}

struct RunResult {
    double km = 0.0;
    std::vector<double> gains;
    std::vector<NodeP> rules;
};

// One faithful NS-GP run, fitness on the factory.
static std::vector<NodeP> evolve_one(const SimData& sd, const EvolutionConfig& cfg, int seed,
                                     FactoryContext& ctx, const std::vector<int64_t>& tmask,
                                     const TreeConfig& tc) {
    Mt19937 rng((uint64_t)seed);
    std::vector<NodeP> pop = ramped_half_and_half(tc, rng, cfg.population_size);
    for (int gen = 0; gen < cfg.generations; gen++) {
        std::vector<double> sampled = gen_durations(sd, cfg.train_samples, (long long)seed * 10000 + gen);
        PackedTrees pk = pack_trees(pop);
        std::vector<double> flat = ctx.evaluate(pk, sampled, cfg.train_samples, tmask);
        std::vector<double> f(pop.size());
        for (size_t p = 0; p < pop.size(); p++) {
            double acc = 0.0;
            for (int s = 0; s < cfg.train_samples; s++) acc += flat[p * cfg.train_samples + s];
            f[p] = acc / cfg.train_samples;
        }
        if (gen < cfg.generations - 1) pop = gp_step(pop, f, rng, tc, cfg);
    }
    return pop;
}

// Sequential per-seed ruler protocol.
static RunResult exhaust(const std::string& name, int n_seeds,
                         const std::vector<std::string>& vocab, bool with_rmod) {
    SimData sd(load_instance(name), 3, 0.2);
    EvolutionConfig cfg;
    std::vector<double> zp = zp_of(sd);
    FactoryContext ctx(sd, zp);
    std::vector<int64_t> tmask = term_mask_from(vocab);
    TreeConfig tc;
    tc.terminals = vocab;
    tc.functions = with_rmod ? FUNCS_RMOD : FUNCTIONS;
    RunResult rr;
    rr.km = nsgp_mean_30(name);
    std::vector<double> test_sampled = gen_durations(sd, N_TEST, TEST_SEED);
    for (int seed = 0; seed < n_seeds; seed++) {
        double t0 = now_s();
        std::vector<NodeP> pop = evolve_one(sd, cfg, seed, ctx, tmask, tc);
        long long fseed = (long long)seed * 10000 + cfg.generations;
        std::vector<double> fs = gen_durations(sd, cfg.train_samples, fseed);
        PackedTrees pk = pack_trees(pop);
        std::vector<double> ff = ctx.evaluate(pk, fs, cfg.train_samples, tmask);
        int bi = 0;
        for (size_t p = 1; p < pop.size(); p++)
            if (ff[p] > ff[bi]) bi = (int)p;
        NodeP rule = pop[bi];
        PackedTrees one = pack_trees({rule});
        std::vector<double> ts = ctx.evaluate(one, test_sampled, N_TEST, tmask);
        double got = 0.0;
        for (int s = 0; s < N_TEST; s++) got += ts[s];
        got /= N_TEST;
        double g = 100.0 * (got - rr.km) / rr.km;
        rr.gains.push_back(g);
        rr.rules.push_back(rule);
        std::string bb;
        for (auto& b : beyond_baseline(to_prefix(rule))) bb += (bb.empty() ? "" : ",") + b;
        std::printf("    seed %2d: test-gain=%+.2f%%  beyond=%s  (%.0fs)\n",
                    seed, g, bb.empty() ? "-" : bb.c_str(), now_s() - t0);
        std::fflush(stdout);
    }
    return rr;
}

// Seed-batched: all seeds evolve in parallel, one grouped launch per
// generation.
static RunResult exhaust_batched(const std::string& name, int n_seeds,
                                 const std::vector<std::string>& vocab, bool with_rmod) {
    SimData sd(load_instance(name), 3, 0.2);
    EvolutionConfig cfg;
    std::vector<double> zp = zp_of(sd);
    FactoryContext ctx(sd, zp);
    std::vector<int64_t> tmask = term_mask_from(vocab);
    TreeConfig tc;
    tc.terminals = vocab;
    tc.functions = with_rmod ? FUNCS_RMOD : FUNCTIONS;
    RunResult rr;
    rr.km = nsgp_mean_30(name);
    std::vector<double> test_sampled = gen_durations(sd, N_TEST, TEST_SEED);
    int ps = cfg.population_size;

    std::vector<Mt19937> rngs;
    rngs.reserve(n_seeds);
    for (int s = 0; s < n_seeds; s++) rngs.emplace_back((uint64_t)s);
    std::vector<std::vector<NodeP>> pops(n_seeds);
    for (int s = 0; s < n_seeds; s++) pops[s] = ramped_half_and_half(tc, rngs[s], ps);

    auto grouped_fit = [&](std::vector<long long> gseeds) {
        std::vector<NodeP> allt;
        allt.reserve((size_t)n_seeds * ps);
        for (auto& pop : pops)
            for (auto& t : pop) allt.push_back(t);
        std::vector<double> grouped((size_t)n_seeds * cfg.train_samples * sd.n);
        for (int g = 0; g < n_seeds; g++) {
            std::vector<double> one = gen_durations(sd, cfg.train_samples, gseeds[g]);
            std::copy(one.begin(), one.end(),
                      grouped.begin() + (size_t)g * cfg.train_samples * sd.n);
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
        auto fits = grouped_fit(gseeds);
        if (gen < cfg.generations - 1)
            for (int s = 0; s < n_seeds; s++) pops[s] = gp_step(pops[s], fits[s], rngs[s], tc, cfg);
    }
    std::printf("    batched %d seeds x %d gens em %.0fs\n", n_seeds, cfg.generations, now_s() - t0);
    std::fflush(stdout);
    std::vector<long long> fseeds(n_seeds);
    for (int s = 0; s < n_seeds; s++) fseeds[s] = (long long)s * 10000 + cfg.generations;
    auto ff = grouped_fit(fseeds);
    for (int s = 0; s < n_seeds; s++) {
        int bi = 0;
        for (int p = 1; p < ps; p++)
            if (ff[s][p] > ff[s][bi]) bi = p;
        NodeP rule = pops[s][bi];
        PackedTrees one = pack_trees({rule});
        std::vector<double> ts = ctx.evaluate(one, test_sampled, N_TEST, tmask);
        double got = 0.0;
        for (int t = 0; t < N_TEST; t++) got += ts[t];
        got /= N_TEST;
        rr.gains.push_back(100.0 * (got - rr.km) / rr.km);
        rr.rules.push_back(rule);
    }
    return rr;
}

static std::string json_str_list(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + ("\"" + v[i] + "\"");
    return s + "]";
}

int main(int argc, char** argv) {
    std::vector<std::string> insts;
    int n_seeds = 30;
    std::string terms_arg, vocab_arg, tag;
    bool with_rmod = false, seq = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seeds") n_seeds = std::atoi(argv[++i]);
        else if (a == "--terms") terms_arg = argv[++i];
        else if (a == "--vocab") vocab_arg = argv[++i];
        else if (a == "--tag") tag = argv[++i];
        else if (a == "--rmod") with_rmod = true;
        else if (a == "--seq") seq = true;
        else insts.push_back(a);
    }
    if (insts.empty()) insts.push_back("r101");

    auto split_csv = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream ss(s);
        std::string w;
        while (std::getline(ss, w, ',')) {
            size_t a = w.find_first_not_of(' ');
            size_t b = w.find_last_not_of(' ');
            if (a != std::string::npos) out.push_back(w.substr(a, b - a + 1));
        }
        return out;
    };

    std::vector<std::string> vocab, extra;
    if (!vocab_arg.empty()) {
        vocab = split_csv(vocab_arg);
        for (auto& t : vocab)
            if (std::find(NS_TERMINALS.begin(), NS_TERMINALS.end(), t) == NS_TERMINALS.end())
                extra.push_back(t);
    } else {
        extra = terms_arg.empty() ? EXTRA : split_csv(terms_arg);
        vocab = NS_TERMINALS;
        for (auto& t : extra) vocab.push_back(t);
    }
    for (auto& t : vocab)
        if (t != "ERC" && term_index(t) < 0) {
            std::fprintf(stderr, "unknown terminal %s (see vocab.hpp term_map)\n", t.c_str());
            return 2;
        }

    mkdir("results", 0755);
    std::string vocab_desc;
    for (auto& t : extra) vocab_desc += (vocab_desc.empty() ? "" : ",") + t;
    std::printf("# gp_engine | faithful NS-GP (pop1024 gen51 T=1-rot, %s, %s) | vocab = 11 baseline + [%s] | seeds=%d\n",
                with_rmod ? "6 ops + rmod" : "6 ops (baseline)", seq ? "seq" : "seed-batched",
                vocab_desc.c_str(), n_seeds);
    std::fflush(stdout);

    for (size_t ii = 0; ii < insts.size(); ii++) {
        const std::string& name = insts[ii];
        double t0 = now_s();
        std::printf("[%zu/%zu] %s ...\n", ii + 1, insts.size(), name.c_str());
        std::fflush(stdout);
        RunResult rr = seq ? exhaust(name, n_seeds, vocab, with_rmod)
                           : exhaust_batched(name, n_seeds, vocab, with_rmod);
        int n = (int)rr.gains.size();
        double mean_g = 0.0;
        for (double g : rr.gains) mean_g += g;
        mean_g /= n;
        double var = 0.0;
        for (double g : rr.gains) var += (g - mean_g) * (g - mean_g);
        double std_g = n > 1 ? std::sqrt(var / (n - 1)) : 0.0;
        int pos = 0;
        double best_g = rr.gains[0];
        for (double g : rr.gains) { if (g > 0) pos++; if (g > best_g) best_g = g; }
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return rr.gains[a] > rr.gains[b]; });

        std::map<std::string, int> freq, full_freq;
        std::vector<int> freq_rank;
        for (int k = 0; k < 5 && k < n; k++) {
            for (auto& t : beyond_baseline(to_prefix(rr.rules[order[k]]))) freq[t]++;
            for (auto& t : terminals_used(to_prefix(rr.rules[order[k]]))) full_freq[t]++;
        }
        std::vector<std::string> base_off;
        for (auto& t : NS_TERMINALS)
            if (!full_freq.count(t)) base_off.push_back(t);

        // per-instance result JSON
        std::string jp = "results/gp_data_" + name + (tag.empty() ? "" : "_" + tag) + ".json";
        std::ofstream jf(jp);
        jf << "{\n \"mean_gain\": " << std::fixed;
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2f", mean_g); jf << buf << ",\n";
        std::snprintf(buf, sizeof buf, "%.2f", std_g); jf << " \"std\": " << buf << ",\n";
        jf << " \"seeds_positive\": \"" << pos << "/" << n << "\",\n";
        std::snprintf(buf, sizeof buf, "%.2f", best_g); jf << " \"best_seed_gain\": " << buf << ",\n";
        jf << " \"top5_terminal_freq\": {";
        {
            std::vector<std::pair<std::string, int>> fs(freq.begin(), freq.end());
            std::stable_sort(fs.begin(), fs.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < fs.size(); i++)
                jf << (i ? ", " : "") << "\"" << fs[i].first << "\": " << fs[i].second;
        }
        jf << "},\n \"full_terminal_freq\": {";
        {
            std::vector<std::pair<std::string, int>> fs(full_freq.begin(), full_freq.end());
            std::stable_sort(fs.begin(), fs.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < fs.size(); i++)
                jf << (i ? ", " : "") << "\"" << fs[i].first << "\": " << fs[i].second;
        }
        jf << "},\n \"baseline_off_candidates\": " << json_str_list(base_off) << ",\n";
        jf << " \"top5_rules\": [";
        for (int k = 0; k < 5 && k < n; k++)
            jf << (k ? ", " : "") << "\"" << to_prefix(rr.rules[order[k]]) << "\"";
        jf << "],\n \"per_seed\": [";
        for (int k = 0; k < n; k++) {
            std::snprintf(buf, sizeof buf, "%.2f", rr.gains[order[k]]);
            jf << (k ? ", " : "") << "{\"gain\": " << buf << ", \"terminals\": "
               << json_str_list(terminals_used(to_prefix(rr.rules[order[k]]))) << "}";
        }
        jf << "],\n \"best_rule\": \"" << to_prefix(rr.rules[order[0]]) << "\",\n";
        jf << " \"vocab_extra\": " << json_str_list(extra) << ",\n";
        jf << " \"rmod\": " << (with_rmod ? "true" : "false") << "\n}\n";
        jf.close();

        std::printf("[%zu/%zu] %-6s base=%8.1f | RULER mean-over-%d = %+.2f%% (std %.2f, %d/%d pos) | best-seed %+.2f%%  (%.0fs)\n",
                    ii + 1, insts.size(), name.c_str(), rr.km, n, mean_g, std_g, pos, n, best_g, now_s() - t0);
        std::string fs_s;
        {
            std::vector<std::pair<std::string, int>> fs(freq.begin(), freq.end());
            std::stable_sort(fs.begin(), fs.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (auto& kv : fs) fs_s += (fs_s.empty() ? "" : ", ") + kv.first + ": " + std::to_string(kv.second);
        }
        std::printf("        top-5 extra freq: {%s}\n", fs_s.c_str());
        std::printf("        best-seed rule: %s\n", to_prefix(rr.rules[order[0]]).c_str());
        std::fflush(stdout);
    }
    std::printf("GP_DONE\n");
    return 0;
}
