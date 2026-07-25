// Structured-population genetic algorithm over 0/1 terminal-subset
// chromosomes: a population of 13 on a ternary tree of overlapping
// leader/supporter clusters (one leader, three supporters each). Every
// generation each leader crosses with each of its three supporters (12
// uniform crossovers), children are evaluated in lockstep batches (one
// launch per inner generation, per-genome terminal masks), and each child
// replaces the worse of its two parents when it improves it; swaps restore
// the hierarchy. Initialization is random; convergence is declared by the
// stagnation cutoff only, the generation cap is a guard. Finalists are
// deduplicated and the champion is confirmed at the full seed count.
//     ga <inst> --universe A,B,... [--seeds random|neutral]
//        [--pgens 51] [--proxy 16] [--confirm 30] [--pm 0.5] [--gens 50]
//        [--stag 5] [--bmax 10] [--top 4] [--nofull] [--gate] [--cache F]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../engine/factory.cuh"
#include "../engine/gp.hpp"

static constexpr long long TRAIN_SEED = 2000000;
static constexpr int N_TRAIN = 64;

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string w;
    while (std::getline(ss, w, ',')) {
        size_t a = w.find_first_not_of(' ');
        size_t b = w.find_last_not_of(' ');
        if (a != std::string::npos) out.push_back(w.substr(a, b - a + 1));
    }
    return out;
}

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string inst, uni_arg, seed_mode = "random", cache_path;
    int pgens = 51, proxy = 16, confirm = 30, gens = 50, stag_k = 5, bmax = 10, top_n = 4;
    double pm_mult = 0.5;
    bool gate = false, nofull = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--universe") uni_arg = argv[++i];
        else if (a == "--seeds") seed_mode = argv[++i];
        else if (a == "--pgens") pgens = std::atoi(argv[++i]);
        else if (a == "--proxy") proxy = std::atoi(argv[++i]);
        else if (a == "--confirm") confirm = std::atoi(argv[++i]);
        else if (a == "--pm") pm_mult = std::atof(argv[++i]);
        else if (a == "--gens") gens = std::atoi(argv[++i]);
        else if (a == "--stag") stag_k = std::atoi(argv[++i]);
        else if (a == "--bmax") bmax = std::atoi(argv[++i]);
        else if (a == "--gate") gate = true;
        else if (a == "--nofull") nofull = true;
        else if (a == "--top") top_n = std::atoi(argv[++i]);
        else if (a == "--cache") cache_path = argv[++i];
        else inst = a;
    }
    std::vector<std::string> uni = split_csv(uni_arg);
    int nb = (int)uni.size();
    if (inst.empty() || nb == 0 || nb > 30) {
        std::fprintf(stderr, "usage: ga <inst> --universe A,B,... [--seeds random|neutral]\n");
        return 2;
    }
    for (auto& t : uni)
        if (term_index(t) < 0) { std::fprintf(stderr, "unknown terminal %s\n", t.c_str()); return 2; }

    SimData sd(load_instance(inst), 3, 0.2);
    FactoryContext ctx(sd, zp_of(sd));
    std::vector<double> blk = gen_durations(sd, N_TRAIN, TRAIN_SEED);
    EvolutionConfig cfg;
    int ps = cfg.population_size;
    Mt19937 rng(20260714);

    auto vocab_of = [&](uint32_t g) {
        std::vector<std::string> v = NS_TERMINALS;
        for (int b = 0; b < nb; b++)
            if (g >> b & 1) v.push_back(uni[b]);
        return v;
    };
    auto name_of = [&](uint32_t g) {
        std::string s;
        for (int b = 0; b < nb; b++)
            if (g >> b & 1) s += (s.empty() ? "" : ",") + uni[b];
        return s.empty() ? std::string("(base)") : s;
    };
    auto nbits = [](uint32_t g) { int c = 0; while (g) { c += g & 1; g >>= 1; } return c; };

    // Lockstep retrain of a batch of genomes: per inner generation one launch
    // over all genomes x seeds x population; finals scored on the train block.
    // Returns per-genome per-seed train-block means.
    long evals = 0;
    double gpu_s = 0.0;
    auto batch_score = [&](const std::vector<uint32_t>& gs, int n, int ng) {
        int K = (int)gs.size();
        std::vector<std::vector<double>> res(K);
        for (int k0 = 0; k0 < K; k0 += bmax) {
            int kb = std::min(bmax, K - k0);
            double t0 = now_s();
            std::vector<TreeConfig> tcs(kb);
            std::vector<std::vector<int64_t>> masks(kb);
            for (int k = 0; k < kb; k++) {
                tcs[k].terminals = vocab_of(gs[k0 + k]);
                tcs[k].functions = FUNCTIONS;
                masks[k] = term_mask_from(tcs[k].terminals);
            }
            std::vector<int64_t> mask_flat;
            for (int k = 0; k < kb; k++)
                mask_flat.insert(mask_flat.end(), masks[k].begin(), masks[k].end());
            std::vector<std::vector<Mt19937>> rngs(kb);
            std::vector<std::vector<std::vector<NodeP>>> pops(kb);
            for (int k = 0; k < kb; k++) {
                for (int s = 0; s < n; s++) rngs[k].emplace_back((uint64_t)s);
                pops[k].resize(n);
            }
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int k = 0; k < kb; k++)
                for (int s = 0; s < n; s++)
                    pops[k][s] = ramped_half_and_half(tcs[k], rngs[k][s], ps);
            auto gfit = [&](int gen_id) {
                std::vector<NodeP> all;
                all.reserve((size_t)kb * n * ps);
                for (int k = 0; k < kb; k++)
                    for (int s = 0; s < n; s++)
                        for (auto& t : pops[k][s]) all.push_back(t);
                std::vector<double> grouped((size_t)kb * n * cfg.train_samples * sd.n);
                for (int k = 0; k < kb; k++)
                    for (int s = 0; s < n; s++) {
                        std::vector<double> one = gen_durations(sd, cfg.train_samples,
                                                                (long long)s * 10000 + gen_id);
                        std::copy(one.begin(), one.end(),
                                  grouped.begin() + ((size_t)k * n + s) * cfg.train_samples * sd.n);
                    }
                PackedTrees pk = pack_trees(all);
                std::vector<double> out = ctx.evaluate_grouped_masks(pk, grouped, cfg.train_samples,
                                                                     ps, n * ps, mask_flat);
                std::vector<std::vector<std::vector<double>>> f(kb);
                for (int k = 0; k < kb; k++) {
                    f[k].assign(n, std::vector<double>(ps));
                    for (int s = 0; s < n; s++)
                        for (int p = 0; p < ps; p++) {
                            double acc = 0.0;
                            for (int t = 0; t < cfg.train_samples; t++)
                                acc += out[(((size_t)k * n + s) * ps + p) * cfg.train_samples + t];
                            f[k][s][p] = acc / cfg.train_samples;
                        }
                }
                return f;
            };
            for (int gen = 0; gen < ng; gen++) {
                auto f = gfit(gen);
                if (gen < ng - 1) {
                    #pragma omp parallel for collapse(2) schedule(dynamic)
                    for (int k = 0; k < kb; k++)
                        for (int s = 0; s < n; s++)
                            pops[k][s] = gp_step(pops[k][s], f[k][s], rngs[k][s], tcs[k], cfg);
                }
            }
            auto ff = gfit(ng);
            std::vector<NodeP> finals;
            for (int k = 0; k < kb; k++)
                for (int s = 0; s < n; s++) {
                    int b = 0;
                    for (int p = 1; p < ps; p++)
                        if (ff[k][s][p] > ff[k][s][b]) b = p;
                    finals.push_back(pops[k][s][b]);
                }
            std::vector<double> blk_rep;
            for (int k = 0; k < kb; k++)
                blk_rep.insert(blk_rep.end(), blk.begin(), blk.end());
            PackedTrees pk = pack_trees(finals);
            std::vector<double> fits = ctx.evaluate_grouped_masks(pk, blk_rep, N_TRAIN, n, n,
                                                                  mask_flat);
            for (int k = 0; k < kb; k++) {
                std::vector<double> per(n);
                for (int s = 0; s < n; s++) {
                    double acc = 0.0;
                    for (int t = 0; t < N_TRAIN; t++)
                        acc += fits[((size_t)k * n + s) * N_TRAIN + t];
                    per[s] = acc / N_TRAIN;
                }
                res[k0 + k] = per;
            }
            evals += kb;
            double dt = now_s() - t0;
            gpu_s += dt;
            std::printf("  batch %d genomes @%d seeds | %.1fs | %.2fs/eval\n", kb, n, dt, dt / kb);
        }
        return res;
    };

    // Sequential reference path (one genome, global mask).
    auto seq_score = [&](uint32_t g, int n, int ng) {
        std::vector<std::string> vocab = vocab_of(g);
        std::vector<int64_t> tm = term_mask_from(vocab);
        TreeConfig tc;
        tc.terminals = vocab;
        tc.functions = FUNCTIONS;
        std::vector<Mt19937> rngs;
        for (int s = 0; s < n; s++) rngs.emplace_back((uint64_t)s);
        std::vector<std::vector<NodeP>> pops(n);
        for (int s = 0; s < n; s++) pops[s] = ramped_half_and_half(tc, rngs[s], ps);
        auto gfit = [&](const std::vector<long long>& gseeds) {
            std::vector<NodeP> all;
            for (auto& p : pops)
                for (auto& t : p) all.push_back(t);
            std::vector<double> grouped((size_t)n * cfg.train_samples * sd.n);
            for (int s = 0; s < n; s++) {
                std::vector<double> one = gen_durations(sd, cfg.train_samples, gseeds[s]);
                std::copy(one.begin(), one.end(), grouped.begin() + (size_t)s * cfg.train_samples * sd.n);
            }
            PackedTrees pk = pack_trees(all);
            std::vector<double> out = ctx.evaluate_grouped(pk, grouped, n, cfg.train_samples, ps, tm);
            std::vector<std::vector<double>> f(n, std::vector<double>(ps));
            for (int s = 0; s < n; s++)
                for (int p = 0; p < ps; p++) {
                    double acc = 0.0;
                    for (int t = 0; t < cfg.train_samples; t++)
                        acc += out[((size_t)s * ps + p) * cfg.train_samples + t];
                    f[s][p] = acc / cfg.train_samples;
                }
            return f;
        };
        for (int gen = 0; gen < ng; gen++) {
            std::vector<long long> gseeds(n);
            for (int s = 0; s < n; s++) gseeds[s] = (long long)s * 10000 + gen;
            auto f = gfit(gseeds);
            if (gen < ng - 1)
                for (int s = 0; s < n; s++) pops[s] = gp_step(pops[s], f[s], rngs[s], tc, cfg);
        }
        std::vector<long long> fs(n);
        for (int s = 0; s < n; s++) fs[s] = (long long)s * 10000 + ng;
        auto ff = gfit(fs);
        std::vector<NodeP> finals;
        for (int s = 0; s < n; s++) {
            int b = 0;
            for (int p = 1; p < ps; p++)
                if (ff[s][p] > ff[s][b]) b = p;
            finals.push_back(pops[s][b]);
        }
        PackedTrees pk = pack_trees(finals);
        std::vector<double> fits = ctx.evaluate(pk, blk, N_TRAIN, tm);
        std::vector<double> per(n);
        for (int s = 0; s < n; s++) {
            double acc = 0.0;
            for (int t = 0; t < N_TRAIN; t++) acc += fits[(size_t)s * N_TRAIN + t];
            per[s] = acc / N_TRAIN;
        }
        return per;
    };

    if (gate) {
        uint32_t full = nb >= 32 ? 0xffffffffu : ((1u << nb) - 1u);
        std::vector<uint32_t> gset = {0u, full, 1u << 0, 1u << (nb > 7 ? 7 : nb - 1),
                                      (1u << 1) | (1u << (nb > 4 ? 4 : 0)) | (1u << (nb > 9 ? 9 : 0))};
        int gn = 4, gng = 5;
        std::printf("# gate %s | %zu genomes | %d seeds x %d gens\n", inst.c_str(), gset.size(), gn, gng);
        auto bat = batch_score(gset, gn, gng);
        long long cmp = 0, bad = 0;
        double mx = 0.0;
        for (size_t i = 0; i < gset.size(); i++) {
            auto ref = seq_score(gset[i], gn, gng);
            for (int s = 0; s < gn; s++) {
                double d = std::fabs(bat[i][s] - ref[s]);
                mx = std::max(mx, d);
                cmp++;
                if (bat[i][s] != ref[s]) bad++;
            }
            std::printf("  genome [%s] batched %.6f seq %.6f\n",
                        name_of(gset[i]).c_str(), bat[i][0], ref[0]);
        }
        std::printf("GATE compared %lld | mismatches %lld | max|diff| %.3e | %s\n",
                    cmp, bad, mx, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
        return bad == 0 ? 0 : 1;
    }

    // Cache genome -> per-seed proxy vector; mean is derived.
    std::map<uint32_t, std::vector<double>> cache;
    std::map<uint32_t, std::vector<double>> cache_conf;
    auto mean_of = [](const std::vector<double>& v) {
        double a = 0.0;
        for (double x : v) a += x;
        return a / v.size();
    };

    // Persistent evaluation cache (--cache F): runs are deterministic, so a
    // relaunch retraces the same genome sequence and every batch already paid
    // for is a hit -- an effective resume after a power cut. The cache is a
    // pure function store (genome -> per-seed fitness), so entries survive
    // changes to pm/stag/seed-mode; the header pins what does change the
    // function (instance, universe order behind the bitmask, proxy/pgens,
    // confirm/inner generations) and a mismatch discards the file. Fitness is
    // stored as hexfloat for exact round-trip; a line truncated by the cut is
    // skipped and its batch recomputed.
    std::string uni_join;
    for (int b = 0; b < nb; b++) uni_join += (b ? "," : "") + uni[b];
    char cache_hdr[512];
    std::snprintf(cache_hdr, sizeof cache_hdr, "# gacache v1 %s uni=%s proxy=%d pgens=%d confirm=%d cgens=%d",
                  inst.c_str(), uni_join.c_str(), proxy, pgens, confirm, cfg.generations);
    std::FILE* cache_fp = nullptr;
    if (!cache_path.empty()) {
        long hits = 0;
        {
            std::ifstream in(cache_path);
            std::string line;
            if (in && std::getline(in, line) && line == cache_hdr) {
                while (std::getline(in, line)) {
                    std::istringstream ls(line);
                    std::string kind, hex, tok;
                    int n;
                    if (!(ls >> kind >> hex >> n) || (kind != "P" && kind != "C")) continue;
                    uint32_t g = (uint32_t)std::strtoul(hex.c_str(), nullptr, 16);
                    std::vector<double> v;
                    v.reserve(n);
                    while ((int)v.size() < n && ls >> tok) v.push_back(std::strtod(tok.c_str(), nullptr));
                    if ((int)v.size() != n) continue;
                    (kind == "P" ? cache : cache_conf)[g] = v;
                    hits++;
                }
            }
        }
        cache_fp = std::fopen(cache_path.c_str(), hits > 0 ? "a" : "w");
        if (cache_fp && hits == 0) std::fprintf(cache_fp, "%s\n", cache_hdr);
        if (hits > 0) std::printf("# cache %s | %ld entries loaded\n", cache_path.c_str(), hits);
    }
    auto cache_append = [&](const char* kind, uint32_t g, const std::vector<double>& v) {
        if (!cache_fp) return;
        std::fprintf(cache_fp, "%s %x %d", kind, g, (int)v.size());
        for (double x : v) std::fprintf(cache_fp, " %a", x);
        std::fprintf(cache_fp, "\n");
    };

    auto ensure = [&](std::vector<uint32_t> want) {
        std::sort(want.begin(), want.end());
        want.erase(std::unique(want.begin(), want.end()), want.end());
        std::vector<uint32_t> miss;
        for (uint32_t g : want)
            if (!cache.count(g)) miss.push_back(g);
        if (miss.empty()) return;
        auto res = batch_score(miss, proxy, pgens);
        for (size_t i = 0; i < miss.size(); i++) {
            cache[miss[i]] = res[i];
            cache_append("P", miss[i], res[i]);
        }
        if (cache_fp) std::fflush(cache_fp);
    };

    std::vector<uint32_t> seeds;
    uint32_t full = nb >= 32 ? 0xffffffffu : ((1u << nb) - 1u);
    if (seed_mode == "neutral") {
        seeds.push_back(0);
        if (!nofull) seeds.push_back(full);
        for (int b = 0; b < nb; b++) seeds.push_back(1u << b);
    }
    // seed_mode "random": no seeded genomes, the fill below draws all 13.
    std::sort(seeds.begin(), seeds.end());
    seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());

    // Ternary tree of 13: slot 0 root, 1-3 mids, 4-12 leaves; cluster i>0 =
    // mid i with leaves 4+3(i-1)..6+3(i-1); cluster 0 = root with the mids.
    const int NP = 13;
    std::vector<uint32_t> pool = seeds;
    while ((int)pool.size() < NP) {
        uint32_t g = 0;
        for (int b = 0; b < nb; b++)
            if (rng.random() < 0.35) g |= 1u << b;
        pool.push_back(g);
    }
    std::printf("# ga %s | universe %d | seeds %s (%zu)%s | tree-13 uniform-x pm %.2f | gens %d stag %d\n",
                inst.c_str(), nb, seed_mode.c_str(), seeds.size(), nofull ? " nofull" : "",
                pm_mult, gens, stag_k);
    double t_start = now_s();
    ensure(pool);
    std::stable_sort(pool.begin(), pool.end(), [&](uint32_t a, uint32_t b) {
        double fa = mean_of(cache[a]), fb = mean_of(cache[b]);
        if (fa != fb) return fa > fb;
        return nbits(a) < nbits(b);
    });
    pool.resize(NP);
    std::vector<uint32_t> tree(pool.begin(), pool.end());

    auto restore = [&]() {
        for (int pass = 0; pass < NP; pass++) {
            bool moved = false;
            for (int c = 3; c >= 0; c--) {
                int lead = c == 0 ? 0 : c;
                int base = c == 0 ? 1 : 4 + 3 * (c - 1);
                int cnt = 3;
                int best = -1;
                for (int j = 0; j < cnt; j++) {
                    int idx = base + j;
                    if (mean_of(cache[tree[idx]]) > mean_of(cache[tree[lead]]))
                        if (best < 0 || mean_of(cache[tree[idx]]) > mean_of(cache[tree[best]]))
                            best = idx;
                }
                if (best >= 0) { std::swap(tree[lead], tree[best]); moved = true; }
            }
            if (!moved) break;
        }
    };
    restore();

    // Uniform crossover (common genes always inherited, divergent genes drawn
    // 50/50) followed by per-gene mutation at pm/L.
    auto cross_mut = [&](uint32_t a, uint32_t b) {
        uint32_t child = 0;
        for (int k = 0; k < nb; k++)
            child |= (rng.random() < 0.5 ? a : b) & (1u << k);
        for (int k = 0; k < nb; k++)
            if (rng.random() < pm_mult / nb) child ^= 1u << k;
        return child;
    };

    double best_seen = mean_of(cache[tree[0]]);
    int stag = 0;
    for (int gen = 0; gen < gens; gen++) {
        std::set<uint32_t> uniq(tree.begin(), tree.end());
        double pmean = 0.0;
        for (int i = 0; i < NP; i++) pmean += mean_of(cache[tree[i]]);
        pmean /= NP;
        std::printf("gen %d | best %.2f [%s] | mean %.2f | unique %zu | evals %ld\n",
                    gen, mean_of(cache[tree[0]]), name_of(tree[0]).c_str(), pmean, uniq.size(), evals);
        if ((int)uniq.size() <= 6) {
            std::vector<uint32_t> fresh;
            for (int r = 0; r < 2; r++) {
                uint32_t g = 0;
                for (int b = 0; b < nb; b++)
                    if (rng.random() < 0.35) g |= 1u << b;
                tree[12 - r] = g;
                fresh.push_back(g);
            }
            ensure(fresh);
            std::printf("  reinject 2\n");
        }
        // Full intra-cluster reproduction: every leader crosses with each of
        // its three supporters, 12 children per generation.
        std::vector<uint32_t> children;
        std::vector<std::pair<int, int>> pairs;
        for (int c = 0; c < 4; c++) {
            int lead = c == 0 ? 0 : c;
            int base = c == 0 ? 1 : 4 + 3 * (c - 1);
            for (int j = 0; j < 3; j++) {
                int sup = base + j;
                children.push_back(cross_mut(tree[lead], tree[sup]));
                pairs.push_back({lead, sup});
            }
        }
        ensure(children);
        for (size_t i = 0; i < children.size(); i++) {
            int lead = pairs[i].first, sup = pairs[i].second;
            int wp = mean_of(cache[tree[sup]]) <= mean_of(cache[tree[lead]]) ? sup : lead;
            if (mean_of(cache[children[i]]) > mean_of(cache[tree[wp]]))
                tree[wp] = children[i];
        }
        restore();
        double cur = mean_of(cache[tree[0]]);
        if (cur > best_seen) { best_seen = cur; stag = 0; }
        else stag++;
        if (stag >= stag_k) {
            std::printf("stop: stagnant %d gens\n", stag);
            break;
        }
    }
    {
        double pmean = 0.0;
        for (int i = 0; i < NP; i++) pmean += mean_of(cache[tree[i]]);
        pmean /= NP;
        std::printf("gen end | best %.2f [%s] | mean %.2f | evals %ld\n",
                    mean_of(cache[tree[0]]), name_of(tree[0]).c_str(), pmean, evals);
    }

    std::vector<uint32_t> fin_g(tree.begin(), tree.end());
    std::sort(fin_g.begin(), fin_g.end());
    fin_g.erase(std::unique(fin_g.begin(), fin_g.end()), fin_g.end());
    std::stable_sort(fin_g.begin(), fin_g.end(), [&](uint32_t a, uint32_t b) {
        double fa = mean_of(cache[a]), fb = mean_of(cache[b]);
        if (fa != fb) return fa > fb;
        return nbits(a) < nbits(b);
    });
    if ((int)fin_g.size() > top_n) fin_g.resize(top_n);
    std::printf("\nconfirming top-%zu at %d seeds:\n", fin_g.size(), confirm);
    std::vector<uint32_t> cmiss;
    for (uint32_t g : fin_g)
        if (!cache_conf.count(g)) cmiss.push_back(g);
    if (!cmiss.empty()) {
        auto cres = batch_score(cmiss, confirm, cfg.generations);
        for (size_t i = 0; i < cmiss.size(); i++) {
            cache_conf[cmiss[i]] = cres[i];
            cache_append("C", cmiss[i], cres[i]);
        }
        if (cache_fp) std::fflush(cache_fp);
    }
    std::vector<std::vector<double>> conf;
    for (uint32_t g : fin_g) conf.push_back(cache_conf[g]);
    int bi = 0;
    for (size_t i = 1; i < fin_g.size(); i++) {
        double a = mean_of(conf[i]), b = mean_of(conf[bi]);
        if (a > b || (a == b && nbits(fin_g[i]) < nbits(fin_g[bi]))) bi = (int)i;
    }
    for (size_t i = 0; i < fin_g.size(); i++)
        std::printf("  %.2f (proxy %.2f) [%s]\n", mean_of(conf[i]),
                    mean_of(cache[fin_g[i]]), name_of(fin_g[i]).c_str());
    double wall = now_s() - t_start;
    std::printf("\nCHAMPION train-%d %.2f | evals %ld | terms: %s\n",
                confirm, mean_of(conf[bi]), evals, name_of(fin_g[bi]).c_str());
    std::printf("PERF evals %ld | wall %.0fs | %.1fs/eval | batch-gpu %.0fs\n",
                evals, wall, wall / std::max(1L, evals), gpu_s);
    if (cache_fp) std::fclose(cache_fp);
    return 0;
}
