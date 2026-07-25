// Host-side simulation context: instance arrays with the (n+1)x(n+1)
// Euclidean matrix, stochastic service sampling D(p) ~ N(d(p), sigma d(p))
// clipped at 0 (Mei 2018 SIV.A) with one fresh generator per scenario (common
// random numbers), the CPU batch runner, and the term-mask and z-stats helpers.
#pragma once
#include <vector>
#include "instance.hpp"
#include "rng.hpp"
#include "sim_core.hpp"
#include "vocab.hpp"

struct SimData {
    int n = 0, m = 3, ld = 0;
    double sigma = 0.2, tmax = 0.0;
    std::vector<double> scores, durations, open_t, close_t, dist, dist_to_depot;

    SimData() = default;
    explicit SimData(const Instance& inst, int m_ = 3, double sigma_ = 0.2) : m(m_), sigma(sigma_) {
        n = inst.n_customers();
        ld = n + 1;
        tmax = inst.tmax;
        scores.resize(n); durations.resize(n); open_t.resize(n); close_t.resize(n);
        for (int i = 0; i < n; i++) {
            const POI& p = inst.pois[i + 1];
            scores[i] = p.score; durations[i] = p.duration;
            open_t[i] = p.open_time; close_t[i] = p.close_time;
        }
        std::vector<double> cx(ld), cy(ld);
        cx[0] = inst.depot().x; cy[0] = inst.depot().y;
        for (int i = 0; i < n; i++) { cx[i + 1] = inst.pois[i + 1].x; cy[i + 1] = inst.pois[i + 1].y; }
        dist.resize((size_t)ld * ld);
        for (int a = 0; a < ld; a++)
            for (int b = 0; b < ld; b++) {
                double dx = cx[a] - cx[b], dy = cy[a] - cy[b];
                dist[(size_t)a * ld + b] = std::sqrt(dx * dx + dy * dy);
            }
        dist_to_depot.resize(n);
        for (int i = 0; i < n; i++) dist_to_depot[i] = dist[(size_t)(i + 1) * ld + 0];
    }
};

// Zero-duration visits draw nothing from the generator.
inline double sample_duration(double expected, double sigma, Mt19937& rng) {
    if (expected <= 0.0) return 0.0;
    double v = rng.gauss(expected, sigma * expected);
    return v > 0.0 ? v : 0.0;
}

// One fresh generator per scenario, seeded base_seed + s.
inline std::vector<double> gen_durations(const SimData& sd, int n_samples, long long base_seed) {
    std::vector<double> out((size_t)n_samples * sd.n);
    for (int s = 0; s < n_samples; s++) {
        Mt19937 rng((uint64_t)(base_seed + s));
        for (int i = 0; i < sd.n; i++)
            out[(size_t)s * sd.n + i] = sample_duration(sd.durations[i], sd.sigma, rng);
    }
    return out;
}

struct SimBuffers {
    std::vector<char> visited, scratch;
    std::vector<double> terms, stack, struct_buf;
    explicit SimBuffers(int n)
        : visited(n), scratch(n), terms(N_TERMS), stack(MAX_STACK), struct_buf(4) {}
};

inline std::vector<double> sim_batch(const SimData& sd, const Compiled& c,
                                     const std::vector<double>& all_sampled, int n_samples,
                                     const std::vector<int64_t>& term_mask,
                                     const std::vector<double>& zp) {
    SimBuffers buf(sd.n);
    std::vector<double> out(n_samples);
    for (int s = 0; s < n_samples; s++) {
        out[s] = sim_one(c.ops.data(), &c.args[0][0], (int)c.ops.size(),
                         sd.scores.data(), sd.durations.data(), sd.open_t.data(), sd.close_t.data(),
                         sd.dist.data(), sd.ld, sd.dist_to_depot.data(), sd.tmax,
                         &all_sampled[(size_t)s * sd.n], sd.m, sd.n,
                         term_mask.data(), zp.data(),
                         (bool*)buf.visited.data(), buf.terms.data(), buf.stack.data(),
                         buf.struct_buf.data(), (bool*)buf.scratch.data());
    }
    return out;
}

inline std::vector<int64_t> term_mask_from(const std::vector<std::string>& terms) {
    std::vector<int64_t> m(N_TERMS, 0);
    for (auto& nm : terms) {
        if (nm == "ERC") continue;
        int ti = term_index(nm);
        if (ti < 0) throw std::runtime_error("unknown terminal " + nm);
        m[ti] = 1;
    }
    return m;
}

inline std::vector<double> zp_of(const SimData& sd) {
    std::vector<double> zm(5), zs(5);
    std::vector<char> vis(sd.n);
    zstats(sd.scores.data(), sd.durations.data(), sd.open_t.data(), sd.close_t.data(),
           sd.dist.data(), sd.ld, sd.dist_to_depot.data(), sd.tmax, sd.n,
           zm.data(), zs.data(), (bool*)vis.data());
    std::vector<double> zp(10);
    for (int k = 0; k < 5; k++) { zp[k] = zm[k]; zp[5 + k] = zs[k]; }
    return zp;
}

inline std::vector<double> zp_unit() {
    std::vector<double> zp(10, 0.0);
    for (int k = 5; k < 10; k++) zp[k] = 1.0;
    return zp;
}
