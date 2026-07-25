// All terminal functions of the engine, one source for CPU and GPU
// (__host__ __device__ under nvcc, plain inline otherwise): the NS
// neighbourhood family (Jackson & Mei 2020 Alg 1), the k-step reachability and
// chain families, and the structural window-cascade family (closed-form
// Gaussian tails, no simulation).
#pragma once
#include <cmath>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

inline constexpr double SIM_SIGMA = 0.2;   // == terms_cpu._SIGMA
inline constexpr double SQRT2_C = 1.4142135623730951;

HD inline double t_ns(int i, double finish_time_at_p, const double* scores, const double* durations,
                      const double* open_t, const double* close_t, const double* dist, int ld,
                      const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_id = i + 1;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_id * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_time_at_p + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        total += scores[j] / t_p_nb;
    }
    return total;
}

HD inline double t_regret(int i, int current, double current_time, double finish_i,
                          const double* scores, const double* durations, const double* open_t,
                          const double* close_t, const double* dist, int ld,
                          const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double arr_now = current_time + dist[current * ld + j + 1];
        if (arr_now > close_t[j]) continue;
        double start_now = arr_now > open_t[j] ? arr_now : open_t[j];
        if (start_now + durations[j] + dist_to_depot[j] > tmax) continue;
        double arr_after = finish_i + dist[poi_i * ld + j + 1];
        if (arr_after > close_t[j]) { total += scores[j]; continue; }
        double start_after = arr_after > open_t[j] ? arr_after : open_t[j];
        if (start_after + durations[j] + dist_to_depot[j] > tmax) total += scores[j];
    }
    return total;
}

HD inline double t_ns_rob(int i, double finish_i, double dur_i, const double* scores,
                          const double* durations, const double* open_t, const double* close_t,
                          const double* dist, int ld, const double* dist_to_depot,
                          const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double finish_rob = finish_i + dur_i * SIM_SIGMA;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_rob + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] * (1.0 + SIM_SIGMA) + dist_to_depot[j] > tmax) continue;
        total += scores[j] / t_p_nb;
    }
    return total;
}

HD inline double t_maxn(int i, double finish_i, const double* scores, const double* durations,
                        const double* open_t, const double* close_t, const double* dist, int ld,
                        const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double best = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_i + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        double v = scores[j] / t_p_nb;
        if (v > best) best = v;
    }
    return best;
}

HD inline double t_reach2(int i, double finish_i, const double* scores, const double* durations,
                          const double* open_t, const double* close_t, const double* dist, int ld,
                          const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double best_v = 0.0, finish_j = 0.0;
    int best_j = -1;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t = dist[poi_i * ld + j + 1];
        if (t < 1e-10) continue;
        double arrival = finish_i + t;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        double v = scores[j] / t;
        if (v > best_v) { best_v = v; best_j = j; finish_j = start + durations[j]; }
    }
    if (best_j < 0) return 0.0;
    int poi_j = best_j + 1;
    double best_v2 = 0.0, s2 = 0.0;
    for (int k = 0; k < n; k++) {
        if (k == i || k == best_j || visited[k]) continue;
        double t = dist[poi_j * ld + k + 1];
        if (t < 1e-10) continue;
        double arrival = finish_j + t;
        if (arrival > close_t[k]) continue;
        double start = arrival > open_t[k] ? arrival : open_t[k];
        if (start + durations[k] + dist_to_depot[k] > tmax) continue;
        double v = scores[k] / t;
        if (v > best_v2) { best_v2 = v; s2 = scores[k]; }
    }
    return scores[best_j] + s2;
}

HD inline double t_skiploss(int i, int cur, double current_time, const double* scores,
                            const double* durations, const double* open_t, const double* close_t,
                            const double* dist, int ld, const double* dist_to_depot,
                            const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double best_v = 0.0, finish_q = 0.0;
    int best_q = -1;
    for (int q = 0; q < n; q++) {
        if (q == i || visited[q]) continue;
        double t = dist[cur * ld + q + 1];
        if (t < 1e-10) continue;
        double arrival = current_time + t;
        if (arrival > close_t[q]) continue;
        double start = arrival > open_t[q] ? arrival : open_t[q];
        if (start + durations[q] + dist_to_depot[q] > tmax) continue;
        double v = scores[q] / t;
        if (v > best_v) { best_v = v; best_q = q; finish_q = start + durations[q]; }
    }
    if (best_q < 0) return 0.0;
    double t_qi = dist[(best_q + 1) * ld + poi_i];
    double arr_i = finish_q + t_qi;
    if (arr_i > close_t[i]) return scores[i];
    double start_i = arr_i > open_t[i] ? arr_i : open_t[i];
    if (start_i + durations[i] + dist_to_depot[i] > tmax) return scores[i];
    return 0.0;
}

HD inline double t_reach2_rob(int i, double finish_i, double dur_i, const double* scores,
                              const double* durations, const double* open_t, const double* close_t,
                              const double* dist, int ld, const double* dist_to_depot,
                              const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double finish_rob = finish_i + dur_i * SIM_SIGMA;
    double best_v = 0.0, finish_j = 0.0;
    int best_j = -1;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t = dist[poi_i * ld + j + 1];
        if (t < 1e-10) continue;
        double arrival = finish_rob + t;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] * (1.0 + SIM_SIGMA) + dist_to_depot[j] > tmax) continue;
        double v = scores[j] / t;
        if (v > best_v) { best_v = v; best_j = j; finish_j = start + durations[j] * (1.0 + SIM_SIGMA); }
    }
    if (best_j < 0) return 0.0;
    int poi_j = best_j + 1;
    double best_v2 = 0.0, s2 = 0.0;
    for (int k = 0; k < n; k++) {
        if (k == i || k == best_j || visited[k]) continue;
        double t = dist[poi_j * ld + k + 1];
        if (t < 1e-10) continue;
        double arrival = finish_j + t;
        if (arrival > close_t[k]) continue;
        double start = arrival > open_t[k] ? arrival : open_t[k];
        if (start + durations[k] * (1.0 + SIM_SIGMA) + dist_to_depot[k] > tmax) continue;
        double v = scores[k] / t;
        if (v > best_v2) { best_v2 = v; s2 = scores[k]; }
    }
    return scores[best_j] + s2;
}

HD inline double t_nstw(int i, double finish_i, const double* scores, const double* durations,
                        const double* open_t, const double* close_t, const double* dist, int ld,
                        const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t = dist[poi_i * ld + j + 1];
        if (t < 1e-10) continue;
        double arrival = finish_i + t;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        double wait = open_t[j] - arrival;
        if (wait < 0.0) wait = 0.0;
        double win = close_t[j] - open_t[j];
        double weight = win > 1e-10 ? 1.0 - wait / win : 1.0;
        if (weight < 0.0) weight = 0.0;
        total += (scores[j] / t) * weight;
    }
    return total;
}

HD inline double t_rand(int i) {
    long long x = (((long long)i + 1) * 1103515245LL + 12345LL) & 0x7FFFFFFFLL;
    x = (x * 1103515245LL + 12345LL) & 0x7FFFFFFFLL;
    return (double)(x % 100000LL) / 1000.0;
}

HD inline double norm_cdf(double z) {
    return 0.5 * (1.0 + erf(z * 0.7071067811865476));
}

HD inline double t_eloss_chance(int c, int current, double current_time, const double* scores,
                                const double* durations, const double* open_t, const double* close_t,
                                const double* dist, int ld, const double* dist_to_depot,
                                double tmax, const bool* visited, int n) {
    const double SIG = 0.2;
    int poi_c = c + 1;
    double arr_c = current_time + dist[current * ld + poi_c];
    double start_c = arr_c > open_t[c] ? arr_c : open_t[c];
    double mean_finish_c = start_c + durations[c];
    double sd_c = SIG * durations[c];
    double eloss = 0.0;
    for (int j = 0; j < n; j++) {
        if (visited[j] || j == c) continue;
        int pj = j + 1;
        double arr_now = current_time + dist[current * ld + pj];
        if (arr_now > close_t[j]) continue;
        double mean_arr = mean_finish_c + dist[poi_c * ld + pj];
        double p_win;
        if (sd_c > 1e-9) p_win = norm_cdf((close_t[j] - mean_arr) / sd_c);
        else p_win = mean_arr <= close_t[j] ? 1.0 : 0.0;
        double mean_start, var_start;
        if (mean_arr > open_t[j]) { mean_start = mean_arr; var_start = sd_c * sd_c; }
        else { mean_start = open_t[j]; var_start = 0.0; }
        double sd_j = SIG * durations[j];
        double var_bud = var_start + sd_j * sd_j;
        double mean_bud = mean_start + durations[j] + dist_to_depot[j];
        double p_bud;
        if (var_bud > 1e-18) p_bud = norm_cdf((tmax - mean_bud) / pow(var_bud, 0.5));
        else p_bud = mean_bud <= tmax ? 1.0 : 0.0;
        eloss += scores[j] * (1.0 - p_win * p_bud);
    }
    return eloss;
}

HD inline double t_ns_a2(int i, double finish_time_at_p, const double* scores, const double* durations,
                         const double* open_t, const double* close_t, const double* dist, int ld,
                         const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t = dist[poi_i * ld + j + 1];
        if (t < 1e-10) continue;
        double arrival = finish_time_at_p + t;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        total += scores[j] / (t * t);
    }
    return total;
}

HD inline double t_ns_a05(int i, double finish_time_at_p, const double* scores, const double* durations,
                          const double* open_t, const double* close_t, const double* dist, int ld,
                          const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t = dist[poi_i * ld + j + 1];
        if (t < 1e-10) continue;
        double arrival = finish_time_at_p + t;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        total += scores[j] / sqrt(t);
    }
    return total;
}

HD inline double t_ns_rob_m(int i, double finish_i, double dur_i, const double* scores,
                            const double* durations, const double* open_t, const double* close_t,
                            const double* dist, int ld, const double* dist_to_depot,
                            const bool* visited, double tmax, int n, double m) {
    int poi_i = i + 1;
    double ms = m * SIM_SIGMA;
    double finish_rob = finish_i + dur_i * ms;
    double total = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_rob + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] * (1.0 + ms) + dist_to_depot[j] > tmax) continue;
        total += scores[j] / t_p_nb;
    }
    return total;
}

HD inline double t_nnfwd(int i, double finish_i, const double* scores, const double* durations,
                         const double* open_t, const double* close_t, const double* dist, int ld,
                         const double* dist_to_depot, const bool* visited, double tmax, int n) {
    (void)scores;
    int poi_i = i + 1;
    double dmin = -1.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_i + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        if (dmin < 0.0 || t_p_nb < dmin) dmin = t_p_nb;
    }
    if (dmin < 0.0) return 0.0;
    return 1.0 / (1.0 + dmin);
}

HD inline double t_nnk3(int i, double finish_i, const double* scores, const double* durations,
                        const double* open_t, const double* close_t, const double* dist, int ld,
                        const double* dist_to_depot, const bool* visited, double tmax, int n) {
    (void)scores;
    int poi_i = i + 1;
    double d1 = -1.0, d2 = -1.0, d3 = -1.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_i + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        if (d1 < 0.0 || t_p_nb < d1) { d3 = d2; d2 = d1; d1 = t_p_nb; }
        else if (d2 < 0.0 || t_p_nb < d2) { d3 = d2; d2 = t_p_nb; }
        else if (d3 < 0.0 || t_p_nb < d3) { d3 = t_p_nb; }
    }
    if (d1 < 0.0) return 0.0;
    double tot = d1, k = 1.0;
    if (d2 >= 0.0) { tot += d2; k += 1.0; }
    if (d3 >= 0.0) { tot += d3; k += 1.0; }
    return 1.0 / (1.0 + tot / k);
}

HD inline double t_detour(int i, int current, double finish_i, double ta_i, const double* scores,
                          const double* durations, const double* open_t, const double* close_t,
                          const double* dist, int ld, const double* dist_to_depot,
                          const bool* visited, double tmax, int n) {
    (void)scores;
    int poi_i = i + 1;
    double total = 0.0, cnt = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_i + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        total += ta_i + t_p_nb - dist[current * ld + j + 1];
        cnt += 1.0;
    }
    if (cnt == 0.0) return 0.0;
    return total / cnt;
}

HD inline double t_scorerank(int i, double finish_i, const double* scores, const double* durations,
                             const double* open_t, const double* close_t, const double* dist, int ld,
                             const double* dist_to_depot, const bool* visited, double tmax, int n) {
    int poi_i = i + 1;
    double below = 0.0, tot = 0.0;
    for (int j = 0; j < n; j++) {
        if (j == i || visited[j]) continue;
        double t_p_nb = dist[poi_i * ld + j + 1];
        if (t_p_nb < 1e-10) continue;
        double arrival = finish_i + t_p_nb;
        if (arrival > close_t[j]) continue;
        double start = arrival > open_t[j] ? arrival : open_t[j];
        if (start + durations[j] + dist_to_depot[j] > tmax) continue;
        tot += 1.0;
        if (scores[j] < scores[i]) below += 1.0;
    }
    if (tot == 0.0) return 1.0;
    return below / tot;
}

// REACH-K: K greedy waypoints ahead by score/dist under expected durations.
// scratch = caller buffer of size >= n.
HD inline double t_reach_k(int c, int current, double current_time, const double* scores,
                           const double* durations, const double* open_t, const double* close_t,
                           const double* dist, int ld, const double* dist_to_depot, double tmax,
                           const bool* visited, int n, int K, bool* scratch) {
    for (int j = 0; j < n; j++) scratch[j] = visited[j];
    scratch[c] = true;
    double ta_c = dist[current * ld + c + 1];
    double arr_c = current_time + ta_c;
    double start_c = arr_c > open_t[c] ? arr_c : open_t[c];
    int cur = c + 1;
    double t = start_c + durations[c];
    double total = 0.0;
    for (int step = 0; step < K; step++) {
        int best = -1;
        double best_v = -1.0;
        for (int q = 0; q < n; q++) {
            if (scratch[q]) continue;
            double ta = dist[cur * ld + q + 1];
            double arr = t + ta;
            if (arr > close_t[q]) continue;
            double start = arr > open_t[q] ? arr : open_t[q];
            if (start + durations[q] + dist_to_depot[q] > tmax) continue;
            double tt = ta > 1e-10 ? ta : 1e-10;
            double v = scores[q] / tt;
            if (v > best_v) { best_v = v; best = q; }
        }
        if (best < 0) break;
        double ta = dist[cur * ld + best + 1];
        double arr = t + ta;
        double start = arr > open_t[best] ? arr : open_t[best];
        t = start + durations[best];
        cur = best + 1;
        total += scores[best];
        scratch[best] = true;
    }
    return total;
}

HD inline double t_reach_k_rob(int c, int current, double current_time, const double* scores,
                               const double* durations, const double* open_t, const double* close_t,
                               const double* dist, int ld, const double* dist_to_depot, double tmax,
                               const bool* visited, int n, int K, bool* scratch) {
    for (int j = 0; j < n; j++) scratch[j] = visited[j];
    scratch[c] = true;
    double ta_c = dist[current * ld + c + 1];
    double arr_c = current_time + ta_c;
    double start_c = arr_c > open_t[c] ? arr_c : open_t[c];
    int cur = c + 1;
    double t = start_c + durations[c] * (1.0 + SIM_SIGMA);
    double total = 0.0;
    for (int step = 0; step < K; step++) {
        int best = -1;
        double best_v = -1.0;
        for (int q = 0; q < n; q++) {
            if (scratch[q]) continue;
            double ta = dist[cur * ld + q + 1];
            double arr = t + ta;
            if (arr > close_t[q]) continue;
            double start = arr > open_t[q] ? arr : open_t[q];
            if (start + durations[q] * (1.0 + SIM_SIGMA) + dist_to_depot[q] > tmax) continue;
            double tt = ta > 1e-10 ? ta : 1e-10;
            double v = scores[q] / tt;
            if (v > best_v) { best_v = v; best = q; }
        }
        if (best < 0) break;
        double ta = dist[cur * ld + best + 1];
        double arr = t + ta;
        double start = arr > open_t[best] ? arr : open_t[best];
        t = start + durations[best] * (1.0 + SIM_SIGMA);
        cur = best + 1;
        total += scores[best];
        scratch[best] = true;
    }
    return total;
}

HD inline double t_nnchain(int c, int current, double current_time, const double* scores,
                           const double* durations, const double* open_t, const double* close_t,
                           const double* dist, int ld, const double* dist_to_depot, double tmax,
                           const bool* visited, int n, int K, bool* scratch) {
    (void)scores;
    for (int j = 0; j < n; j++) scratch[j] = visited[j];
    scratch[c] = true;
    double ta_c = dist[current * ld + c + 1];
    double arr_c = current_time + ta_c;
    double start_c = arr_c > open_t[c] ? arr_c : open_t[c];
    int cur = c + 1;
    double t = start_c + durations[c];
    double total_d = 0.0;
    int done = 0;
    for (int step = 0; step < K; step++) {
        int best = -1;
        double best_d = -1.0;
        for (int q = 0; q < n; q++) {
            if (scratch[q]) continue;
            double ta = dist[cur * ld + q + 1];
            if (ta < 1e-10) continue;
            double arr = t + ta;
            if (arr > close_t[q]) continue;
            double start = arr > open_t[q] ? arr : open_t[q];
            if (start + durations[q] + dist_to_depot[q] > tmax) continue;
            if (best_d < 0.0 || ta < best_d) { best_d = ta; best = q; }
        }
        if (best < 0) break;
        double arr = t + best_d;
        double start = arr > open_t[best] ? arr : open_t[best];
        t = start + durations[best];
        cur = best + 1;
        total_d += best_d;
        done++;
        scratch[best] = true;
    }
    if (done == 0) return 0.0;
    return 1.0 / (1.0 + total_d / done);
}

// The 4 structural window-cascade features: FRAGCNT/FRAGSCORE/DOWNSLACK/PBUST.
HD inline bool sf_feasible(int q, int cur, double cur_t, const double* durations,
                           const double* open_t, const double* close_t, const double* dist, int ld,
                           const double* dist_to_depot, double tmax) {
    double ta = dist[cur * ld + q + 1];
    double arrival = cur_t + ta;
    if (arrival > close_t[q]) return false;
    double start = arrival > open_t[q] ? arrival : open_t[q];
    if (start + durations[q] + dist_to_depot[q] > tmax) return false;
    return true;
}

HD inline void struct_feats_one(int c, int current, double current_time, double sigma,
                                const double* scores, const double* durations, const double* open_t,
                                const double* close_t, const double* dist, int ld,
                                const double* dist_to_depot, double tmax, const bool* visited,
                                int n, double* out) {
    double ta_c = dist[current * ld + c + 1];
    double arr_c = current_time + ta_c;
    double start_c = arr_c > open_t[c] ? arr_c : open_t[c];
    double finish_c = start_c + durations[c];
    int cpos = c + 1;

    double frag_cnt = 0.0, frag_score = 0.0, downslack = 1e18;
    double best_v = -1.0;
    int best_q = -1;
    for (int q = 0; q < n; q++) {
        if (visited[q] || q == c) continue;
        bool feas_now = sf_feasible(q, current, current_time, durations, open_t, close_t, dist, ld, dist_to_depot, tmax);
        bool feas_after = sf_feasible(q, cpos, finish_c, durations, open_t, close_t, dist, ld, dist_to_depot, tmax);
        if (feas_now && !feas_after) { frag_cnt += 1.0; frag_score += scores[q]; }
        if (feas_after) {
            double arr_q = finish_c + dist[cpos * ld + q + 1];
            double slack = close_t[q] - arr_q;
            if (slack < downslack) downslack = slack;
            double t = dist[cpos * ld + q + 1];
            if (t > 1e-10) {
                double v = scores[q] / t;
                if (v > best_v) { best_v = v; best_q = q; }
            }
        }
    }
    if (downslack > 1e17) downslack = 0.0;

    double pbust = 0.0;
    if (best_q >= 0) {
        double arr_best = finish_c + dist[cpos * ld + best_q + 1];
        double std = sigma * durations[c];
        if (std > 1e-12) {
            double z = (close_t[best_q] - arr_best) / (std * SQRT2_C);
            pbust = 0.5 * erfc(z);
        } else {
            pbust = arr_best > close_t[best_q] ? 1.0 : 0.0;
        }
    }
    out[0] = frag_cnt;
    out[1] = frag_score;
    out[2] = downslack;
    out[3] = pbust;
}
