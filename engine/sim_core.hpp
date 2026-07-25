// The simulation core, one source for CPU and GPU: postfix tree evaluation
// (opcodes -1 terminal, -2 constant, 0..5 arithmetic, 6 rmod), the decision
// step over the 61 terminal slots behind term_mask, the online rollout (Mei
// 2018 Alg 2: each vehicle closes its route greedily, then the next; the
// sampled duration only advances time), a round-robin variant, and the
// instance-level z-statistics.
#pragma once
#include "sim_terms.hpp"

// dist is the (n+1)x(n+1) row-major matrix with the depot at index 0; ld = n+1.
HD inline double eval_tree(const int32_t* ops, const int32_t* args, int len,
                           const double* terms, double* stack) {
    int sp = 0;
    for (int k = 0; k < len; k++) {
        int o = ops[k];
        if (o == -1) {
            stack[sp++] = terms[args[2 * k]];
        } else if (o == -2) {
            stack[sp++] = args[2 * k] / 1000.0;
        } else {
            double b = stack[sp - 1];
            double a = stack[sp - 2];
            sp -= 2;
            if (o == 0) stack[sp] = a + b;
            else if (o == 1) stack[sp] = a - b;
            else if (o == 2) stack[sp] = a * b;
            else if (o == 3) stack[sp] = fabs(b) < 1e-10 ? 1.0 : a / b;
            else if (o == 4) stack[sp] = a < b ? a : b;
            else if (o == 5) stack[sp] = a > b ? a : b;
            else {   // o == 6: RMOD(base=a, regime=b) = max(a,0) ^ (1 + clip(b,0,1))
                double base = a > 0.0 ? a : 0.0;
                double e = b;
                if (e < 0.0) e = 0.0;
                else if (e > 1.0) e = 1.0;
                stack[sp] = pow(base, 1.0 + e);
            }
            sp++;
        }
    }
    return stack[0];
}

HD inline int pick_best(int current, double current_time, int remaining,
                        const int32_t* ops, const int32_t* args, int len,
                        const double* scores, const double* durations,
                        const double* open_t, const double* close_t,
                        const double* dist, int ld, const double* dist_to_depot,
                        double tmax, const bool* visited, int n,
                        const int64_t* term_mask, const double* zp,
                        double* terms, double* stack, double* struct_buf, bool* scratch) {
    bool need_ns = term_mask[10] || term_mask[28] || term_mask[29];
    bool need_struct = term_mask[17] || term_mask[18] || term_mask[19] || term_mask[20]
                       || term_mask[28] || term_mask[29];

    double sat = tmax > 0.0 ? current_time / tmax : 0.0;
    double vfrac = 0.0, nfeas = 0.0, fscore = 0.0, nfeasrob = 0.0;
    if (term_mask[25] || term_mask[26] || term_mask[51] || term_mask[52]) {
        double nvis = 0.0;
        for (int j = 0; j < n; j++) {
            if (visited[j]) { nvis += 1.0; continue; }
            double ta_j = dist[current * ld + j + 1];
            double arr_j = current_time + ta_j;
            if (arr_j > close_t[j]) continue;
            double st_j = arr_j > open_t[j] ? arr_j : open_t[j];
            if (st_j + durations[j] + dist_to_depot[j] > tmax) continue;
            nfeas += 1.0;
            fscore += scores[j];
            if (st_j + durations[j] * (1.0 + SIM_SIGMA) + dist_to_depot[j] <= tmax) nfeasrob += 1.0;
        }
        vfrac = nvis / n;
    }

    int best = -1;
    double best_pri = -1e300;
    for (int i = 0; i < n; i++) {
        if (visited[i]) continue;
        int poi_id = i + 1;
        double ta = dist[current * ld + poi_id];
        double arrival = current_time + ta;
        if (arrival > close_t[i]) continue;
        double start = arrival > open_t[i] ? arrival : open_t[i];
        if (start + durations[i] + dist_to_depot[i] > tmax) continue;

        double to = open_t[i] - current_time;
        double tc = close_t[i] - current_time;
        double tsv = to > ta ? to : ta;
        double tfv = tsv + durations[i];
        terms[0] = scores[i];
        terms[1] = durations[i];
        terms[2] = to;
        terms[3] = tc;
        terms[4] = ta;
        terms[5] = dist_to_depot[i];
        terms[6] = tsv;
        terms[7] = tfv;
        terms[8] = tc - ta;
        terms[9] = remaining * tmax + (tmax - current_time);

        double fin = current_time + tfv;
        double ns_val = 0.0;
        if (need_ns)
            ns_val = t_ns(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n);
        terms[10] = term_mask[10] ? ns_val : 0.0;
        terms[11] = (term_mask[11] || term_mask[33])
            ? t_regret(i, current, current_time, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[12] = term_mask[12]
            ? t_ns_rob(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[13] = term_mask[13]
            ? t_maxn(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[14] = (term_mask[14] || term_mask[32])
            ? t_reach2(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[15] = term_mask[15]
            ? t_nstw(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[16] = term_mask[16] ? t_rand(i) : 0.0;

        double frag_score = 0.0;
        if (need_struct) {
            struct_feats_one(i, current, current_time, SIM_SIGMA, scores, durations, open_t,
                             close_t, dist, ld, dist_to_depot, tmax, visited, n, struct_buf);
            frag_score = struct_buf[1];
        }
        terms[17] = term_mask[17] ? struct_buf[0] : 0.0;
        terms[18] = term_mask[18] ? struct_buf[1] : 0.0;
        terms[19] = term_mask[19] ? struct_buf[2] : 0.0;
        terms[20] = term_mask[20] ? struct_buf[3] : 0.0;

        terms[21] = term_mask[21]
            ? t_reach_k(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 1, scratch) : 0.0;
        terms[22] = term_mask[22]
            ? t_reach_k(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 3, scratch) : 0.0;
        terms[23] = term_mask[23]
            ? t_reach_k(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 5, scratch) : 0.0;

        terms[24] = term_mask[24] ? sat : 0.0;
        terms[25] = term_mask[25] ? vfrac : 0.0;
        terms[26] = term_mask[26] ? nfeas : 0.0;
        terms[27] = term_mask[27] ? (double)remaining : 0.0;
        terms[28] = term_mask[28] ? frag_score / (ns_val + 1.0) : 0.0;
        terms[29] = term_mask[29] ? frag_score - ns_val : 0.0;
        terms[30] = (term_mask[30] || term_mask[34])
            ? t_eloss_chance(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n) : 0.0;
        terms[31] = term_mask[31] ? (terms[3] - zp[0]) / zp[5] : 0.0;
        terms[32] = term_mask[32] ? (terms[14] - zp[1]) / zp[6] : 0.0;
        terms[33] = term_mask[33] ? (terms[11] - zp[2]) / zp[7] : 0.0;
        terms[34] = term_mask[34] ? (terms[30] - zp[3]) / zp[8] : 0.0;
        terms[35] = term_mask[35] ? (terms[5] - zp[4]) / zp[9] : 0.0;
        terms[36] = term_mask[36] ? 0.1 : 0.0;
        terms[37] = term_mask[37] ? 0.2 : 0.0;
        terms[38] = term_mask[38] ? 0.3 : 0.0;
        terms[39] = term_mask[39] ? 0.4 : 0.0;
        terms[40] = term_mask[40] ? 0.5 : 0.0;
        terms[41] = term_mask[41]
            ? t_reach2_rob(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        {
            double denom = tmax - current_time;
            if (denom < 1.0) denom = 1.0;
            terms[42] = term_mask[42] ? (ta + durations[i]) / denom : 0.0;
        }
        terms[43] = term_mask[43]
            ? (t_reach2_rob(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n)
               - t_regret(i, current, current_time, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n)) : 0.0;
        terms[44] = term_mask[44]
            ? t_skiploss(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[45] = term_mask[45]
            ? (t_ns(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n)
               - t_ns_rob(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n)) : 0.0;
        {
            double denom = ta + dist_to_depot[i];
            if (denom < 1.0) denom = 1.0;
            terms[46] = term_mask[46]
                ? (scores[i] + t_ns_rob(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n)) / denom : 0.0;
        }
        terms[47] = term_mask[47]
            ? t_reach_k_rob(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 3, scratch) : 0.0;
        terms[48] = term_mask[48]
            ? t_reach_k_rob(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 5, scratch) : 0.0;
        terms[49] = term_mask[49]
            ? t_ns_a2(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[50] = term_mask[50]
            ? t_ns_a05(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[51] = term_mask[51] ? fscore : 0.0;
        terms[52] = term_mask[52] ? nfeasrob : 0.0;
        terms[53] = term_mask[53]
            ? t_ns_rob_m(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n, 0.5) : 0.0;
        terms[54] = term_mask[54]
            ? t_ns_rob_m(i, fin, durations[i], scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n, 2.0) : 0.0;
        terms[55] = term_mask[55]
            ? t_nnfwd(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[56] = term_mask[56]
            ? t_nnk3(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[57] = term_mask[57]
            ? t_nnchain(i, current, current_time, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited, n, 3, scratch) : 0.0;
        terms[58] = term_mask[58] ? (dist[current * ld + 0] + dist_to_depot[i] - ta) : 0.0;
        terms[59] = term_mask[59]
            ? t_detour(i, current, fin, ta, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;
        terms[60] = term_mask[60]
            ? t_scorerank(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited, tmax, n) : 0.0;

        double pri = eval_tree(ops, args, len, terms, stack);
        if (pri != pri) pri = -1e300;   // NaN
        if (pri > best_pri) { best_pri = pri; best = i; }
    }
    return best;
}

// V0 (Mei protocol): vehicle v closes its whole route, then v+1.
HD inline double sim_one(const int32_t* ops, const int32_t* args, int len,
                         const double* scores, const double* durations,
                         const double* open_t, const double* close_t,
                         const double* dist, int ld, const double* dist_to_depot,
                         double tmax, const double* sampled, int m, int n,
                         const int64_t* term_mask, const double* zp,
                         bool* visited, double* terms, double* stack,
                         double* struct_buf, bool* scratch) {
    for (int i = 0; i < n; i++) visited[i] = false;
    double total = 0.0;
    for (int v = 0; v < m; v++) {
        int remaining = m - v - 1;
        int current = 0;
        double current_time = 0.0;
        while (true) {
            int best = pick_best(current, current_time, remaining, ops, args, len, scores,
                                 durations, open_t, close_t, dist, ld, dist_to_depot,
                                 tmax, visited, n, term_mask, zp, terms, stack, struct_buf, scratch);
            if (best < 0) break;
            int poi_id = best + 1;
            double ta = dist[current * ld + poi_id];
            double arrival = current_time + ta;
            double start = arrival > open_t[best] ? arrival : open_t[best];
            double finish = start + sampled[best];
            current = poi_id;
            current_time = finish;
            total += scores[best];
            visited[best] = true;
        }
    }
    return total;
}

// V2 (round-robin ablation arm): the m vehicles advance one POI per round.
HD inline double sim_one_rr(const int32_t* ops, const int32_t* args, int len,
                            const double* scores, const double* durations,
                            const double* open_t, const double* close_t,
                            const double* dist, int ld, const double* dist_to_depot,
                            double tmax, const double* sampled, int m, int n,
                            const int64_t* term_mask, const double* zp,
                            bool* visited, double* terms, double* stack,
                            double* struct_buf, bool* scratch,
                            int* current_v, double* time_v, bool* active) {
    for (int i = 0; i < n; i++) visited[i] = false;
    for (int v = 0; v < m; v++) { current_v[v] = 0; time_v[v] = 0.0; active[v] = true; }
    double total = 0.0;
    int n_active = m;
    while (n_active > 0) {
        for (int v = 0; v < m; v++) {
            if (!active[v]) continue;
            int remaining = m - v - 1;
            int best = pick_best(current_v[v], time_v[v], remaining, ops, args, len, scores,
                                 durations, open_t, close_t, dist, ld, dist_to_depot,
                                 tmax, visited, n, term_mask, zp, terms, stack, struct_buf, scratch);
            if (best < 0) { active[v] = false; n_active--; continue; }
            int poi_id = best + 1;
            double ta = dist[current_v[v] * ld + poi_id];
            double arrival = time_v[v] + ta;
            double start = arrival > open_t[best] ? arrival : open_t[best];
            double finish = start + sampled[best];
            current_v[v] = poi_id;
            time_v[v] = finish;
            total += scores[best];
            visited[best] = true;
        }
    }
    return total;
}

// sim_cpu._zstats: instance-level mean/std of the 5 z-features at the initial state.
HD inline void zstats(const double* scores, const double* durations, const double* open_t,
                      const double* close_t, const double* dist, int ld,
                      const double* dist_to_depot, double tmax, int n,
                      double* zmean, double* zstd, bool* visited_buf) {
    double zsum[5] = {0, 0, 0, 0, 0}, zsq[5] = {0, 0, 0, 0, 0};
    double cnt = 0.0;
    for (int i = 0; i < n; i++) visited_buf[i] = false;
    for (int i = 0; i < n; i++) {
        double ta = dist[0 * ld + i + 1];
        double arr = ta;
        if (arr > close_t[i]) continue;
        double start = arr > open_t[i] ? arr : open_t[i];
        if (start + durations[i] + dist_to_depot[i] > tmax) continue;
        double to = open_t[i];
        double tsv = to > ta ? to : ta;
        double fin = tsv + durations[i];
        double f0 = close_t[i];
        double f1 = t_reach2(i, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited_buf, tmax, n);
        double f2 = t_regret(i, 0, 0.0, fin, scores, durations, open_t, close_t, dist, ld, dist_to_depot, visited_buf, tmax, n);
        double f3 = t_eloss_chance(i, 0, 0.0, scores, durations, open_t, close_t, dist, ld, dist_to_depot, tmax, visited_buf, n);
        double f4 = dist_to_depot[i];
        zsum[0] += f0; zsq[0] += f0 * f0;
        zsum[1] += f1; zsq[1] += f1 * f1;
        zsum[2] += f2; zsq[2] += f2 * f2;
        zsum[3] += f3; zsq[3] += f3 * f3;
        zsum[4] += f4; zsq[4] += f4 * f4;
        cnt += 1.0;
    }
    if (cnt < 1.0) cnt = 1.0;
    for (int k = 0; k < 5; k++) {
        zmean[k] = zsum[k] / cnt;
        double var = zsq[k] / cnt - zmean[k] * zmean[k];
        if (var < 0.0) var = 0.0;
        zstd[k] = sqrt(var) + 1e-9;
    }
}
