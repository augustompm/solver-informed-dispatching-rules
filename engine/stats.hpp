// Statistics for the paired stage: the Wilcoxon signed-rank test (two-sided;
// zero differences dropped; exact distribution up to n=50 when ranks have no
// ties, otherwise the normal approximation with tie-corrected variance).
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

inline double norm_sf(double z) { return 0.5 * std::erfc(z / 1.4142135623730951); }

// average ranks over ties
inline std::vector<double> rankdata_avg(const std::vector<double>& v) {
    size_t n = v.size();
    std::vector<int> idx(n);
    for (size_t i = 0; i < n; i++) idx[i] = (int)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] < v[b]; });
    std::vector<double> r(n);
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) j++;
        double avg = (double)(i + 1 + j + 1) / 2.0;
        for (size_t k = i; k <= j; k++) r[idx[k]] = avg;
        i = j + 1;
    }
    return r;
}

// two-sided p-value; NaN when every diff is zero.
inline double wilcoxon_p(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> d;
    size_t n_zero = 0;
    for (size_t i = 0; i < x.size(); i++) {
        double v = x[i] - y[i];
        if (v != 0.0) d.push_back(v);   // zero differences are dropped
        else n_zero++;
    }
    size_t n = d.size();
    if (n == 0) return std::nan("");
    std::vector<double> ad(n);
    for (size_t i = 0; i < n; i++) ad[i] = std::fabs(d[i]);
    std::vector<double> r = rankdata_avg(ad);
    double r_plus = 0.0, r_minus = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (d[i] > 0) r_plus += r[i];
        else r_minus += r[i];
    }
    bool has_ties = false;
    {
        std::vector<double> s = ad;
        std::sort(s.begin(), s.end());
        for (size_t i = 1; i < n; i++)
            if (s[i] == s[i - 1]) { has_ties = true; break; }
    }
    if (n <= 50 && !has_ties && n_zero == 0) {
        // exact distribution of W+ over subsets of ranks 1..n (no ties, no zeros)
        long long W = (long long)std::llround(std::min(r_plus, r_minus));
        long long maxw = (long long)n * (n + 1) / 2;
        std::vector<double> cnt(maxw + 1, 0.0);
        cnt[0] = 1.0;
        for (long long k = 1; k <= (long long)n; k++)
            for (long long w = maxw; w >= k; w--) cnt[w] += cnt[w - k];
        double total = std::pow(2.0, (double)n);
        double cdf = 0.0;
        for (long long w = 0; w <= W; w++) cdf += cnt[w];
        double p = 2.0 * cdf / total;
        return p > 1.0 ? 1.0 : p;
    }
    // normal approximation, tie-corrected variance, no continuity correction
    double T = std::min(r_plus, r_minus);
    double cnt_n = (double)n;
    double mn = cnt_n * (cnt_n + 1.0) * 0.25;
    double se = cnt_n * (cnt_n + 1.0) * (2.0 * cnt_n + 1.0);
    {
        std::vector<double> s = ad;
        std::sort(s.begin(), s.end());
        size_t i = 0;
        while (i < n) {
            size_t j = i;
            while (j + 1 < n && s[j + 1] == s[i]) j++;
            double t = (double)(j - i + 1);
            if (t > 1.0) se -= 0.5 * t * (t * t - 1.0);
            i = j + 1;
        }
    }
    se = std::sqrt(se / 24.0);
    double z = (T - mn) / se;
    double p = 2.0 * norm_sf(std::fabs(z));
    return p > 1.0 ? 1.0 : p;
}
