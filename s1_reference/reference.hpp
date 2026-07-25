// Stage 1, our side of the reference solver: the instance reader, the
// stochastic service sampler and the fixed-route replay (window misses and
// no-time-home skips). The SOLVE itself is an external third-party solver
// (PyVRP HGS on CPU, cuOpt LNS on GPU); its route arrives as JSON.
#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../engine/rng.hpp"

struct RefInstance {
    int n = 0;                       // rows including the depot
    std::vector<double> prize, open_t, close_t, service;
    std::vector<double> dist;        // n x n, np.hypot
    double tmax = 0.0;

    explicit RefInstance(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot open " + path);
        std::vector<std::vector<double>> rows;
        std::string ln;
        bool after = false;
        while (std::getline(f, ln)) {
            std::istringstream ss(ln);
            std::vector<std::string> parts;
            std::string w;
            while (ss >> w) parts.push_back(w);
            if (!after) {
                after = !parts.empty() && parts[0].size() >= 4 &&
                        (parts[0].rfind("CUST", 0) == 0 || parts[0].rfind("cust", 0) == 0 ||
                         parts[0].rfind("Cust", 0) == 0);
                continue;
            }
            if (parts.size() >= 7) {
                std::string t = parts[0];
                size_t q = t.find_first_not_of('-');
                t = (q == std::string::npos) ? "" : t.substr(q);
                bool digit = !t.empty();
                for (char c : t) if (c < '0' || c > '9') { digit = false; break; }
                if (!digit) continue;
                std::vector<double> r;
                for (int k = 0; k < 7; k++) r.push_back(std::strtod(parts[k].c_str(), nullptr));
                rows.push_back(r);
            }
        }
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a[0] < b[0]; });
        n = (int)rows.size();
        prize.resize(n); open_t.resize(n); close_t.resize(n); service.resize(n);
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; i++) {
            x[i] = rows[i][1]; y[i] = rows[i][2];
            prize[i] = rows[i][3];
            open_t[i] = rows[i][4]; close_t[i] = rows[i][5];
            service[i] = rows[i][6];
        }
        prize[0] = 0.0;
        tmax = close_t[0];
        dist.resize((size_t)n * n);
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++)
                dist[(size_t)a * n + b] = std::hypot(x[a] - x[b], y[a] - y[b]);
    }

    // One gaussian per customer in id order; zero service draws nothing.
    std::vector<double> sample_service(long long seed) const {
        Mt19937 rng((uint64_t)seed);
        std::vector<double> out(n, 0.0);
        for (int c = 1; c < n; c++) {
            double m = service[c];
            if (m > 0.0) {
                double v = rng.gauss(m, 0.2 * m);
                out[c] = v > 0.0 ? v : 0.0;
            }
        }
        return out;
    }
};

// Replay a fixed route: window missed or no time home -> skip the customer.
inline double deploy_route(const RefInstance& inst, const std::vector<std::vector<int>>& route,
                           const std::vector<double>& service_times) {
    double total = 0.0;
    for (auto& lane : route) {
        int cur = 0;
        double now = 0.0;
        for (int c : lane) {
            double arrival = now + inst.dist[(size_t)cur * inst.n + c];
            if (arrival > inst.close_t[c] || now + inst.dist[(size_t)cur * inst.n + 0] > inst.tmax)
                continue;
            double start = arrival > inst.open_t[c] ? arrival : inst.open_t[c];
            now = start + service_times[c];
            total += inst.prize[c];
            cur = c;
        }
    }
    return total;
}

// Ceiling of a solved route: mean prize over sampled service realizations,
// seed base 1M.
inline double route_prize_mean(const RefInstance& inst, const std::vector<std::vector<int>>& route,
                               int scenarios = 30, long long seed = 1000000) {
    double acc = 0.0;
    for (int s = 0; s < scenarios; s++)
        acc += deploy_route(inst, route, inst.sample_service(seed + s));
    return acc / scenarios;
}
