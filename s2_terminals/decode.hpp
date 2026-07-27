// Shared pieces of the stage-2 route readers: the sigma=0 greedy decoder
// (Jackson feasibility + the 11 Mei/Jackson terminals + tree evaluation over a
// name map), the fixed-route replay, and the reference-route locator.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "../engine/instance.hpp"
#include "../engine/tree.hpp"

// Reference route the reading logs were produced with, largest solve time
// limit available for the instance (file name <inst>_cuopt_route_<seconds>.json).
// These live under s2: the reading predates the final reference-gap routes.
inline std::string ref_route_path(const std::string& inst) {
    std::string dir = "results/s2_terminals/routes_read";
    std::string prefix = inst + "_cuopt_route_";
    int best_tl = -1;
    std::string best;
    for (auto& e : std::filesystem::directory_iterator(dir)) {
        std::string fn = e.path().filename().string();
        if (fn.rfind(prefix, 0) == 0 && fn.size() > prefix.size() + 5) {
            int tl = std::atoi(fn.substr(prefix.size(), fn.size() - prefix.size() - 5).c_str());
            if (tl > best_tl) { best_tl = tl; best = e.path().string(); }
        }
    }
    if (best.empty()) {
        std::fprintf(stderr, "no reference route for %s in %s\n", inst.c_str(), dir.c_str());
        std::exit(2);
    }
    return best;
}

inline double evaluate_map(const NodeP& node, const std::map<std::string, double>& terms) {
    if (node->is_terminal()) return terms.at(node->label);
    double a = evaluate_map(node->children[0], terms);
    double b = evaluate_map(node->children[1], terms);
    const std::string& op = node->label;
    if (op == "+") return a + b;
    if (op == "-") return a - b;
    if (op == "*") return a * b;
    if (op == "/") return protected_div(a, b);
    if (op == "min") return a < b ? a : b;
    if (op == "max") return a > b ? a : b;
    return rmod_fn(a, b);
}

// Neighbourhood score (Jackson & Mei 2020 Alg 1).
inline double ns_poi(const POI& cand, double finish_at_p, const Instance& inst,
                     const std::set<int>& visited) {
    double total = 0.0;
    for (int k = 1; k < (int)inst.pois.size(); k++) {
        const POI& nb = inst.pois[k];
        if (nb.id == cand.id || visited.count(nb.id)) continue;
        double t = euclidean(cand, nb);
        if (t < 1e-10) continue;
        double arrival = finish_at_p + t;
        if (arrival > nb.close_time) continue;
        double start = arrival > nb.open_time ? arrival : nb.open_time;
        if (start + nb.duration + euclidean(nb, inst.depot()) > inst.tmax) continue;
        total += nb.score / t;
    }
    return total;
}

// The 11 Mei/Jackson terminal values for one candidate.
inline std::map<std::string, double> compute_terminals_ns(const POI& cand, const POI& current,
                                                          double current_time, const Instance& inst,
                                                          const std::set<int>& visited,
                                                          int remaining_vehicles) {
    double ta = euclidean(current, cand);
    double tc = cand.close_time - current_time;
    double to = cand.open_time - current_time;
    double tr = euclidean(cand, inst.depot());
    double tsv = to > ta ? to : ta;
    double tfv = tsv + cand.duration;
    std::map<std::string, double> terms = {
        {"SCORE", cand.score}, {"DUR", cand.duration}, {"TO", to}, {"TC", tc},
        {"TA", ta}, {"TR", tr}, {"TSV", tsv}, {"TFV", tfv}, {"SL", tc - ta},
        {"RemT", remaining_vehicles * inst.tmax + (inst.tmax - current_time)},
    };
    terms["NS"] = ns_poi(cand, current_time + tfv, inst, visited);
    return terms;
}

// The sigma=0 greedy decoder, m vehicles in sequence.
inline std::vector<std::vector<int>> route_of(const Instance& inst, const NodeP& tree, int m = 3) {
    std::set<int> visited;
    std::vector<std::vector<int>> routes;
    for (int v = 0; v < m; v++) {
        int remaining = m - v - 1;
        const POI* current = &inst.depot();
        double t = 0.0;
        std::vector<int> route;
        while (true) {
            const POI* chosen = nullptr;
            double best = -1e308;
            bool any = false;
            for (int k = 1; k < (int)inst.pois.size(); k++) {
                const POI& p = inst.pois[k];
                if (visited.count(p.id) || !is_feasible(p, *current, t, inst)) continue;
                any = true;
                auto terms = compute_terminals_ns(p, *current, t, inst, visited, remaining);
                double pri = evaluate_map(tree, terms);
                if (pri != pri) pri = -1e308;
                if (pri > best) { best = pri; chosen = &p; }
            }
            if (!any || !chosen) break;
            double travel = euclidean(*current, *chosen);
            double arr = t + travel;
            double finish = (arr > chosen->open_time ? arr : chosen->open_time) + chosen->duration;
            current = chosen;
            t = finish;
            visited.insert(chosen->id);
            route.push_back(chosen->id);
        }
        routes.push_back(route);
    }
    return routes;
}

struct Replay {
    int n;
    double score, travel, wait, makespan;
    std::vector<int> late;
};

// Sigma=0 replay of a fixed route.
inline Replay replay_route(const Instance& inst, const std::vector<int>& route) {
    std::map<int, const POI*> cust;
    for (int k = 1; k < (int)inst.pois.size(); k++) cust[inst.pois[k].id] = &inst.pois[k];
    const POI* current = &inst.depot();
    double t = 0.0, travel_tot = 0.0, wait_tot = 0.0, score = 0.0;
    std::vector<int> late;
    for (int cid : route) {
        const POI* p = cust.at(cid);
        double tv = euclidean(*current, *p);
        double arr = t + tv;
        if (arr > p->close_time + 1e-9) late.push_back(cid);
        double w = p->open_time - arr;
        wait_tot += w > 0.0 ? w : 0.0;
        t = (arr > p->open_time ? arr : p->open_time) + p->duration;
        travel_tot += tv;
        score += p->score;
        current = p;
    }
    double back = euclidean(*current, inst.depot());
    return {(int)route.size(), score, travel_tot + back, wait_tot, t + back, late};
}
