// Move-by-move reading of the reference route against the 30 NS-GP rules
// (sigma=0): coverage, which customers the rules lose and their profile, route
// mechanics, and where on the reference timeline the lost customers live. The
// human-guided reading loop follows Ferreira et al. 2021.
//     lookit rc205
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/json.hpp"
#include "decode.hpp"

namespace fs = std::filesystem;

static JPtr load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "not found: %s\n", path.c_str()); std::exit(2); }
    std::stringstream buf;
    buf << f.rdbuf();
    return json_parse(buf.str());
}

static double mean_of(const std::vector<double>& v) {
    double a = 0.0;
    for (double x : v) a += x;
    return a / v.size();
}

static double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

int main(int argc, char** argv) {
    std::string INST = argc > 1 ? argv[1] : "rc205";
    Instance inst = load_instance(INST);
    std::map<int, const POI*> cust;
    for (int k = 1; k < (int)inst.pois.size(); k++) cust[inst.pois[k].id] = &inst.pois[k];

    auto prof = [&](const std::set<int>& ids) -> std::string {
        if (ids.empty()) return "empty";
        std::vector<double> S, W, O, D, U;
        for (int i : ids) {
            const POI* p = cust.at(i);
            S.push_back(p->score);
            W.push_back(p->close_time - p->open_time);
            O.push_back(p->open_time);
            D.push_back(euclidean(*p, inst.depot()));
            U.push_back(p->duration);
        }
        char buf[160];
        std::snprintf(buf, sizeof buf,
                      "score %.1f | window %.0f | open %.0f | dist-depot %.1f | dur %.0f",
                      mean_of(S), mean_of(W), mean_of(O), mean_of(D), mean_of(U));
        return buf;
    };

    JPtr ref = load_json(ref_route_path(INST));

    std::vector<std::vector<int>> ref_routes;
    for (size_t v = 0; v < ref->at("routes").size(); v++) {
        std::vector<int> lane;
        for (size_t k = 0; k < ref->at("routes")[v].size(); k++)
            lane.push_back((int)ref->at("routes")[v][k].as_int() + 1);   // cuOpt is 0-indexed; Solomon 1..100
        ref_routes.push_back(lane);
    }
    std::set<int> ref_served;
    for (auto& r : ref_routes)
        for (int i : r) ref_served.insert(i);

    std::printf("# LOOKIT (reference solver vs NS-GP) %s | tmax=%.0f m=3 | reference: %lld/%lld served\n",
                INST.c_str(), inst.tmax, ref->at("n_served").as_int(), ref->at("n").as_int());
    std::printf("\n## Reference route, sigma=0 replay (mechanics)\n");
    for (size_t k = 0; k < ref_routes.size(); k++) {
        Replay rp = replay_route(inst, ref_routes[k]);
        std::string tag;
        if (!rp.late.empty()) {
            tag = " LATE=[";
            for (size_t j = 0; j < rp.late.size(); j++) tag += (j ? ", " : "") + std::to_string(rp.late[j]);
            tag += "]";
        }
        std::printf("  veh%zu: n=%2d score=%4.0f travel=%5.0f wait=%5.0f makespan=%5.0f%s\n",
                    k, rp.n, rp.score, rp.travel, rp.wait, rp.makespan, tag.c_str());
    }

    std::printf("\n## The 30 NS-GP rules (sigma=0 decoder)\n");
    struct Row { int seed; std::set<int> served; double score; int n; double travel, wait; };
    std::vector<Row> rows;
    std::vector<std::string> seed_files;
    for (auto& e : fs::directory_iterator("results/baseline/nsgp/" + INST))
        if (e.path().filename().string().rfind("seed", 0) == 0) seed_files.push_back(e.path().string());
    std::sort(seed_files.begin(), seed_files.end());
    for (auto& f : seed_files) {
        JPtr d = load_json(f);
        std::string stem = fs::path(f).stem().string();
        int sd = std::atoi(stem.substr(4).c_str());
        auto routes = route_of(inst, parse_prefix(d->at("best_tree").str));
        std::set<int> served;
        for (auto& r : routes)
            for (int i : r) served.insert(i);
        double sc = 0.0;
        for (int i : served) sc += cust.at(i)->score;
        double travel = 0.0, wait = 0.0;
        for (auto& r : routes) {
            Replay rp = replay_route(inst, r);
            travel += rp.travel;
            wait += rp.wait;
        }
        rows.push_back({sd, served, sc, (int)served.size(), travel, wait});
    }
    std::vector<double> ns_v, scs;
    for (auto& r : rows) { ns_v.push_back(r.n); scs.push_back(r.score); }
    double ref_score = 0.0;
    for (int i : ref_served) ref_score += cust.at(i)->score;
    std::printf("  served: min %.0f | median %.0f | max %.0f   (reference: %zu)\n",
                *std::min_element(ns_v.begin(), ns_v.end()), median_of(ns_v),
                *std::max_element(ns_v.begin(), ns_v.end()), ref_served.size());
    std::printf("  score sigma=0: min %.0f | median %.0f | max %.0f   (reference: %.0f)\n",
                *std::min_element(scs.begin(), scs.end()), median_of(scs),
                *std::max_element(scs.begin(), scs.end()), ref_score);
    std::vector<int> order(rows.size());
    for (size_t i = 0; i < rows.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return rows[a].score < rows[b].score; });
    const Row& med = rows[order[rows.size() / 2]];
    std::map<int, int> byc;
    for (auto& r : rows)
        for (int i : r.served) byc[i]++;
    std::set<int> always, never_served;
    for (auto& kv : byc)
        if (kv.second == 30) always.insert(kv.first);
    for (auto& kv : cust)
        if (!byc.count(kv.first)) never_served.insert(kv.first);
    std::printf("  NS-GP median seed%d: %d served, score %.0f, travel %.0f, wait %.0f\n",
                med.seed, med.n, med.score, med.travel, med.wait);
    double kt_travel = 0.0, kt_wait = 0.0;
    for (auto& r : ref_routes) {
        Replay rp = replay_route(inst, r);
        kt_travel += rp.travel;
        kt_wait += rp.wait;
    }
    std::printf("  reference total: travel %.0f, wait %.0f\n", kt_travel, kt_wait);

    std::printf("\n## The diff: reference-served customers NS-GP does not reach\n");
    std::set<int> lost_always, lost_med;
    for (int i : ref_served) {
        if (!byc.count(i)) lost_always.insert(i);
        if (!med.served.count(i)) lost_med.insert(i);
    }
    std::string la = "[";
    {
        size_t j = 0;
        for (int i : lost_always) { la += (j++ ? ", " : "") + std::to_string(i); }
        la += "]";
    }
    std::printf("  lost by ALL 30 seeds: %zu -> %s\n", lost_always.size(), la.c_str());
    std::printf("    profile: %s\n", prof(lost_always).c_str());
    std::printf("  lost by the median: %zu\n", lost_med.size());
    std::printf("    profile: %s\n", prof(lost_med).c_str());
    std::printf("  always-served (30/30): %zu | never served by NS-GP: %zu\n", always.size(), never_served.size());
    std::printf("  profile of always-served: %s\n", prof(always).c_str());
    std::set<int> all_ids;
    for (auto& kv : cust) all_ids.insert(kv.first);
    std::printf("  instance profile overall: %s\n", prof(all_ids).c_str());

    std::printf("\n## Where on the reference route the lost-by-all live (position = route fraction)\n");
    for (size_t k = 0; k < ref_routes.size(); k++) {
        auto& r = ref_routes[k];
        std::string line;
        for (size_t j = 0; j < r.size(); j++)
            if (lost_always.count(r[j])) {
                char b[48];
                std::snprintf(b, sizeof b, "%d@%.2f", r[j], (double)j / (r.size() > 1 ? r.size() - 1 : 1));
                line += (line.empty() ? "" : " ") + std::string(b);
            }
        if (!line.empty()) std::printf("  veh%zu: %s\n", k, line.c_str());
    }
    return 0;
}
