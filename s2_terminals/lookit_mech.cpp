// Measure literature mechanisms on reference routes vs the NS-GP rules before
// implementing a terminal: M1 far-first (corr position x dist-to-depot), M2
// productive waiting, M3 angular sweep (Gillett & Miller), M4 mean
// Clarke-Wright step savings.
//     lookit_mech rc205
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
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

static double corr(const std::vector<double>& xs, const std::vector<double>& ys) {
    size_t n = xs.size();
    if (n < 3) return 0.0;
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (size_t i = 0; i < n; i++) {
        cov += (xs[i] - mx) * (ys[i] - my);
        vx += (xs[i] - mx) * (xs[i] - mx);
        vy += (ys[i] - my) * (ys[i] - my);
    }
    return (vx > 0 && vy > 0) ? cov / std::sqrt(vx * vy) : 0.0;
}

struct Mech { double m1; int n_wait, n; double m3, m4, wait_tot; };

int main(int argc, char** argv) {
    std::string INST = argc > 1 ? argv[1] : "rc205";
    Instance inst = load_instance(INST);
    std::map<int, const POI*> cust;
    for (int k = 1; k < (int)inst.pois.size(); k++) cust[inst.pois[k].id] = &inst.pois[k];
    const POI& DEPOT = inst.depot();

    auto route_metrics = [&](const std::vector<int>& route) -> Mech {
        std::vector<double> d0, pos;
        for (size_t j = 0; j < route.size(); j++) {
            d0.push_back(euclidean(*cust.at(route[j]), DEPOT));
            pos.push_back((double)j);
        }
        double m1 = corr(pos, d0);
        const POI* current = &DEPOT;
        double t = 0.0;
        std::vector<double> waits;
        for (int cid : route) {
            const POI* p = cust.at(cid);
            double arr = t + euclidean(*current, *p);
            double w = p->open_time - arr;
            waits.push_back(w > 0.0 ? w : 0.0);
            t = (arr > p->open_time ? arr : p->open_time) + p->duration;
            current = p;
        }
        int n_wait = 0;
        double wait_tot = 0.0;
        for (double w : waits) { if (w > 1e-9) n_wait++; wait_tot += w; }
        std::vector<double> angs;
        for (int cid : route)
            angs.push_back(std::atan2(cust.at(cid)->y - DEPOT.y, cust.at(cid)->x - DEPOT.x));
        std::vector<int> dirs;
        for (size_t j = 1; j < angs.size(); j++) {
            double d = angs[j] - angs[j - 1];
            while (d > M_PI) d -= 2 * M_PI;
            while (d < -M_PI) d += 2 * M_PI;
            if (std::fabs(d) > 1e-9) dirs.push_back(d > 0 ? 1 : -1);
        }
        int same = 0;
        for (size_t j = 1; j < dirs.size(); j++)
            if (dirs[j] == dirs[j - 1]) same++;
        double m3 = dirs.size() > 1 ? (double)same / (dirs.size() - 1) : 0.0;
        std::vector<double> savs;
        const POI* prev = &DEPOT;
        for (int cid : route) {
            const POI* p = cust.at(cid);
            savs.push_back(euclidean(*prev, DEPOT) + euclidean(DEPOT, *p) - euclidean(*prev, *p));
            prev = p;
        }
        double m4 = 0.0;
        if (savs.size() > 1) {
            for (size_t j = 1; j < savs.size(); j++) m4 += savs[j];
            m4 /= (savs.size() - 1);
        }
        return {m1, n_wait, (int)route.size(), m3, m4, wait_tot};
    };

    JPtr ref = load_json(ref_route_path(INST));
    std::vector<std::vector<int>> ref_routes;
    for (size_t v = 0; v < ref->at("routes").size(); v++) {
        std::vector<int> lane;
        for (size_t k = 0; k < ref->at("routes")[v].size(); k++)
            lane.push_back((int)ref->at("routes")[v][k].as_int() + 1);
        ref_routes.push_back(lane);
    }

    std::printf("# MECHANISMS %s (M1 corr-pos-d0 [<0=far-first] | M2 waits | M3 sweep [1=perfect] | M4 mean savings)\n", INST.c_str());
    std::printf("## reference (cuOpt)\n");
    for (size_t k = 0; k < ref_routes.size(); k++) {
        Mech m = route_metrics(ref_routes[k]);
        std::printf("  veh%zu: M1 %+.2f | M2 %2d/%d waits (tot %4.0f) | M3 %.2f | M4 %6.1f\n",
                    k, m.m1, m.n_wait, m.n, m.wait_tot, m.m3, m.m4);
    }

    std::printf("## NS-GP (5 representative seeds: 2 best, median, 2 worst by sigma=0 score)\n");
    struct Row { double score; int seed; std::vector<std::vector<int>> routes; };
    std::vector<Row> rows;
    std::vector<std::string> seed_files;
    for (auto& e : fs::directory_iterator("results/baseline/nsgp/" + INST))
        if (e.path().filename().string().rfind("seed", 0) == 0) seed_files.push_back(e.path().string());
    std::sort(seed_files.begin(), seed_files.end());
    for (auto& f : seed_files) {
        JPtr d = load_json(f);
        auto routes = route_of(inst, parse_prefix(d->at("best_tree").str));
        std::map<int, bool> served;
        double sc = 0.0;
        for (auto& r : routes)
            for (int i : r)
                if (!served[i]) { served[i] = true; sc += cust.at(i)->score; }
        std::string stem = fs::path(f).stem().string();
        rows.push_back({sc, std::atoi(stem.substr(4).c_str()), routes});
    }
    std::stable_sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.score > b.score; });
    std::vector<const Row*> picks = {&rows[0], &rows[1], &rows[rows.size() / 2],
                                     &rows[rows.size() - 2], &rows[rows.size() - 1]};
    for (const Row* r : picks) {
        std::vector<Mech> ms;
        for (auto& rt : r->routes)
            if (rt.size() >= 3) ms.push_back(route_metrics(rt));
        double m1 = 0.0, m3 = 0.0, m4 = 0.0, wt = 0.0;
        int nw = 0, n = 0;
        for (auto& m : ms) {
            m1 += m.m1; m3 += m.m3; m4 += m.m4;
            nw += m.n_wait; n += m.n; wt += m.wait_tot;
        }
        m1 /= ms.size(); m3 /= ms.size(); m4 /= ms.size();
        std::printf("  seed%2d (score %4.0f): M1 %+.2f | M2 %2d/%d (tot %4.0f) | M3 %.2f | M4 %6.1f\n",
                    r->seed, r->score, m1, nw, n, wt, m3, m4);
    }
    return 0;
}
