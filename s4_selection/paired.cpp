// Paired test per seed: the chosen column vs the NS-GP rule of the SAME seed
// (shared seed index and common random numbers, so the Wilcoxon signed-rank
// paired test applies).
//     paired c202 <trace.json>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/baseline_csv.hpp"
#include "../engine/json.hpp"
#include "../engine/stats.hpp"

static JPtr load_json(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::fprintf(stderr, "not found: %s\n", p.c_str()); std::exit(2); }
    std::stringstream b;
    b << f.rdbuf();
    return json_parse(b.str());
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: paired <inst> <trace.json>\n"); return 2; }
    std::string inst = argv[1], tag = argv[2];
    double km = nsgp_mean_30(inst);
    JPtr tr = load_json(tag);
    if (tr->at("instance").str != inst) { std::fprintf(stderr, "trace instance mismatch\n"); return 2; }
    const JValue& finals = tr->at("finals");
    std::vector<std::pair<int, double>> by_seed;
    for (size_t i = 0; i < finals.size(); i++)
        by_seed.push_back({(int)finals[i].at("seed").as_int(), finals[i].at("gain").num});
    std::stable_sort(by_seed.begin(), by_seed.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<double> ours, nsgp;
    for (auto& [sd, gain] : by_seed) ours.push_back(km * (1 + gain / 100.0));
    for (size_t s = 0; s < ours.size(); s++) {
        JPtr d = load_json(nsgp_seed_path(inst, (int)s));
        nsgp.push_back(d->at("test_mean").num);
    }
    int w = 0, l = 0, t = 0;
    double md = 0.0;
    for (size_t i = 0; i < ours.size(); i++) {
        double d = ours[i] - nsgp[i];
        md += d;
        if (d > 1e-9) w++;
        else if (d < -1e-9) l++;
        else t++;
    }
    md /= ours.size();
    double p = (w + l) ? wilcoxon_p(ours, nsgp) : std::nan("");
    std::printf("paired %s %s: %dW/%dL/%dT | Wilcoxon p=%.4g | mean-delta %+.2f (%+.2f%%) | vs-MEAN %+.2f%%\n",
                inst.c_str(), tag.c_str(), w, l, t, p, md, 100.0 * md / km, tr->at("mean_gain").num);
    return 0;
}
