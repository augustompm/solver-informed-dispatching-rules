// Redeploy gate: re-parse the trace's best rule from its serialized prefix
// form and re-score it on the held-out 500@1M, checking that the deployed
// rule reproduces the trace gain. delta <= 0.005pp = sealed.
//     redeploy r101 <trace.json>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/baseline_csv.hpp"
#include "../engine/factory.cuh"
#include "../engine/json.hpp"

static constexpr long long TEST_SEED = 1000000;
static constexpr int N_TEST = 500;

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: redeploy <inst> <trace.json>\n"); return 2; }
    std::string inst_name = argv[1], tag = argv[2];
    std::ifstream f(tag);
    if (!f) { std::fprintf(stderr, "trace not found\n"); return 2; }
    std::stringstream b;
    b << f.rdbuf();
    JPtr tr = json_parse(b.str());
    if (tr->at("instance").str != inst_name) { std::fprintf(stderr, "trace instance mismatch\n"); return 2; }
    const JValue& finals = tr->at("finals");
    const JValue* best = &finals[0];
    for (size_t i = 1; i < finals.size(); i++)
        if (finals[i].at("gain").num > best->at("gain").num) best = &finals[i];
    NodeP rule = parse_prefix(best->at("rule").str);

    SimData sd(load_instance(inst_name), 3, 0.2);
    FactoryContext ctx(sd, zp_of(sd));
    std::vector<std::string> vocab = NS_TERMINALS;
    for (size_t i = 0; i < tr->at("vocab_extra").size(); i++)
        vocab.push_back(tr->at("vocab_extra")[i].str);
    std::vector<int64_t> tmask = term_mask_from(vocab);
    std::vector<double> test = gen_durations(sd, N_TEST, TEST_SEED);
    PackedTrees pk = pack_trees({rule});
    std::vector<double> ts = ctx.evaluate(pk, test, N_TEST, tmask);
    double got = 0.0;
    for (int s = 0; s < N_TEST; s++) got += ts[s];
    got /= N_TEST;
    double km = nsgp_mean_30(inst_name);
    double gain = 100.0 * (got - km) / km;
    double delta = std::fabs(gain - best->at("gain").num);
    std::printf("redeploy %s %s: trace-gain %+.2f | re-deploy CUDA %+.2f | delta %.4fpp -> %s\n",
                inst_name.c_str(), tag.c_str(), best->at("gain").num, gain, delta,
                delta <= 0.005 ? "SEALED" : "FAILED");
    return 0;
}
