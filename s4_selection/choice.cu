// Cross-validatory choice per instance (Stone 1974): select the column by the
// mean train score of its 30 final rules on a common TRAIN block (2M x 64),
// read the held-out once. The NS-GP baseline enters as a backup member (Xu et
// al. 2008), so the choice is never worse by construction.
//     choice c101 <trace.json> ... [--backup]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../engine/baseline_csv.hpp"
#include "../engine/factory.cuh"
#include "../engine/json.hpp"

static constexpr long long TRAIN_SEED = 2000000;
static constexpr int N_TRAIN = 64;

static JPtr load_json(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::fprintf(stderr, "not found: %s\n", p.c_str()); std::exit(2); }
    std::stringstream b;
    b << f.rdbuf();
    return json_parse(b.str());
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: choice <inst> <trace.json>... [--backup]\n"); return 2; }
    std::string inst_name = argv[1];
    std::vector<std::string> paths;
    bool with_backup = false;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--backup") with_backup = true;
        else paths.push_back(argv[i]);
    }
    SimData sd(load_instance(inst_name), 3, 0.2);
    FactoryContext ctx(sd, zp_of(sd));
    std::vector<double> train = gen_durations(sd, N_TRAIN, TRAIN_SEED);
    std::printf("# choice %s | train block 2M x %d | columns: %zu\n",
                inst_name.c_str(), N_TRAIN, paths.size() + (with_backup ? 1 : 0));
    struct Row { std::string tag; double mtrain, heldout; };
    std::vector<Row> rows;

    auto mean_train = [&](const std::vector<NodeP>& rules, const std::vector<int64_t>& tmask) {
        PackedTrees pk = pack_trees(rules);
        std::vector<double> fits = ctx.evaluate(pk, train, N_TRAIN, tmask);
        double acc = 0.0;
        for (size_t p = 0; p < rules.size(); p++) {
            double m = 0.0;
            for (int s = 0; s < N_TRAIN; s++) m += fits[p * N_TRAIN + s];
            acc += m / N_TRAIN;
        }
        return acc / rules.size();
    };

    if (with_backup) {
        std::vector<NodeP> rules;
        for (int s = 0; s < 30; s++) {
            JPtr d = load_json(nsgp_seed_path(inst_name, s));
            rules.push_back(parse_prefix(d->at("best_tree").str));
        }
        double mtrain = mean_train(rules, term_mask_from(NS_TERMINALS));
        rows.push_back({"NS-GP(backup)", mtrain, 0.0});
        std::printf("  %-24s mean-TRAIN %8.2f | held-out +0.00\n", "NS-GP (backup)", mtrain);
    }
    for (auto& path : paths) {
        JPtr t = load_json(path);
        const JValue& finals = t->at("finals");
        std::vector<std::pair<int, std::string>> by_seed;
        for (size_t i = 0; i < finals.size(); i++)
            by_seed.push_back({(int)finals[i].at("seed").as_int(), finals[i].at("rule").str});
        std::stable_sort(by_seed.begin(), by_seed.end(),
                         [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<NodeP> rules;
        for (auto& [sd_, r] : by_seed) rules.push_back(parse_prefix(r));
        std::vector<std::string> vocab = NS_TERMINALS;
        for (size_t i = 0; i < t->at("vocab_extra").size(); i++)
            vocab.push_back(t->at("vocab_extra")[i].str);
        std::string label = path;
        size_t sl = label.find_last_of("/\\");
        if (sl != std::string::npos) label = label.substr(sl + 1);
        size_t dot = label.find_last_of('.');
        if (dot != std::string::npos) label = label.substr(0, dot);
        double mtrain = mean_train(rules, term_mask_from(vocab));
        rows.push_back({label, mtrain, t->at("mean_gain").num});
        std::printf("  %-24s mean-TRAIN %8.2f | held-out %+.2f\n", label.c_str(), mtrain,
                    t->at("mean_gain").num);
    }
    const Row* choice = &rows[0];
    for (auto& r : rows)
        if (r.mtrain > choice->mtrain) choice = &r;
    std::printf("\nCROSS-VALIDATORY CHOICE (train-side): %s (train %.2f) -> held-out %+.2f\n",
                choice->tag.c_str(), choice->mtrain, choice->heldout);
    const Row* vbs = &rows[0];
    for (auto& r : rows)
        if (r.heldout > vbs->heldout) vbs = &r;
    std::printf("virtual best (labeled): %s -> %+.2f | choice==VBS? %s\n",
                vbs->tag.c_str(), vbs->heldout,
                choice->tag == vbs->tag ? "True" : "False");
    return 0;
}
