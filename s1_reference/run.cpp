// Stage 1 driver: replay a reference route under random service and print the
// ceiling numbers. Route JSONs in this repository are 0-indexed over customers
// (+1 here); --pyvrp-indexing reads raw 1-based PyVRP exports instead.
//     run <instance> <route.json> [--pyvrp-indexing] [--scenarios N]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "../engine/json.hpp"
#include "reference.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: run <instance> <route.json> [--pyvrp-indexing] [--scenarios N]\n"); return 2; }
    std::string name = argv[1], route_path = argv[2];
    bool pyvrp_idx = false;
    int scenarios = 30;
    for (int a = 3; a < argc; a++) {
        std::string s = argv[a];
        if (s == "--pyvrp-indexing") pyvrp_idx = true;
        else if (s == "--scenarios" && a + 1 < argc) scenarios = std::atoi(argv[++a]);
    }
    RefInstance inst(load_instance(name));

    std::ifstream f(route_path);
    if (!f) { std::fprintf(stderr, "route json not found: %s\n", route_path.c_str()); return 2; }
    std::stringstream buf;
    buf << f.rdbuf();
    JPtr doc = json_parse(buf.str());
    std::vector<std::vector<int>> route;
    int visited = 0;
    for (size_t v = 0; v < doc->at("routes").size(); v++) {
        std::vector<int> lane;
        const JValue& jr = doc->at("routes")[v];
        for (size_t k = 0; k < jr.size(); k++) {
            int cid = (int)jr[k].as_int();
            lane.push_back(pyvrp_idx ? cid : cid + 1);   // cuOpt is 0-indexed; Solomon 1..100
            visited++;
        }
        route.push_back(lane);
    }
    double prize = route_prize_mean(inst, route, scenarios);

    std::printf("instance      %s\n", name.c_str());
    std::printf("customers     %d\n", inst.n - 1);
    std::printf("visited       %d\n", visited);
    std::printf("route prize   %.2f   (under random service, held-out seed)\n", prize);
    std::printf("\n");
    std::printf("The consolidated per-instance reference gap (500 scenarios, common random\n");
    std::printf("numbers) is results/s1_reference/reference_gap.csv.\n");
    return 0;
}
