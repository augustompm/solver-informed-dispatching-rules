// Stage 1 driver: replay a reference route under random service and print the
// ceiling numbers. The route comes from the external solver's JSON (cuOpt
// routes are 0-indexed over customers, +1 here; PyVRP visits are 1-based).
//     run <instance> <route.json> [--pyvrp-indexing]
#include <cstdio>
#include <fstream>
#include <sstream>
#include "../engine/json.hpp"
#include "reference.hpp"

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: run <instance> <route.json> [--pyvrp-indexing]\n"); return 2; }
    std::string name = argv[1], route_path = argv[2];
    bool pyvrp_idx = argc > 3 && std::string(argv[3]) == "--pyvrp-indexing";
    const char* env = std::getenv("GITC_DATA");
    std::string data = env ? env : "instances";
    RefInstance inst(data + "/solomon/" + name + ".txt");

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
    double prize = route_prize_mean(inst, route);

    std::printf("instance      %s\n", name.c_str());
    std::printf("customers     %d\n", inst.n - 1);
    std::printf("visited       %d\n", visited);
    std::printf("route prize   %.2f   (under random service, held-out seed)\n", prize);
    std::printf("\n");
    std::printf("A quick solve gives a feasible route. The full reference gap in results/ used\n");
    std::printf("the paper's time budget, minutes per instance, and cuOpt on GPU.\n");
    return 0;
}
