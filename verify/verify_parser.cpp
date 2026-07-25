// Gate: the instance readers (core/instance.hpp) against the golden/parser.txt
// fixtures. Every field of every POI of every instance is compared by bit
// pattern.
//     verify_parser golden/parser.txt
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "../engine/instance.hpp"

static std::string dhex(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)u);
    return buf;
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "golden/parser.txt");
    if (!f) { std::fprintf(stderr, "golden file not found\n"); return 2; }
    std::string line, tag;
    Instance inst;
    size_t poi_i = 0;
    long total = 0, bad = 0;
    int n_inst = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ss >> tag;
        if (tag == "#") {
            std::string w, nm;
            ss >> w >> nm;
            inst = load_instance(nm);
            poi_i = 0;
            n_inst++;
            continue;
        }
        total++;
        if (tag == "name") {
            std::string want; ss >> want;
            if (inst.name != want) { std::printf("MISMATCH name: %s vs %s\n", inst.name.c_str(), want.c_str()); bad++; }
        } else if (tag == "n") {
            size_t want; ss >> want;
            if (inst.pois.size() != want) { std::printf("MISMATCH n: %zu vs %zu\n", inst.pois.size(), want); bad++; }
        } else if (tag == "tmax") {
            std::string want; ss >> want;
            if (dhex(inst.tmax) != want) { std::printf("MISMATCH tmax %s\n", inst.name.c_str()); bad++; }
        } else if (tag == "poi") {
            int id; std::string xs, ys, sc, ot, ct, du;
            ss >> id >> xs >> ys >> sc >> ot >> ct >> du;
            const POI& p = inst.pois[poi_i++];
            bool ok = p.id == id && dhex(p.x) == xs && dhex(p.y) == ys && dhex(p.score) == sc
                      && dhex(p.open_time) == ot && dhex(p.close_time) == ct && dhex(p.duration) == du;
            if (!ok) {
                if (bad < 10) std::printf("MISMATCH poi %s id=%d\n", inst.name.c_str(), id);
                bad++;
            }
        }
    }
    std::printf("verify_parser: %d instances, %ld fields compared, %ld mismatches -> %s\n",
                n_inst, total, bad, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
