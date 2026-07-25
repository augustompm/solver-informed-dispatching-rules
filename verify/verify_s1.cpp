// Gate: the stage-1 replay (reference.hpp) against the golden/s1.txt fixtures
// (route embedded in the file; 30 sampled-service scenarios, prizes compared
// bit by bit).
//     verify_s1 golden/s1.txt
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../s1_reference/reference.hpp"

static std::string dhex(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)u);
    return buf;
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "golden/s1.txt");
    if (!f) { std::fprintf(stderr, "s1 golden not found\n"); return 2; }
    const char* env = std::getenv("GITC_DATA");
    std::string data = env ? env : "instances";
    RefInstance inst(data + "/solomon/r101.txt");
    std::vector<std::vector<int>> route;
    std::string line;
    long total = 0, bad = 0;
    int s = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("route ", 0) == 0) {
            std::istringstream ss(line.substr(6));
            std::vector<int> lane;
            int c;
            while (ss >> c) lane.push_back(c);
            route.push_back(lane);
        } else if (line.rfind("prize ", 0) == 0) {
            std::string want = line.substr(6);
            double got = deploy_route(inst, route, inst.sample_service(1000000 + s));
            total++;
            if (dhex(got) != want) {
                if (bad < 5) std::printf("MISMATCH scenario %d: %.17g\n", s, got);
                bad++;
            }
            s++;
        }
    }
    std::printf("verify_s1: %ld prizes compared, %ld mismatches -> %s\n",
                total, bad, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
