// Gate: the MT19937 stream (core/rng.hpp) against the golden/rng.txt fixtures.
// Replays the exact draw sequence of make_golden.py and compares every value;
// doubles compared by bit pattern, never by tolerance.
//     verify_rng golden/rng.txt
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include "../engine/rng.hpp"

static std::string dhex(double x) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)u);
    return buf;
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "golden/rng.txt");
    if (!f) { std::fprintf(stderr, "golden file not found\n"); return 2; }
    Mt19937* rng = nullptr;
    std::string line, tag;
    long total = 0, bad = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        ss >> tag;
        if (tag == "#") {
            std::string w; uint64_t seed;
            ss >> w >> seed;
            delete rng;
            rng = new Mt19937(seed);
            continue;
        }
        total++;
        std::string got, want;
        if (tag == "random" || tag == "mix_random") { ss >> want; got = dhex(rng->random()); }
        else if (tag.rfind("getrandbits_", 0) == 0) {
            int k = std::stoi(tag.substr(12));
            ss >> want; got = std::to_string(rng->getrandbits(k));
        }
        else if (tag == "randint_2_8" || tag == "mix_randint") { ss >> want; got = std::to_string(rng->randint(2, 8)); }
        else if (tag == "choice29") { ss >> want; got = std::to_string(rng->choice_index(29)); }
        else if (tag == "mix_choice") { ss >> want; got = std::to_string(rng->choice_index(61)); }
        else if (tag == "sample1024_7" || tag == "sample50_7") {
            auto s = rng->sample(tag == "sample1024_7" ? 1024 : 50, 7);
            std::string rest;
            std::getline(ss, rest);
            std::ostringstream o;
            for (size_t i = 0; i < s.size(); i++) o << (i ? " " : "") << s[i];
            want = rest.substr(1);   // skip leading space
            got = o.str();
        }
        else if (tag == "gauss") { ss >> want; got = dhex(rng->gauss(10.0, 2.0)); }
        else if (tag == "mix_gauss") { ss >> want; got = dhex(rng->gauss(5.0, 1.5)); }
        else { std::fprintf(stderr, "unknown tag %s\n", tag.c_str()); return 2; }
        if (got != want) {
            if (bad < 10) std::printf("MISMATCH %s: got %s want %s\n", tag.c_str(), got.c_str(), want.c_str());
            bad++;
        }
    }
    delete rng;
    std::printf("verify_rng: %ld draws compared, %ld mismatches -> %s\n",
                total, bad, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
