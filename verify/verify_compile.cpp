// Gate: parse_prefix + compile_rmod (core/tree.hpp, core/vocab.hpp) against
// the golden/compile.txt fixtures. Opcode/argument arrays and the prefix
// round-trip are compared exactly.
//     verify_compile golden/compile.txt
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include "../engine/tree.hpp"
#include "../engine/vocab.hpp"

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "golden/compile.txt");
    if (!f) { std::fprintf(stderr, "golden file not found\n"); return 2; }
    std::string line;
    long trees = 0, bad = 0;
    std::string cur_tree;
    Compiled c;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("tree ", 0) == 0) {
            cur_tree = line.substr(5);
            NodeP t = parse_prefix(cur_tree);
            if (to_prefix(t) != cur_tree) {
                std::printf("MISMATCH roundtrip: %s\n", cur_tree.c_str());
                bad++;
            }
            c = compile_rmod(t);
            trees++;
        } else if (line.rfind("ops ", 0) == 0) {
            std::istringstream ss(line.substr(4));
            std::string got;
            for (size_t i = 0; i < c.ops.size(); i++) got += (i ? " " : "") + std::to_string(c.ops[i]);
            std::string want = line.substr(4);
            if (got != want) {
                if (bad < 5) std::printf("MISMATCH ops: %s\n  got  %s\n  want %s\n", cur_tree.c_str(), got.c_str(), want.c_str());
                bad++;
            }
        } else if (line.rfind("args ", 0) == 0) {
            std::string got;
            for (size_t i = 0; i < c.args.size(); i++) {
                if (i) got += " ";
                got += std::to_string(c.args[i][0]) + " " + std::to_string(c.args[i][1]);
            }
            std::string want = line.substr(5);
            if (got != want) {
                if (bad < 5) std::printf("MISMATCH args: %s\n", cur_tree.c_str());
                bad++;
            }
        }
    }
    std::printf("verify_compile: %ld trees compared, %ld mismatches -> %s\n",
                trees, bad, bad == 0 ? "GATE_PASS" : "GATE_FAIL");
    return bad == 0 ? 0 : 1;
}
