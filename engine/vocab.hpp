// Terminal vocabulary: the canonical slot order of the engine terminals,
// aliases, the 11 NS-GP baseline terminals (Mei 2018 Tab I + Jackson 2020 NS),
// the 18-terminal constructed menu, and the compiler from trees to flat opcode
// arrays (op -1 terminal, -2 constant in thousandths, 0..5 arithmetic, 6 rmod).
#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "tree.hpp"

inline constexpr int N_TERMS = 61;
inline constexpr int MAX_N = 512;     // fixed per-thread buffer bounds
inline constexpr int MAX_TREE = 1024;
inline constexpr int MAX_STACK = 32;

inline const std::unordered_map<std::string, int>& term_map() {
    static const std::unordered_map<std::string, int> m = {
        {"SCORE", 0}, {"DUR", 1}, {"TO", 2}, {"TC", 3}, {"TA", 4}, {"TR", 5},
        {"TSV", 6}, {"TFV", 7}, {"SL", 8}, {"RemT", 9}, {"NS", 10}, {"REGRET", 11},
        {"NSROB", 12}, {"MAXN", 13}, {"REACH2", 14}, {"NSTW", 15}, {"RAND", 16},
        {"FRAGCNT", 17}, {"FRAGSCORE", 18}, {"DOWNSLACK", 19}, {"PBUST", 20},
        {"REACH1", 21}, {"REACH3", 22}, {"REACH5", 23},
        {"SAT", 24}, {"VFRAC", 25}, {"NFEAS", 26}, {"VEH", 27},
        {"HARVEST", 28}, {"BOLSAO", 29},
        {"ATRISK", 30},
        {"ZTC", 31}, {"ZREACH", 32}, {"ZWLOSS", 33}, {"ZATRISK", 34}, {"ZD2D", 35},
        {"C1", 36}, {"C2", 37}, {"C3", 38}, {"C4", 39}, {"C5", 40},
        {"REACH2ROB", 41},
        {"RELCOST", 42},
        {"NETREACH", 43},
        {"SKIPLOSS", 44},
        {"FRAGILITY", 45},
        {"SROB", 46},
        {"REACH3ROB", 47},
        {"REACH5ROB", 48},
        {"NSA2", 49},
        {"NSA05", 50},
        {"FEASCORE", 51},
        {"NFEASROB", 52},
        {"NSROB05", 53},
        {"NSROB2", 54},
        {"NNFWD", 55},
        {"NNK3", 56},
        {"NNCHAIN3", 57},
        {"SAVINGS", 58},
        {"DETOUR", 59},
        {"SCORERANK", 60},
    };
    return m;
}

inline const std::unordered_map<std::string, int>& term_aliases() {
    static const std::unordered_map<std::string, int> m = {
        {"D2D", 5}, {"SLACK", 8}, {"WLOSS", 11},
    };
    return m;
}

inline const std::vector<std::string> NS_TERMINALS = {
    "SCORE", "DUR", "TO", "TC", "TA", "TR", "TSV", "TFV", "SL", "RemT", "NS"};

// The 18-terminal constructed menu beyond the 11 baseline.
inline const std::vector<std::string> EXTRA = {
    "REGRET", "NSROB", "MAXN", "REACH2", "NSTW", "FRAGCNT", "FRAGSCORE", "PBUST",
    "REACH1", "REACH3", "REACH5", "SAT", "VFRAC", "NFEAS", "HARVEST", "BOLSAO",
    "ATRISK", "REACH2ROB"};

inline int term_index(const std::string& label) {
    auto& tm = term_map();
    auto it = tm.find(label);
    if (it != tm.end()) return it->second;
    auto& ta = term_aliases();
    auto jt = ta.find(label);
    if (jt != ta.end()) return jt->second;
    return -1;
}

struct Compiled {
    std::vector<int32_t> ops;
    std::vector<std::array<int32_t, 2>> args;
};

inline int op_code(const std::string& label) {
    if (label == "+") return 0;
    if (label == "-") return 1;
    if (label == "*") return 2;
    if (label == "/") return 3;
    if (label == "min") return 4;
    if (label == "max") return 5;
    if (label == "rmod") return 6;
    throw std::runtime_error("unknown operator " + label);
}

// Post-order flatten: terminal -> (-1, slot); numeric label (ERC) ->
// (-2, round(value*1000)); operator -> opcode + child indices.
inline void compile_visit(const NodeP& nd, Compiled& out) {
    if (nd->is_terminal()) {
        int ti = term_index(nd->label);
        if (ti >= 0) {
            out.ops.push_back(-1);
            out.args.push_back({ti, 0});
        } else {
            double v = std::strtod(nd->label.c_str(), nullptr);
            out.ops.push_back(-2);
            out.args.push_back({(int32_t)std::llround(v * 1000.0), 0});
        }
        return;
    }
    compile_visit(nd->children[0], out);
    int32_t left = (int32_t)out.ops.size() - 1;
    compile_visit(nd->children[1], out);
    int32_t right = (int32_t)out.ops.size() - 1;
    out.ops.push_back(op_code(nd->label));
    out.args.push_back({left, right});
}

inline Compiled compile_rmod(const NodeP& tree) {
    Compiled c;
    compile_visit(tree, c);
    return c;
}
