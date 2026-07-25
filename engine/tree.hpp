// GP expression tree: immutable nodes with shared subtrees, the binary
// function set (Mei 2018 SIII.A), protected division, the rmod power operator,
// size/depth and prefix (de)serialization.
#pragma once
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct Node;
using NodeP = std::shared_ptr<const Node>;

struct Node {
    std::string label;
    std::vector<NodeP> children;
    bool is_terminal() const { return children.empty(); }
};

inline NodeP mk_node(std::string label) {
    auto n = std::make_shared<Node>();
    n->label = std::move(label);
    return n;
}

inline NodeP mk_node(std::string label, std::vector<NodeP> children) {
    auto n = std::make_shared<Node>();
    n->label = std::move(label);
    n->children = std::move(children);
    return n;
}

inline const std::vector<std::string> FUNCTIONS = {"+", "-", "*", "/", "min", "max"};
inline const std::vector<std::string> FUNCS_RMOD = {"+", "-", "*", "/", "min", "max", "rmod"};

// Mei §III.A: '/' returns 1 if divided by (near-)zero.
inline double protected_div(double a, double b) {
    return std::fabs(b) < 1e-10 ? 1.0 : a / b;
}

// RMOD(base=a, regime=b) = max(a,0) ^ (1 + clip(b,0,1)) — opcode 6 of the engine.
inline double rmod_fn(double a, double b) {
    double base = a > 0.0 ? a : 0.0;
    double e = b < 0.0 ? 0.0 : (b > 1.0 ? 1.0 : b);
    return std::pow(base, 1.0 + e);
}

inline int tree_size(const NodeP& n) {
    if (n->is_terminal()) return 1;
    int s = 1;
    for (auto& c : n->children) s += tree_size(c);
    return s;
}

inline int tree_depth(const NodeP& n) {
    if (n->is_terminal()) return 0;
    int d = 0;
    for (auto& c : n->children) d = std::max(d, tree_depth(c));
    return 1 + d;
}

inline std::string to_prefix(const NodeP& n) {
    if (n->is_terminal()) return n->label;
    std::string out = "(" + n->label;
    for (auto& c : n->children) out += " " + to_prefix(c);
    return out + ")";
}

inline NodeP parse_prefix_at(const std::vector<std::string>& toks, size_t& pos) {
    const std::string& tok = toks[pos];
    if (tok == "(") {
        pos++;
        std::string symbol = toks[pos++];
        std::vector<NodeP> children;
        while (toks[pos] != ")") children.push_back(parse_prefix_at(toks, pos));
        pos++;
        return mk_node(symbol, std::move(children));
    }
    pos++;
    return mk_node(tok);
}

inline NodeP parse_prefix(const std::string& s) {
    std::string t;
    for (char c : s) {
        if (c == '(') t += " ( ";
        else if (c == ')') t += " ) ";
        else t += c;
    }
    std::istringstream ss(t);
    std::vector<std::string> toks;
    std::string w;
    while (ss >> w) toks.push_back(w);
    size_t pos = 0;
    return parse_prefix_at(toks, pos);
}
