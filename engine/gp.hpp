// The GP loop: ramped half-and-half initialization (Mei Alg 3 line 1), Koza
// 90/10 point selection (Koza 1992 S6, the ECJ default), subtree crossover and
// mutation under the depth-8 limit, tournament selection of size 7, elitism
// with stable ordering, and one generation step at the Mei/Jackson rates.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "rng.hpp"
#include "tree.hpp"

struct TreeConfig {
    std::vector<std::string> terminals;
    std::vector<std::string> functions = FUNCTIONS;
    int max_depth = 8;   // Mei Tab II
    int min_depth = 2;
};

struct EvolutionConfig {
    int population_size = 1024;
    int generations = 51;
    int max_depth = 8;
    int min_depth = 2;
    int tournament_size = 7;
    int elitism = 10;
    double crossover_rate = 0.80;
    double mutation_rate = 0.15;
    double reproduction_rate = 0.05;
    int num_vehicles = 3;
    double sigma = 0.2;
    int train_samples = 1;   // Mei §III.C: 1 sample per generation, rotated
    int test_samples = 500;  // Mei §IV.A
};

// 'ERC' in the terminal set instantiates a random constant, label "%.3f".
inline NodeP pick_terminal(const TreeConfig& cfg, Mt19937& rng) {
    const std::string& t = cfg.terminals[rng.choice_index(cfg.terminals.size())];
    if (t == "ERC") {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%.3f", rng.random());
        return mk_node(buf);
    }
    return mk_node(t);
}

inline NodeP generate_full(const TreeConfig& cfg, Mt19937& rng, int depth) {
    if (depth <= 0) return pick_terminal(cfg, rng);
    const std::string& symbol = cfg.functions[rng.choice_index(cfg.functions.size())];
    NodeP a = generate_full(cfg, rng, depth - 1);
    NodeP b = generate_full(cfg, rng, depth - 1);
    return mk_node(symbol, {a, b});
}

inline NodeP generate_grow(const TreeConfig& cfg, Mt19937& rng, int depth) {
    if (depth <= 0) return pick_terminal(cfg, rng);
    double n_term = (double)cfg.terminals.size();
    double n_func = (double)cfg.functions.size();
    if (rng.random() < n_term / (n_term + n_func)) return pick_terminal(cfg, rng);
    const std::string& symbol = cfg.functions[rng.choice_index(cfg.functions.size())];
    NodeP a = generate_grow(cfg, rng, depth - 1);
    NodeP b = generate_grow(cfg, rng, depth - 1);
    return mk_node(symbol, {a, b});
}

// Mei Alg 3 line 1: depths ramp min..max, alternating full/grow blocks.
inline std::vector<NodeP> ramped_half_and_half(const TreeConfig& cfg, Mt19937& rng, int pop_size) {
    std::vector<NodeP> population;
    population.reserve(pop_size);
    int n_depths = cfg.max_depth - cfg.min_depth + 1;
    for (int i = 0; i < pop_size; i++) {
        int d = cfg.min_depth + i % n_depths;
        if ((i / n_depths) % 2 == 0) population.push_back(generate_full(cfg, rng, d));
        else population.push_back(generate_grow(cfg, rng, d));
    }
    return population;
}

inline void all_nodes_into(const NodeP& node, std::vector<const Node*>& out) {
    out.push_back(node.get());
    for (auto& c : node->children) all_nodes_into(c, out);
}

inline std::vector<const Node*> all_nodes(const NodeP& root) {
    std::vector<const Node*> out;
    all_nodes_into(root, out);
    return out;
}

inline NodeP get_subtree_at(const NodeP& node, int index, int& counter) {
    if (++counter == index) return node;
    for (auto& c : node->children) {
        NodeP r = get_subtree_at(c, index, counter);
        if (r) return r;
    }
    return nullptr;
}

inline NodeP get_subtree(const NodeP& root, int index) {
    int counter = -1;
    return get_subtree_at(root, index, counter);
}

inline NodeP replace_subtree_rec(const NodeP& n, int index, const NodeP& replacement, int& counter) {
    if (++counter == index) return replacement;
    if (n->is_terminal()) return n;
    std::vector<NodeP> children;
    children.reserve(n->children.size());
    for (auto& c : n->children) children.push_back(replace_subtree_rec(c, index, replacement, counter));
    return mk_node(n->label, std::move(children));
}

inline NodeP replace_subtree(const NodeP& root, int index, const NodeP& replacement) {
    int counter = -1;
    return replace_subtree_rec(root, index, replacement, counter);
}

// Koza 90/10 internal/terminal point selection; the rng is consumed only
// when both node classes are present.
inline int select_crossover_point(const NodeP& tree, Mt19937& rng, double p_internal = 0.9) {
    std::vector<const Node*> nodes = all_nodes(tree);
    std::vector<int> internals, leaves;
    for (int i = 0; i < (int)nodes.size(); i++) {
        if (nodes[i]->is_terminal()) leaves.push_back(i);
        else internals.push_back(i);
    }
    if (!internals.empty() && (leaves.empty() || rng.random() < p_internal))
        return internals[rng.choice_index(internals.size())];
    if (!leaves.empty()) return leaves[rng.choice_index(leaves.size())];
    return internals[rng.choice_index(internals.size())];
}

inline std::pair<NodeP, NodeP> subtree_crossover(const NodeP& parent1, const NodeP& parent2,
                                                 Mt19937& rng, int max_depth) {
    int pt1 = select_crossover_point(parent1, rng);
    int pt2 = select_crossover_point(parent2, rng);
    NodeP sub1 = get_subtree(parent1, pt1);
    NodeP sub2 = get_subtree(parent2, pt2);
    NodeP child1 = replace_subtree(parent1, pt1, sub2);
    NodeP child2 = replace_subtree(parent2, pt2, sub1);
    if (tree_depth(child1) > max_depth) child1 = parent1;
    if (tree_depth(child2) > max_depth) child2 = parent2;
    return {child1, child2};
}

inline NodeP subtree_mutation(const NodeP& individual, const TreeConfig& cfg, Mt19937& rng) {
    int pt = select_crossover_point(individual, rng);
    int new_depth = (int)rng.randint(cfg.min_depth, cfg.max_depth);
    NodeP new_subtree = generate_grow(cfg, rng, new_depth);
    NodeP mutant = replace_subtree(individual, pt, new_subtree);
    if (tree_depth(mutant) > cfg.max_depth) return individual;
    return mutant;
}

// Tournament selection: k distinct contenders, first maximum in contender
// order.
inline NodeP tournament_selection(const std::vector<NodeP>& population,
                                  const std::vector<double>& fitness,
                                  int size_, Mt19937& rng) {
    size_t k = std::min((size_t)size_, population.size());
    std::vector<size_t> contenders = rng.sample(population.size(), k);
    size_t best = contenders[0];
    for (size_t idx = 1; idx < contenders.size(); idx++)
        if (fitness[contenders[idx]] > fitness[best]) best = contenders[idx];
    return population[best];
}

// One generation: stable sort desc by fitness (ties keep index order),
// elitism, then crossover/mutation/reproduction by rate.
inline std::vector<NodeP> gp_step(const std::vector<NodeP>& pop, const std::vector<double>& fit,
                                  Mt19937& rng, const TreeConfig& tree_cfg, const EvolutionConfig& cfg) {
    std::vector<int> order(pop.size());
    for (size_t i = 0; i < pop.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return fit[a] > fit[b]; });
    std::vector<NodeP> new_pop;
    new_pop.reserve(cfg.population_size);
    for (int e = 0; e < cfg.elitism && e < (int)order.size(); e++) new_pop.push_back(pop[order[e]]);
    while ((int)new_pop.size() < cfg.population_size) {
        double r = rng.random();
        if (r < cfg.crossover_rate) {
            NodeP p1 = tournament_selection(pop, fit, cfg.tournament_size, rng);
            NodeP p2 = tournament_selection(pop, fit, cfg.tournament_size, rng);
            auto [c1, c2] = subtree_crossover(p1, p2, rng, cfg.max_depth);
            new_pop.push_back(c1);
            if ((int)new_pop.size() < cfg.population_size) new_pop.push_back(c2);
        } else if (r < cfg.crossover_rate + cfg.mutation_rate) {
            new_pop.push_back(subtree_mutation(tournament_selection(pop, fit, cfg.tournament_size, rng), tree_cfg, rng));
        } else {
            new_pop.push_back(tournament_selection(pop, fit, cfg.tournament_size, rng));
        }
    }
    new_pop.resize(cfg.population_size);
    return new_pop;
}
