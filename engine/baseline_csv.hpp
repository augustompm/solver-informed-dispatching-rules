// NS-GP baseline access: nsgp_mean_30 reads the per-instance mean-30 from the
// baseline CSV ($GITC_CAP2), nsgp_seed_path locates a per-seed run JSON.
// Quoted CSV fields hold commas (equations). Missing file or instance is a
// hard error, never a default.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::string nsgp_seed_path(const std::string& inst, int seed) {
    return "results/baseline/nsgp/" + inst + "/seed" + std::to_string(seed) + ".json";
}

inline double nsgp_mean_30(const std::string& name) {
    const char* env = std::getenv("GITC_CAP2");
    std::string path = env ? env : "results/baseline/cap2-results.csv";
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "baseline csv not found: %s\n", path.c_str()); std::exit(2); }
    std::string header;
    std::getline(f, header);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    std::vector<std::string> cols;
    {
        std::istringstream ss(header);
        std::string c;
        while (std::getline(ss, c, ',')) cols.push_back(c);
    }
    int ci = -1, cv = -1;
    for (int i = 0; i < (int)cols.size(); i++) {
        if (cols[i] == "instance") ci = i;
        if (cols[i] == "baseline_nsgp_mean30") cv = i;
    }
    if (ci < 0 || cv < 0) { std::fprintf(stderr, "columns missing in %s\n", path.c_str()); std::exit(2); }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> parts;
        bool inq = false;
        std::string cur;
        for (char ch : line) {
            if (ch == '"') inq = !inq;
            else if (ch == ',' && !inq) { parts.push_back(cur); cur.clear(); }
            else cur += ch;
        }
        parts.push_back(cur);
        if ((int)parts.size() > (ci > cv ? ci : cv) && parts[ci] == name)
            return std::strtod(parts[cv].c_str(), nullptr);
    }
    std::fprintf(stderr, "instance %s not in %s\n", name.c_str(), path.c_str());
    std::exit(2);
}
