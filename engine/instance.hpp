// STOPTW instance: POI, Instance, Euclidean distance, Solomon and Cordeau
// readers (score = the demand column, Mei 2018 convention; depot id 0; Tmax =
// the depot close), and the Jackson 2020 SIII.A feasibility rule. Data dir:
// $GITC_DATA, default instances/ under the repo root.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct POI {
    int id;
    double x, y, score, open_time, close_time, duration;
};

struct Instance {
    std::string name;
    std::vector<POI> pois;   // sorted by id; pois[0] = depot
    double tmax;
    std::string source;

    const POI& depot() const { return pois[0]; }
    int n_customers() const { return (int)pois.size() - 1; }
};

inline double euclidean(const POI& a, const POI& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

inline bool starts_with_digit(const std::string& tok) {
    std::string t = tok;
    size_t p = t.find_first_not_of('-');
    t = (p == std::string::npos) ? "" : t.substr(p);
    return !t.empty() && std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; });
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream ss(s);
    std::vector<std::string> out;
    std::string w;
    while (ss >> w) out.push_back(w);
    return out;
}

inline Instance parse_solomon(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f) throw std::runtime_error("cannot open " + filepath);
    std::vector<std::string> lines;
    std::string ln;
    while (std::getline(f, ln)) {
        size_t a = ln.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = ln.find_last_not_of(" \t\r\n");
        lines.push_back(ln.substr(a, b - a + 1));
    }
    std::string name = lines[0];
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    size_t cust_idx = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        std::string up = lines[i];
        std::transform(up.begin(), up.end(), up.begin(), ::toupper);
        if (up.find("CUST") != std::string::npos) { cust_idx = i; break; }
    }
    Instance inst;
    inst.name = name;
    inst.source = filepath;
    for (size_t i = cust_idx + 1; i < lines.size(); i++) {
        auto parts = split_ws(lines[i]);
        if (parts.size() < 7 || !starts_with_digit(parts[0])) continue;
        POI p;
        p.id = std::stoi(parts[0]);
        p.x = std::strtod(parts[1].c_str(), nullptr);
        p.y = std::strtod(parts[2].c_str(), nullptr);
        double demand = std::strtod(parts[3].c_str(), nullptr);
        p.open_time = std::strtod(parts[4].c_str(), nullptr);
        p.close_time = std::strtod(parts[5].c_str(), nullptr);
        p.duration = std::strtod(parts[6].c_str(), nullptr);
        p.score = (p.id == 0) ? 0.0 : demand;
        inst.pois.push_back(p);
    }
    std::sort(inst.pois.begin(), inst.pois.end(), [](const POI& a, const POI& b) { return a.id < b.id; });
    inst.tmax = inst.pois[0].close_time;
    return inst;
}

inline Instance parse_cordeau(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f) throw std::runtime_error("cannot open " + filepath);
    Instance inst;
    std::string stem = filepath;
    size_t sl = stem.find_last_of("/\\");
    if (sl != std::string::npos) stem = stem.substr(sl + 1);
    size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);
    std::transform(stem.begin(), stem.end(), stem.begin(), ::tolower);
    inst.name = stem;
    inst.source = filepath;
    std::string ln;
    while (std::getline(f, ln)) {
        auto parts = split_ws(ln);
        if (parts.size() < 9 || !starts_with_digit(parts[0])) continue;
        int cid = std::stoi(parts[0]);
        if (cid < 0) continue;
        POI p;
        p.id = cid;
        p.x = std::strtod(parts[1].c_str(), nullptr);
        p.y = std::strtod(parts[2].c_str(), nullptr);
        p.duration = std::strtod(parts[3].c_str(), nullptr);
        p.score = (cid == 0) ? 0.0 : std::strtod(parts[4].c_str(), nullptr);
        p.open_time = std::strtod(parts[parts.size() - 2].c_str(), nullptr);
        p.close_time = std::strtod(parts[parts.size() - 1].c_str(), nullptr);
        inst.pois.push_back(p);
    }
    std::sort(inst.pois.begin(), inst.pois.end(), [](const POI& a, const POI& b) { return a.id < b.id; });
    inst.tmax = inst.pois[0].close_time;
    return inst;
}

inline std::string data_dir() {
    const char* env = std::getenv("GITC_DATA");
    if (env) return env;
    return "instances";   // run from the repo root, or set GITC_DATA
}

inline Instance load_instance(const std::string& name) {
    if (name.rfind("pr", 0) == 0) return parse_cordeau(data_dir() + "/cordeau/" + name + ".txt");
    return parse_solomon(data_dir() + "/solomon/" + name + ".txt");
}

// Jackson 2020 SIII.A: the 3 feasibility conditions for a candidate.
inline bool is_feasible(const POI& candidate, const POI& current, double current_time, const Instance& inst) {
    if (candidate.id == 0 || candidate.id == current.id) return false;
    double arrival = current_time + euclidean(current, candidate);
    if (arrival > candidate.close_time) return false;
    double start = std::max(arrival, candidate.open_time);
    double finish = start + candidate.duration;
    return finish + euclidean(candidate, inst.depot()) <= inst.tmax;
}
