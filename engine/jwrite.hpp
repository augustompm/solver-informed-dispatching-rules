// Compact JSON writing helpers: ", " and ": " separators, floats as shortest
// round-trip strings, and decimal round-half-even.
#pragma once
#include <charconv>
#include <cmath>
#include <string>
#include <vector>

// Decimal rounding to k places, half to even.
inline double round_half_even(double x, int k) {
    double m = std::pow(10.0, k);
    double v = x * m;
    double f = std::floor(v);
    double diff = v - f;
    double r;
    if (diff > 0.5) r = f + 1;
    else if (diff < 0.5) r = f;
    else r = (std::fmod(f, 2.0) == 0.0) ? f : f + 1;
    return r / m;
}

inline std::string jnum(double x) {
    if (x == (long long)x && std::abs(x) < 1e15) {
        // integral doubles print with a trailing .0
        std::string s = std::to_string((long long)x);
        return s + ".0";
    }
    char buf[32];
    auto r = std::to_chars(buf, buf + sizeof buf, x);
    return std::string(buf, r.ptr);
}

inline std::string jint(long long v) { return std::to_string(v); }

inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o + "\"";
}

inline std::string jlist(const std::vector<std::string>& parts) {
    std::string o = "[";
    for (size_t i = 0; i < parts.size(); i++) o += (i ? ", " : "") + parts[i];
    return o + "]";
}

inline std::string jstrlist(const std::vector<std::string>& v) {
    std::vector<std::string> p;
    for (auto& s : v) p.push_back(jstr(s));
    return jlist(p);
}
