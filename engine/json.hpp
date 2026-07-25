// Minimal JSON parser for the project's own files (route JSONs, seed and
// result files). Objects, arrays, strings, numbers, true/false/null. Unknown
// syntax is a hard error, never a default value.
#pragma once
#include <cctype>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct JValue;
using JPtr = std::shared_ptr<JValue>;

struct JValue {
    enum Kind { OBJ, ARR, STR, NUM, BOOL, NUL } kind = NUL;
    std::map<std::string, JPtr> obj;
    std::vector<JPtr> arr;
    std::string str;
    double num = 0.0;
    bool b = false;

    const JValue& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json key missing: " + k);
        return *it->second;
    }
    bool has(const std::string& k) const { return obj.count(k) > 0; }
    const JValue& operator[](size_t i) const { return *arr.at(i); }
    size_t size() const { return arr.size(); }
    long long as_int() const { return (long long)num; }
};

struct JParser {
    const std::string& s;
    size_t p = 0;
    explicit JParser(const std::string& s_) : s(s_) {}

    void ws() { while (p < s.size() && std::isspace((unsigned char)s[p])) p++; }
    char peek() { ws(); if (p >= s.size()) throw std::runtime_error("json eof"); return s[p]; }

    JPtr parse() {
        char c = peek();
        if (c == '{') return obj();
        if (c == '[') return arr();
        if (c == '"') return str();
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') { p += 4; return std::make_shared<JValue>(); }
        return num();
    }
    JPtr obj() {
        auto v = std::make_shared<JValue>();
        v->kind = JValue::OBJ;
        p++;   // {
        if (peek() == '}') { p++; return v; }
        while (true) {
            JPtr k = str();
            ws();
            if (s[p] != ':') throw std::runtime_error("json ':' expected");
            p++;
            v->obj[k->str] = parse();
            char c = peek();
            p++;
            if (c == '}') return v;
            if (c != ',') throw std::runtime_error("json ',' expected in object");
        }
    }
    JPtr arr() {
        auto v = std::make_shared<JValue>();
        v->kind = JValue::ARR;
        p++;   // [
        if (peek() == ']') { p++; return v; }
        while (true) {
            v->arr.push_back(parse());
            char c = peek();
            p++;
            if (c == ']') return v;
            if (c != ',') throw std::runtime_error("json ',' expected in array");
        }
    }
    JPtr str() {
        ws();
        if (s[p] != '"') throw std::runtime_error("json string expected");
        p++;
        auto v = std::make_shared<JValue>();
        v->kind = JValue::STR;
        while (s[p] != '"') {
            if (s[p] == '\\') {
                p++;
                char c = s[p++];
                if (c == 'n') v->str += '\n';
                else if (c == 't') v->str += '\t';
                else if (c == 'u') { v->str += '?'; p += 4; }   // not used by our files
                else v->str += c;
            } else v->str += s[p++];
        }
        p++;
        return v;
    }
    JPtr num() {
        ws();
        size_t start = p;
        while (p < s.size() && (std::isdigit((unsigned char)s[p]) || s[p] == '-' || s[p] == '+'
                                || s[p] == '.' || s[p] == 'e' || s[p] == 'E')) p++;
        auto v = std::make_shared<JValue>();
        v->kind = JValue::NUM;
        v->num = std::strtod(s.substr(start, p - start).c_str(), nullptr);
        return v;
    }
    JPtr boolean() {
        auto v = std::make_shared<JValue>();
        v->kind = JValue::BOOL;
        if (s[p] == 't') { v->b = true; p += 4; }
        else { v->b = false; p += 5; }
        return v;
    }
};

inline JPtr json_parse(const std::string& text) {
    JParser jp(text);
    return jp.parse();
}
