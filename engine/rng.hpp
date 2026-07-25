// Mersenne Twister MT19937 (Matsumoto & Nishimura 1998) with array seeding,
// 53-bit uniform doubles, bounded draws by rejection sampling, sampling of
// distinct indices, Fisher-Yates shuffle and Box-Muller gaussians with the
// spare value cached. Every stochastic draw of the engine flows through this
// class; tools/verify_rng checks the stream against recorded fixtures.
#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

class Mt19937 {
    static constexpr int N = 624, M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu, UPPER = 0x80000000u, LOWER = 0x7fffffffu;
    uint32_t mt[N];
    int mti = N + 1;
    bool has_gauss = false;
    double gauss_next = 0.0;

    void init_genrand(uint32_t s) {
        mt[0] = s;
        for (mti = 1; mti < N; mti++)
            mt[mti] = 1812433253u * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + (uint32_t)mti;
    }

    void init_by_array(const std::vector<uint32_t>& key) {
        init_genrand(19650218u);
        size_t i = 1, j = 0;
        size_t k = (N > key.size()) ? N : key.size();
        for (; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525u)) + key[j] + (uint32_t)j;
            i++; j++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            if (j >= key.size()) j = 0;
        }
        for (k = N - 1; k; k--) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941u)) - (uint32_t)i;
            i++;
            if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
        }
        mt[0] = 0x80000000u;
    }

public:
    // Seed from a non-negative integer: key = 32-bit words, LSB first.
    explicit Mt19937(uint64_t seed) {
        std::vector<uint32_t> key;
        if (seed == 0) key.push_back(0);
        while (seed) { key.push_back((uint32_t)(seed & 0xffffffffu)); seed >>= 32; }
        init_by_array(key);
    }

    uint32_t genrand() {
        uint32_t y;
        if (mti >= N) {
            for (int kk = 0; kk < N - M; kk++) {
                y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (int kk = N - M; kk < N - 1; kk++) {
                y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            mti = 0;
        }
        y = mt[mti++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // 53-bit double in [0,1).
    double random() {
        uint32_t a = genrand() >> 5, b = genrand() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // k in 1..32.
    uint32_t getrandbits(int k) { return genrand() >> (32 - k); }

    // Uniform draw below n by rejection on the top bits.
    uint64_t randbelow(uint64_t n) {
        if (n == 0) return 0;
        int k = std::bit_width(n);           // == n.bit_length()
        uint64_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // Uniform index into a sequence of the given length.
    size_t choice_index(size_t len) { return (size_t)randbelow(len); }

    // Uniform integer in [a, b].
    long long randint(long long a, long long b) { return a + (long long)randbelow((uint64_t)(b - a + 1)); }

    // k distinct indices from [0, n): pool copy for small n, selection set otherwise.
    std::vector<size_t> sample(size_t n, size_t k) {
        std::vector<size_t> result(k);
        size_t setsize = 21;
        if (k > 5) setsize += (size_t)std::pow(4.0, std::ceil(std::log((double)(k * 3)) / std::log(4.0)));
        if (n <= setsize) {
            std::vector<size_t> pool(n);
            for (size_t i = 0; i < n; i++) pool[i] = i;
            for (size_t i = 0; i < k; i++) {
                size_t j = (size_t)randbelow(n - i);
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            std::unordered_set<size_t> selected;
            for (size_t i = 0; i < k; i++) {
                size_t j = (size_t)randbelow(n);
                while (selected.count(j)) j = (size_t)randbelow(n);
                selected.insert(j);
                result[i] = j;
            }
        }
        return result;
    }

    // Fisher-Yates shuffle, high index down.
    template <class T> void shuffle(std::vector<T>& x) {
        for (size_t i = x.size() - 1; i > 0; i--) {
            size_t j = (size_t)randbelow(i + 1);
            std::swap(x[i], x[j]);
        }
    }

    // Gaussian by the Box-Muller trigonometric pair; the second value is cached.
    double gauss(double mu, double sigma) {
        double z;
        if (has_gauss) { z = gauss_next; has_gauss = false; }
        else {
            constexpr double TWOPI = 2.0 * 3.14159265358979323846;
            double x2pi = random() * TWOPI;
            double g2rad = std::sqrt(-2.0 * std::log(1.0 - random()));
            z = std::cos(x2pi) * g2rad;
            gauss_next = std::sin(x2pi) * g2rad;
            has_gauss = true;
        }
        return mu + z * sigma;
    }
};
