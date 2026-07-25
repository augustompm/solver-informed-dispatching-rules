// The GPU factory: P formulas x N scenarios per launch, one thread = one
// complete stochastic route simulation. Variable-size formulas concatenated
// into flat opcode buffers with offsets/lengths; instance arrays resident on
// the device (WarpDrive's one-time data transfer; EvoGP's full-kernel
// residency); the seed-batched grouped kernel fills the GPU with G groups x
// population trees x T scenarios.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include "sim_core.hpp"
#include "sim_host.hpp"
#include "vocab.hpp"

#define CUDA_CHECK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

struct PackedTrees {
    std::vector<int32_t> ops, args;      // args flat: 2 per node
    std::vector<int32_t> offsets, lengths;
};

inline PackedTrees pack_trees(const std::vector<NodeP>& trees) {
    PackedTrees pk;
    int32_t off = 0;
    for (auto& t : trees) {
        Compiled c = compile_rmod(t);
        pk.offsets.push_back(off);
        pk.lengths.push_back((int32_t)c.ops.size());
        for (size_t k = 0; k < c.ops.size(); k++) {
            pk.ops.push_back(c.ops[k]);
            pk.args.push_back(c.args[k][0]);
            pk.args.push_back(c.args[k][1]);
        }
        off += (int32_t)c.ops.size();
    }
    return pk;
}

__global__ static void factory_kernel(const int32_t* all_ops, const int32_t* all_args,
                                      const int32_t* offsets, const int32_t* lengths,
                                      int P, int N,
                                      const double* scores, const double* durations,
                                      const double* open_t, const double* close_t,
                                      const double* dist, int ld, const double* dist_to_depot,
                                      double tmax, const double* all_sampled, int m, int n,
                                      const int64_t* term_mask, const double* zp, double* out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)P * N) return;
    int p = (int)(idx / N), s = (int)(idx % N);
    bool visited[MAX_N];
    double terms[N_TERMS], stack[MAX_STACK], struct_buf[4];
    bool scratch[MAX_N];
    int off = offsets[p], L = lengths[p];
    out[(size_t)p * N + s] = sim_one(all_ops + off, all_args + 2 * (size_t)off, L,
                                     scores, durations, open_t, close_t, dist, ld,
                                     dist_to_depot, tmax, all_sampled + (size_t)s * n, m, n,
                                     term_mask, zp, visited, terms, stack, struct_buf, scratch);
}

// Seed batching: tree p belongs to group p/group_size and is evaluated only
// on its group's scenarios.
__global__ static void grouped_kernel(const int32_t* all_ops, const int32_t* all_args,
                                      const int32_t* offsets, const int32_t* lengths,
                                      int P, int group_size, int T,
                                      const double* scores, const double* durations,
                                      const double* open_t, const double* close_t,
                                      const double* dist, int ld, const double* dist_to_depot,
                                      double tmax, const double* grouped_sampled, int m, int n,
                                      const int64_t* term_mask, const double* zp, double* out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)P * T) return;
    int p = (int)(idx / T), s = (int)(idx % T);
    int g = p / group_size;
    bool visited[MAX_N];
    double terms[N_TERMS], stack[MAX_STACK], struct_buf[4];
    bool scratch[MAX_N];
    int off = offsets[p], L = lengths[p];
    out[(size_t)p * T + s] = sim_one(all_ops + off, all_args + 2 * (size_t)off, L,
                                     scores, durations, open_t, close_t, dist, ld,
                                     dist_to_depot, tmax,
                                     grouped_sampled + ((size_t)g * T + s) * n, m, n,
                                     term_mask, zp, visited, terms, stack, struct_buf, scratch);
}

// As grouped_kernel, with one terminal mask per block of mask_group trees:
// tree p reads scenarios of group p/group_size and mask p/mask_group.
__global__ static void grouped_masks_kernel(const int32_t* all_ops, const int32_t* all_args,
                                            const int32_t* offsets, const int32_t* lengths,
                                            int P, int group_size, int mask_group, int T,
                                            const double* scores, const double* durations,
                                            const double* open_t, const double* close_t,
                                            const double* dist, int ld, const double* dist_to_depot,
                                            double tmax, const double* grouped_sampled, int m, int n,
                                            const int64_t* masks, const double* zp, double* out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)P * T) return;
    int p = (int)(idx / T), s = (int)(idx % T);
    int g = p / group_size;
    const int64_t* tm = masks + (size_t)(p / mask_group) * N_TERMS;
    bool visited[MAX_N];
    double terms[N_TERMS], stack[MAX_STACK], struct_buf[4];
    bool scratch[MAX_N];
    int off = offsets[p], L = lengths[p];
    out[(size_t)p * T + s] = sim_one(all_ops + off, all_args + 2 * (size_t)off, L,
                                     scores, durations, open_t, close_t, dist, ld,
                                     dist_to_depot, tmax,
                                     grouped_sampled + ((size_t)g * T + s) * n, m, n,
                                     tm, zp, visited, terms, stack, struct_buf, scratch);
}

template <class T>
static T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// Instance arrays are copied once; per evaluate() only the formulas, the
// scenarios and the mask move.
struct FactoryContext {
    const SimData& sd;
    double *d_sc, *d_du, *d_ot, *d_ct, *d_dm, *d_d2, *d_zp;
    std::vector<double> zp_host;

    FactoryContext(const SimData& sd_, const std::vector<double>& zp) : sd(sd_), zp_host(zp) {
        d_sc = to_device(sd.scores);
        d_du = to_device(sd.durations);
        d_ot = to_device(sd.open_t);
        d_ct = to_device(sd.close_t);
        d_dm = to_device(sd.dist);
        d_d2 = to_device(sd.dist_to_depot);
        d_zp = to_device(zp);
    }
    ~FactoryContext() {
        cudaFree(d_sc); cudaFree(d_du); cudaFree(d_ot); cudaFree(d_ct);
        cudaFree(d_dm); cudaFree(d_d2); cudaFree(d_zp);
    }

    // out[p*N+s]; all_sampled = N scenarios x n durations.
    std::vector<double> evaluate(const PackedTrees& pk, const std::vector<double>& all_sampled,
                                 int N, const std::vector<int64_t>& term_mask, int threads = 128) {
        int P = (int)pk.offsets.size();
        int32_t *d_ops = to_device(pk.ops), *d_args = to_device(pk.args);
        int32_t *d_off = to_device(pk.offsets), *d_len = to_device(pk.lengths);
        double* d_sp = to_device(all_sampled);
        int64_t* d_mask = to_device(term_mask);
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)P * N * sizeof(double)));
        long long tot = (long long)P * N;
        int blocks = (int)((tot + threads - 1) / threads);
        factory_kernel<<<blocks, threads>>>(d_ops, d_args, d_off, d_len, P, N,
                                            d_sc, d_du, d_ot, d_ct, d_dm, sd.ld, d_d2, sd.tmax,
                                            d_sp, sd.m, sd.n, d_mask, d_zp, d_out);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> out((size_t)P * N);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_ops); cudaFree(d_args); cudaFree(d_off); cudaFree(d_len);
        cudaFree(d_sp); cudaFree(d_mask); cudaFree(d_out);
        return out;
    }

    // grouped_sampled = G groups x T scenarios x n; tree p -> group p/group_size.
    std::vector<double> evaluate_grouped(const PackedTrees& pk, const std::vector<double>& grouped_sampled,
                                         int G, int T, int group_size,
                                         const std::vector<int64_t>& term_mask, int threads = 128) {
        (void)G;
        int P = (int)pk.offsets.size();
        int32_t *d_ops = to_device(pk.ops), *d_args = to_device(pk.args);
        int32_t *d_off = to_device(pk.offsets), *d_len = to_device(pk.lengths);
        double* d_sp = to_device(grouped_sampled);
        int64_t* d_mask = to_device(term_mask);
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)P * T * sizeof(double)));
        long long tot = (long long)P * T;
        int blocks = (int)((tot + threads - 1) / threads);
        grouped_kernel<<<blocks, threads>>>(d_ops, d_args, d_off, d_len, P, group_size, T,
                                            d_sc, d_du, d_ot, d_ct, d_dm, sd.ld, d_d2, sd.tmax,
                                            d_sp, sd.m, sd.n, d_mask, d_zp, d_out);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> out((size_t)P * T);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_ops); cudaFree(d_args); cudaFree(d_off); cudaFree(d_len);
        cudaFree(d_sp); cudaFree(d_mask); cudaFree(d_out);
        return out;
    }

    // Batched variant: masks holds one N_TERMS row per block of mask_group
    // trees, so genomes with different vocabularies share a single launch.
    std::vector<double> evaluate_grouped_masks(const PackedTrees& pk,
                                               const std::vector<double>& grouped_sampled,
                                               int T, int group_size, int mask_group,
                                               const std::vector<int64_t>& masks, int threads = 128) {
        int P = (int)pk.offsets.size();
        int32_t *d_ops = to_device(pk.ops), *d_args = to_device(pk.args);
        int32_t *d_off = to_device(pk.offsets), *d_len = to_device(pk.lengths);
        double* d_sp = to_device(grouped_sampled);
        int64_t* d_mask = to_device(masks);
        double* d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out, (size_t)P * T * sizeof(double)));
        long long tot = (long long)P * T;
        int blocks = (int)((tot + threads - 1) / threads);
        grouped_masks_kernel<<<blocks, threads>>>(d_ops, d_args, d_off, d_len, P, group_size,
                                                  mask_group, T,
                                                  d_sc, d_du, d_ot, d_ct, d_dm, sd.ld, d_d2, sd.tmax,
                                                  d_sp, sd.m, sd.n, d_mask, d_zp, d_out);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> out((size_t)P * T);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(d_ops); cudaFree(d_args); cudaFree(d_off); cudaFree(d_len);
        cudaFree(d_sp); cudaFree(d_mask); cudaFree(d_out);
        return out;
    }
};
