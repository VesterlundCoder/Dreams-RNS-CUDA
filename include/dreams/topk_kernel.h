#ifndef DREAMS_TOPK_KERNEL_H
#define DREAMS_TOPK_KERNEL_H

#include "config.h"

namespace dreams {

// Compute delta proxy from RNS representation (partial CRT)
// Uses first K_small primes for rough BigInt estimation
void compute_delta_proxy(
    double* deltas,                 // Output: [B]
    double* log_qs,                 // Output: [B]
    const u32* p_rns,               // p values in RNS: [K, B]
    const u32* q_rns,               // q values in RNS: [K, B]
    const PrimeMeta* primes,        // Primes: [K]
    double target,                  // Target constant (e.g., pi)
    int K,
    int B,
    int K_small = 8                 // Primes to use for rough CRT
);

// Compute delta from float64 shadow run (faster but approximate)
void compute_delta_float(
    double* deltas,                 // Output: [B]
    const double* P_float,          // Trajectory: [B, m*m]
    const double* log_scale,        // Log scale: [B]
    double target,
    int m,
    int B,
    int p_row = 0,
    int p_col = -1,
    int q_row = 1,
    int q_col = -1
);

// TopK selection on GPU
// Maintains top K candidates by smallest delta
void topk_select(
    Hit* topk_hits,                 // Output: [topk]
    int* num_hits,                  // Output: number of hits found
    const double* deltas,           // Delta values: [B]
    const double* log_qs,           // Log q values: [B]
    const i32* shifts,              // Shifts: [B, dim]
    int cmf_idx,
    int depth,
    int dim,
    int B,
    int topk,
    double delta_threshold = 1e-6  // Only keep if delta < threshold
);

// Merge TopK results from multiple batches
void topk_merge(
    Hit* global_topk,               // In/Out: [topk]
    int* global_num_hits,
    const Hit* batch_topk,          // Batch results: [topk]
    int batch_num_hits,
    int topk
);

// Full CRT reconstruction for final verification (CPU)
void crt_reconstruct_verify(
    const u32* p_rns,               // p in RNS: [K]
    const u32* q_rns,               // q in RNS: [K]
    const PrimeMeta* primes,
    int K,
    double target,
    double* exact_delta             // Output: exact delta value
);

} // namespace dreams

#endif // DREAMS_TOPK_KERNEL_H
