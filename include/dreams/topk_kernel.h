#ifndef DREAMS_TOPK_KERNEL_H
#define DREAMS_TOPK_KERNEL_H

#include "config.h"

namespace dreams {

// ============================================================================
// GPU PARTIAL CRT SCORING
// ============================================================================
// Uses iterative CRT with K_small primes to reconstruct p and q into 256-bit
// integers, converts to float for ratio estimation, and computes delta proxy.

// Compute delta proxy from RNS representation using partial CRT
// This is the PRIMARY scoring method (not float shadow)
void compute_delta_proxy(
    double* deltas,                 // Output: absolute error [B]
    double* log_qs,                 // Output: log|q| estimates [B]
    const u32* P_rns,               // Trajectory in RNS: [K, B, m*m]
    const PrimeMeta* primes,        // Primes: [K]
    double target,                  // Target constant (e.g., pi)
    int K,                          // Total primes
    int B,                          // Batch size
    int m,                          // Matrix dimension
    int K_small,                    // Primes for partial CRT (default: 8)
    int p_row = 0,                  // Row for p in matrix
    int p_col = -1,                 // Col for p (-1 = last)
    int q_row = 1,                  // Row for q in matrix
    int q_col = -1                  // Col for q (-1 = last)
);

// Compute delta from float64 shadow run (OPTIONAL, for debugging only)
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

// ============================================================================
// TOPK SELECTION
// ============================================================================

// TopK selection on GPU
// Maintains top K candidates by smallest delta (absolute error)
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
    double delta_threshold          // Only keep if delta < threshold
);

// Merge TopK results from multiple batches
void topk_merge(
    Hit* global_topk,               // In/Out: [topk]
    int* global_num_hits,
    const Hit* batch_topk,          // Batch results: [topk]
    int batch_num_hits,
    int topk
);

// ============================================================================
// CPU VERIFICATION (for final topK hits)
// ============================================================================

// Full CRT reconstruction for final verification (CPU side)
// Uses all K primes for exact integer reconstruction
void crt_reconstruct_full_cpu(
    const u32* p_rns,               // p in RNS: [K]
    const u32* q_rns,               // q in RNS: [K]
    const PrimeMeta* primes,
    int K,
    double target,
    double* exact_delta,            // Output: exact delta value
    double* exact_log_q             // Output: exact log|q|
);

} // namespace dreams

#endif // DREAMS_TOPK_KERNEL_H
