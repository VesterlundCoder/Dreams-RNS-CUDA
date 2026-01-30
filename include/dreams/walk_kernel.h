#ifndef DREAMS_WALK_KERNEL_H
#define DREAMS_WALK_KERNEL_H

#include "config.h"

namespace dreams {

// Initialize trajectory matrix P to identity
void init_trajectory(
    u32* P_rns,                     // Output: [K, B, m*m]
    const PrimeMeta* primes,
    int m,
    int K,
    int B
);

// Update trajectory: P = P @ M (batched modular matrix multiply)
void update_trajectory(
    u32* P_rns,                     // In/Out: [K, B, m*m]
    const u32* M_rns,               // Step matrix: [K, B, m*m]
    const PrimeMeta* primes,
    int m,
    int K,
    int B
);

// Extract p and q from trajectory matrix (typically P[0,-1] and P[1,-1])
void extract_convergents(
    u32* p_rns,                     // Output: [K, B]
    u32* q_rns,                     // Output: [K, B]
    const u32* P_rns,               // Trajectory: [K, B, m*m]
    int m,
    int K,
    int B,
    int p_row = 0,
    int p_col = -1,                 // -1 means last column
    int q_row = 1,
    int q_col = -1
);

// Shadow run in float64 for quick delta estimation (parallel to RNS)
void update_trajectory_float(
    double* P_float,                // In/Out: [B, m*m]
    const double* M_float,          // Step matrix: [B, m*m]
    double* log_scale,              // Log scaling factor per shift: [B]
    int m,
    int B
);

// Run complete walk loop (persistent kernel approach)
void run_walk_loop(
    u32* P_rns,                     // Trajectory in RNS: [K, B, m*m]
    double* P_float,                // Shadow trajectory: [B, m*m] (optional)
    double* log_scale,              // Log scale: [B]
    const DeviceProgram* program,   // CMF program
    const i32* shifts,              // Shifts: [B, dim]
    const PrimeMeta* primes,        // Primes: [K]
    const WalkConfig& config,
    Hit* d_hits,                    // Output hits: [topk]
    int* d_num_hits                 // Number of hits found
);

} // namespace dreams

#endif // DREAMS_WALK_KERNEL_H
