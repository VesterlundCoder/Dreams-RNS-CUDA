#include "dreams/config.h"
#include "dreams/cmf_program.h"
#include "dreams/eval_kernel.h"
#include "dreams/walk_kernel.h"
#include "dreams/topk_kernel.h"
#include <vector>
#include <cmath>

#ifdef DREAMS_HAS_GPU
#include <cuda_runtime.h>

namespace dreams {

// Constant memory for program and primes
__constant__ DeviceProgram d_program;
__constant__ PrimeMeta d_primes[MAX_K];

// ============================================================================
// MODULAR ARITHMETIC PRIMITIVES (SAFE IMPLEMENTATIONS)
// ============================================================================

// Barrett reduction: compute a mod p using precomputed mu
// INVARIANT: a < p * 2^32 (satisfied when a is product of two u32 residues)
__device__ __forceinline__ u32 barrett_reduce(u64 a, u32 p, u64 mu) {
    u64 q = __umul64hi(a, mu);
    u64 r = a - q * p;
    return (r >= p) ? (u32)(r - p) : (u32)r;
}

// Modular addition: (a + b) mod p where a, b < p
__device__ __forceinline__ u32 mod_add(u32 a, u32 b, u32 p) {
    u64 sum = (u64)a + (u64)b;  // Safe: max 2^32 - 2
    return (sum >= p) ? (u32)(sum - p) : (u32)sum;
}

// Modular subtraction: (a - b) mod p where a, b < p
__device__ __forceinline__ u32 mod_sub(u32 a, u32 b, u32 p) {
    return (a >= b) ? (a - b) : (p - b + a);
}

// Modular multiplication with Barrett: (a * b) mod p
__device__ __forceinline__ u32 mod_mul(u32 a, u32 b, u32 p, u64 mu) {
    return barrett_reduce((u64)a * (u64)b, p, mu);
}

// Modular negation: (-a) mod p
__device__ __forceinline__ u32 mod_neg(u32 a, u32 p) {
    return (a == 0) ? 0 : p - a;
}

// Extended GCD for modular inverse
// Returns 0 if a == 0 (caller should mark lane as dead)
__device__ u32 mod_inv(u32 a, u32 p, bool* alive) {
    if (a == 0) {
        *alive = false;
        return 0;
    }
    i32 t = 0, newt = 1;
    i32 r = (i32)p, newr = (i32)a;
    while (newr != 0) {
        i32 quotient = r / newr;
        i32 tmp = t - quotient * newt;
        t = newt; newt = tmp;
        tmp = r - quotient * newr;
        r = newr; newr = tmp;
    }
    return (t < 0) ? (u32)(t + (i32)p) : (u32)t;
}

// Overload without alive tracking (for compatibility)
__device__ u32 mod_inv(u32 a, u32 p) {
    bool dummy = true;
    return mod_inv(a, p, &dummy);
}

// Evaluate bytecode program for one (prime, shift) pair
// Returns m*m matrix entries
__device__ void eval_program_single(
    u32* out,                       // Output: m*m entries
    const DeviceProgram* prog,
    const i32* shift,               // Shift for this batch element [dim]
    int step_n,                     // Current step
    int k,                          // Prime index
    u32 p,
    u64 mu
) {
    u32 regs[MAX_REGS];
    int m = prog->m;
    
    // Initialize output to zero
    for (int i = 0; i < m * m; i++) out[i] = 0;
    
    // Execute bytecode
    for (int i = 0; i < prog->n_opcodes; i++) {
        const Instr& instr = prog->opcodes[i];
        switch (instr.op) {
            case Opcode::NOP:
                break;
            case Opcode::LOAD_X: {
                // axis value = shift[axis] + step_n * direction[axis]
                int axis = instr.arg0;
                i64 val = (i64)shift[axis] + (i64)step_n * prog->directions[axis];
                // Reduce to positive mod p
                val = ((val % (i64)p) + p) % p;
                regs[instr.arg1] = (u32)val;
                break;
            }
            case Opcode::LOAD_C: {
                regs[instr.arg1] = prog->constants_rns[k][instr.arg0];
                break;
            }
            case Opcode::LOAD_N: {
                regs[instr.arg0] = step_n % p;
                break;
            }
            case Opcode::ADD: {
                regs[instr.arg2] = mod_add(regs[instr.arg0], regs[instr.arg1], p);
                break;
            }
            case Opcode::SUB: {
                regs[instr.arg2] = mod_sub(regs[instr.arg0], regs[instr.arg1], p);
                break;
            }
            case Opcode::MUL: {
                regs[instr.arg2] = mod_mul(regs[instr.arg0], regs[instr.arg1], p, mu);
                break;
            }
            case Opcode::NEG: {
                regs[instr.arg1] = mod_neg(regs[instr.arg0], p);
                break;
            }
            case Opcode::POW2: {
                u32 a = regs[instr.arg0];
                regs[instr.arg1] = mod_mul(a, a, p, mu);
                break;
            }
            case Opcode::POW3: {
                u32 a = regs[instr.arg0];
                u32 a2 = mod_mul(a, a, p, mu);
                regs[instr.arg1] = mod_mul(a2, a, p, mu);
                break;
            }
            case Opcode::INV: {
                regs[instr.arg1] = mod_inv(regs[instr.arg0], p);
                break;
            }
            case Opcode::STORE: {
                int idx = instr.arg1 * m + instr.arg2;
                out[idx] = regs[instr.arg0];
                break;
            }
            case Opcode::END:
                return;
        }
    }
}

// ============================================================================
// SAFE MODULAR MATRIX MULTIPLY
// ============================================================================
// CRITICAL: For m > 4, we MUST reduce after each multiply-add to prevent
// u64 overflow. For m <= 4, we can accumulate then reduce once at the end.
// Worst case without reduction: m * (p-1)^2 ≈ m * 2^62
// For m=4: 4 * 2^62 = 2^64 (just barely safe)
// For m=6: 6 * 2^62 > 2^64 (OVERFLOW - must reduce per iteration)
// ============================================================================

// 2x2 modular matrix multiply (m=2, safe: 2 * 2^62 < 2^64)
__device__ void matmul_2x2_mod(
    u32* C,
    const u32* A,
    const u32* B,
    u32 p,
    u64 mu
) {
    u32 tmp[4];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            u64 acc = 0;
            #pragma unroll
            for (int k = 0; k < 2; k++) {
                acc += (u64)A[i * 2 + k] * (u64)B[k * 2 + j];
            }
            tmp[i * 2 + j] = barrett_reduce(acc, p, mu);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) C[i] = tmp[i];
}

// 4x4 modular matrix multiply (m=4, safe: 4 * 2^62 = 2^64)
__device__ void matmul_4x4_mod(
    u32* C,
    const u32* A,
    const u32* B,
    u32 p,
    u64 mu
) {
    u32 tmp[16];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            u64 acc = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                acc += (u64)A[i * 4 + k] * (u64)B[k * 4 + j];
            }
            tmp[i * 4 + j] = barrett_reduce(acc, p, mu);
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) C[i] = tmp[i];
}

// 6x6 modular matrix multiply (m=6, UNSAFE without per-step reduction)
// Must reduce after each multiply-add
__device__ void matmul_6x6_mod(
    u32* C,
    const u32* A,
    const u32* B,
    u32 p,
    u64 mu
) {
    u32 tmp[36];
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        #pragma unroll
        for (int j = 0; j < 6; j++) {
            u64 acc = 0;
            #pragma unroll
            for (int k = 0; k < 6; k++) {
                // Reduce product before accumulating to prevent overflow
                u32 prod = mod_mul(A[i * 6 + k], B[k * 6 + j], p, mu);
                acc += prod;
            }
            // acc <= 6 * (p-1) < 6 * 2^31 < 2^34, safe for final reduction
            tmp[i * 6 + j] = (u32)(acc % p);
        }
    }
    #pragma unroll
    for (int i = 0; i < 36; i++) C[i] = tmp[i];
}

// 8x8 modular matrix multiply (m=8, UNSAFE without per-step reduction)
__device__ void matmul_8x8_mod(
    u32* C,
    const u32* A,
    const u32* B,
    u32 p,
    u64 mu
) {
    u32 tmp[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            u64 acc = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                u32 prod = mod_mul(A[i * 8 + k], B[k * 8 + j], p, mu);
                acc += prod;
            }
            tmp[i * 8 + j] = (u32)(acc % p);
        }
    }
    #pragma unroll
    for (int i = 0; i < 64; i++) C[i] = tmp[i];
}

// General m x m modular matrix multiply (safe for any m)
__device__ void matmul_mod(
    u32* C,
    const u32* A,
    const u32* B,
    int m,
    u32 p,
    u64 mu
) {
    u32 tmp[MAX_M * MAX_M];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            u64 acc = 0;
            for (int k = 0; k < m; k++) {
                // Always reduce per-step for safety in general case
                u32 prod = mod_mul(A[i * m + k], B[k * m + j], p, mu);
                acc += prod;
            }
            // acc <= m * (p-1) < m * 2^31, safe to reduce
            tmp[i * m + j] = (u32)(acc % p);
        }
    }
    for (int i = 0; i < m * m; i++) C[i] = tmp[i];
}

// Initialize identity matrix in RNS
__device__ void init_identity_mod(u32* P, int m, u32 p) {
    for (int i = 0; i < m * m; i++) P[i] = 0;
    for (int i = 0; i < m; i++) P[i * m + i] = 1;
}

// ============================================================================
// SHADOW-FLOAT EVALUATION (mirrors RNS evaluation but uses double)
// ============================================================================

// Evaluate bytecode program in float mode (for shadow trajectory)
__device__ void eval_program_float(
    double* out,                    // Output: m*m entries
    const DeviceProgram* prog,
    const i32* shift,               // Shift for this batch element [dim]
    int step_n                      // Current step
) {
    double regs[MAX_REGS];
    int m = prog->m;
    
    // Initialize output to zero
    for (int i = 0; i < m * m; i++) out[i] = 0.0;
    
    // Execute bytecode in float mode
    for (int i = 0; i < prog->n_opcodes; i++) {
        const Instr& instr = prog->opcodes[i];
        switch (instr.op) {
            case Opcode::NOP:
                break;
            case Opcode::LOAD_X: {
                int axis = instr.arg0;
                double val = (double)shift[axis] + (double)step_n * prog->directions[axis];
                regs[instr.arg1] = val;
                break;
            }
            case Opcode::LOAD_C: {
                // Constants stored as integers, convert to float
                regs[instr.arg1] = (double)prog->constants[instr.arg0];
                break;
            }
            case Opcode::LOAD_N: {
                regs[instr.arg0] = (double)step_n;
                break;
            }
            case Opcode::ADD: {
                regs[instr.arg2] = regs[instr.arg0] + regs[instr.arg1];
                break;
            }
            case Opcode::SUB: {
                regs[instr.arg2] = regs[instr.arg0] - regs[instr.arg1];
                break;
            }
            case Opcode::MUL: {
                regs[instr.arg2] = regs[instr.arg0] * regs[instr.arg1];
                break;
            }
            case Opcode::NEG: {
                regs[instr.arg1] = -regs[instr.arg0];
                break;
            }
            case Opcode::POW2: {
                double a = regs[instr.arg0];
                regs[instr.arg1] = a * a;
                break;
            }
            case Opcode::POW3: {
                double a = regs[instr.arg0];
                regs[instr.arg1] = a * a * a;
                break;
            }
            case Opcode::INV: {
                double a = regs[instr.arg0];
                regs[instr.arg1] = (a != 0.0) ? (1.0 / a) : 0.0;
                break;
            }
            case Opcode::STORE: {
                int idx = instr.arg1 * m + instr.arg2;
                out[idx] = regs[instr.arg0];
                break;
            }
            case Opcode::END:
                return;
        }
    }
}

// Float matrix multiply (for shadow trajectory)
__device__ void matmul_float(double* C, const double* A, const double* B, int m) {
    double tmp[MAX_M * MAX_M];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < m; k++) {
                acc += A[i * m + k] * B[k * m + j];
            }
            tmp[i * m + j] = acc;
        }
    }
    for (int i = 0; i < m * m; i++) C[i] = tmp[i];
}

// Initialize identity matrix (float)
__device__ void init_identity_float(double* P, int m) {
    for (int i = 0; i < m * m; i++) P[i] = 0.0;
    for (int i = 0; i < m; i++) P[i * m + i] = 1.0;
}

// Normalize float matrix and return max abs value
__device__ double normalize_matrix(double* P, int m) {
    double max_val = 0.0;
    for (int i = 0; i < m * m; i++) {
        double absv = fabs(P[i]);
        if (absv > max_val) max_val = absv;
    }
    if (max_val > 1.0) {
        for (int i = 0; i < m * m; i++) {
            P[i] /= max_val;
        }
    }
    return max_val;
}

// ============================================================================
// DREAMS DELTA COMPUTATION (shadow-float based)
// ============================================================================
// delta = -(1 + log(|err|) / log(|q|))
// where err = |est - target|, est = p/q from trajectory matrix

__device__ double compute_dreams_delta(
    const double* P_float,          // Trajectory matrix [m*m]
    double log_scale,               // Accumulated log scale
    int m,
    double target,
    double* out_log_q               // Output: log|q|
) {
    // Extract p and q from trajectory (convention: p = P[0, m-1], q = P[1, m-1])
    double p_val = P_float[0 * m + (m - 1)];
    double q_val = P_float[1 * m + (m - 1)];
    
    // Guard against invalid values
    if (!isfinite(p_val) || !isfinite(q_val) || fabs(q_val) < 1e-300) {
        *out_log_q = 0.0;
        return -1e10;  // Invalid, will be filtered
    }
    
    // Compute estimate
    double est = p_val / q_val;
    double abs_err = fabs(est - target);
    
    // Compute log|q| including accumulated scale
    double log_abs_q = log_scale + log(fabs(q_val));
    *out_log_q = log_abs_q;
    
    // Guard against edge cases
    if (abs_err < 1e-300 || log_abs_q <= 0.0) {
        return (abs_err < 1e-300) ? 100.0 : -1e10;  // Perfect match or invalid
    }
    
    // Dreams delta formula
    double delta = -(1.0 + log(abs_err) / log_abs_q);
    
    return delta;
}

// ============================================================================
// PERSISTENT KERNEL: Runs entire walk on GPU with shadow-float
// ============================================================================
// Key insight: RNS trajectory is per-(k,b), but shadow-float is per-b only.
// Only k==0 threads update the shadow-float trajectory.

#define NORMALIZE_EVERY 50  // Normalize shadow-float every N steps

__global__ void k_persistent_walk(
    u32* P_rns,                     // RNS trajectory: [K, B, m*m]
    double* P_float,                // Shadow trajectory: [B, m*m]
    double* log_scale,              // Log scale: [B]
    double* deltas,                 // Output Dreams delta: [B]
    double* log_qs,                 // Output log|q|: [B]
    const i32* shifts,              // Shifts: [B, dim]
    int K, int B, int m, int dim,
    int depth,
    double target,
    int snapshot_depth_lo,          // First snapshot (e.g., 200)
    int snapshot_depth_hi           // Second snapshot (e.g., 2000)
) {
    // Thread indexing: parallelize over (k, b) pairs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = tid / B;                // Prime index
    int b = tid % B;                // Batch index
    
    if (k >= K || b >= B) return;
    
    u32 p = d_primes[k].p;
    u64 mu = d_primes[k].mu;
    
    // Local storage for RNS
    u32 P_local[MAX_M * MAX_M];
    u32 M_local[MAX_M * MAX_M];
    
    // Local storage for shadow-float (only used by k==0)
    double P_float_local[MAX_M * MAX_M];
    double M_float_local[MAX_M * MAX_M];
    double local_log_scale = 0.0;
    
    // Get shift for this batch element
    i32 shift_local[MAX_AXES];
    for (int d = 0; d < dim; d++) {
        shift_local[d] = shifts[b * dim + d];
    }
    
    // Initialize P to identity (RNS)
    init_identity_mod(P_local, m, p);
    
    // Initialize shadow-float (only k==0)
    if (k == 0) {
        init_identity_float(P_float_local, m);
    }
    
    // Main walk loop
    for (int step = 0; step < depth; step++) {
        // ===== RNS PATH (all k threads) =====
        // Evaluate step matrix M in RNS
        eval_program_single(M_local, &d_program, shift_local, step, k, p, mu);
        
        // Update RNS trajectory: P = P @ M
        switch (m) {
            case 2:
                matmul_2x2_mod(P_local, P_local, M_local, p, mu);
                break;
            case 4:
                matmul_4x4_mod(P_local, P_local, M_local, p, mu);
                break;
            case 6:
                matmul_6x6_mod(P_local, P_local, M_local, p, mu);
                break;
            case 8:
                matmul_8x8_mod(P_local, P_local, M_local, p, mu);
                break;
            default:
                matmul_mod(P_local, P_local, M_local, m, p, mu);
                break;
        }
        
        // ===== SHADOW-FLOAT PATH (only k==0) =====
        if (k == 0) {
            // Evaluate step matrix M in float
            eval_program_float(M_float_local, &d_program, shift_local, step);
            
            // Update shadow trajectory: P_float = P_float @ M_float
            matmul_float(P_float_local, P_float_local, M_float_local, m);
            
            // Normalize periodically to prevent overflow
            if ((step + 1) % NORMALIZE_EVERY == 0) {
                double scale = normalize_matrix(P_float_local, m);
                if (scale > 1.0) {
                    local_log_scale += log(scale);
                }
            }
            
            // Compute Dreams delta at snapshot depths
            if (step + 1 == snapshot_depth_lo || step + 1 == snapshot_depth_hi) {
                double log_q;
                double delta = compute_dreams_delta(P_float_local, local_log_scale, 
                                                    m, target, &log_q);
                
                // Store (only at final snapshot or if this is better)
                if (step + 1 == snapshot_depth_hi) {
                    deltas[b] = delta;
                    log_qs[b] = log_q;
                }
            }
        }
    }
    
    // Write final RNS P back to global memory (for optional CRT verification)
    int mm = m * m;
    for (int i = 0; i < mm; i++) {
        P_rns[k * B * mm + b * mm + i] = P_local[i];
    }
    
    // Write final shadow-float state (only k==0)
    if (k == 0) {
        for (int i = 0; i < mm; i++) {
            P_float[b * mm + i] = P_float_local[i];
        }
        log_scale[b] = local_log_scale;
    }
}

// ============================================================================
// GPU PARTIAL CRT RECONSTRUCTION (256-bit)
// ============================================================================
// Reconstructs integer from residues using iterative CRT (Chinese Remainder Theorem)
// Result is stored in U256 (4 x u64 limbs)

// Precomputed constants for 256-bit mod p operations
struct CrtConstants {
    u64 R1;  // 2^64 mod p
    u64 R2;  // 2^128 mod p
    u64 R3;  // 2^192 mod p
};

// Compute x mod p where x is a 256-bit integer (4 limbs)
__device__ __forceinline__ u32 u256_mod_p(const U256& x, u32 p, const CrtConstants& c) {
    // x mod p = (x0 + R1*x1 + R2*x2 + R3*x3) mod p
    u64 acc = x.limbs[0] % p;
    acc += (c.R1 * (x.limbs[1] % p)) % p;
    acc += (c.R2 * (x.limbs[2] % p)) % p;
    acc += (c.R3 * (x.limbs[3] % p)) % p;
    return (u32)(acc % p);
}

// Add U256 + u64 * u64 (x = x + M * t where M and t are both < 2^64 for simplicity)
// For full 256-bit M, we need multi-precision arithmetic
__device__ void u256_add_mul(U256& x, const U256& M, u64 t) {
    // Compute M * t (256-bit * 64-bit = 320-bit, but we'll truncate to 256)
    // For simplicity with K_small <= 8, M stays within 256 bits
    
    // Multiply M by t and add to x
    u64 carry = 0;
    for (int i = 0; i < 4; i++) {
        // Compute M.limbs[i] * t + carry + x.limbs[i]
        // This needs 128-bit intermediate
        u64 lo = M.limbs[i] * t;  // Low 64 bits of product
        u64 hi = __umul64hi(M.limbs[i], t);  // High 64 bits
        
        // Add to x with carry propagation
        u64 sum = x.limbs[i] + lo + carry;
        carry = hi + (sum < x.limbs[i] ? 1 : 0) + (sum < lo ? 1 : 0);
        x.limbs[i] = sum;
    }
}

// Multiply U256 by u32 (for updating M = M * p)
__device__ void u256_mul_u32(U256& M, u32 p) {
    u64 carry = 0;
    u64 p64 = (u64)p;
    for (int i = 0; i < 4; i++) {
        // M.limbs[i] * p + carry
        // u64 * u32 = 96 bits max, plus carry (32 bits) = 97 bits
        // We need to handle this carefully
        u64 lo = M.limbs[i] * p64;
        u64 hi = __umul64hi(M.limbs[i], p64);
        
        // Add carry to lo, propagate to hi if overflow
        u64 sum = lo + carry;
        if (sum < lo) hi++;
        
        M.limbs[i] = sum;
        carry = hi;
    }
}

// Simplified iterative CRT for K_small primes
// x = CRT(residues[0..K_small-1], primes[0..K_small-1])
__device__ void crt_reconstruct_partial(
    U256& result,
    const u32* residues,    // K_small residues
    int K_small
) {
    // Start with x = residues[0], M = primes[0]
    result = U256(residues[0]);
    U256 M;
    M.limbs[0] = d_primes[0].p;
    
    for (int i = 1; i < K_small; i++) {
        u32 p_i = d_primes[i].p;
        u32 a_i = residues[i];
        
        // Compute x mod p_i
        CrtConstants c;
        c.R1 = ((u64)1 << 32) % p_i;
        c.R1 = (c.R1 * c.R1) % p_i;  // 2^64 mod p_i
        c.R2 = (c.R1 * c.R1) % p_i;  // 2^128 mod p_i
        c.R3 = (c.R2 * c.R1) % p_i;  // 2^192 mod p_i
        
        u32 x_mod_pi = u256_mod_p(result, p_i, c);
        
        // Compute M mod p_i
        u32 M_mod_pi = u256_mod_p(M, p_i, c);
        
        // Compute t = (a_i - x_mod_pi) * inv(M_mod_pi) mod p_i
        u32 diff = (a_i >= x_mod_pi) ? (a_i - x_mod_pi) : (p_i - x_mod_pi + a_i);
        u32 M_inv = mod_inv(M_mod_pi, p_i);
        u32 t = mod_mul(diff, M_inv, p_i, d_primes[i].mu);
        
        // Update x = x + M * t
        u256_add_mul(result, M, t);
        
        // Update M = M * p_i
        u256_mul_u32(M, p_i);
    }
}

// Convert U256 to double (approximate, for delta computation)
__device__ double u256_to_double(const U256& x) {
    // Find highest non-zero limb
    int top_idx = 3;
    while (top_idx > 0 && x.limbs[top_idx] == 0) top_idx--;
    
    if (top_idx == 0 && x.limbs[0] == 0) return 0.0;
    
    // Convert with appropriate scaling
    double result = (double)x.limbs[top_idx];
    result = ldexp(result, 64 * top_idx);
    
    // Add contribution from lower limbs for better precision
    if (top_idx > 0) {
        result += ldexp((double)x.limbs[top_idx - 1], 64 * (top_idx - 1));
    }
    
    return result;
}

// Compute bit length of U256 (for log|q| estimation)
__device__ int u256_bitlen(const U256& x) {
    int top_idx = 3;
    while (top_idx > 0 && x.limbs[top_idx] == 0) top_idx--;
    
    if (x.limbs[top_idx] == 0) return 0;
    
    // Count leading zeros in top limb
    int lz = __clzll(x.limbs[top_idx]);
    return 64 * (top_idx + 1) - lz;
}

// Kernel to compute delta proxy using proper partial CRT
__global__ void k_compute_delta_proxy(
    double* deltas,
    double* log_qs,
    const u32* P_rns,               // [K, B, m*m]
    int K, int B, int m,
    double target,
    int p_row, int p_col,
    int q_row, int q_col,
    int K_small                     // Number of primes for CRT
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Clamp K_small
    if (K_small > K) K_small = K;
    if (K_small > MAX_K_SMALL) K_small = MAX_K_SMALL;
    
    // Extract p and q residues from P matrix
    int mm = m * m;
    int p_idx = p_row * m + ((p_col < 0) ? m - 1 : p_col);
    int q_idx = q_row * m + ((q_col < 0) ? m - 1 : q_col);
    
    u32 p_residues[MAX_K_SMALL];
    u32 q_residues[MAX_K_SMALL];
    
    for (int k = 0; k < K_small; k++) {
        p_residues[k] = P_rns[k * B * mm + b * mm + p_idx];
        q_residues[k] = P_rns[k * B * mm + b * mm + q_idx];
    }
    
    // Reconstruct p and q using partial CRT
    U256 p_val, q_val;
    crt_reconstruct_partial(p_val, p_residues, K_small);
    crt_reconstruct_partial(q_val, q_residues, K_small);
    
    // Convert to double for ratio computation
    double p_float = u256_to_double(p_val);
    double q_float = u256_to_double(q_val);
    
    // Handle edge cases
    if (q_float == 0.0 || !isfinite(p_float) || !isfinite(q_float)) {
        deltas[b] = 1e10;
        log_qs[b] = 0.0;
        return;
    }
    
    // Compute estimate and error
    double estimate = p_float / q_float;
    double err = fabs(estimate - target);
    
    // Compute log|q| from bit length
    int q_bitlen = u256_bitlen(q_val);
    double log_abs_q = q_bitlen * 0.693147180559945;  // ln(2)
    
    // Compute delta proxy: delta = -(1 + log(err) / log|q|)
    // Higher delta = better (closer to limit)
    if (err == 0.0 || log_abs_q == 0.0) {
        deltas[b] = (err == 0.0) ? -100.0 : 1e10;  // Perfect match or bad
        log_qs[b] = log_abs_q;
        return;
    }
    
    double log_err = log(err);
    double delta = -(1.0 + log_err / log_abs_q);
    
    deltas[b] = err;  // Store absolute error for now (simpler threshold)
    log_qs[b] = log_abs_q;
}

// TopK selection kernel - selects by HIGHEST delta (Dreams convention)
// In Dreams, delta > 0 indicates good convergence, higher is better
__global__ void k_topk_select(
    Hit* hits,
    int* hit_count,
    const double* deltas,
    const double* log_qs,
    const i32* shifts,
    int cmf_idx,
    int depth,
    int dim,
    int B,
    int max_hits,
    double min_delta                // Minimum delta threshold (e.g., 0.0 or -0.5)
) {
    __shared__ Hit shared_hits[256];
    __shared__ int shared_count;
    
    if (threadIdx.x == 0) shared_count = 0;
    __syncthreads();
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Dreams: select hits where delta > min_delta (higher delta = better)
    // Also filter out invalid deltas (< -1e9)
    if (b < B && deltas[b] > min_delta && deltas[b] > -1e9) {
        int idx = atomicAdd(&shared_count, 1);
        if (idx < 256) {
            shared_hits[idx].cmf_idx = cmf_idx;
            shared_hits[idx].depth = depth;
            shared_hits[idx].delta = deltas[b];
            shared_hits[idx].log_q = log_qs[b];
            for (int d = 0; d < dim && d < MAX_AXES; d++) {
                shared_hits[idx].shift[d] = shifts[b * dim + d];
            }
        }
    }
    __syncthreads();
    
    // Thread 0 writes to global hits
    if (threadIdx.x == 0 && shared_count > 0) {
        int to_write = min(shared_count, 256);
        int global_idx = atomicAdd(hit_count, to_write);
        for (int i = 0; i < to_write && global_idx + i < max_hits; i++) {
            hits[global_idx + i] = shared_hits[i];
        }
    }
}

// ============================================================================
// Host-side API
// ============================================================================

void run_walk_loop(
    u32* P_rns,
    double* P_float,
    double* log_scale,
    const DeviceProgram* program,
    const i32* shifts,
    const PrimeMeta* primes,
    const WalkConfig& config,
    Hit* d_hits,
    int* d_num_hits
) {
    // Copy program and primes to constant memory
    cudaMemcpyToSymbol(d_program, program, sizeof(DeviceProgram));
    cudaMemcpyToSymbol(d_primes, primes, config.K * sizeof(PrimeMeta));
    
    // Allocate delta arrays
    double* d_deltas;
    double* d_log_qs;
    cudaMalloc(&d_deltas, config.B * sizeof(double));
    cudaMalloc(&d_log_qs, config.B * sizeof(double));
    
    // Initialize deltas to invalid (will be set by persistent kernel)
    cudaMemset(d_deltas, 0, config.B * sizeof(double));
    cudaMemset(d_log_qs, 0, config.B * sizeof(double));
    
    // Initialize hit count
    cudaMemset(d_num_hits, 0, sizeof(int));
    
    // Launch persistent kernel with shadow-float Dreams delta
    // This computes delta directly in the kernel using shadow-float trajectory
    int total_threads = config.K * config.B;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    k_persistent_walk<<<blocks, BLOCK_SIZE>>>(
        P_rns, P_float, log_scale, d_deltas, d_log_qs,
        shifts,
        config.K, config.B, config.m, config.dim,
        config.depth,
        config.target,
        config.snapshot_depths[0],  // snapshot_depth_lo (e.g., 200)
        config.snapshot_depths[1]   // snapshot_depth_hi (e.g., 2000)
    );
    
    // TopK selection - select hits with delta > delta_threshold
    // In Dreams, higher delta = better convergence
    int blocks_b = (config.B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_topk_select<<<blocks_b, BLOCK_SIZE>>>(
        d_hits, d_num_hits,
        d_deltas, d_log_qs, shifts,
        0,                          // cmf_idx (set by caller)
        config.depth,
        config.dim,
        config.B,
        config.topk,
        config.delta_threshold      // min_delta threshold (Dreams: higher = better)
    );
    
    cudaFree(d_deltas);
    cudaFree(d_log_qs);
}

// Generate primes for RNS
void generate_rns_primes(PrimeMeta* primes, int K) {
    // Start from largest 31-bit prime and work down
    u32 candidate = PRIME_MAX;
    int count = 0;
    
    auto is_prime = [](u32 n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (u32 i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    };
    
    while (count < K && candidate >= PRIME_MIN) {
        if (is_prime(candidate)) {
            primes[count].p = candidate;
            // Barrett mu = floor(2^64 / p)
            primes[count].mu = ((u64)1 << 63) / candidate * 2;
            count++;
        }
        candidate -= 2;  // Only check odd numbers
    }
}

} // namespace dreams

#else // CPU fallback

namespace dreams {

void run_walk_loop(
    u32* P_rns,
    double* P_float,
    double* log_scale,
    const DeviceProgram* program,
    const i32* shifts,
    const PrimeMeta* primes,
    const WalkConfig& config,
    Hit* d_hits,
    int* d_num_hits
) {
    // CPU fallback implementation would go here
    // For now, just a placeholder
}

void generate_rns_primes(PrimeMeta* primes, int K) {
    u32 candidate = PRIME_MAX;
    int count = 0;
    
    auto is_prime = [](u32 n) {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        for (u32 i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    };
    
    while (count < K && candidate >= PRIME_MIN) {
        if (is_prime(candidate)) {
            primes[count].p = candidate;
            primes[count].mu = ((u64)1 << 63) / candidate * 2;
            count++;
        }
        candidate -= 2;
    }
}

} // namespace dreams

#endif // DREAMS_HAS_GPU
