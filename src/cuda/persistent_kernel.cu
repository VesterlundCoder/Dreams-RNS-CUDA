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

// Barrett reduction: compute a mod p using precomputed mu
__device__ __forceinline__ u32 barrett_reduce(u64 a, u32 p, u64 mu) {
    u64 q = __umul64hi(a, mu);
    u64 r = a - q * p;
    return (r >= p) ? (u32)(r - p) : (u32)r;
}

// Modular addition
__device__ __forceinline__ u32 mod_add(u32 a, u32 b, u32 p) {
    u32 r = a + b;
    return (r >= p) ? r - p : r;
}

// Modular subtraction
__device__ __forceinline__ u32 mod_sub(u32 a, u32 b, u32 p) {
    return (a >= b) ? a - b : p - b + a;
}

// Modular multiplication with Barrett
__device__ __forceinline__ u32 mod_mul(u32 a, u32 b, u32 p, u64 mu) {
    return barrett_reduce((u64)a * b, p, mu);
}

// Modular negation
__device__ __forceinline__ u32 mod_neg(u32 a, u32 p) {
    return (a == 0) ? 0 : p - a;
}

// Extended GCD for modular inverse
__device__ u32 mod_inv(u32 a, u32 p) {
    i32 t = 0, newt = 1;
    i32 r = p, newr = a;
    while (newr != 0) {
        i32 quotient = r / newr;
        i32 tmp = t - quotient * newt;
        t = newt; newt = tmp;
        tmp = r - quotient * newr;
        r = newr; newr = tmp;
    }
    return (t < 0) ? (u32)(t + p) : (u32)t;
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

// 4x4 modular matrix multiply (optimized for small m)
__device__ void matmul_4x4_mod(
    u32* C,                         // Output: 16 entries
    const u32* A,                   // Input: 16 entries
    const u32* B,                   // Input: 16 entries
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
                acc += (u64)A[i * 4 + k] * B[k * 4 + j];
            }
            tmp[i * 4 + j] = barrett_reduce(acc, p, mu);
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) C[i] = tmp[i];
}

// General m x m modular matrix multiply
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
                acc += (u64)A[i * m + k] * B[k * m + j];
            }
            tmp[i * m + j] = barrett_reduce(acc, p, mu);
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
// PERSISTENT KERNEL: Runs entire walk on GPU
// ============================================================================

__global__ void k_persistent_walk(
    u32* P_rns,                     // Trajectory: [K, B, m*m]
    double* P_float,                // Shadow trajectory: [B, m*m]
    double* log_scale,              // Log scale: [B]
    double* deltas,                 // Output deltas: [B]
    double* log_qs,                 // Output log_q: [B]
    const i32* shifts,              // Shifts: [B, dim]
    int K, int B, int m, int dim,
    int depth,
    double target,
    int snapshot_depth
) {
    // Thread indexing: we parallelize over (k, b) pairs
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k = tid / B;                // Prime index
    int b = tid % B;                // Batch index
    
    if (k >= K || b >= B) return;
    
    u32 p = d_primes[k].p;
    u64 mu = d_primes[k].mu;
    
    // Local storage for this thread
    u32 P_local[MAX_M * MAX_M];
    u32 M_local[MAX_M * MAX_M];
    
    // Get shift for this batch element
    i32 shift_local[MAX_AXES];
    for (int d = 0; d < dim; d++) {
        shift_local[d] = shifts[b * dim + d];
    }
    
    // Initialize P to identity
    init_identity_mod(P_local, m, p);
    
    // Main walk loop
    for (int step = 0; step < depth; step++) {
        // Evaluate step matrix M for this step
        eval_program_single(M_local, &d_program, shift_local, step, k, p, mu);
        
        // Update trajectory: P = P @ M
        if (m == 4) {
            matmul_4x4_mod(P_local, P_local, M_local, p, mu);
        } else {
            matmul_mod(P_local, P_local, M_local, m, p, mu);
        }
        
        // At snapshot depth, compute delta (only prime 0 does this to avoid races)
        if (step + 1 == snapshot_depth && k == 0) {
            // This is just placeholder - real delta computation needs all primes
            // We'll do proper delta computation in a separate kernel call
        }
    }
    
    // Write final P back to global memory
    int mm = m * m;
    for (int i = 0; i < mm; i++) {
        P_rns[k * B * mm + b * mm + i] = P_local[i];
    }
}

// Kernel to compute delta proxy using partial CRT
__global__ void k_compute_delta_proxy(
    double* deltas,
    double* log_qs,
    const u32* P_rns,               // [K, B, m*m]
    int K, int B, int m,
    double target,
    int p_row, int p_col,
    int q_row, int q_col,
    int K_small                     // Number of primes for rough CRT
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Extract p and q entries from P matrix
    int mm = m * m;
    int p_idx = p_row * m + ((p_col < 0) ? m - 1 : p_col);
    int q_idx = q_row * m + ((q_col < 0) ? m - 1 : q_col);
    
    // Rough CRT reconstruction using first K_small primes
    // This gives approximate values good enough for delta estimation
    double p_approx = 0.0;
    double q_approx = 0.0;
    double scale = 1.0;
    
    for (int k = 0; k < K_small && k < K; k++) {
        u32 pk = d_primes[k].p;
        u32 p_val = P_rns[k * B * mm + b * mm + p_idx];
        u32 q_val = P_rns[k * B * mm + b * mm + q_idx];
        
        // Simple weighted average (not true CRT, but fast approximation)
        p_approx = p_approx * pk + p_val;
        q_approx = q_approx * pk + q_val;
        scale *= pk;
    }
    
    // Normalize and compute ratio
    if (q_approx == 0.0) {
        deltas[b] = 1e10;
        log_qs[b] = 0.0;
        return;
    }
    
    double ratio = p_approx / q_approx;
    deltas[b] = fabs(ratio - target);
    log_qs[b] = log(fabs(q_approx));
}

// TopK selection kernel
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
    int topk,
    double threshold
) {
    __shared__ Hit shared_hits[256];
    __shared__ int shared_count;
    
    if (threadIdx.x == 0) shared_count = 0;
    __syncthreads();
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < B && deltas[b] < threshold) {
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
    
    // Thread 0 writes to global hits (simple approach, could be improved)
    if (threadIdx.x == 0 && shared_count > 0) {
        int global_idx = atomicAdd(hit_count, min(shared_count, topk));
        for (int i = 0; i < min(shared_count, topk) && global_idx + i < topk; i++) {
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
    
    // Initialize hit count
    cudaMemset(d_num_hits, 0, sizeof(int));
    
    // Launch persistent kernel
    int total_threads = config.K * config.B;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    k_persistent_walk<<<blocks, BLOCK_SIZE>>>(
        P_rns, P_float, log_scale, d_deltas, d_log_qs,
        shifts,
        config.K, config.B, config.m, config.dim,
        config.depth,
        config.target,
        config.snapshot_depths[1]
    );
    
    // Compute delta proxy
    int blocks_b = (config.B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_compute_delta_proxy<<<blocks_b, BLOCK_SIZE>>>(
        d_deltas, d_log_qs, P_rns,
        config.K, config.B, config.m,
        config.target,
        0, -1, 1, -1,               // p_row, p_col, q_row, q_col
        8                           // K_small
    );
    
    // TopK selection
    k_topk_select<<<blocks_b, BLOCK_SIZE>>>(
        d_hits, d_num_hits,
        d_deltas, d_log_qs, shifts,
        0,                          // cmf_idx (set by caller)
        config.depth,
        config.dim,
        config.B,
        config.topk,
        1e-6                        // threshold
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
