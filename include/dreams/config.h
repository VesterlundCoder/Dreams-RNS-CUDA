#ifndef DREAMS_CONFIG_H
#define DREAMS_CONFIG_H

#include <cstdint>

#ifdef DREAMS_HAS_GPU
  #include <cuda_runtime.h>
  #define DREAMS_HOST_DEVICE __host__ __device__
  #define DREAMS_DEVICE __device__
  #define DREAMS_GLOBAL __global__
  #define DREAMS_SHARED __shared__
  #define DREAMS_CONSTANT __constant__
  #define DREAMS_FORCEINLINE __forceinline__
#else
  #define DREAMS_HOST_DEVICE
  #define DREAMS_DEVICE
  #define DREAMS_GLOBAL
  #define DREAMS_SHARED
  #define DREAMS_CONSTANT
  #define DREAMS_FORCEINLINE inline
#endif

namespace dreams {

// ============================================================================
// NUMERIC TYPES
// ============================================================================
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

// 128-bit unsigned integer (host only, use helpers on device)
#ifndef __CUDA_ARCH__
using u128 = unsigned __int128;
#endif

// 256-bit integer as 4 x u64 limbs (for CRT reconstruction)
struct alignas(32) U256 {
    u64 limbs[4];  // limbs[0] = low, limbs[3] = high
    
    DREAMS_HOST_DEVICE U256() : limbs{0, 0, 0, 0} {}
    DREAMS_HOST_DEVICE U256(u64 v) : limbs{v, 0, 0, 0} {}
};

// ============================================================================
// NUMERIC INVARIANTS (CRITICAL FOR CORRECTNESS)
// ============================================================================
// - Residues are stored in u32 (values in [0, p-1] where p < 2^31)
// - Intermediate multiply: u64 product = (u64)a * (u64)b < 2^62
// - Accumulation: For matrix size m, we accumulate m products.
//   Worst case: m * (p-1)^2 ≈ m * 2^62
//   For m <= 4: safe in u64 without intermediate reduction
//   For m > 4: MUST reduce after each multiply-add to prevent u64 overflow
// - Barrett reduction requires: input < p * 2^32 (satisfied by u64 products)
// ============================================================================

// RNS Configuration
constexpr int MAX_K = 128;              // Max primes
constexpr int DEFAULT_K = 64;           // Default primes (64 * 31 = 1984 bits)
constexpr int PRIME_BITS = 31;
constexpr u32 PRIME_MIN = 1u << 30;
constexpr u32 PRIME_MAX = (1u << 31) - 1;

// Partial CRT Configuration (for GPU delta scoring)
constexpr int DEFAULT_K_SMALL = 8;      // Primes for partial CRT (8 * 31 = 248 bits)
constexpr int MAX_K_SMALL = 16;         // Max primes for partial CRT

// Safe accumulation threshold: for m > SAFE_M_THRESHOLD, reduce every iteration
constexpr int SAFE_M_THRESHOLD = 4;

// Matrix Configuration
constexpr int MAX_M = 8;                // Max matrix dimension
constexpr int DEFAULT_M = 4;            // Default matrix dimension

// Walk Configuration
constexpr int DEFAULT_DEPTH = 2000;     // Default walk depth
constexpr int SNAPSHOT_DEPTH1 = 200;    // First snapshot
constexpr int SNAPSHOT_DEPTH2 = 2000;   // Second snapshot

// Batch Configuration
constexpr int MAX_BATCH = 4096;         // Max shifts per batch
constexpr int DEFAULT_BATCH = 1000;     // Default shifts per CMF

// Bytecode Configuration
constexpr int MAX_OPCODES = 1024;       // Max opcodes per program
constexpr int MAX_CONSTANTS = 256;      // Max constants per program
constexpr int MAX_REGS = 64;            // Max registers
constexpr int MAX_AXES = 8;             // Max axes (dimensions)

// TopK Configuration
constexpr int DEFAULT_TOPK = 100;       // Default top-k hits to keep

// GPU Configuration
constexpr int BLOCK_SIZE = 256;         // Threads per block
constexpr int WARP_SIZE = 32;

// Opcodes for bytecode evaluator
enum class Opcode : u32 {
    NOP = 0,
    LOAD_X,      // Load axis value: args = (axis_idx, dest_reg)
    LOAD_C,      // Load constant: args = (const_idx, dest_reg)
    LOAD_N,      // Load step number n: args = (dest_reg)
    ADD,         // Add: args = (src1, src2, dest)
    SUB,         // Subtract: args = (src1, src2, dest)
    MUL,         // Multiply: args = (src1, src2, dest)
    NEG,         // Negate: args = (src, dest)
    POW2,        // Square: args = (src, dest)
    POW3,        // Cube: args = (src, dest)
    INV,         // Modular inverse: args = (src, dest)
    STORE,       // Store to matrix: args = (src, row, col)
    END          // End of program
};

// Instruction encoding: opcode (8 bits) + 3 args (8 bits each)
struct Instr {
    Opcode op;
    u32 arg0;
    u32 arg1;
    u32 arg2;
};

// Prime metadata for modular arithmetic
struct PrimeMeta {
    u32 p;        // Prime value
    u64 mu;       // Barrett reduction constant
    u32 r2;       // Montgomery R² mod p
    u32 p_inv;    // -p⁻¹ mod 2³²
};

// Hit result structure
struct Hit {
    i32 cmf_idx;
    i32 shift[MAX_AXES];
    i32 depth;
    double delta;
    double log_q;
};

// CMF Program structure
struct CmfProgram {
    int m;                          // Matrix dimension
    int dim;                        // Number of axes
    int n_opcodes;                  // Number of opcodes
    int n_constants;                // Number of constants
    Instr opcodes[MAX_OPCODES];     // Bytecode
    i64 constants[MAX_CONSTANTS];   // Constants (will be converted to RNS)
    i32 directions[MAX_AXES];       // Walk directions per axis (+1 or -1)
};

// Walk configuration
struct WalkConfig {
    int K;                          // Number of primes
    int K_small;                    // Primes for partial CRT scoring
    int B;                          // Batch size (shifts)
    int m;                          // Matrix dimension
    int dim;                        // Number of axes
    int depth;                      // Walk depth
    int topk;                       // Top-K to keep
    double target;                  // Target constant (e.g., pi)
    double delta_threshold;         // Hit threshold
    int snapshot_depths[2];         // Depths for delta computation
    bool use_float_shadow;          // Optional float64 shadow for debugging
};

// Lane status for dead-lane tracking (e.g., division by zero)
struct LaneStatus {
    bool alive;                     // false if lane encountered error (e.g., inv(0))
};

} // namespace dreams

#endif // DREAMS_CONFIG_H
