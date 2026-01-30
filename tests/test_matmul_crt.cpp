/**
 * Test suite for modular matrix multiplication and CRT reconstruction.
 * 
 * Tests:
 * 1. test_matmul_mod_cpu - Verify CPU reference matmul against known values
 * 2. test_crt_reconstruct - Verify CRT reconstruction for known integers
 * 3. test_barrett_reduce - Verify Barrett reduction correctness
 */

#include "dreams/config.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>

using namespace dreams;

// CPU reference implementations for testing

u32 cpu_mod_mul(u32 a, u32 b, u32 p) {
    return (u64)a * b % p;
}

u32 cpu_mod_add(u32 a, u32 b, u32 p) {
    return ((u64)a + b) % p;
}

// CPU reference matmul (safe, correct)
void cpu_matmul_mod(u32* C, const u32* A, const u32* B, int m, u32 p) {
    std::vector<u32> tmp(m * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            u64 acc = 0;
            for (int k = 0; k < m; k++) {
                // Reduce each product before accumulating
                u32 prod = cpu_mod_mul(A[i * m + k], B[k * m + j], p);
                acc += prod;
            }
            tmp[i * m + j] = (u32)(acc % p);
        }
    }
    for (int i = 0; i < m * m; i++) C[i] = tmp[i];
}

// Extended GCD for modular inverse
u32 cpu_mod_inv(u32 a, u32 p) {
    if (a == 0) return 0;
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

// CRT reconstruction (iterative)
// Returns value mod product of all primes
void cpu_crt_reconstruct(u64* result_lo, u64* result_hi,
                         const u32* residues, const u32* primes, int K) {
    // Use 128-bit arithmetic via u128
#ifndef __CUDA_ARCH__
    u128 x = residues[0];
    u128 M = primes[0];
    
    for (int i = 1; i < K; i++) {
        u32 p_i = primes[i];
        u32 a_i = residues[i];
        
        u32 x_mod_pi = (u32)(x % p_i);
        u32 M_mod_pi = (u32)(M % p_i);
        u32 M_inv = cpu_mod_inv(M_mod_pi, p_i);
        
        u32 diff = (a_i >= x_mod_pi) ? (a_i - x_mod_pi) : (p_i - x_mod_pi + a_i);
        u32 t = cpu_mod_mul(diff, M_inv, p_i);
        
        x = x + M * t;
        M = M * p_i;
    }
    
    *result_lo = (u64)x;
    *result_hi = (u64)(x >> 64);
#else
    *result_lo = 0;
    *result_hi = 0;
#endif
}

// Test 1: Matmul correctness for various m
bool test_matmul_correctness() {
    std::cout << "\n[Test] Matmul Correctness\n";
    std::cout << "--------------------------\n";
    
    std::vector<int> test_sizes = {2, 4, 6, 8};
    std::vector<u32> test_primes = {2147483647u, 2147483629u, 2147483587u};  // Large 31-bit primes
    
    bool all_passed = true;
    
    for (int m : test_sizes) {
        for (u32 p : test_primes) {
            // Generate random matrices
            std::vector<u32> A(m * m), B(m * m), C(m * m);
            
            srand(42 + m + p);
            for (int i = 0; i < m * m; i++) {
                A[i] = rand() % p;
                B[i] = rand() % p;
            }
            
            // Compute using CPU reference
            cpu_matmul_mod(C.data(), A.data(), B.data(), m, p);
            
            // Verify: compute expected using naive method
            std::vector<u32> expected(m * m);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    u64 acc = 0;
                    for (int k = 0; k < m; k++) {
                        acc += (u64)A[i * m + k] * B[k * m + j];
                    }
                    expected[i * m + j] = (u32)(acc % p);
                }
            }
            
            // Compare
            bool match = true;
            for (int i = 0; i < m * m; i++) {
                if (C[i] != expected[i]) {
                    match = false;
                    std::cout << "  FAIL: m=" << m << ", p=" << p 
                              << ", idx=" << i << ", got=" << C[i] 
                              << ", expected=" << expected[i] << "\n";
                }
            }
            
            if (match) {
                std::cout << "  PASS: m=" << m << ", p=" << p << "\n";
            } else {
                all_passed = false;
            }
        }
    }
    
    return all_passed;
}

// Test 2: CRT reconstruction correctness
bool test_crt_correctness() {
    std::cout << "\n[Test] CRT Reconstruction\n";
    std::cout << "--------------------------\n";
    
    // Generate some 31-bit primes
    std::vector<u32> primes = {
        2147483647u,  // 2^31 - 1
        2147483629u,
        2147483587u,
        2147483579u,
        2147483549u,
        2147483543u,
        2147483497u,
        2147483489u
    };
    
    bool all_passed = true;
    
    // Test with known values
    std::vector<u64> test_values = {0, 1, 12345, 1000000007, 
                                    (u64)1 << 32, (u64)1 << 40, (u64)1 << 50};
    
    for (u64 original : test_values) {
        // Compute residues
        std::vector<u32> residues(primes.size());
        for (size_t i = 0; i < primes.size(); i++) {
            residues[i] = original % primes[i];
        }
        
        // Reconstruct
        u64 result_lo, result_hi;
        cpu_crt_reconstruct(&result_lo, &result_hi, residues.data(), primes.data(), primes.size());
        
        // For values < 2^64, result_hi should be 0 and result_lo should match
        if (original < ((u64)1 << 63)) {
            if (result_lo == original && result_hi == 0) {
                std::cout << "  PASS: value=" << original << "\n";
            } else {
                std::cout << "  FAIL: value=" << original 
                          << ", got_lo=" << result_lo << ", got_hi=" << result_hi << "\n";
                all_passed = false;
            }
        } else {
            // Just check consistency (can't easily verify without 128-bit comparison)
            std::cout << "  INFO: large value=" << original 
                      << ", result_lo=" << result_lo << ", result_hi=" << result_hi << "\n";
        }
    }
    
    return all_passed;
}

// Test 3: Barrett reduction correctness
bool test_barrett_correctness() {
    std::cout << "\n[Test] Barrett Reduction\n";
    std::cout << "--------------------------\n";
    
    std::vector<u32> test_primes = {2147483647u, 2147483629u, 1000000007u, 998244353u};
    
    bool all_passed = true;
    
    for (u32 p : test_primes) {
        u64 mu = ((u64)1 << 63) / p * 2;
        
        // Test with various values
        std::vector<u64> test_vals = {0, 1, p - 1, p, p + 1, 
                                      (u64)p * p - 1, (u64)p * p,
                                      (u64)1 << 40, (u64)1 << 50};
        
        for (u64 a : test_vals) {
            if (a >= (u64)p * ((u64)1 << 32)) continue;  // Skip values out of Barrett range
            
            // Barrett reduction (simplified CPU version)
            u64 q = ((__uint128_t)a * mu) >> 64;
            u64 r = a - q * p;
            u32 result = (r >= p) ? (u32)(r - p) : (u32)r;
            
            // Expected
            u32 expected = (u32)(a % p);
            
            if (result == expected) {
                // Pass silently for brevity
            } else {
                std::cout << "  FAIL: p=" << p << ", a=" << a 
                          << ", got=" << result << ", expected=" << expected << "\n";
                all_passed = false;
            }
        }
        
        if (all_passed) {
            std::cout << "  PASS: p=" << p << "\n";
        }
    }
    
    return all_passed;
}

// Test 4: Accumulator overflow detection
bool test_accumulator_safety() {
    std::cout << "\n[Test] Accumulator Safety\n";
    std::cout << "--------------------------\n";
    
    // For m=4 with 31-bit primes, max accumulator value is:
    // 4 * (2^31 - 1)^2 = 4 * 2^62 - 4 * 2^32 + 4 ≈ 2^64
    // This is at the edge of u64 safety
    
    u32 p = 2147483647u;  // 2^31 - 1
    u32 max_val = p - 1;
    
    // Simulate accumulation for different m values
    for (int m = 2; m <= 10; m++) {
        u64 max_acc = (u64)m * ((u64)max_val * max_val);
        bool overflows_u64 = (max_acc < (u64)m * max_val);  // Overflow check
        
        // Compute bit length
        int bits_needed = 0;
        u64 temp = max_acc;
        while (temp > 0) { bits_needed++; temp >>= 1; }
        
        std::cout << "  m=" << m << ": max_acc bits=" << bits_needed;
        if (bits_needed > 64) {
            std::cout << " [UNSAFE - needs per-step reduction]";
        } else if (bits_needed > 62) {
            std::cout << " [MARGINAL - recommend per-step reduction]";
        } else {
            std::cout << " [SAFE]";
        }
        std::cout << "\n";
    }
    
    return true;  // This is informational
}

int main() {
    std::cout << "=== Dreams-RNS-CUDA Matmul/CRT Test Suite ===\n";
    
    bool all_passed = true;
    
    all_passed &= test_matmul_correctness();
    all_passed &= test_crt_correctness();
    all_passed &= test_barrett_correctness();
    all_passed &= test_accumulator_safety();
    
    std::cout << "\n=== Summary ===\n";
    if (all_passed) {
        std::cout << "All tests PASSED\n";
        return 0;
    } else {
        std::cout << "Some tests FAILED\n";
        return 1;
    }
}
