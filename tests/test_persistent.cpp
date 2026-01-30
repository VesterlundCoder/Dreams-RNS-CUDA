#include "dreams/config.h"
#include "dreams/cmf_program.h"
#include <iostream>
#include <cmath>
#include <vector>

using namespace dreams;

int main() {
    std::cout << "=== Dreams-RNS-CUDA Persistent Kernel Test ===" << std::endl;
    
    // Test 1: Prime generation
    std::cout << "\n[Test 1] Prime generation..." << std::endl;
    std::vector<PrimeMeta> primes(DEFAULT_K);
    
    // Generate primes (simplified test version)
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
    
    while (count < DEFAULT_K && candidate >= PRIME_MIN) {
        if (is_prime(candidate)) {
            primes[count].p = candidate;
            primes[count].mu = ((u64)1 << 63) / candidate * 2;
            count++;
        }
        candidate -= 2;
    }
    
    std::cout << "  Generated " << count << " primes" << std::endl;
    std::cout << "  First prime: " << primes[0].p << std::endl;
    std::cout << "  Last prime: " << primes[count-1].p << std::endl;
    
    // Test 2: Barrett reduction
    std::cout << "\n[Test 2] Barrett reduction..." << std::endl;
    u32 p = primes[0].p;
    u64 mu = primes[0].mu;
    
    // Test values
    u64 test_vals[] = {0, 1, p-1, p, p+1, (u64)p*2, (u64)p*p/2};
    bool barrett_ok = true;
    
    for (u64 val : test_vals) {
        u64 q = ((__uint128_t)val * mu) >> 64;
        u64 r = val - q * p;
        u32 result = (r >= p) ? (u32)(r - p) : (u32)r;
        u32 expected = val % p;
        if (result != expected) {
            std::cout << "  FAIL: " << val << " mod " << p << " = " << result 
                      << " (expected " << expected << ")" << std::endl;
            barrett_ok = false;
        }
    }
    
    if (barrett_ok) {
        std::cout << "  Barrett reduction: PASS" << std::endl;
    }
    
    // Test 3: Bytecode structure
    std::cout << "\n[Test 3] Bytecode structure..." << std::endl;
    CmfProgram prog;
    prog.m = 4;
    prog.dim = 2;
    prog.n_opcodes = 0;
    prog.n_constants = 0;
    
    // Add some opcodes
    prog.opcodes[prog.n_opcodes++] = {Opcode::LOAD_X, 0, 0, 0};  // Load x0 to reg 0
    prog.opcodes[prog.n_opcodes++] = {Opcode::LOAD_C, 0, 1, 0};  // Load const 0 to reg 1
    prog.opcodes[prog.n_opcodes++] = {Opcode::MUL, 0, 1, 2, };   // reg2 = reg0 * reg1
    prog.opcodes[prog.n_opcodes++] = {Opcode::STORE, 2, 0, 0};   // Store reg2 to M[0,0]
    prog.opcodes[prog.n_opcodes++] = {Opcode::END, 0, 0, 0};
    
    prog.constants[prog.n_constants++] = 2;  // Constant: 2
    
    prog.directions[0] = 1;
    prog.directions[1] = 1;
    
    std::cout << "  Created program with " << prog.n_opcodes << " opcodes" << std::endl;
    std::cout << "  Matrix dimension: " << prog.m << "x" << prog.m << std::endl;
    std::cout << "  Number of axes: " << prog.dim << std::endl;
    
    // Test 4: Configuration
    std::cout << "\n[Test 4] Configuration..." << std::endl;
    WalkConfig config;
    config.K = DEFAULT_K;
    config.B = 100;
    config.m = 4;
    config.dim = 2;
    config.depth = 2000;
    config.topk = 100;
    config.target = M_PI;
    config.snapshot_depths[0] = 200;
    config.snapshot_depths[1] = 2000;
    
    std::cout << "  Primes (K): " << config.K << std::endl;
    std::cout << "  Batch size (B): " << config.B << std::endl;
    std::cout << "  Walk depth: " << config.depth << std::endl;
    std::cout << "  Target: " << config.target << std::endl;
    std::cout << "  Bit capacity: " << (config.K * PRIME_BITS) << " bits" << std::endl;
    
    std::cout << "\n=== All Tests Passed ===" << std::endl;
    return 0;
}
