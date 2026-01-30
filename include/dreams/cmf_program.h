#ifndef DREAMS_CMF_PROGRAM_H
#define DREAMS_CMF_PROGRAM_H

#include "config.h"
#include <vector>

namespace dreams {

// Device-side program storage (will be copied to constant memory)
struct DeviceProgram {
    int m;
    int dim;
    int n_opcodes;
    int n_constants;
    Instr opcodes[MAX_OPCODES];
    u32 constants_rns[MAX_K][MAX_CONSTANTS];  // Constants in RNS form
    i32 directions[MAX_AXES];
};

// Host-side program builder
class CmfProgramBuilder {
public:
    CmfProgramBuilder(int matrix_dim, int num_axes);
    
    // Add opcodes
    void load_axis(int axis_idx, int dest_reg);
    void load_const(int const_idx, int dest_reg);
    void load_n(int dest_reg);
    void add(int src1, int src2, int dest);
    void sub(int src1, int src2, int dest);
    void mul(int src1, int src2, int dest);
    void neg(int src, int dest);
    void pow2(int src, int dest);
    void pow3(int src, int dest);
    void inv(int src, int dest);
    void store(int src, int row, int col);
    void end();
    
    // Add constants
    int add_constant(i64 value);
    
    // Set directions
    void set_direction(int axis, int dir);
    
    // Build final program
    CmfProgram build() const;
    
private:
    int m_;
    int dim_;
    std::vector<Instr> opcodes_;
    std::vector<i64> constants_;
    std::vector<i32> directions_;
    int next_reg_ = 0;
};

// Convert constants to RNS representation
void convert_constants_to_rns(
    const CmfProgram& program,
    const PrimeMeta* primes,
    int K,
    DeviceProgram& d_program
);

// Serialize/deserialize programs for file I/O
std::vector<uint8_t> serialize_program(const CmfProgram& program);
CmfProgram deserialize_program(const uint8_t* data, size_t size);

} // namespace dreams

#endif // DREAMS_CMF_PROGRAM_H
