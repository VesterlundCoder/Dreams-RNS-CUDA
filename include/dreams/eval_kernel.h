#ifndef DREAMS_EVAL_KERNEL_H
#define DREAMS_EVAL_KERNEL_H

#include "config.h"
#include "cmf_program.h"

namespace dreams {

// Evaluate bytecode program to produce matrix entries in RNS
// Output: M_rns[K, B, m*m] - step matrix for each prime and shift
void eval_step_matrix(
    u32* M_rns,                     // Output: [K, B, m*m]
    const DeviceProgram* program,   // CMF program
    const i32* shifts,              // Shift values [B, dim]
    const PrimeMeta* primes,        // Prime metadata [K]
    int step_n,                     // Current step number
    int K,                          // Number of primes
    int B                           // Batch size
);

// Evaluate single axis matrix
void eval_axis_matrix(
    u32* A_rns,                     // Output: [K, B, m*m]
    const DeviceProgram* program,   // CMF program
    int axis_idx,                   // Which axis to evaluate
    const i32* axis_values,         // Axis values [B]
    const PrimeMeta* primes,        // Prime metadata [K]
    int K,
    int B
);

// Compose step matrix from axis matrices: M = A_0 @ A_1 @ ... @ A_{dim-1}
void compose_step_matrix(
    u32* M_rns,                     // Output: [K, B, m*m]
    const u32* const* A_rns,        // Axis matrices [dim][K, B, m*m]
    const PrimeMeta* primes,
    int dim,
    int m,
    int K,
    int B
);

} // namespace dreams

#endif // DREAMS_EVAL_KERNEL_H
