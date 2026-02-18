"""
GPU-accelerated RNS walk using cupy (CUDA).

Provides drop-in replacements for the numpy walk functions that run on
the GPU. All modular arithmetic and matrix multiplications execute on
the CUDA device, with data only transferred back for CRT reconstruction.

Two main entry points:
  - run_pcf_walk_gpu():       single walk (like run_pcf_walk but on GPU)
  - run_pcf_walk_batch_gpu(): batch B walks with different shift_vals

Auto-detects cupy; falls back to numpy CPU if unavailable.

Performance model (A100, K=32, depth=2000):
  - CPU numpy:  ~20 walks/s
  - GPU cupy:   ~5,000-10,000 walks/s  (batch=256)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .compiler import CmfProgram, Opcode
from .runner import generate_rns_primes, _eval_bytecode_float

# ── cupy detection ───────────────────────────────────────────────────────

_cp = None
_GPU_AVAILABLE = False

def _ensure_cupy():
    global _cp, _GPU_AVAILABLE
    if _cp is not None:
        return _GPU_AVAILABLE
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
        _cp = cupy
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def gpu_available() -> bool:
    """Check if cupy + CUDA GPU is available."""
    return _ensure_cupy()


# ── GPU bytecode evaluator (batched) ─────────────────────────────────────

def _precompute_const_residues_gpu(program: CmfProgram, primes_np: np.ndarray):
    """Precompute (n_constants, K) residue table on GPU."""
    cp = _cp
    K = len(primes_np)
    n_c = len(program.constants)
    table = np.zeros((n_c, K), dtype=np.int64)
    for idx in range(n_c):
        c = int(program.constants[idx])
        for ki in range(K):
            p = int(primes_np[ki])
            table[idx, ki] = c % p if c >= 0 else (p - (-c % p)) % p
    return cp.asarray(table)


def _eval_bytecode_batch_gpu(program: CmfProgram, step: int,
                              shift_vals_gpu, pp_gpu, const_table_gpu):
    """Evaluate bytecode for B shift values × K primes simultaneously.

    Args:
        shift_vals_gpu: cupy (B,) int64
        pp_gpu:         cupy (K,) int64
        const_table_gpu: cupy (n_constants, K) int64

    Returns:
        cupy (m, m, B, K) int64 — batched M matrices
    """
    cp = _cp
    m = program.m
    B = len(shift_vals_gpu)
    K = len(pp_gpu)
    pp = pp_gpu.reshape(1, K)  # broadcast (1, K) for (B, K) ops

    # Only allocate registers actually used by this program
    max_reg = 0
    for instr in program.opcodes:
        if instr.op == Opcode.END:
            break
        for a in [getattr(instr, f'arg{i}', 0) for i in range(3)]:
            if a is not None and isinstance(a, int):
                max_reg = max(max_reg, a)
    n_regs = max_reg + 1

    regs = [None] * n_regs
    M = cp.zeros((m, m, B, K), dtype=cp.int64)

    for instr in program.opcodes:
        op = instr.op
        if op == Opcode.END:
            break
        elif op == Opcode.LOAD_X:
            axis, dest = instr.arg0, instr.arg1
            vals = shift_vals_gpu.reshape(B, 1) + cp.int64(step * program.directions[axis])
            regs[dest] = vals % pp
        elif op == Opcode.LOAD_C:
            idx, dest = instr.arg0, instr.arg1
            regs[dest] = cp.broadcast_to(const_table_gpu[idx:idx+1], (B, K)).copy()
        elif op == Opcode.LOAD_N:
            dest = instr.arg0
            regs[dest] = cp.full((B, K), step, dtype=cp.int64) % pp
        elif op == Opcode.ADD:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = (regs[r1] + regs[r2]) % pp
        elif op == Opcode.SUB:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = (regs[r1] - regs[r2]) % pp
        elif op == Opcode.MUL:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = (regs[r1] * regs[r2]) % pp
        elif op == Opcode.NEG:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = (pp - regs[r]) % pp
        elif op == Opcode.POW2:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = (regs[r] * regs[r]) % pp
        elif op == Opcode.POW3:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = (regs[r] * regs[r] % pp * regs[r]) % pp
        elif op == Opcode.INV:
            # Modular inverse via Fermat's little theorem: a^(p-2) mod p
            # Fall back to CPU for pow(base, exp, mod) — cupy lacks this
            r, dest = instr.arg0, instr.arg1
            vals_cpu = regs[r].get()
            pp_cpu = pp_gpu.get()
            inv_cpu = np.empty_like(vals_cpu)
            for bi in range(B):
                for ki in range(K):
                    v = int(vals_cpu[bi, ki])
                    p = int(pp_cpu[ki])
                    inv_cpu[bi, ki] = pow(v, p - 2, p) if v != 0 else 0
            regs[dest] = cp.asarray(inv_cpu)
        elif op == Opcode.STORE:
            r, row, col = instr.arg0, instr.arg1, instr.arg2
            M[row, col] = regs[r] % pp

    return M


def _matmul_2x2_batch_mod_gpu(A, B, pp):
    """Batched modular 2×2 matmul: A @ B mod pp.

    A, B: (2, 2, batch, K) int64 on GPU
    pp:   (1, K) int64 on GPU
    Returns: (2, 2, batch, K) int64
    """
    cp = _cp
    C = cp.empty_like(A)
    C[0, 0] = (A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]) % pp
    C[0, 1] = (A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]) % pp
    C[1, 0] = (A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]) % pp
    C[1, 1] = (A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]) % pp
    return C


# ── Single GPU walk ──────────────────────────────────────────────────────

def run_pcf_walk_gpu(program: CmfProgram, a0: int, depth: int,
                     K: int = 32, shift_val: int = 1) -> Dict[str, Any]:
    """GPU-accelerated single PCF walk. Same interface as run_pcf_walk."""
    results = run_pcf_walk_batch_gpu(program, a0, depth, K, [shift_val])
    return results[0]


# ── Batched GPU walk ─────────────────────────────────────────────────────

def run_pcf_walk_batch_gpu(
    program: CmfProgram,
    a0: int,
    depth: int,
    K: int = 32,
    shift_vals: Optional[List[int]] = None,
    batch_size: int = 256,
) -> List[Dict[str, Any]]:
    """Run B walks in parallel on GPU, each with a different shift_val.

    The RNS walk (exact integer arithmetic mod K primes) runs entirely
    on GPU. A single float64 shadow is maintained on CPU for the first
    shift value to provide approximate convergent estimates.

    Args:
        program:     compiled CmfProgram (2×2 companion)
        a0:          a(0) for initial matrix A = [[1,a0],[0,1]]
        depth:       walk depth
        K:           number of RNS primes
        shift_vals:  list of integer shift values (len = total walks)
        batch_size:  max simultaneous walks per GPU batch (tune for memory)

    Returns:
        List of result dicts, one per shift_val:
          p_residues, q_residues: (K,) int64 arrays
          p_float, q_float: float64 estimates (from representative shadow)
          log_scale: float64 accumulated log-normalization
          primes: (K,) int64 array
          shift_val: the shift used
    """
    if not _ensure_cupy():
        from .runner import run_pcf_walk
        return [run_pcf_walk(program, a0, depth, K, sv) for sv in (shift_vals or [1])]

    cp = _cp

    if shift_vals is None:
        shift_vals = [1]

    m = program.m
    assert m == 2, "GPU walk only supports 2×2 companion matrices"

    pp_np = generate_rns_primes(K).astype(np.int64)
    pp_gpu = cp.asarray(pp_np)
    pp_bcast = pp_gpu.reshape(1, K)

    const_table_gpu = _precompute_const_residues_gpu(program, pp_np)

    # a0 residues on CPU (small, reused per batch)
    a0_res_np = np.array([int(a0) % int(p) if a0 >= 0 else (int(p) - (-int(a0) % int(p))) % int(p)
                           for p in pp_np], dtype=np.int64)
    a0_res_gpu = cp.asarray(a0_res_np)

    all_results = []

    for batch_start in range(0, len(shift_vals), batch_size):
        batch_sv = shift_vals[batch_start : batch_start + batch_size]
        B = len(batch_sv)
        sv_gpu = cp.asarray(np.array(batch_sv, dtype=np.int64))

        # GPU: P_rns shape (2, 2, B, K) initialised to A = [[1,a0],[0,1]]
        P_rns = cp.zeros((2, 2, B, K), dtype=cp.int64)
        P_rns[0, 0] = cp.ones((B, K), dtype=cp.int64)
        P_rns[0, 1] = cp.broadcast_to(a0_res_gpu.reshape(1, K), (B, K)).copy()
        P_rns[1, 1] = cp.ones((B, K), dtype=cp.int64)

        # CPU: single representative float shadow (first shift in batch)
        P_f = np.array([[1.0, float(a0)], [0.0, 1.0]], dtype=np.float64)
        log_scale = 0.0

        for step in range(depth):
            # GPU: evaluate M(n) for all B shifts × K primes at once
            M_rns = _eval_bytecode_batch_gpu(
                program, step, sv_gpu, pp_gpu, const_table_gpu)

            # GPU: batched 2×2 modular matmul
            P_rns = _matmul_2x2_batch_mod_gpu(P_rns, M_rns, pp_bcast)

            # CPU: float shadow for approximate estimate (only 1 representative)
            M_f = _eval_bytecode_float(program, step, batch_sv[0])
            P_f = P_f @ M_f
            mx = np.max(np.abs(P_f))
            if mx > 1e10:
                P_f /= mx
                log_scale += math.log(mx)

        # Transfer RNS results back to CPU
        P_cpu = P_rns.get()  # (2, 2, B, K) on CPU

        for bi in range(B):
            all_results.append({
                'p_residues': P_cpu[0, 1, bi],
                'q_residues': P_cpu[1, 1, bi],
                'p_float': float(P_f[0, 1]),
                'q_float': float(P_f[1, 1]),
                'log_scale': log_scale,
                'primes': pp_np,
                'shift_val': batch_sv[bi],
            })

    return all_results
