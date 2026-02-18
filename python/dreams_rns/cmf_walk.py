"""
Full CMF walk pipeline: compile and walk r×r companion matrices
with multi-axis trajectory + shift support.

For a pFq CMF with p a-params and q b-params:
  - rank r = max(p, q) + 1
  - dim = p + q (number of axes)
  - Axes: x0..x_{p-1} for a-params, y0..y_{q-1} for b-params
  - Trajectory: direction vector in Z^dim (one int per axis)
  - Shift: rational offset vector in Q^dim (one rational per axis)

At walk step t, axis i evaluates to: shift[i] + t * trajectory[i]

The companion matrix M(step) is r×r, evaluated at the current axis values.
Walk: P(N) = M(1) · M(2) · ... · M(N), starting from P(0) = Identity.
Convergent: p = P[0, r-1], q = P[r-1, r-1].
"""

from __future__ import annotations

import math
import itertools
from fractions import Fraction
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .compiler import CmfCompiler, CmfProgram, Opcode
from .runner import generate_rns_primes, crt_reconstruct, centered


# ── Build per-axis companion matrix ──────────────────────────────────────

def _build_per_axis_companion(
    p: int,
    q: int,
    a_params: List[Fraction],
    b_params: List[Fraction],
) -> Tuple[int, Dict[str, str], List[str]]:
    """Build r×r companion matrix using per-axis variables.

    Instead of (n + a_i), uses (x_i + a_i) where x_i is axis i.
    This allows each parameter to evolve independently along the trajectory.

    Returns:
        (rank, matrix_dict, axis_names)
        matrix_dict maps "row,col" -> sympy expression string
        axis_names is ['x0', 'x1', ..., 'y0', 'y1', ...]
    """
    r = max(p, q) + 1
    axis_names = [f"x{i}" for i in range(p)] + [f"y{j}" for j in range(q)]

    # Per-axis Pochhammer factors: (axis_var + param)
    a_factors = [f"({axis_names[i]} + {a_params[i]})" for i in range(p)]
    b_factors = [f"({axis_names[p + j]} + {b_params[j]})" for j in range(q)]

    matrix: Dict[str, str] = {}

    # Fill with zeros and sub-diagonal ones
    for i in range(r):
        for j in range(r):
            if i == j + 1:
                matrix[f"{i},{j}"] = "1"
            else:
                matrix[f"{i},{j}"] = "0"

    # Last column: recurrence coefficients
    if r == 2:
        an_prod = " * ".join(a_factors) if a_factors else "1"
        bn_prod = " * ".join(b_factors) if b_factors else "1"
        matrix[f"0,{r-1}"] = an_prod
        matrix[f"1,{r-1}"] = bn_prod
    else:
        # Row 0: numerator product
        an_prod = " * ".join(a_factors) if a_factors else "1"
        matrix[f"0,{r-1}"] = an_prod

        # Middle rows: elementary symmetric polynomial differences
        for row in range(1, r - 1):
            k = r - 1 - row
            # e_k of b-factors
            bn_terms = []
            if k <= len(b_factors):
                for combo in itertools.combinations(b_factors, k):
                    bn_terms.append(" * ".join(combo))
            bn_ek = " + ".join(bn_terms) if bn_terms else "0"
            # e_k of a-factors
            an_terms = []
            if k <= len(a_factors):
                for combo in itertools.combinations(a_factors, k):
                    an_terms.append(" * ".join(combo))
            an_ek = " + ".join(an_terms) if an_terms else "0"
            matrix[f"{row},{r-1}"] = f"({bn_ek}) - ({an_ek})"

        # Last row: denominator product
        bn_prod = " * ".join(b_factors) if b_factors else "1"
        matrix[f"{r-1},{r-1}"] = bn_prod

    return r, matrix, axis_names


# ── Compile CMF spec to bytecode ─────────────────────────────────────────

def compile_cmf_spec(spec: Dict, trajectory: Optional[List[int]] = None) -> Optional[CmfProgram]:
    """Compile a pFq CMF spec to r×r bytecode with per-axis variables.

    Args:
        spec: CMF spec dict with keys: p, q, a_params, b_params, rank, dim
        trajectory: direction vector (len = dim). If None, uses all-ones.

    Returns:
        CmfProgram with m=rank, dim=p+q, directions=trajectory.
        Or None if compilation fails.
    """
    import sympy as sp

    p_val = spec['p']
    q_val = spec['q']
    a_params = [Fraction(x) for x in spec['a_params']]
    b_params = [Fraction(x) for x in spec['b_params']]

    rank, matrix, axis_names = _build_per_axis_companion(
        p_val, q_val, a_params, b_params)

    dim = p_val + q_val
    if trajectory is None:
        trajectory = [1] * dim
    assert len(trajectory) == dim, f"trajectory len {len(trajectory)} != dim {dim}"

    compiler = CmfCompiler(m=rank, dim=dim, directions=list(trajectory))

    # Map axis names to axis indices
    axis_symbols = {name: idx for idx, name in enumerate(axis_names)}

    for i in range(rank):
        for j in range(rank):
            expr_str = matrix[f"{i},{j}"]
            if expr_str == "0":
                continue
            if expr_str == "1":
                expr = sp.Integer(1)
            else:
                # Parse with axis symbols as sympy Symbols
                local_ns = {name: sp.Symbol(name) for name in axis_names}
                expr = sp.sympify(expr_str, locals=local_ns)

            if expr.has(sp.I):
                return None

            compiler.compile_matrix_entry(i, j, expr, axis_symbols)

    return compiler.build()


# ── Per-axis shift conversion ────────────────────────────────────────────

def shift_to_axis_offsets(shift: Dict, p: int) -> List[int]:
    """Convert a shift dict to per-axis integer offsets.

    The shift dict has 'nums' and 'dens' arrays.
    For the RNS walker, we need integer starting values per axis.
    We use: offset = 1 + nums[i]  (ensure positive starting n ≥ 1).
    """
    nums = shift.get('nums', [])
    dim = len(nums)
    offsets = []
    for i in range(dim):
        offsets.append(max(1, 1 + nums[i]))
    return offsets


# ── General r×r modular matmul ───────────────────────────────────────────

def _matmul_rxr_mod(A: np.ndarray, B: np.ndarray, pp: np.ndarray, r: int) -> np.ndarray:
    """Modular r×r matrix multiply A @ B mod pp, vectorised over K primes.

    A, B: shape (r, r, K) int64
    pp:   shape (K,) int64
    Returns: (r, r, K) int64
    """
    C = np.zeros_like(A)
    for i in range(r):
        for j in range(r):
            acc = np.zeros_like(pp)
            for k in range(r):
                acc = (acc + A[i, k] * B[k, j]) % pp
            C[i, j] = acc
    return C


# ── General r×r bytecode evaluator with per-axis shifts ──────────────────

def _eval_bytecode_allprimes_multiaxis(
    program: CmfProgram,
    step: int,
    shift_vals: List[int],
    primes: np.ndarray,
    const_table: np.ndarray,
) -> np.ndarray:
    """Evaluate bytecode producing (m, m, K) matrix with per-axis shift values.

    At LOAD_X for axis i: value = shift_vals[i] + step * direction[i]
    """
    m = program.m
    K = len(primes)
    pp = primes

    regs = [np.zeros(K, dtype=np.int64) for _ in range(512)]
    M = np.zeros((m, m, K), dtype=np.int64)

    for instr in program.opcodes:
        op = instr.op
        if op == Opcode.END:
            break
        elif op == Opcode.LOAD_X:
            axis, dest = instr.arg0, instr.arg1
            val = np.int64(shift_vals[axis] + step * program.directions[axis])
            regs[dest] = val % pp
        elif op == Opcode.LOAD_C:
            idx, dest = instr.arg0, instr.arg1
            regs[dest] = const_table[idx].copy()
        elif op == Opcode.LOAD_N:
            dest = instr.arg0
            regs[dest] = np.int64(step) % pp
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
            r, dest = instr.arg0, instr.arg1
            vals = regs[r]
            regs[dest] = np.array([pow(int(v), int(p) - 2, int(p))
                                    for v, p in zip(vals, pp)], dtype=np.int64)
        elif op == Opcode.STORE:
            r, row, col = instr.arg0, instr.arg1, instr.arg2
            M[row, col] = regs[r] % pp

    return M


def _eval_bytecode_float_multiaxis(
    program: CmfProgram,
    step: int,
    shift_vals: List[int],
) -> np.ndarray:
    """Evaluate bytecode producing (m, m) float64 matrix with per-axis shifts."""
    m = program.m
    regs = [0.0] * 512
    M = np.zeros((m, m), dtype=np.float64)

    for instr in program.opcodes:
        op = instr.op
        if op == Opcode.END:
            break
        elif op == Opcode.LOAD_X:
            axis, dest = instr.arg0, instr.arg1
            regs[dest] = float(shift_vals[axis] + step * program.directions[axis])
        elif op == Opcode.LOAD_C:
            idx, dest = instr.arg0, instr.arg1
            regs[dest] = float(program.constants[idx])
        elif op == Opcode.LOAD_N:
            dest = instr.arg0
            regs[dest] = float(step)
        elif op == Opcode.ADD:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = regs[r1] + regs[r2]
        elif op == Opcode.SUB:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = regs[r1] - regs[r2]
        elif op == Opcode.MUL:
            r1, r2, dest = instr.arg0, instr.arg1, instr.arg2
            regs[dest] = regs[r1] * regs[r2]
        elif op == Opcode.NEG:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = -regs[r]
        elif op == Opcode.POW2:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = regs[r] ** 2
        elif op == Opcode.POW3:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = regs[r] ** 3
        elif op == Opcode.INV:
            r, dest = instr.arg0, instr.arg1
            regs[dest] = 1.0 / regs[r] if abs(regs[r]) > 1e-300 else 0.0
        elif op == Opcode.STORE:
            r, row, col = instr.arg0, instr.arg1, instr.arg2
            M[row, col] = regs[r]

    return M


# ── Precompute constant residues ─────────────────────────────────────────

def _precompute_const_residues(program: CmfProgram, primes: np.ndarray) -> np.ndarray:
    K = len(primes)
    pp_list = [int(p) for p in primes]
    n_c = len(program.constants)
    table = np.zeros((n_c, K), dtype=np.int64)
    for idx in range(n_c):
        c = int(program.constants[idx])
        table[idx] = np.array([c % p if c >= 0 else (p - (-c % p)) % p
                                for p in pp_list], dtype=np.int64)
    return table


# ── Main CMF walk function ───────────────────────────────────────────────

def run_cmf_walk(
    program: CmfProgram,
    depth: int,
    K: int,
    shift_vals: List[int],
) -> Dict[str, Any]:
    """Walk an r×r companion matrix with per-axis shift values.

    Args:
        program:    compiled CmfProgram (r×r, multi-axis)
        depth:      walk steps
        K:          number of RNS primes
        shift_vals: per-axis starting offsets (len = program.dim)

    Returns:
        dict with p_residues, q_residues, p_float, q_float, log_scale, primes
    """
    r = program.m
    pp = generate_rns_primes(K).astype(np.int64)
    const_table = _precompute_const_residues(program, pp)

    # RNS accumulator: P[i,j,k] = entry (i,j) mod prime k, init to identity
    P_rns = np.zeros((r, r, K), dtype=np.int64)
    for i in range(r):
        P_rns[i, i] = np.ones(K, dtype=np.int64)

    # Float shadow: identity
    P_f = np.eye(r, dtype=np.float64)
    log_scale = 0.0

    for step in range(depth):
        M_rns = _eval_bytecode_allprimes_multiaxis(
            program, step, shift_vals, pp, const_table)
        M_f = _eval_bytecode_float_multiaxis(program, step, shift_vals)

        P_rns = _matmul_rxr_mod(P_rns, M_rns, pp, r)

        P_f = P_f @ M_f
        mx = np.max(np.abs(P_f))
        if mx > 1e10:
            P_f /= mx
            log_scale += math.log(mx)

    # Extract last column: p = P[0, r-1], q = P[r-1, r-1]
    p_res = P_rns[0, r - 1]
    q_res = P_rns[r - 1, r - 1]
    p_float = P_f[0, r - 1]
    q_float = P_f[r - 1, r - 1]

    return {
        'p_residues': p_res,
        'q_residues': q_res,
        'p_float': p_float,
        'q_float': q_float,
        'log_scale': log_scale,
        'primes': pp,
    }
