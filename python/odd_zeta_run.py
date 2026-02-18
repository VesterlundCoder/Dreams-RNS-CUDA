#!/usr/bin/env python3
"""
Lightweight odd-zeta runner for Lambda Labs / remote GPU machines.

Self-contained: generates CMF spec, compiles, and runs a small batch.
No separate spec file needed. Resume-safe via JSONL append.

Usage:
    # Quick smoke test (ζ(5), 10 walks)
    python odd_zeta_run.py --zeta 5 --n-shifts 2 --n-traj 5 --depth 200

    # Medium run (ζ(5), 1000 walks)
    python odd_zeta_run.py --zeta 5 --n-shifts 20 --n-traj 50 --depth 1000

    # Chunk a larger run into pages (page 0 = first 1000 traj, page 1 = next 1000, ...)
    python odd_zeta_run.py --zeta 5 --n-shifts 50 --n-traj 1000 --depth 1000 --page 0 --page-size 200
    python odd_zeta_run.py --zeta 5 --n-shifts 50 --n-traj 1000 --depth 1000 --page 1 --page-size 200

    # All odd zetas from ζ(5) to ζ(21)
    for z in 5 7 9 11 13 15 17 19 21; do
        python odd_zeta_run.py --zeta $z --n-shifts 10 --n-traj 50 --depth 1000 &
    done
    wait
"""

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns.compiler import compile_cmf_from_dict
from dreams_rns.cmf_walk import run_cmf_walk_vec


# ── Inline CMF spec generation (no external file needed) ──────────────

def build_zeta_cmf(n: int):
    """Build ζ(2n+1) CMF spec inline. Returns (matrix_dict, rank, dim, axis_names, dirs, shifts)."""
    import sympy as sp

    d = 2 * n  # matrix rank
    xs = [sp.Symbol(f'x{i}') for i in range(3 * n)]
    axis_names = [f'x{i}' for i in range(3 * n)]
    dim = 3 * n

    # Directions and default shifts for standard HPHP08 construction
    directions = []
    default_shifts = []
    for j in range(n):
        directions.extend([1, 2])       # num_j, den_j
        default_shifts.extend([2, 3])   # default offsets
    directions.append(1)                # coupling axis
    default_shifts.append(1)
    # remaining coupling axes for j >= 2
    for j in range(1, n):
        directions.append(1)
        default_shifts.append(2)

    # Pad to exactly dim
    while len(directions) < dim:
        directions.append(1)
    while len(default_shifts) < dim:
        default_shifts.append(1)
    directions = directions[:dim]
    default_shifts = default_shifts[:dim]

    # Build matrix entries
    matrix_dict = {}

    # L-shaped matrix: U-rows (0..n-1) and accumulator rows (n..d-1)
    # Row j in U-block (j = 0..n-1):
    #   M[j, j]   = num_j / den_j   (diagonal)
    #   M[j, n]   = coupling_j       (off-diagonal to accumulator)
    # Accumulator block: M[n+i, n+i] = 1 (identity), M[n, n] += accumulator term

    for j in range(n):
        num_sym = xs[2 * j]       # x_{2j}
        den_sym = xs[2 * j + 1]   # x_{2j+1}

        # Diagonal: num/den with k-dependent polynomial
        # Standard: (k+1)^2 * (k+1)^(2j) ... simplified to product form
        # Use generic per-row factor: num_j * (num_j + 1) for numerator part
        if j == 0:
            numer = -(num_sym * (num_sym + 1))**2
            denom = den_sym**2
        else:
            numer = -(num_sym * (num_sym + 1))**2
            denom = den_sym**2

        matrix_dict[(j, j)] = str(sp.expand(numer / denom))

        # Coupling to accumulator row
        coupling_idx = 2 * n + min(j, n - 1)
        coupling_sym = xs[coupling_idx] if coupling_idx < dim else sp.Integer(1)
        matrix_dict[(j, n)] = str(coupling_sym)

    # Accumulator rows: identity diagonal
    for i in range(n, d):
        matrix_dict[(i, i)] = '1'

    # Accumulator (row n): receives contributions
    if n >= 1:
        matrix_dict[(n, 0)] = str(xs[0])  # feedback from first U-row

    # Convert to string keys for JSON compat
    str_matrix = {}
    for (r, c), expr in matrix_dict.items():
        str_matrix[f"{r},{c}"] = expr

    return {
        'name': f'zeta_{2*n+1}',
        'zeta_val': 2 * n + 1,
        'n': n,
        'rank': d,
        'dim': dim,
        'matrix': str_matrix,
        'axis_names': axis_names,
        'directions': directions,
        'default_shifts': default_shifts,
        'accumulator_idx': n,
        'constant_idx': n + 1,
    }


def build_zeta_cmf_from_generator(n: int):
    """Use the full generator if available, fall back to inline version."""
    try:
        from odd_zeta_cmf_generator import build_cmf_matrix_2n
        matrix_dict, dim, axis_names, dirs, def_shifts, target = build_cmf_matrix_2n(n)
        str_matrix = {}
        for (r, c), expr_str in matrix_dict.items():
            str_matrix[f"{r},{c}"] = expr_str
        return {
            'name': f'zeta_{target}',
            'zeta_val': target,
            'n': n,
            'rank': 2 * n,
            'dim': dim,
            'matrix': str_matrix,
            'axis_names': axis_names,
            'directions': dirs,
            'default_shifts': def_shifts,
            'accumulator_idx': n,
            'constant_idx': n + 1,
        }
    except ImportError:
        return build_zeta_cmf(n)


# ── Initial state ─────────────────────────────────────────────────────

def compute_initial_state(n: int) -> List[Fraction]:
    """Compute initial state v(0) for ζ(2n+1) in exact rationals."""
    d = 2 * n

    def A_k(k):
        from math import comb
        return Fraction((-1)**(k-1), comb(2*k, k))

    def e_list(k_minus_1, nn):
        e = [Fraction(0)] * nn
        e[0] = Fraction(1)
        for m in range(1, k_minus_1 + 1):
            xm = Fraction(1, m**2)
            for j in reversed(range(1, nn)):
                e[j] = e[j] + xm * e[j-1]
        return e

    def kernel_coeffs(k, nn):
        coeffs = [Fraction(0)] * nn
        coeffs[nn-1] += Fraction(5) * Fraction((-1)**(nn-1))
        for j in range(0, nn-1):
            t = (nn-1) - j
            coeffs[j] += Fraction(4) * Fraction((-1)**j) / Fraction(k**(2*t))
        return coeffs

    def direct_term(k, nn):
        Ak = A_k(k)
        e = e_list(k-1, nn)
        c = kernel_coeffs(k, nn)
        s = sum(c[j] * e[j] for j in range(nn))
        return Fraction(1, 2) * Ak * s / Fraction(k**3)

    v = [Fraction(0)] * d
    v[0] = Fraction(1, 2)
    v[n] = direct_term(1, n)
    v[n+1] = Fraction(1)
    return v


# ── Trajectory + shift generation (small, deterministic) ──────────────

TRAJ_MULTS = [
    Fraction(1, 3), Fraction(1, 2), Fraction(2, 3),
    Fraction(3, 2), Fraction(2), Fraction(3),
    Fraction(-1), Fraction(-1, 2), Fraction(-1, 3),
    Fraction(0),
]

SHIFT_OFFSETS = [
    Fraction(1), Fraction(-1),
    Fraction(1, 2), Fraction(-1, 2),
    Fraction(1, 3), Fraction(-1, 3),
    Fraction(2), Fraction(-2),
    Fraction(1, 4), Fraction(-1, 4),
]


def gen_trajs(dim, default_dirs, n_traj):
    """Generate trajectory vectors: standard + single-axis + two-axis."""
    base = [Fraction(d) for d in default_dirs]
    out = [list(base)]
    seen = {_key(base)}

    # Single-axis
    for ax in range(dim):
        for m in TRAJ_MULTS:
            v = list(base)
            v[ax] = base[ax] * m
            k = _key(v)
            if k not in seen:
                out.append(v)
                seen.add(k)
                if len(out) >= n_traj:
                    return out

    # Two-axis
    for a1 in range(dim):
        for a2 in range(a1 + 1, dim):
            for m1 in TRAJ_MULTS[:5]:
                for m2 in TRAJ_MULTS[:5]:
                    v = list(base)
                    v[a1] = base[a1] * m1
                    v[a2] = base[a2] * m2
                    k = _key(v)
                    if k not in seen:
                        out.append(v)
                        seen.add(k)
                        if len(out) >= n_traj:
                            return out
    return out[:n_traj]


def gen_shifts(dim, default_shifts, n_shifts):
    """Generate shift vectors: standard + single-axis offsets."""
    base = [Fraction(s) for s in default_shifts]
    out = [list(base)]
    seen = {_key(base)}

    for ax in range(dim):
        for off in SHIFT_OFFSETS:
            v = list(base)
            v[ax] = base[ax] + off
            k = _key(v)
            if k not in seen:
                out.append(v)
                seen.add(k)
                if len(out) >= n_shifts:
                    return out

    # Two-axis
    for a1 in range(dim):
        for a2 in range(a1 + 1, dim):
            for o1 in SHIFT_OFFSETS[:3]:
                for o2 in SHIFT_OFFSETS[:3]:
                    v = list(base)
                    v[a1] = base[a1] + o1
                    v[a2] = base[a2] + o2
                    k = _key(v)
                    if k not in seen:
                        out.append(v)
                        seen.add(k)
                        if len(out) >= n_shifts:
                            return out
    return out[:n_shifts]


def _key(v):
    return "|".join(str(x) for x in v)


def _vstr(v):
    return "[" + ",".join(str(x) for x in v) + "]"


# ── Constants (inline, no external file) ──────────────────────────────

def load_zeta_constants():
    """Small set of constants relevant to odd zeta values."""
    import mpmath as mp
    mp.mp.dps = 50
    consts = []
    for k in range(3, 25, 2):
        val = float(mp.zeta(k))
        consts.append({'name': f'zeta({k})', 'value_float': val})
    # A few extras
    consts.append({'name': 'pi', 'value_float': float(mp.pi)})
    consts.append({'name': 'pi^2/6', 'value_float': float(mp.pi**2 / 6)})
    consts.append({'name': 'ln2', 'value_float': float(mp.log(2))})
    consts.append({'name': 'euler_gamma', 'value_float': float(mp.euler)})
    consts.append({'name': 'catalan', 'value_float': float(mp.catalan)})
    return consts


# ── Resume ────────────────────────────────────────────────────────────

def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add(f"{r['shift']}|{r['trajectory']}")
                except Exception:
                    pass
    return done


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lightweight odd-zeta CMF runner for Lambda Labs"
    )
    parser.add_argument("--zeta", type=int, required=True,
                        help="Odd zeta value: 5, 7, 9, 11, 13, ...")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n-shifts", type=int, default=10,
                        help="Number of shift vectors (default 10)")
    parser.add_argument("--n-traj", type=int, default=50,
                        help="Number of trajectory vectors (default 50)")
    parser.add_argument("--page", type=int, default=-1,
                        help="Trajectory page index for chunked runs (-1=all)")
    parser.add_argument("--page-size", type=int, default=200,
                        help="Trajectories per page (default 200)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSONL (default: zeta_N_results.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed combos")
    args = parser.parse_args()

    zeta_val = args.zeta
    if zeta_val < 5 or zeta_val % 2 == 0:
        print(f"ERROR: --zeta must be an odd integer >= 5, got {zeta_val}")
        sys.exit(1)
    n = (zeta_val - 1) // 2  # ζ(2n+1)

    if not args.output:
        args.output = f"zeta_{zeta_val}_results.jsonl"

    print(f"Building ζ({zeta_val}) CMF (n={n})...")
    spec = build_zeta_cmf_from_generator(n)
    dim = spec['dim']
    rank = spec['rank']
    print(f"  rank={rank}, dim={dim}, axes={spec['axis_names']}")

    # Compile
    matrix_dict = {}
    for key, expr_str in spec['matrix'].items():
        r, c = key.split(',')
        matrix_dict[(int(r), int(c))] = expr_str

    program = compile_cmf_from_dict(
        matrix_dict, m=rank, dim=dim,
        axis_names=spec['axis_names'],
        directions=spec['directions'],
    )
    print(f"  compiled: {len(program.opcodes)} instructions")

    # Initial state
    v0 = compute_initial_state(n)

    # Generate vectors
    all_trajs = gen_trajs(dim, spec['directions'], args.n_traj)
    shifts = gen_shifts(dim, spec['default_shifts'], args.n_shifts)

    # Page slicing for trajectories
    if args.page >= 0:
        start = args.page * args.page_size
        end = start + args.page_size
        trajs = all_trajs[start:end]
        page_info = f" (page {args.page}: traj {start}-{min(end, len(all_trajs))})"
    else:
        trajs = all_trajs
        page_info = ""

    total = len(shifts) * len(trajs)
    print(f"\n  shifts:  {len(shifts)}")
    print(f"  trajs:   {len(trajs)}{page_info}")
    print(f"  total:   {total} walks")
    print(f"  depth:   {args.depth}, K={args.K}")
    print(f"  output:  {args.output}")

    # Constants
    constants = load_zeta_constants()

    # Resume
    done = load_done(args.output) if args.resume else set()
    if done:
        print(f"  resume:  {len(done)} already done")

    acc_idx = spec.get('accumulator_idx', n)
    const_idx = spec.get('constant_idx', n + 1)

    print(f"\n{'='*70}")
    print(f"  ζ({zeta_val}) sweep: {len(shifts)} shifts × {len(trajs)} traj "
          f"= {total} walks")
    print(f"{'='*70}\n")

    fout = open(args.output, 'a')
    n_done = 0
    n_hits = 0
    best_digits = -1
    best_result = None
    t0 = time.time()

    try:
        for ti, traj in enumerate(trajs):
            traj_str = _vstr(traj)

            for si, shift in enumerate(shifts):
                shift_str = _vstr(shift)
                key = f"{shift_str}|{traj_str}"
                if key in done:
                    continue

                try:
                    res = run_cmf_walk_vec(
                        program, args.depth, args.K,
                        shift_vals=shift,
                        initial_state=v0,
                        acc_idx=acc_idx,
                        const_idx=const_idx,
                        trajectory_vals=traj,
                    )

                    pf = res['p_float']
                    qf = res['q_float']
                    est = pf / qf if abs(qf) > 1e-300 else float('nan')
                    if not math.isfinite(est):
                        continue

                    # Match
                    bc, bd = "none", 0.0
                    for c in constants:
                        err = abs(est - c['value_float'])
                        if err == 0:
                            d = 16.0
                        elif err > 0 and c['value_float'] != 0:
                            d = -math.log10(err / max(abs(c['value_float']), 1e-300))
                        else:
                            d = 0.0
                        if d > bd:
                            bd = d
                            bc = c['name']

                    result = {
                        'cmf': f'zeta_{zeta_val}',
                        'shift': shift_str,
                        'trajectory': traj_str,
                        'est': round(est, 15),
                        'best_const': bc,
                        'match_digits': round(max(bd, 0), 1),
                        'depth': args.depth,
                    }
                    fout.write(json.dumps(result) + "\n")
                    n_done += 1
                    done.add(key)

                    if bd >= 6:
                        n_hits += 1
                    if bd > best_digits:
                        best_digits = bd
                        best_result = result

                except Exception as e:
                    if ti == 0 and si == 0:
                        print(f"  ERROR: {e}")
                        import traceback; traceback.print_exc()
                    continue

            # Progress every 10 trajectories
            if (ti + 1) % 10 == 0 or ti == len(trajs) - 1:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 0.001)
                pct = (ti + 1) / len(trajs) * 100
                print(f"  [{ti+1:>4}/{len(trajs)}] {pct:5.1f}%  "
                      f"{n_done:>5} walks  {n_hits} hits(≥6d)  "
                      f"{rate:.1f}/s", flush=True)

    except KeyboardInterrupt:
        print("\n  Interrupted — partial results saved.")
    finally:
        fout.flush()
        fout.close()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE: {n_done} walks in {elapsed:.1f}s ({n_done/max(elapsed,1):.1f}/s)")
    print(f"  Hits (≥6 digits): {n_hits}")
    if best_result:
        print(f"  Best: {best_result['match_digits']} digits → {best_result['best_const']}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
