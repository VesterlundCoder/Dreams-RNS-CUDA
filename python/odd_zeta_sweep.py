#!/usr/bin/env python3
"""
Odd-Zeta CMF Sweep: compiled RNS walk for ζ(2n+1) CMFs.

Multi-dimensional exploration: each CMF has dim = 3n axes (per-row decomposition).
Supports rational shifts AND rational trajectories via modular inverse.
Float shadow (~15 digit precision) is the primary matching metric.

The 1000 trajectory vectors explore different "angles" in the multi-dim parameter
space by perturbing the standard direction vector one or two axes at a time.
Shifts perturb the standard starting offsets similarly.

All completed (shift, trajectory) combos are tracked for resume.

Pipeline:
  1. THIS SCRIPT → fast compiled-bytecode sweep
  2. odd_zeta_exact.py (CPU) → deep exact big-integer analysis of hits

Usage:
    python odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl

    # Full: 50 shift vectors × 1000 trajectory vectors in multi-dim space
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --cmf-name zeta_5 \
        --n-shifts 50 --n-traj 1000 --depth 1000 --K 16

    # Quick test
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --depth 200 \
        --n-shifts 3 --n-traj 5 --max-cmfs 1
"""

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns.compiler import compile_cmf_from_dict
from dreams_rns.cmf_walk import run_cmf_walk_vec
from dreams_rns.constants import load_constants, compute_match_digits

import sympy as sp
import mpmath as mp


# ── Compile odd-zeta CMF to bytecode ────────────────────────────────────

def compile_odd_zeta_cmf(spec: Dict):
    """Compile an odd-zeta CMF spec into a CmfProgram (once per CMF).

    Directions are stored in the program but can be overridden at walk time
    via trajectory_vals parameter.
    """
    matrix_dict = {}
    for key, expr_str in spec['matrix'].items():
        r, c = key.split(',')
        matrix_dict[(int(r), int(c))] = expr_str

    program = compile_cmf_from_dict(
        matrix_dict=matrix_dict,
        m=spec['rank'],
        dim=spec['dim'],
        axis_names=spec.get('axis_names', ['k']),
        directions=spec.get('directions', [1]),
    )
    return program


# ── Initial state for ζ(2n+1) ──────────────────────────────────────────

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


# ── Sphere coverage: auto-scale with dimensionality ─────────────────

def compute_sphere_coverage(dim: int, budget: int) -> Tuple[int, int]:
    """Compute (n_traj, n_shifts) to cover a D-dimensional sphere within budget.

    Enumerable directions on a D-dim sphere:
      Layer 1 (single-axis):  D × 10 mults = 10D
      Layer 2 (two-axis):     C(D,2) × 25 pairs = ~12.5 D²
      Layer 3 (three-axis):   C(D,3) × 27 triples (capped)

    Shifts need fewer (shifts modulate starting point, not direction):
      Layer 1: D × 10 offsets
      Layer 2: C(D,2) × 9 pairs

    Budget is a HARD CAP: n_traj × n_shifts <= budget.
    Trajectories get ~80% of sqrt-budget, shifts get ~20%.
    """
    import math as _m

    # Max enumerable per type
    traj_L1 = 1 + dim * 10                       # single-axis: complete
    traj_L2 = dim * (dim - 1) // 2 * 25          # two-axis pairs
    traj_L3 = dim * (dim - 1) * (dim - 2) // 6 * 27  # three-axis
    traj_full = traj_L1 + traj_L2 + traj_L3

    shift_L1 = 1 + dim * 10
    shift_L2 = dim * (dim - 1) // 2 * 9
    shift_full = shift_L1 + shift_L2

    # Strategy: trajectories scale with O(D²) to cover pair directions on
    # the sphere; shifts get the remaining budget.
    #
    # Target traj: max(L1, 5·D²) — covers all single-axis + many pairs
    # Target shifts: budget / n_traj — whatever is left
    # Both capped at their enumerable maximum.

    traj_target = max(traj_L1, 5 * dim * dim)
    n_traj = min(traj_full, traj_target)

    if budget <= n_traj:
        # Budget too small for target — all to traj, 1 shift
        return min(budget, traj_full), 1

    n_shifts = max(1, min(shift_full, budget // n_traj))

    # If shifts are saturated and budget remains, pour rest into traj
    if n_shifts >= shift_full:
        n_shifts = shift_full
        n_traj = min(traj_full, budget // n_shifts)

    # Hard cap
    while n_traj * n_shifts > budget and n_shifts > 1:
        n_shifts -= 1
    while n_traj * n_shifts > budget and n_traj > 1:
        n_traj -= 1

    return n_traj, n_shifts


# ── Multi-dim trajectory generation ────────────────────────────────────

# Perturbation multipliers for trajectory exploration
TRAJ_MULTIPLIERS = [
    Fraction(1, 3), Fraction(1, 2), Fraction(2, 3),
    Fraction(3, 2), Fraction(2), Fraction(3),
    Fraction(-1), Fraction(-1, 2), Fraction(-1, 3),
    Fraction(0),
]


def generate_trajectory_vectors(
    dim: int,
    default_dirs: List[int],
    n_traj: int,
) -> List[List[Fraction]]:
    """Generate trajectory vectors covering a D-dim sphere around default_dirs.

    Systematic enumeration in layers:
      Layer 0: standard trajectory (1 vector)
      Layer 1: single-axis perturbations (D × 10 vectors)
      Layer 2: two-axis perturbations (C(D,2) × 25 vectors)
      Layer 3: three-axis perturbations (C(D,3) × 27 vectors)

    Generates ALL vectors in each layer before moving to the next,
    ensuring uniform sphere coverage up to n_traj.
    """
    base = [Fraction(d) for d in default_dirs]
    trajs = [list(base)]
    seen = {_vec_key(base)}

    def _try_add(vec):
        if len(trajs) >= n_traj:
            return False
        key = _vec_key(vec)
        if key not in seen:
            trajs.append(vec)
            seen.add(key)
        return True

    # Layer 1: ALL single-axis perturbations (complete sphere axes)
    for axis in range(dim):
        for mult in TRAJ_MULTIPLIERS:
            vec = list(base)
            vec[axis] = base[axis] * mult
            if not _try_add(vec) and len(trajs) >= n_traj:
                break

    # Layer 2: ALL two-axis perturbations
    if len(trajs) < n_traj:
        for a1 in range(dim):
            for a2 in range(a1 + 1, dim):
                for m1 in TRAJ_MULTIPLIERS[:5]:
                    for m2 in TRAJ_MULTIPLIERS[:5]:
                        vec = list(base)
                        vec[a1] = base[a1] * m1
                        vec[a2] = base[a2] * m2
                        if not _try_add(vec):
                            pass
                        if len(trajs) >= n_traj:
                            return trajs[:n_traj]

    # Layer 3: ALL three-axis perturbations (full dim, no caps)
    if len(trajs) < n_traj:
        for a1 in range(dim):
            for a2 in range(a1 + 1, dim):
                for a3 in range(a2 + 1, dim):
                    for m1 in TRAJ_MULTIPLIERS[:3]:
                        for m2 in TRAJ_MULTIPLIERS[:3]:
                            for m3 in TRAJ_MULTIPLIERS[:3]:
                                vec = list(base)
                                vec[a1] = base[a1] * m1
                                vec[a2] = base[a2] * m2
                                vec[a3] = base[a3] * m3
                                _try_add(vec)
                                if len(trajs) >= n_traj:
                                    return trajs[:n_traj]

    return trajs[:n_traj]


# ── Multi-dim shift generation ─────────────────────────────────────

SHIFT_OFFSETS = [
    Fraction(1), Fraction(-1),
    Fraction(1, 2), Fraction(-1, 2),
    Fraction(1, 3), Fraction(-1, 3),
    Fraction(2), Fraction(-2),
    Fraction(1, 4), Fraction(-1, 4),
]


def generate_shift_vectors(
    dim: int,
    default_shifts: List[int],
    n_shifts: int,
) -> List[List[Fraction]]:
    """Generate shift vectors covering a D-dim sphere around default_shifts.

    Layers:
      Layer 0: standard shift (1 vector)
      Layer 1: single-axis offsets (D × 10 vectors)
      Layer 2: two-axis offsets (C(D,2) × 9 vectors)
    """
    base = [Fraction(s) for s in default_shifts]
    shifts = [list(base)]
    seen = {_vec_key(base)}

    def _try_add(vec):
        if len(shifts) >= n_shifts:
            return False
        key = _vec_key(vec)
        if key not in seen:
            shifts.append(vec)
            seen.add(key)
        return True

    # Layer 1: ALL single-axis offsets
    for axis in range(dim):
        for offset in SHIFT_OFFSETS:
            vec = list(base)
            vec[axis] = base[axis] + offset
            _try_add(vec)

    # Layer 2: ALL two-axis offsets
    if len(shifts) < n_shifts:
        pair_offsets = [Fraction(1), Fraction(-1), Fraction(1, 2)]
        for a1 in range(dim):
            for a2 in range(a1 + 1, dim):
                for o1 in pair_offsets:
                    for o2 in pair_offsets:
                        vec = list(base)
                        vec[a1] = base[a1] + o1
                        vec[a2] = base[a2] + o2
                        _try_add(vec)
                        if len(shifts) >= n_shifts:
                            return shifts[:n_shifts]

    return shifts[:n_shifts]


def _vec_key(vec: List[Fraction]) -> str:
    """Hashable key for a Fraction vector."""
    return "|".join(str(v) for v in vec)


def _vec_str(vec: List[Fraction]) -> str:
    """Compact string representation for results."""
    return "[" + ",".join(str(v) for v in vec) + "]"


# ── Resume: load already-computed keys ─────────────────────────────────

def load_completed_keys(results_path: str) -> Set[str]:
    """Load (cmf, shift_str, traj_str) keys from existing JSONL results."""
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    key = f"{r['cmf']}|{r['shift']}|{r['trajectory']}"
                    done.add(key)
                except Exception:
                    pass
    return done


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Odd-Zeta CMF Sweep — compiled RNS walk, multi-dimensional"
    )
    parser.add_argument("--specs", type=str, required=True,
                        help="CMF specs JSONL from odd_zeta_cmf_generator.py")
    parser.add_argument("--depth", type=int, default=1000,
                        help="Walk depth (default 1000)")
    parser.add_argument("--budget", type=int, default=50000,
                        help="Max walks per CMF. Auto-scales shifts/traj with dim.")
    parser.add_argument("--n-shifts", type=int, default=0,
                        help="Override shift count (0=auto from budget+dim)")
    parser.add_argument("--n-traj", type=int, default=0,
                        help="Override traj count (0=auto from budget+dim)")
    parser.add_argument("--K", type=int, default=16,
                        help="RNS primes (default 16; float shadow is primary)")
    parser.add_argument("--dps", type=int, default=50,
                        help="mpmath dps for constant evaluation")
    parser.add_argument("--max-cmfs", type=int, default=0,
                        help="Max CMFs to process (0=all)")
    parser.add_argument("--skip-cmfs", type=int, default=0,
                        help="Skip first N CMFs")
    parser.add_argument("--cmf-name", type=str, default="",
                        help="Process only this CMF, e.g. zeta_5")
    parser.add_argument("--output", type=str, default="odd_zeta_results/",
                        help="Output directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-computed combos in output")
    args = parser.parse_args()

    # Load constants
    constants = load_constants(args.dps)
    print(f"Constants bank: {len(constants)} constants")

    # Load CMF specs
    specs = []
    with open(args.specs) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))

    if args.cmf_name:
        specs = [s for s in specs if s['name'] == args.cmf_name]
        if not specs:
            print(f"ERROR: no CMF named '{args.cmf_name}' found")
            sys.exit(1)
    else:
        if args.skip_cmfs > 0:
            specs = specs[args.skip_cmfs:]
        if args.max_cmfs > 0:
            specs = specs[:args.max_cmfs]

    print(f"\nOdd-Zeta CMF Sweep — compiled RNS, multi-dimensional")
    print(f"  CMFs:        {len(specs)}")
    print(f"  Budget/CMF:  {args.budget} walks")
    print(f"  Depth:       {args.depth}")
    print(f"  K:           {args.K}")
    print(f"  Constants:   {len(constants)}")
    print(f"  Resume:      {args.resume}")
    # Show per-CMF scaling
    for s in specs:
        d = s['dim']
        nt, ns = compute_sphere_coverage(d, args.budget)
        if args.n_traj > 0: nt = args.n_traj
        if args.n_shifts > 0: ns = args.n_shifts
        print(f"    {s['name']:>10}: dim={d:>2} → {ns} shifts × {nt} traj = {ns*nt} walks")
    print(f"{'='*80}")

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "odd_zeta_results.jsonl")

    done_keys = load_completed_keys(results_path) if args.resume else set()
    if done_keys:
        print(f"  Resuming: {len(done_keys)} results already computed")

    results_file = open(results_path, 'a')
    total_new = 0
    total_hits = 0

    try:
        for ci, spec in enumerate(specs):
            name = spec['name']
            zeta_val = spec['zeta_val']
            rank = spec['rank']
            n = spec['n']
            dim = spec['dim']
            acc_idx = spec.get('accumulator_idx', n)
            const_idx = spec.get('constant_idx', n + 1)
            default_dirs = spec.get('directions', [1])
            default_shifts = spec.get('default_shifts', [1] * dim)

            print(f"\n{'━'*80}")
            print(f"[{ci+1}/{len(specs)}] {name}: ζ({zeta_val}), "
                  f"{rank}×{rank} matrix, {dim} axes")

            # Compute initial state
            v0 = compute_initial_state(n)

            # Compile bytecode ONCE per CMF (trajectory passed at walk time)
            try:
                program = compile_odd_zeta_cmf(spec)
            except Exception as e:
                print(f"  COMPILE ERROR: {e}")
                continue

            # Auto-scale shifts/traj based on dim and budget
            auto_nt, auto_ns = compute_sphere_coverage(dim, args.budget)
            nt = args.n_traj if args.n_traj > 0 else auto_nt
            ns = args.n_shifts if args.n_shifts > 0 else auto_ns

            shift_vecs = generate_shift_vectors(dim, default_shifts, ns)
            traj_vecs = generate_trajectory_vectors(dim, default_dirs, nt)
            print(f"  Generated {len(shift_vecs)} shift vecs × "
                  f"{len(traj_vecs)} traj vecs = "
                  f"{len(shift_vecs) * len(traj_vecs)} walks "
                  f"(dim={dim}, budget={args.budget})")
            print(f"  Sphere coverage: L1={dim*10} axis, "
                  f"L2={dim*(dim-1)//2} pairs, "
                  f"L3={dim*(dim-1)*(dim-2)//6} triples")

            t_cmf_start = time.time()
            cmf_new = 0
            cmf_hits = 0
            best_hit = None
            n_tasks = len(shift_vecs) * len(traj_vecs)

            for ti, traj_vec in enumerate(traj_vecs):
                traj_str = _vec_str(traj_vec)

                for si, shift_vec in enumerate(shift_vecs):
                    shift_str = _vec_str(shift_vec)
                    key = f"{name}|{shift_str}|{traj_str}"
                    if key in done_keys:
                        continue

                    try:
                        res = run_cmf_walk_vec(
                            program, args.depth, args.K,
                            shift_vals=shift_vec,
                            initial_state=v0,
                            acc_idx=acc_idx,
                            const_idx=const_idx,
                            trajectory_vals=traj_vec,
                        )

                        pf = res['p_float']
                        qf = res['q_float']
                        est = pf / qf if abs(qf) > 1e-300 else float('nan')

                        if not math.isfinite(est):
                            continue

                        # Match against all constants (ratio-aware near 1.0)
                        best_const, best_digits = compute_match_digits(
                            est, constants)

                        # Float delta (secondary)
                        log_q = res['log_scale'] + (
                            math.log(abs(qf)) if abs(qf) > 1e-300 else 0)
                        if best_const != "none" and log_q > 1:
                            target_f = next(
                                c['value_float'] for c in constants
                                if c['name'] == best_const)
                            err = abs(est - target_f)
                            best_delta = -(1.0 + math.log(err) / log_q) if err > 0 else float('inf')
                        else:
                            best_delta = float('nan')

                        result = {
                            'cmf': name,
                            'zeta_val': zeta_val,
                            'rank': rank,
                            'dim': dim,
                            'shift': shift_str,
                            'trajectory': traj_str,
                            'depth': args.depth,
                            'est': round(est, 15),
                            'best_const': best_const,
                            'best_delta': round(best_delta, 6) if math.isfinite(best_delta) else str(best_delta),
                            'match_digits': round(best_digits, 1),
                            'K': args.K,
                        }

                        results_file.write(json.dumps(result, default=str) + "\n")
                        cmf_new += 1
                        total_new += 1
                        done_keys.add(key)

                        if best_digits >= 6:
                            cmf_hits += 1
                            total_hits += 1

                        if best_digits > (best_hit['match_digits'] if best_hit else -1):
                            best_hit = result

                    except Exception as e:
                        if ti == 0 and si == 0:
                            print(f"  WALK ERROR: {e}")
                            import traceback; traceback.print_exc()
                        continue

                # Progress
                if (ti + 1) % 50 == 0 or ti == len(traj_vecs) - 1:
                    elapsed = time.time() - t_cmf_start
                    rate = cmf_new / max(elapsed, 0.001)
                    done_pct = (ti + 1) * len(shift_vecs) / n_tasks * 100
                    print(f"  traj {ti+1}/{len(traj_vecs)} ({done_pct:.0f}%): "
                          f"{cmf_new} walks, {cmf_hits} hits (≥6d), "
                          f"{rate:.1f} walks/s", flush=True)

            results_file.flush()
            elapsed = time.time() - t_cmf_start

            if best_hit:
                print(f"  Best: {best_hit['match_digits']} digits → "
                      f"{best_hit['best_const']} "
                      f"(est={best_hit['est']})")
            print(f"  Total: {cmf_new} new walks in {elapsed:.1f}s")

    finally:
        results_file.close()

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"  New walks:  {total_new}")
    print(f"  Hits (≥6d): {total_hits}")
    print(f"  Results:    {results_path}")
    print(f"  Total keys: {len(done_keys)}")


if __name__ == "__main__":
    main()
