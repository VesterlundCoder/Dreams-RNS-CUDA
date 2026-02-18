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
from dreams_rns.constants import load_constants

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
    """Generate n_traj trajectory vectors in dim-dimensional space.

    Strategy:
      1. Standard trajectory (default_dirs)
      2. Single-axis perturbations: scale one axis by rational multipliers
      3. Two-axis perturbations: scale pairs of axes
      4. Fill remaining with more combinations

    Returns list of Fraction vectors, each of length dim.
    """
    base = [Fraction(d) for d in default_dirs]
    trajs = [list(base)]  # index 0 = standard
    seen = {_vec_key(base)}

    # Phase 1: single-axis perturbations
    for axis in range(dim):
        for mult in TRAJ_MULTIPLIERS:
            if len(trajs) >= n_traj:
                break
            vec = list(base)
            vec[axis] = base[axis] * mult
            key = _vec_key(vec)
            if key not in seen:
                trajs.append(vec)
                seen.add(key)

    # Phase 2: two-axis perturbations
    for a1 in range(dim):
        for a2 in range(a1 + 1, dim):
            for m1 in TRAJ_MULTIPLIERS[:5]:
                for m2 in TRAJ_MULTIPLIERS[:5]:
                    if len(trajs) >= n_traj:
                        break
                    vec = list(base)
                    vec[a1] = base[a1] * m1
                    vec[a2] = base[a2] * m2
                    key = _vec_key(vec)
                    if key not in seen:
                        trajs.append(vec)
                        seen.add(key)
                if len(trajs) >= n_traj:
                    break
            if len(trajs) >= n_traj:
                break
        if len(trajs) >= n_traj:
            break

    # Phase 3: three-axis perturbations if still room
    if len(trajs) < n_traj:
        for a1 in range(min(dim, 6)):
            for a2 in range(a1 + 1, min(dim, 8)):
                for a3 in range(a2 + 1, min(dim, 10)):
                    for m1 in TRAJ_MULTIPLIERS[:3]:
                        for m2 in TRAJ_MULTIPLIERS[:3]:
                            for m3 in TRAJ_MULTIPLIERS[:3]:
                                if len(trajs) >= n_traj:
                                    break
                                vec = list(base)
                                vec[a1] = base[a1] * m1
                                vec[a2] = base[a2] * m2
                                vec[a3] = base[a3] * m3
                                key = _vec_key(vec)
                                if key not in seen:
                                    trajs.append(vec)
                                    seen.add(key)
                            if len(trajs) >= n_traj:
                                break
                        if len(trajs) >= n_traj:
                            break
                    if len(trajs) >= n_traj:
                        break
                if len(trajs) >= n_traj:
                    break
            if len(trajs) >= n_traj:
                break

    return trajs[:n_traj]


# ── Multi-dim shift generation ─────────────────────────────────────────

SHIFT_OFFSETS = [
    Fraction(0),
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
    """Generate n_shifts shift vectors in dim-dimensional space.

    Strategy:
      1. Standard shift (default_shifts)
      2. Single-axis offsets: add ±1/d to one axis
      3. Two-axis offsets
    """
    base = [Fraction(s) for s in default_shifts]
    shifts = [list(base)]
    seen = {_vec_key(base)}

    # Single-axis offsets
    for axis in range(dim):
        for offset in SHIFT_OFFSETS:
            if offset == 0:
                continue
            if len(shifts) >= n_shifts:
                break
            vec = list(base)
            vec[axis] = base[axis] + offset
            key = _vec_key(vec)
            if key not in seen:
                shifts.append(vec)
                seen.add(key)

    # Two-axis offsets
    for a1 in range(dim):
        for a2 in range(a1 + 1, dim):
            for o1 in SHIFT_OFFSETS[1:4]:
                for o2 in SHIFT_OFFSETS[1:4]:
                    if len(shifts) >= n_shifts:
                        break
                    vec = list(base)
                    vec[a1] = base[a1] + o1
                    vec[a2] = base[a2] + o2
                    key = _vec_key(vec)
                    if key not in seen:
                        shifts.append(vec)
                        seen.add(key)
                if len(shifts) >= n_shifts:
                    break
            if len(shifts) >= n_shifts:
                break
        if len(shifts) >= n_shifts:
            break

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
    parser.add_argument("--n-shifts", type=int, default=50,
                        help="Number of multi-dim shift vectors to generate")
    parser.add_argument("--n-traj", type=int, default=1000,
                        help="Number of multi-dim trajectory vectors to generate")
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
    print(f"  Shift vecs:  {args.n_shifts}")
    print(f"  Traj vecs:   {args.n_traj}")
    print(f"  Tasks/CMF:   {args.n_shifts * args.n_traj}")
    print(f"  Depth:       {args.depth}")
    print(f"  K:           {args.K}")
    print(f"  Constants:   {len(constants)}")
    print(f"  Resume:      {args.resume}")
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

            # Generate multi-dim shift and trajectory vectors
            shift_vecs = generate_shift_vectors(dim, default_shifts, args.n_shifts)
            traj_vecs = generate_trajectory_vectors(dim, default_dirs, args.n_traj)
            print(f"  Generated {len(shift_vecs)} shift vecs × "
                  f"{len(traj_vecs)} traj vecs = "
                  f"{len(shift_vecs) * len(traj_vecs)} walks")

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

                        # Match against all constants
                        best_const = "none"
                        best_digits = -1.0

                        for c in constants:
                            err = abs(est - c['value_float'])
                            if err == 0:
                                digits = 16.0
                            elif err > 0 and c['value_float'] != 0:
                                digits = -math.log10(
                                    err / max(abs(c['value_float']), 1e-300))
                            else:
                                digits = 0.0
                            if digits > best_digits:
                                best_digits = digits
                                best_const = c['name']
                        best_digits = max(best_digits, 0.0)

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
