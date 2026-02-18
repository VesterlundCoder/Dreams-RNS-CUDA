#!/usr/bin/env python3
"""
Odd-Zeta CMF Sweep: compiled RNS walk for ζ(2n+1) CMFs.

Uses the Dreams RNS compiled bytecode pipeline — NO sympy in the hot loop.
Supports rational shifts (±1/d) via modular inverse in the bytecode evaluator.
Float shadow (~15 digit precision) is the primary matching metric for sweep.

Shifts and trajectories explored are tracked in the output JSONL so runs
can be resumed and extended without repeating work.

Pipeline:
  1. THIS SCRIPT → fast compiled-bytecode sweep, many shifts × trajectories
  2. odd_zeta_exact.py (CPU) → deep exact big-integer analysis of hits
  3. Manual algebraic proof of irrationality

Usage:
    python odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl

    # Exhaustive: 50 rational shifts × 1000 trajectories
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --cmf-name zeta_5 \
        --rational-shifts 50 --max-traj 1000 --depth 1000 --K 16

    # Quick test
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --depth 500 \
        --shifts 10 --max-traj 5 --max-cmfs 1
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

def compile_odd_zeta_cmf(spec: Dict, trajectory: int = 1):
    """Compile an odd-zeta CMF spec into a CmfProgram with given trajectory."""
    matrix_dict = {}
    for key, expr_str in spec['matrix'].items():
        r, c = key.split(',')
        matrix_dict[(int(r), int(c))] = expr_str

    program = compile_cmf_from_dict(
        matrix_dict=matrix_dict,
        m=spec['rank'],
        dim=spec['dim'],
        axis_names=spec.get('axis_names', ['k']),
        directions=[trajectory],
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


# ── Shift generation ───────────────────────────────────────────────────

def generate_rational_shifts(n: int) -> List[Fraction]:
    """Generate n rational shifts: ±1/d for d=1,2,3,...

    Pattern: 1, -1, 1/2, -1/2, 1/3, -1/3, ...
    """
    shifts = []
    d = 1
    while len(shifts) < n:
        shifts.append(Fraction(1, d))
        if len(shifts) < n:
            shifts.append(Fraction(-1, d))
        d += 1
    return shifts[:n]


def generate_integer_shifts(n: int) -> List[Fraction]:
    """Generate n integer shifts: 1, 2, 3, ..., n as Fractions."""
    return [Fraction(i) for i in range(1, n + 1)]


# ── Resume: load already-computed (shift, traj) combos ─────────────────

def load_completed_keys(results_path: str) -> Set[str]:
    """Load (cmf, shift_str, traj) keys from existing JSONL results."""
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
        description="Odd-Zeta CMF Sweep — compiled RNS walk"
    )
    parser.add_argument("--specs", type=str, required=True,
                        help="CMF specs JSONL from odd_zeta_cmf_generator.py")
    parser.add_argument("--depth", type=int, default=1000,
                        help="Walk depth (default 1000)")
    parser.add_argument("--shifts", type=int, default=0,
                        help="Number of integer shifts (1..N). Ignored if --rational-shifts set.")
    parser.add_argument("--rational-shifts", type=int, default=0,
                        help="Number of rational shifts (±1/d). Takes priority over --shifts.")
    parser.add_argument("--max-traj", type=int, default=10,
                        help="Max trajectory stride (1..max_traj)")
    parser.add_argument("--K", type=int, default=16,
                        help="RNS primes (default 16; float shadow is primary for sweep)")
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
                        help="Skip already-computed (shift, traj) combos in output")
    args = parser.parse_args()

    # Default: 50 rational shifts if neither is set
    if args.rational_shifts == 0 and args.shifts == 0:
        args.rational_shifts = 50

    # Generate shifts
    if args.rational_shifts > 0:
        shifts = generate_rational_shifts(args.rational_shifts)
        shift_mode = "rational"
    else:
        shifts = generate_integer_shifts(args.shifts)
        shift_mode = "integer"

    # Trajectories: 1..max_traj
    trajectories = list(range(1, args.max_traj + 1))

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

    n_tasks = len(shifts) * len(trajectories)
    print(f"\nOdd-Zeta CMF Sweep — compiled RNS bytecode walk")
    print(f"  CMFs:        {len(specs)}")
    print(f"  Shifts:      {len(shifts)} ({shift_mode})")
    print(f"  Trajectories:{len(trajectories)} (1..{args.max_traj})")
    print(f"  Tasks/CMF:   {n_tasks}")
    print(f"  Depth:       {args.depth}")
    print(f"  K:           {args.K} (float shadow is primary)")
    print(f"  Constants:   {len(constants)}")
    print(f"  Resume:      {args.resume}")
    print(f"{'='*80}")

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "odd_zeta_results.jsonl")

    # Resume: load completed keys
    done_keys = load_completed_keys(results_path) if args.resume else set()
    if done_keys:
        print(f"  Resuming: {len(done_keys)} results already computed")

    # Open results in append mode
    results_file = open(results_path, 'a')
    total_new = 0
    total_hits = 0

    try:
        for ci, spec in enumerate(specs):
            name = spec['name']
            zeta_val = spec['zeta_val']
            rank = spec['rank']
            n = spec['n']
            acc_idx = spec.get('accumulator_idx', n)
            const_idx = spec.get('constant_idx', n + 1)

            print(f"\n{'━'*80}")
            print(f"[{ci+1}/{len(specs)}] {name}: ζ({zeta_val}), {rank}×{rank} matrix")

            # Compute initial state (once per CMF)
            v0 = compute_initial_state(n)

            # Compile bytecode — we recompile per trajectory (directions change)
            # But cache by trajectory value
            prog_cache = {}
            t_cmf_start = time.time()
            cmf_new = 0
            cmf_hits = 0
            best_hit = None

            for ti, traj in enumerate(trajectories):
                # Compile (or cache) for this trajectory
                if traj not in prog_cache:
                    try:
                        prog_cache[traj] = compile_odd_zeta_cmf(spec, trajectory=traj)
                    except Exception as e:
                        if traj == 1:
                            print(f"  COMPILE ERROR (traj={traj}): {e}")
                        prog_cache[traj] = None

                program = prog_cache[traj]
                if program is None:
                    continue

                for si, shift in enumerate(shifts):
                    shift_str = str(shift)
                    key = f"{name}|{shift_str}|{traj}"
                    if key in done_keys:
                        continue

                    try:
                        res = run_cmf_walk_vec(
                            program, args.depth, args.K,
                            shift_vals=[shift],
                            initial_state=v0,
                            acc_idx=acc_idx,
                            const_idx=const_idx,
                        )

                        pf = res['p_float']
                        qf = res['q_float']
                        est = pf / qf if abs(qf) > 1e-300 else float('nan')

                        if not math.isfinite(est):
                            continue

                        # Match against all constants (float-level)
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

                        # Float delta (secondary; needs growing q to be meaningful)
                        log_q = res['log_scale'] + (
                            math.log(abs(qf)) if abs(qf) > 1e-300 else 0)
                        if best_const and log_q > 1:
                            target_f = next(
                                c['value_float'] for c in constants
                                if c['name'] == best_const)
                            err = abs(est - target_f)
                            if err > 0:
                                best_delta = -(1.0 + math.log(err) / log_q)
                            else:
                                best_delta = float('inf')
                        else:
                            best_delta = float('nan')

                        result = {
                            'cmf': name,
                            'zeta_val': zeta_val,
                            'rank': rank,
                            'shift': shift_str,
                            'trajectory': traj,
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
                        if traj == 1 and si == 0:
                            print(f"  WALK ERROR (shift={shift}, traj={traj}): {e}")
                            import traceback; traceback.print_exc()
                        continue

                # Progress every 50 trajectories
                if (ti + 1) % 50 == 0 or ti == len(trajectories) - 1:
                    elapsed = time.time() - t_cmf_start
                    rate = cmf_new / max(elapsed, 0.001)
                    print(f"  traj {ti+1}/{len(trajectories)}: "
                          f"{cmf_new} walks, {cmf_hits} hits (≥6 digits), "
                          f"{rate:.0f} walks/s", flush=True)

            results_file.flush()
            elapsed = time.time() - t_cmf_start

            if best_hit:
                print(f"  Best: {best_hit['match_digits']} digits → {best_hit['best_const']} "
                      f"(shift={best_hit['shift']}, traj={best_hit['trajectory']}, "
                      f"est={best_hit['est']})")
            print(f"  Total: {cmf_new} new walks in {elapsed:.1f}s")

    finally:
        results_file.close()

    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"  New walks:  {total_new}")
    print(f"  Hits (≥6d): {total_hits}")
    print(f"  Results:    {results_path}")
    print(f"  Total keys: {len(done_keys)}")
    print(f"\n  Deep analysis of hits:")
    print(f"    python odd_zeta_exact.py --specs {args.specs} --depth 5000 "
          f"--shifts <S> --trajectories <T> --cmf-name <NAME>")


if __name__ == "__main__":
    main()
