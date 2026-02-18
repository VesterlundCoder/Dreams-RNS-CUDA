#!/usr/bin/env python3
"""
Odd-Zeta CMF Deep Analysis: exact big-integer irrationality measurement.

CPU-only tool for deeply analyzing interesting CMF+shift combinations found
by GPU sweeps.  Uses EXACT big-integer arithmetic — no float, no RNS.

Pipeline:
  1. GPU sweep (odd_zeta_sweep.py / LUMI) → finds interesting CMF+shift combos
  2. THIS SCRIPT → deep exact analysis at depth 5000-10000+
  3. Manual algebraic proof of irrationality

At each step k:
  M_int(k) = D(k) · M(k)   — denominator-cleared integer matrix
  u = M_int(k) · u          — exact Python big-integer matvec

After N steps: p = u[acc], q = u[const] are exact ints.
δ = -(1 + log|p/q - ζ| / log|q|)  computed with mpmath at arbitrary precision.

Usage:
    # Deep analysis of ζ(5), depth=5000
    python odd_zeta_exact.py --specs odd_zeta_specs.jsonl --depth 5000 --shifts 3 --max-cmfs 1

    # Ultra-deep single analysis, depth=10000
    python odd_zeta_exact.py --specs odd_zeta_specs.jsonl --depth 10000 --shifts 1 --max-cmfs 1
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns.constants import load_constants

import sympy as sp
import mpmath as mp


def compute_initial_state(n: int):
    """Compute initial state v(1) for ζ(2n+1) in exact rationals.

    v = [U0(1), 0, ..., 0, S(1), 1, 0, ..., 0]
    where U0(1) = A_1 = 1/2, S(1) = T_n(1).
    """
    from fractions import Fraction

    d = 2 * n

    # A_k = (-1)^(k-1) / C(2k, k)
    def A_k(k):
        from math import comb
        return Fraction((-1)**(k-1), comb(2*k, k))

    # Elementary symmetric polynomials e_0..e_{n-1} for x_m = 1/m², m=1..k-1
    def e_list(k_minus_1, nn):
        e = [Fraction(0)] * nn
        e[0] = Fraction(1)
        for m in range(1, k_minus_1 + 1):
            xm = Fraction(1, m**2)
            for j in reversed(range(1, nn)):
                e[j] = e[j] + xm * e[j-1]
        return e

    # Kernel coefficients c_{n,j}(k)
    def kernel_coeffs(k, nn):
        coeffs = [Fraction(0)] * nn
        coeffs[nn-1] += Fraction(5) * Fraction((-1)**(nn-1))
        for j in range(0, nn-1):
            t = (nn-1) - j
            coeffs[j] += Fraction(4) * Fraction((-1)**j) / Fraction(k**(2*t))
        return coeffs

    # T_n(k) = (1/2) * A_k / k³ * Σ_j c_{n,j}(k) * e_j^{(k-1)}
    def direct_term(k, nn):
        Ak = A_k(k)
        e = e_list(k-1, nn)
        c = kernel_coeffs(k, nn)
        s = sum(c[j] * e[j] for j in range(nn))
        return Fraction(1, 2) * Ak * s / Fraction(k**3)

    v = [Fraction(0)] * d
    v[0] = Fraction(1, 2)           # U0(1) = A_1 = 1/2
    v[n] = direct_term(1, n)        # S(1) = T_n(1)
    v[n+1] = Fraction(1)            # constant = 1
    return v


def parse_sympy_matrix(spec: Dict):
    """Parse a spec's matrix dict into a sympy Matrix."""
    rank = spec['rank']
    k = sp.Symbol('k')
    M = sp.zeros(rank, rank)
    for key, expr_str in spec['matrix'].items():
        r, c = key.split(',')
        M[int(r), int(c)] = sp.sympify(expr_str)
    return M


def _lcm(a, b):
    """Least common multiple of two positive integers."""
    from math import gcd
    return a * b // gcd(a, b)


def run_odd_zeta_walk_exact(
    sympy_matrix,
    depth: int,
    shift: int,
    n: int,
    progress_every: int = 200,
) -> Tuple[int, int, int]:
    """Exact big-integer walk for odd-zeta CMF.

    Uses denominator clearing + pure Python big integers.
    NO RNS, NO float, NO approximation.  Result is exact.

    At each step k:
      1. Evaluate M(k) as exact sympy Rationals
      2. D(k) = lcm of all entry denominators
      3. M_int(k) = D(k) · M(k)   — all-integer matrix
      4. u = M_int(k) · u          — exact big-integer matvec

    After N steps:
      p = u[accumulator]  (exact integer)
      q = u[constant]     (exact integer = product of all D values)
      Rational approximation to ζ(2n+1) = p / q

    Returns:
        (p, q, total_bits)  where total_bits = max bit-length of state entries
    """
    k_sym = sp.Symbol('k')
    r = sympy_matrix.rows

    # Initial state as exact Fractions
    v0_frac = compute_initial_state(n)

    # Common denominator of initial state
    D_init = 1
    for f in v0_frac:
        if f.denominator != 0:
            D_init = _lcm(D_init, f.denominator)

    # u(0) = D_init * v(0)  — all exact Python ints
    u = [int(D_init * f) for f in v0_frac]

    # Pre-identify non-zero entries to skip zero muls
    nz_entries = []
    for i in range(r):
        for j in range(r):
            if sympy_matrix[i, j] != 0:
                nz_entries.append((i, j, sympy_matrix[i, j]))

    t0 = time.time()

    for step in range(depth):
        k_val = shift + step

        # Evaluate each non-zero entry as exact Rational, find common denom
        D_k = 1
        entry_vals = {}  # (i,j) -> (numerator, denominator)
        for i, j, expr in nz_entries:
            val = expr.subs(k_sym, k_val)
            rat = sp.Rational(val)
            num = int(rat.p)
            den = abs(int(rat.q))
            D_k = _lcm(D_k, den)
            entry_vals[(i, j)] = (num, den)

        # Build integer matrix M_int = D_k * M_rat
        M_int = {}
        for (i, j), (num, den) in entry_vals.items():
            M_int[(i, j)] = num * (D_k // den)

        # Exact big-integer matrix-vector multiply: u_new = M_int · u
        u_new = [0] * r
        for (i, j), val in M_int.items():
            u_new[i] += val * u[j]
        u = u_new

        # Progress
        if progress_every and (step + 1) % progress_every == 0:
            bits = max((abs(v).bit_length() for v in u if v != 0), default=0)
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (depth - step - 1) / rate
            print(f"    step {step+1}/{depth}: {bits:,} bits, "
                  f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    acc_idx = n
    const_idx = n + 1
    p_big = u[acc_idx]
    q_big = u[const_idx]
    total_bits = max((abs(v).bit_length() for v in u if v != 0), default=0)

    return p_big, q_big, total_bits


def compute_exact_delta(
    p_big: int, q_big: int, target_mp, dps: int = 1000,
) -> Tuple[float, float, int]:
    """Compute exact Dreams delta from big integers p, q.

    δ = -(1 + log|p/q - target| / log|q|)

    Uses mpmath at `dps` decimal places — no float64 anywhere.

    Returns:
        (delta, matching_decimal_digits, q_decimal_digits)
    """
    mp.mp.dps = dps

    if q_big == 0:
        return float('-inf'), 0, 0

    p_mp = mp.mpf(p_big)
    q_mp = mp.mpf(q_big)
    ratio = p_mp / q_mp
    err = abs(ratio - target_mp)
    abs_q = abs(q_mp)

    q_dec_digits = int(mp.log10(abs_q)) if abs_q > 1 else 0

    if err == 0:
        return float('inf'), q_dec_digits, q_dec_digits
    if abs_q <= 1:
        return float('-inf'), 0, q_dec_digits

    match_digits = int(-mp.log10(err)) if err > 0 and err < 1 else 0
    delta = float(-(1 + mp.log(err) / mp.log(abs_q)))
    return delta, match_digits, q_dec_digits


def main():
    parser = argparse.ArgumentParser(
        description="Odd-Zeta CMF Deep Analysis — exact big-integer irrationality measurement"
    )
    parser.add_argument("--specs", type=str, required=True,
                        help="CMF specs JSONL from odd_zeta_cmf_generator.py")
    parser.add_argument("--depth", type=int, default=5000,
                        help="Walk depth (default 5000 for deep exploration)")
    parser.add_argument("--shifts", type=int, default=3,
                        help="Number of shifts (k starts at 1, 2, ..., shifts)")
    parser.add_argument("--dps", type=int, default=0,
                        help="mpmath decimal precision (0 = auto from depth)")
    parser.add_argument("--max-cmfs", type=int, default=0,
                        help="Max CMFs to process (0=all)")
    parser.add_argument("--output", type=str, default="odd_zeta_exact_results/",
                        help="Output directory")
    args = parser.parse_args()

    # Auto-dps: need enough digits to represent the error
    # Convergence rate ~ log10(16)*depth matching digits
    # Plus safety margin for delta computation
    if args.dps == 0:
        args.dps = max(2000, int(args.depth * 1.5))
    mp.mp.dps = args.dps

    # Load constants (extended with ζ(9)..ζ(21))
    constants = load_constants(args.dps)
    print(f"Constants bank: {len(constants)} constants")
    for c in constants:
        print(f"  {c['name']:<15} = {mp.nstr(c['value'], 20)}  ({c['description']})")

    # Load CMF specs
    specs = []
    with open(args.specs) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))

    if args.max_cmfs > 0:
        specs = specs[:args.max_cmfs]

    print(f"\nOdd-Zeta CMF Deep Analysis — EXACT big-integer arithmetic")
    print(f"  CMFs:     {len(specs)}")
    print(f"  Shifts:   {args.shifts}")
    print(f"  Depth:    {args.depth}")
    print(f"  mpmath:   {args.dps} decimal places")
    print(f"  Method:   denominator-cleared integer walk (no float, no RNS)")
    print(f"{'='*90}")

    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, "odd_zeta_exact_results.jsonl")
    summary_path = os.path.join(args.output, "odd_zeta_exact_summary.csv")

    all_results = []

    for ci, spec in enumerate(specs):
        name = spec['name']
        zeta_val = spec['zeta_val']
        rank = spec['rank']
        n = spec['n']

        print(f"\n{'━'*90}")
        print(f"[{ci+1}/{len(specs)}] {name}: ζ({zeta_val}), {rank}×{rank} matrix, n={n}")

        # Parse sympy matrix (once per CMF)
        try:
            sympy_mat = parse_sympy_matrix(spec)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            continue

        cmf_results = []

        for si in range(1, args.shifts + 1):
            print(f"\n  ── Shift {si} ──")
            t0 = time.time()

            try:
                p_big, q_big, total_bits = run_odd_zeta_walk_exact(
                    sympy_mat, args.depth, shift=si, n=n,
                    progress_every=max(args.depth // 10, 100),
                )
            except Exception as e:
                print(f"  WALK ERROR (shift={si}): {e}")
                import traceback; traceback.print_exc()
                continue

            walk_time = time.time() - t0
            p_bits = abs(p_big).bit_length() if p_big else 0
            q_bits = abs(q_big).bit_length() if q_big else 0

            print(f"    Walk complete: {walk_time:.1f}s")
            print(f"    p: {p_bits:,} bits ({p_bits // 3:,} decimal digits)")
            print(f"    q: {q_bits:,} bits ({q_bits // 3:,} decimal digits)")

            # Check against all constants with exact mpmath delta
            best_delta = float('-inf')
            best_const = None
            best_match_digits = 0
            best_q_digits = 0

            for c in constants:
                delta, match_digits, q_digits = compute_exact_delta(
                    p_big, q_big, c['value'], args.dps)

                if delta > best_delta:
                    best_delta = delta
                    best_const = c['name']
                    best_match_digits = match_digits
                    best_q_digits = q_digits

            # Also compute ratio for display
            mp.mp.dps = args.dps
            if q_big != 0:
                ratio = mp.mpf(p_big) / mp.mpf(q_big)
                ratio_str = mp.nstr(ratio, 30)
            else:
                ratio_str = "q=0"

            print(f"    p/q  = {ratio_str}")
            print(f"    Best match: {best_const}")
            print(f"    Matching digits: {best_match_digits:,}")
            print(f"    Denominator digits: {best_q_digits:,}")
            print(f"    δ = {best_delta:.8f}")
            if best_delta > 0:
                print(f"    *** δ > 0 — IRRATIONALITY PROOF TERRITORY ***")
            elif best_delta > -0.1:
                print(f"    (Close to irrationality threshold)")

            result = {
                'cmf': name,
                'zeta_val': zeta_val,
                'rank': rank,
                'n': n,
                'shift': si,
                'depth': args.depth,
                'ratio': str(ratio_str),
                'best_const': best_const,
                'best_delta': best_delta,
                'match_digits': best_match_digits,
                'q_digits': best_q_digits,
                'p_bits': p_bits,
                'q_bits': q_bits,
                'total_bits': total_bits,
                'walk_seconds': round(walk_time, 1),
            }
            cmf_results.append(result)

        # Per-CMF summary
        if cmf_results:
            best = max(cmf_results, key=lambda x: x['best_delta'])
            print(f"\n  ── {name} Summary ──")
            print(f"  Best δ = {best['best_delta']:.8f} (shift={best['shift']}, "
                  f"{best['match_digits']:,} matching / {best['q_digits']:,} q digits)")

        all_results.extend(cmf_results)

    # Write results
    with open(results_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary CSV
    if all_results:
        summary_rows = []
        for spec in specs:
            cmf_name = spec['name']
            cmf_res = [r for r in all_results if r['cmf'] == cmf_name]
            if cmf_res:
                best = max(cmf_res, key=lambda x: x['best_delta'])
                summary_rows.append({
                    'cmf': cmf_name,
                    'zeta': spec['zeta_val'],
                    'rank': spec['rank'],
                    'depth': args.depth,
                    'n_shifts': len(cmf_res),
                    'best_delta': f"{best['best_delta']:.8f}",
                    'match_digits': best['match_digits'],
                    'q_digits': best['q_digits'],
                    'best_shift': best['shift'],
                    'walk_seconds': best['walk_seconds'],
                })

        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n{'='*90}")
    print(f"COMPLETE — Exact big-integer results")
    print(f"  Total walks: {len(all_results)}")
    n_pos = sum(1 for r in all_results if r['best_delta'] > 0)
    if n_pos:
        print(f"  δ > 0 HITS: {n_pos}  ← IRRATIONALITY PROOF CANDIDATES")
    else:
        print(f"  δ > 0 hits: 0")
        if all_results:
            best_overall = max(all_results, key=lambda x: x['best_delta'])
            print(f"  Best δ:     {best_overall['best_delta']:.8f} "
                  f"({best_overall['cmf']}, shift={best_overall['shift']})")
    print(f"  Results:    {results_path}")
    print(f"  Summary:    {summary_path}")


if __name__ == "__main__":
    main()
