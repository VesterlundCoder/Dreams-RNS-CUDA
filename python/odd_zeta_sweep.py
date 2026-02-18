#!/usr/bin/env python3
"""
Odd-Zeta CMF Sweep: RNS walk for ζ(2n+1) CMFs — GPU-scalable.

Large-scale sweep tool.  Uses denominator-cleared RNS integer walk with
float64 shadow.  CRT reconstructs exact p/q when K is large enough;
falls back to float-based delta otherwise.

Pipeline:
  1. THIS SCRIPT (GPU sweep) → scan many CMFs × shifts at depth=5000
  2. odd_zeta_exact.py (CPU) → deep exact analysis of interesting hits
  3. Manual algebraic proof of irrationality

Usage:
    # Generate specs first
    python odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl

    # Sweep all with 512 shifts, depth=5000
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --depth 5000 --shifts 512

    # Quick test: just ζ(5) with 10 shifts
    python odd_zeta_sweep.py --specs odd_zeta_specs.jsonl --depth 500 --shifts 10 --max-cmfs 1
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns.compiler import CmfCompiler, compile_cmf_from_dict
from dreams_rns.runner import generate_rns_primes
from dreams_rns import crt_reconstruct, centered
from dreams_rns.constants import load_constants, match_against_constants, compute_delta_against_constant

import sympy as sp
import mpmath as mp




def compile_odd_zeta_cmf(spec: Dict) -> Any:
    """Compile an odd-zeta CMF spec into a CmfProgram."""
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


def estimate_K_for_cmf(rank: int, depth: int) -> int:
    """Estimate K primes for a rank×rank CMF at given depth.

    The denominator-cleared walk accumulates D(1)*D(2)*...*D(N) in the
    constant slot.  Each D(k) ~ O(k^rank) so log2(product) scales as
    rank * Σ log2(k) ≈ rank * log2(N!).  We need K*31 > total bits.
    Extra 2× safety factor for numerator growth.
    """
    if depth > 1:
        log2_fact = depth * math.log2(depth / math.e) + 0.5 * math.log2(2 * math.pi * depth)
    else:
        log2_fact = 1
    bits_needed = int(rank * log2_fact * 2.0) + 500
    K = max(bits_needed // 31 + 1, 128)
    return K


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


def run_odd_zeta_walk(sympy_matrix, depth: int, K: int, shift: int, n: int,
                      trajectory: int = 1):
    """Denominator-cleared RNS integer walk for odd-zeta CMF.

    Walk: k = shift, shift+traj, shift+2*traj, ...

    At each step k:
      1. Evaluate M(k) as exact sympy Rationals
      2. Find common denominator D(k) = lcm of all entry denominators
      3. Scale M_int(k) = D(k) * M(k)  →  all-integer matrix
      4. u = M_int(k) · u  mod each prime  (RNS integer arithmetic)
      5. Float shadow: v = M_float(k) · v

    Since v[const_idx] = 1 in the original walk, u[const_idx] accumulates
    the product D_init * D(1) * D(2) * ... * D(N) = total denominator.
    CRT gives exact integers: p = u[acc], q = u[const].
    δ = -(1 + log|p/q - L| / log|q|)  — proper Dreams delta.

    Returns:
        dict with p_residues, q_residues, p_float, q_float, log_scale, primes
    """
    k_sym = sp.Symbol('k')
    r = sympy_matrix.rows
    pp = generate_rns_primes(K).astype(np.int64)

    # Initial state as exact rationals
    v0_frac = compute_initial_state(n)

    # Common denominator of initial state
    D_init = 1
    for f in v0_frac:
        if f.denominator != 0:
            D_init = _lcm(D_init, f.denominator)

    # u(0) = D_init * v(0), all integers
    u0 = [int(D_init * f) for f in v0_frac]

    # RNS state: u_rns[i, ki] = u[i] mod prime[ki]
    u_rns = np.zeros((r, K), dtype=np.int64)
    for i in range(r):
        for ki in range(K):
            u_rns[i, ki] = u0[i] % int(pp[ki])

    # Float shadow (original scale, not denominator-cleared)
    v_f = np.array([float(f) for f in v0_frac], dtype=np.float64)
    log_scale = 0.0

    for step in range(depth):
        k_val = shift + step * trajectory

        # Evaluate M(k_val) as exact sympy Rationals
        M_rat = sympy_matrix.subs(k_sym, k_val)

        # Find common denominator of all non-zero entries
        D_k = 1
        for i in range(r):
            for j in range(r):
                entry = M_rat[i, j]
                if entry != 0:
                    rat = sp.Rational(entry)
                    D_k = _lcm(D_k, abs(int(rat.q)))

        # Compute integer matrix entries and convert to RNS
        # M_int[i,j] = D_k * M_rat[i,j] = num * (D_k / den)
        M_rns = np.zeros((r, r, K), dtype=np.int64)
        M_float = np.zeros((r, r), dtype=np.float64)
        for i in range(r):
            for j in range(r):
                entry = M_rat[i, j]
                if entry != 0:
                    rat = sp.Rational(entry)
                    num_ij = int(rat.p)
                    den_ij = abs(int(rat.q))
                    scale_ij = D_k // den_ij  # always exact integer
                    int_val = num_ij * scale_ij
                    M_float[i, j] = float(rat)
                    # Convert to RNS residues
                    for ki in range(K):
                        p = int(pp[ki])
                        M_rns[i, j, ki] = int_val % p

        # u = M_int · u  mod each prime (integer matrix × integer vector)
        new_u_rns = np.zeros_like(u_rns)
        for i in range(r):
            for j in range(r):
                # Only multiply if M_int[i,j] != 0
                if M_float[i, j] != 0.0:
                    new_u_rns[i] = (new_u_rns[i] + M_rns[i, j] * u_rns[j]) % pp
        u_rns = new_u_rns

        # Float shadow (original rational values, not scaled)
        v_f = M_float @ v_f

        # Normalize float shadow
        mx = np.max(np.abs(v_f))
        if mx > 1e10:
            v_f /= mx
            log_scale += math.log(mx)

    # Extract: p = u[acc_idx], q = u[const_idx]
    acc_idx = n
    const_idx = n + 1

    return {
        'p_residues': u_rns[acc_idx],
        'q_residues': u_rns[const_idx],
        'p_float': v_f[acc_idx],
        'q_float': v_f[const_idx],
        'log_scale': log_scale,
        'primes': pp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Odd-Zeta CMF Sweep — RNS walk (GPU-scalable)"
    )
    parser.add_argument("--specs", type=str, required=True,
                        help="CMF specs JSONL from odd_zeta_cmf_generator.py")
    parser.add_argument("--depth", type=int, default=1000,
                        help="Walk depth (default 1000 for sweep; use odd_zeta_exact.py for deep)")
    parser.add_argument("--shifts", type=int, default=512,
                        help="Number of shifts (k starts at 1, 2, ..., shifts)")
    parser.add_argument("--trajectories", type=int, nargs='+', default=[1],
                        help="Trajectory strides, e.g. --trajectories 1 2 3 5")
    parser.add_argument("--K", type=int, default=0,
                        help="RNS primes (0=auto per CMF, sized for depth)")
    parser.add_argument("--dps", type=int, default=200,
                        help="mpmath decimal precision for delta computation")
    parser.add_argument("--max-cmfs", type=int, default=0,
                        help="Max CMFs to process (0=all)")
    parser.add_argument("--skip-cmfs", type=int, default=0,
                        help="Skip first N CMFs (use with --max-cmfs 1 for per-CMF runs)")
    parser.add_argument("--cmf-name", type=str, default="",
                        help="Process only the CMF with this name, e.g. zeta_5")
    parser.add_argument("--output", type=str, default="odd_zeta_results/",
                        help="Output directory")
    args = parser.parse_args()

    # Load constants (extended with ζ(9)..ζ(21))
    constants = load_constants(args.dps)
    print(f"Constants bank: {len(constants)} constants")
    for c in constants:
        print(f"  {c['name']:<15} = {c['value_float']:.15f}  ({c['description']})")

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

    n_tasks_per_cmf = args.shifts * len(args.trajectories)
    print(f"\nOdd-Zeta CMF Sweep — RNS walk (GPU-scalable)")
    print(f"  CMFs:    {len(specs)}")
    print(f"  Shifts:  {args.shifts}")
    print(f"  Trajs:   {args.trajectories}")
    print(f"  Tasks:   {n_tasks_per_cmf} per CMF")
    print(f"  Depth:   {args.depth}")
    print(f"  K:       {'auto' if args.K == 0 else args.K}")
    print(f"  Method:  denominator-cleared RNS + CRT + mpmath delta")
    print(f"{'='*80}")

    os.makedirs(args.output, exist_ok=True)

    # Results file
    results_path = os.path.join(args.output, "odd_zeta_results.jsonl")
    summary_path = os.path.join(args.output, "odd_zeta_summary.csv")

    all_results = []

    for ci, spec in enumerate(specs):
        name = spec['name']
        zeta_val = spec['zeta_val']
        rank = spec['rank']
        n = spec['n']
        acc_idx = spec.get('accumulator_idx', n)
        const_idx = spec.get('constant_idx', n + 1)

        print(f"\n{'─'*80}")
        print(f"[{ci+1}/{len(specs)}] {name}: ζ({zeta_val}), {rank}×{rank} matrix")

        # Parse sympy matrix (once per CMF)
        t0 = time.time()
        try:
            sympy_mat = parse_sympy_matrix(spec)
        except Exception as e:
            print(f"  PARSE ERROR: {e}")
            continue

        # Auto-K: sized for depth
        if args.K > 0:
            K_use = args.K
        else:
            K_use = estimate_K_for_cmf(rank, args.depth)
        print(f"  K={K_use} ({K_use*31} bits CRT range)")

        parse_time = time.time() - t0

        # Walk with each shift
        cmf_results = []
        best_hit = None

        for si in range(1, args.shifts + 1):
          for traj in args.trajectories:
            try:
                res = run_odd_zeta_walk(sympy_mat, args.depth, K_use, shift=si, n=n,
                                        trajectory=traj)

                # CRT reconstruction: p = numerator, q = denominator (both exact integers)
                primes = [int(p) for p in res['primes']]
                p_big, Mp = crt_reconstruct(
                    [int(r) for r in res['p_residues']], primes)
                q_big, _ = crt_reconstruct(
                    [int(r) for r in res['q_residues']], primes)
                p_big = centered(p_big, Mp)
                q_big = centered(q_big, Mp)

                # Float estimate from shadow
                pf = res['p_float']
                qf = res['q_float']
                est_float = pf / qf if abs(qf) > 1e-300 else float('nan')

                if not math.isfinite(est_float):
                    continue

                # CRT overflow check: exact ratio should match float
                mp.mp.dps = args.dps
                crt_ok = True
                if q_big == 0:
                    crt_ok = False
                else:
                    try:
                        cr = float(mp.mpf(p_big) / mp.mpf(q_big))
                        if not math.isfinite(cr):
                            crt_ok = False
                        elif abs(est_float) > 1e-10:
                            crt_ok = abs(cr - est_float) / abs(est_float) < 0.01
                        else:
                            crt_ok = abs(cr - est_float) < 0.01
                    except Exception:
                        crt_ok = False

                # Check all constants — exact CRT delta when possible, float fallback
                best_delta = float('-inf')
                best_const = None
                best_digits = 0

                for c in constants:
                    if crt_ok and q_big != 0:
                        delta = compute_delta_against_constant(
                            p_big, q_big, c['value'], args.dps)
                    else:
                        # Float fallback using log_scale for q estimate
                        try:
                            err = abs(est_float - c['value_float'])
                            log_q = res['log_scale'] + (
                                math.log(abs(qf)) if abs(qf) > 1e-300 else 0)
                            if err > 0 and log_q > 0:
                                delta = -(1.0 + math.log(err) / log_q)
                            elif err == 0:
                                delta = float('inf')
                            else:
                                delta = float('-inf')
                        except Exception:
                            delta = float('-inf')

                    if delta > best_delta:
                        best_delta = delta
                        best_const = c['name']

                # Matching digits (float-level)
                for c in constants:
                    if c['name'] == best_const:
                        try:
                            rel_err = abs(est_float - c['value_float'])
                            if rel_err > 0:
                                best_digits = -math.log10(rel_err / max(abs(c['value_float']), 1e-300))
                            else:
                                best_digits = 16.0
                        except Exception:
                            best_digits = 0
                        break

                result = {
                    'cmf': name,
                    'zeta_val': zeta_val,
                    'rank': rank,
                    'shift': si,
                    'trajectory': traj,
                    'depth': args.depth,
                    'est': est_float,
                    'best_const': best_const,
                    'best_delta': best_delta,
                    'match_digits': round(best_digits, 1),
                    'crt_ok': str(crt_ok),
                    'K': K_use,
                    'p_bits': Mp.bit_length(),
                    'q_bits': q_big.bit_length() if q_big else 0,
                }

                cmf_results.append(result)

                if best_delta > (best_hit['best_delta'] if best_hit else float('-inf')):
                    best_hit = result

            except Exception as e:
                if si == 1 and traj == 1:
                    print(f"  WALK ERROR (shift={si}, traj={traj}): {e}")
                    import traceback; traceback.print_exc()
                continue

        elapsed = time.time() - t0

        # Report — use δ > 0 OR matching_digits > 6 as "hit"
        n_positive = sum(1 for r in cmf_results if r['best_delta'] > 0)
        n_good = sum(1 for r in cmf_results if r['match_digits'] >= 6)
        n_valid = len(cmf_results)

        if best_hit:
            print(f"  {n_valid}/{n_tasks_per_cmf} walks OK, {n_positive} with δ>0, {n_good} with ≥6 digits")
            print(f"  Best: δ={best_hit['best_delta']:.6f} → {best_hit['best_const']} "
                  f"({best_hit['match_digits']} digits, shift={best_hit['shift']}, "
                  f"traj={best_hit['trajectory']}, "
                  f"crt_ok={best_hit['crt_ok']}, est={best_hit['est']:.12f})")
        else:
            print(f"  No valid results")

        print(f"  ({elapsed:.1f}s, {n_valid/max(elapsed,0.001):.0f} walks/s)")

        # Save per-CMF results
        all_results.extend(cmf_results)

    # Write all results
    with open(results_path, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")

    # Write summary CSV
    if all_results:
        summary_rows = []
        for spec in specs:
            name = spec['name']
            zv = spec['zeta_val']
            rk = spec['rank']
            cmf_res = [r for r in all_results if r['cmf'] == name]
            if cmf_res:
                best = max(cmf_res, key=lambda x: x['best_delta'])
                n_pos = sum(1 for r in cmf_res if r['best_delta'] > 0)
                n_good = sum(1 for r in cmf_res if r['match_digits'] >= 6)
                summary_rows.append({
                    'cmf': name,
                    'zeta': zv,
                    'rank': rk,
                    'depth': args.depth,
                    'K': best['K'],
                    'n_shifts': len(cmf_res),
                    'n_positive_delta': n_pos,
                    'n_good_digits': n_good,
                    'best_delta': f"{best['best_delta']:.6f}",
                    'best_digits': best['match_digits'],
                    'best_const': best['best_const'],
                    'best_shift': best['shift'],
                    'best_est': f"{best['est']:.12f}",
                    'crt_ok': best['crt_ok'],
                })

        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"  Total walks: {len(all_results)}")
    print(f"  δ>0 hits:    {sum(1 for r in all_results if r['best_delta'] > 0)}")
    print(f"  CRT valid:   {sum(1 for r in all_results if r['crt_ok'] == 'True')}/{len(all_results)}")
    print(f"  Results:     {results_path}")
    print(f"  Summary:     {summary_path}")
    print(f"\n  Interesting hits? Run deep CPU analysis:")
    print(f"    python odd_zeta_exact.py --specs {args.specs} --depth 10000 --shifts 1 2 3 --trajectories 1 2 3")


if __name__ == "__main__":
    main()
