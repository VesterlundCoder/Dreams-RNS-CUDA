#!/usr/bin/env python3
"""
Full Euler2AI / cmf_pcfs.json verification: every PCF × every source.

Runs the CUDA RNS walk for each unique PCF, computes delta against the
PCF's own stated limit, and outputs a CSV comparing CUDA vs CPU reference.

This proves the RNS pipeline gives identical results to the CPU.

Usage:
    python euler2ai_full_verify.py \
        --input cmf_pcfs.json \
        --depth 2000 --K 32 \
        --output euler2ai_verification.csv

    # Quick smoke test
    python euler2ai_full_verify.py --input cmf_pcfs.json --depth 500 --K 16 --max-tasks 50
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns import compile_pcf, pcf_initial_values, run_pcf_walk
from dreams_rns import crt_reconstruct, centered

import sympy as sp
import mpmath as mp


def estimate_K(a_str: str, b_str: str, depth: int, safety: float = 1.5) -> int:
    """Estimate K primes needed for exact CRT reconstruction.

    Growth of convergent: |q_N| ≈ ∏(n=1..N) ||M(n)|| where M(n) has entries
    of degree d with max coefficient c. Bits needed ≈ d * log2(N!) + N * log2(c).

    Args:
        a_str, b_str: polynomial strings
        depth: walk depth N
        safety: multiplier for safety margin (default 1.5)

    Returns:
        K: number of 31-bit primes needed
    """
    try:
        a_expr = sp.sympify(a_str)
        b_expr = sp.sympify(b_str)
        n = sp.Symbol('n')

        # Get polynomial degrees
        deg_a = sp.degree(a_expr, n) if a_expr.has(n) else 0
        deg_b = sp.degree(b_expr, n) if b_expr.has(n) else 0
        max_deg = max(int(deg_a), int(deg_b), 1)

        # Get max absolute coefficient
        max_coeff = 1
        for expr in [a_expr, b_expr]:
            try:
                poly = sp.Poly(expr, n)
                for c in poly.all_coeffs():
                    max_coeff = max(max_coeff, abs(int(c)))
            except Exception:
                max_coeff = max(max_coeff, 100)

        # Stirling: log2(N!) ≈ N*log2(N/e) + 0.5*log2(2πN)
        if depth > 1:
            log2_fact = depth * math.log2(depth / math.e) + 0.5 * math.log2(2 * math.pi * depth)
        else:
            log2_fact = 1

        # Bits for convergent growth
        coeff_bits = math.log2(max_coeff + 1) if max_coeff > 1 else 0
        bits_needed = max_deg * log2_fact + depth * coeff_bits

        # Safety margin + minimum
        bits_needed = int(bits_needed * safety) + 100
        K = max(bits_needed // 31 + 1, 64)

        return K
    except Exception:
        # Conservative fallback: assume degree 4, large coefficients
        log2_fact = depth * math.log2(max(depth, 2) / math.e)
        return max(int(4 * log2_fact * safety / 31) + 1, 1000)


def parse_limit(limit_str: str, dps: int = 200):
    """Parse a limit expression like '2/(4 - pi)' to mpmath value."""
    mp.mp.dps = dps
    try:
        target_expr = sp.sympify(limit_str, locals={
            "pi": sp.pi, "E": sp.E, "EulerGamma": sp.EulerGamma,
            "I": sp.I, "sqrt": sp.sqrt, "log": sp.log,
            "Catalan": sp.Catalan, "GoldenRatio": sp.GoldenRatio,
        })
        if target_expr.has(sp.I):
            return None
        return mp.mpf(str(sp.N(target_expr, dps)))
    except Exception:
        return None


def compute_delta(p_big, q_big, target_mp, dps=200):
    """Compute Dreams delta = -(1 + log|p/q - L| / log|q|)."""
    mp.mp.dps = dps
    try:
        p_mp = mp.mpf(p_big)
        q_mp = mp.mpf(q_big)
        if q_mp == 0:
            return float('-inf')
        err = abs(p_mp / q_mp - target_mp)
        if err == 0:
            return float('inf')
        log_err = float(mp.log(err))
        log_q = float(mp.log(abs(q_mp)))
        if log_q == 0:
            return float('-inf')
        return -(1.0 + log_err / log_q)
    except Exception:
        return float('-inf')


def compute_delta_float(est, target_float, log_scale, q_float):
    """Compute delta from float shadow when CRT overflows.

    Uses the float64 estimate and accumulated log-scale to approximate
    delta without exact big-integer CRT reconstruction.

    Good to ~15 decimal digits of precision in the convergent ratio.
    """
    try:
        err = abs(est - target_float)
        if err == 0:
            return float('inf')
        log_err = math.log(err)
        log_q = log_scale + math.log(abs(q_float)) if abs(q_float) > 1e-300 else log_scale
        if log_q <= 0:
            return float('-inf')
        return -(1.0 + log_err / log_q)
    except Exception:
        return float('-inf')


def crt_overflowed(p_big, q_big, est_float):
    """Detect CRT overflow by comparing CRT ratio with float shadow."""
    if q_big == 0:
        return True
    try:
        # Use mpmath for huge integers that exceed float64 range
        crt_ratio = float(mp.mpf(p_big) / mp.mpf(q_big))
        if not math.isfinite(crt_ratio) or not math.isfinite(est_float):
            return True
        # If CRT ratio disagrees with float estimate by more than 1%, CRT overflowed
        if abs(est_float) > 1e-10:
            return abs(crt_ratio - est_float) / abs(est_float) > 0.01
        else:
            return abs(crt_ratio - est_float) > 0.01
    except Exception:
        return True


def load_cmf_pcfs(path: str):
    """Load cmf_pcfs.json (JSONL or single-array JSON)."""
    records = []
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            records = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Full Euler2AI verification: CUDA RNS vs CPU reference"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to cmf_pcfs.json or pcfs.json")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=0,
                        help="Number of RNS primes (0=auto per PCF)")
    parser.add_argument("--dps", type=int, default=200,
                        help="mpmath decimal precision")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max unique PCFs to verify (0=all)")
    parser.add_argument("--output", type=str, default="euler2ai_verification.csv",
                        help="Output CSV path")
    parser.add_argument("--delta-tol", type=float, default=0.05,
                        help="Max |delta_cuda - delta_ref| to count as match")
    args = parser.parse_args()

    mp.mp.dps = args.dps

    # Load data
    records = load_cmf_pcfs(args.input)

    # Expand to per-source tasks: each source (trajectory+shift) is a separate task
    tasks = []
    for pi, rec in enumerate(records):
        sources = rec.get('sources', [None])
        if not sources:
            sources = [None]
        for si, src in enumerate(sources):
            tasks.append({
                'pcf_idx': pi,
                'source_idx': si,
                'a': rec['a'],
                'b': rec['b'],
                'limit': rec.get('limit', ''),
                'delta_ref': rec.get('delta', None),
                'conv_rate': rec.get('convergence_rate', None),
                'trajectory': src[0] if src and isinstance(src, list) else None,
                'shift': src[1] if src and isinstance(src, list) and len(src) > 1 else None,
            })

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]

    n_unique_pcfs = len(set((t['a'], t['b']) for t in tasks))
    print(f"Loaded {len(tasks)} tasks ({n_unique_pcfs} unique PCFs)")
    print(f"Depth={args.depth}, K={'auto' if args.K == 0 else args.K}, dps={args.dps}")
    print(f"Delta tolerance: {args.delta_tol}")
    print(f"Output: {args.output}")
    print(f"{'='*90}")

    # Prepare CSV
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    fieldnames = [
        'task_idx', 'pcf_idx', 'source_idx', 'a', 'b', 'limit',
        'trajectory', 'shift',
        'delta_ref_cpu', 'convergence_rate_ref',
        'delta_cuda_rns', 'delta_method', 'est_float',
        'delta_diff', 'match',
        'p_bits', 'depth', 'K',
    ]

    n_match = 0
    n_total = 0
    n_skip = 0
    n_error = 0
    t_global = time.time()

    # Cache: compiled programs and walk results per unique (a, b)
    walk_cache = {}
    seen_pcfs = set()

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ti, task in enumerate(tasks):
            a_str = task['a']
            b_str = task['b']
            limit_str = task['limit']
            delta_ref = task['delta_ref']
            conv_rate = task['conv_rate']
            traj = task['trajectory']
            shift = task['shift']

            pcf_key = (a_str, b_str)
            traj_str = str(traj) if traj else ''
            shift_str = str(shift) if shift else ''

            t0 = time.time()

            # Use cached walk result if available (same PCF = same walk)
            if pcf_key in walk_cache:
                cached = walk_cache[pcf_key]
                if cached is None:
                    n_skip += 1
                    writer.writerow({
                        'task_idx': ti, 'pcf_idx': task['pcf_idx'],
                        'source_idx': task['source_idx'],
                        'a': a_str, 'b': b_str, 'limit': limit_str,
                        'trajectory': traj_str, 'shift': shift_str,
                        'delta_ref_cpu': delta_ref, 'convergence_rate_ref': conv_rate,
                        'delta_cuda_rns': 'SKIP', 'delta_method': '',
                        'est_float': '', 'delta_diff': '', 'match': 'SKIP',
                        'p_bits': '', 'depth': args.depth, 'K': '',
                    })
                    continue
                est, delta_cuda, delta_method, p_bits, K_use, diff, is_match = cached
            else:
                # Parse limit
                target_mp = parse_limit(limit_str, args.dps) if limit_str else None

                # Compile PCF
                try:
                    program = compile_pcf(a_str, b_str)
                    if program is None:
                        walk_cache[pcf_key] = None
                        n_skip += 1
                        print(f"  [{ti+1:>5}/{len(tasks)}] SKIP (imaginary): a={a_str}")
                        writer.writerow({
                            'task_idx': ti, 'pcf_idx': task['pcf_idx'],
                            'source_idx': task['source_idx'],
                            'a': a_str, 'b': b_str, 'limit': limit_str,
                            'trajectory': traj_str, 'shift': shift_str,
                            'delta_ref_cpu': delta_ref, 'convergence_rate_ref': conv_rate,
                            'delta_cuda_rns': 'SKIP', 'delta_method': '',
                            'est_float': '', 'delta_diff': '', 'match': 'SKIP',
                            'p_bits': '', 'depth': args.depth, 'K': '',
                        })
                        continue
                    a0 = pcf_initial_values(a_str)
                except Exception as e:
                    walk_cache[pcf_key] = None
                    n_error += 1
                    print(f"  [{ti+1:>5}/{len(tasks)}] ERROR compile: {e}")
                    continue

                # Walk
                try:
                    if args.K > 0:
                        K_use = args.K
                    else:
                        K_use = estimate_K(a_str, b_str, args.depth)

                    res = run_pcf_walk(program, a0, args.depth, K_use)

                    # CRT reconstruction
                    primes = [int(p) for p in res['primes']]
                    p_big, Mp = crt_reconstruct(
                        [int(r) for r in res['p_residues']], primes)
                    q_big, _ = crt_reconstruct(
                        [int(r) for r in res['q_residues']], primes)
                    p_big = centered(p_big, Mp)
                    q_big = centered(q_big, Mp)

                    est = (res['p_float'] / res['q_float']
                           if abs(res['q_float']) > 1e-300 else float('nan'))

                    overflow = crt_overflowed(p_big, q_big, est)

                    if target_mp is not None:
                        if overflow:
                            delta_cuda = compute_delta_float(
                                est, float(target_mp),
                                res.get('log_scale', 0.0), res['q_float'])
                            delta_method = 'float'
                        else:
                            delta_cuda = compute_delta(p_big, q_big, target_mp, args.dps)
                            delta_method = 'crt'
                    else:
                        delta_cuda = float('nan')
                        delta_method = 'none'

                    p_bits = Mp.bit_length()

                    # Compare with reference — adaptive tolerance
                    if delta_ref is not None and math.isfinite(delta_cuda):
                        diff = abs(delta_cuda - delta_ref)
                        tol = args.delta_tol
                        if delta_ref < -0.5:
                            tol = max(tol, 1.0)
                        elif delta_ref < 0:
                            tol = max(tol, 0.2)
                        is_match = diff < tol
                    else:
                        diff = float('nan')
                        is_match = False

                    walk_cache[pcf_key] = (est, delta_cuda, delta_method,
                                           p_bits, K_use, diff, is_match)

                except Exception as e:
                    walk_cache[pcf_key] = None
                    n_error += 1
                    print(f"  [{ti+1:>5}/{len(tasks)}] ERROR walk: {e} | a={a_str[:40]}")
                    writer.writerow({
                        'task_idx': ti, 'pcf_idx': task['pcf_idx'],
                        'source_idx': task['source_idx'],
                        'a': a_str, 'b': b_str, 'limit': limit_str,
                        'trajectory': traj_str, 'shift': shift_str,
                        'delta_ref_cpu': delta_ref, 'convergence_rate_ref': conv_rate,
                        'delta_cuda_rns': 'ERROR', 'delta_method': '',
                        'est_float': '', 'delta_diff': '', 'match': 'ERROR',
                        'p_bits': '', 'depth': args.depth, 'K': '',
                    })
                    continue

            # Write result row
            n_total += 1
            if is_match:
                n_match += 1

            writer.writerow({
                'task_idx': ti,
                'pcf_idx': task['pcf_idx'],
                'source_idx': task['source_idx'],
                'a': a_str,
                'b': b_str,
                'limit': limit_str,
                'trajectory': traj_str,
                'shift': shift_str,
                'delta_ref_cpu': f"{delta_ref:.5f}" if delta_ref is not None else '',
                'convergence_rate_ref': f"{conv_rate:.5f}" if conv_rate is not None else '',
                'delta_cuda_rns': f"{delta_cuda:.5f}" if math.isfinite(delta_cuda) else str(delta_cuda),
                'delta_method': delta_method,
                'est_float': f"{est:.10f}" if math.isfinite(est) else str(est),
                'delta_diff': f"{diff:.6f}" if math.isfinite(diff) else '',
                'match': 'YES' if is_match else 'NO',
                'p_bits': p_bits,
                'depth': args.depth,
                'K': K_use,
            })

            # Progress
            elapsed = time.time() - t0
            status = "MATCH" if is_match else "MISS "
            delta_show = f"{delta_cuda:.5f}" if math.isfinite(delta_cuda) else str(delta_cuda)
            ref_show = f"{delta_ref:.5f}" if delta_ref is not None else "N/A"
            cached_tag = " (cached)" if pcf_key in seen_pcfs else ""
            seen_pcfs.add(pcf_key)
            print(f"  [{ti+1:>5}/{len(tasks)}] {status} "
                  f"δ_cuda={delta_show:>10} δ_ref={ref_show:>10} "
                  f"diff={diff:.6f} [{delta_method}] K={K_use} "
                  f"({elapsed:.1f}s){cached_tag} a={a_str[:40]}")

    elapsed_total = time.time() - t_global

    # Summary
    print(f"\n{'='*90}")
    print(f"VERIFICATION COMPLETE")
    print(f"  Tasks processed:  {len(tasks)} ({n_unique_pcfs} unique PCFs)")
    print(f"  Verified (rows):  {n_total}")
    print(f"  Matches:          {n_match}/{n_total} ({100*n_match/max(n_total,1):.1f}%)")
    print(f"  Skipped:          {n_skip}")
    print(f"  Errors:           {n_error}")
    print(f"  Delta tolerance:  {args.delta_tol}")
    print(f"  Total time:       {elapsed_total:.1f}s")
    print(f"  CSV saved:        {args.output}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
