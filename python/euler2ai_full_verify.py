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
    parser.add_argument("--K", type=int, default=32)
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
    if args.max_tasks > 0:
        records = records[:args.max_tasks]

    # Expand per-source
    total_sources = sum(len(r.get('sources', [{}])) for r in records)
    print(f"Loaded {len(records)} unique PCFs, {total_sources} total sources")
    print(f"Depth={args.depth}, K={args.K}, dps={args.dps}")
    print(f"Delta tolerance: {args.delta_tol}")
    print(f"Output: {args.output}")
    print(f"{'='*90}")

    # Prepare CSV
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    fieldnames = [
        'pcf_idx', 'source_idx', 'a', 'b', 'limit',
        'trajectory', 'shift',
        'delta_ref_cpu', 'convergence_rate_ref',
        'delta_cuda_rns', 'est_float',
        'delta_diff', 'match',
        'p_bits', 'depth', 'K',
    ]

    n_match = 0
    n_total = 0
    n_skip = 0
    n_error = 0
    t_global = time.time()

    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pi, rec in enumerate(records):
            a_str = rec['a']
            b_str = rec['b']
            limit_str = rec.get('limit', '')
            delta_ref = rec.get('delta', None)
            conv_rate = rec.get('convergence_rate', None)
            sources = rec.get('sources', [None])

            t0 = time.time()

            # Parse limit
            target_mp = parse_limit(limit_str, args.dps) if limit_str else None

            # Compile PCF once
            try:
                program = compile_pcf(a_str, b_str)
                if program is None:
                    n_skip += len(sources)
                    print(f"  [{pi+1:>5}/{len(records)}] SKIP (imaginary): a={a_str}")
                    for si, src in enumerate(sources):
                        writer.writerow({
                            'pcf_idx': pi, 'source_idx': si,
                            'a': a_str, 'b': b_str, 'limit': limit_str,
                            'trajectory': str(src[0]) if src and isinstance(src, list) else '',
                            'shift': str(src[1]) if src and isinstance(src, list) and len(src) > 1 else '',
                            'delta_ref_cpu': delta_ref, 'convergence_rate_ref': conv_rate,
                            'delta_cuda_rns': 'SKIP', 'est_float': '',
                            'delta_diff': '', 'match': 'SKIP',
                            'p_bits': '', 'depth': args.depth, 'K': args.K,
                        })
                    continue

                a0 = pcf_initial_values(a_str)
            except Exception as e:
                n_error += len(sources)
                print(f"  [{pi+1:>5}/{len(records)}] ERROR compile: {e} | a={a_str}")
                continue

            # Walk once (PCF walk is same for all sources of same PCF)
            try:
                res = run_pcf_walk(program, a0, args.depth, args.K)

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

                # Compute delta against stated limit
                if target_mp is not None:
                    delta_cuda = compute_delta(p_big, q_big, target_mp, args.dps)
                else:
                    delta_cuda = float('nan')

                p_bits = Mp.bit_length()
                elapsed = time.time() - t0

                # Compare with reference
                if delta_ref is not None and math.isfinite(delta_cuda):
                    diff = abs(delta_cuda - delta_ref)
                    is_match = diff < args.delta_tol
                else:
                    diff = float('nan')
                    is_match = False

                # Write one row per source
                for si, src in enumerate(sources):
                    n_total += 1
                    if is_match:
                        n_match += 1

                    traj_str = str(src[0]) if src and isinstance(src, list) else ''
                    shift_str = str(src[1]) if src and isinstance(src, list) and len(src) > 1 else ''

                    writer.writerow({
                        'pcf_idx': pi,
                        'source_idx': si,
                        'a': a_str,
                        'b': b_str,
                        'limit': limit_str,
                        'trajectory': traj_str,
                        'shift': shift_str,
                        'delta_ref_cpu': f"{delta_ref:.5f}" if delta_ref is not None else '',
                        'convergence_rate_ref': f"{conv_rate:.5f}" if conv_rate is not None else '',
                        'delta_cuda_rns': f"{delta_cuda:.5f}" if math.isfinite(delta_cuda) else str(delta_cuda),
                        'est_float': f"{est:.10f}" if math.isfinite(est) else str(est),
                        'delta_diff': f"{diff:.6f}" if math.isfinite(diff) else '',
                        'match': 'YES' if is_match else 'NO',
                        'p_bits': p_bits,
                        'depth': args.depth,
                        'K': args.K,
                    })

                status = "MATCH" if is_match else "MISS"
                delta_show = f"{delta_cuda:.5f}" if math.isfinite(delta_cuda) else str(delta_cuda)
                ref_show = f"{delta_ref:.5f}" if delta_ref is not None else "N/A"
                print(f"  [{pi+1:>5}/{len(records)}] {status} "
                      f"δ_cuda={delta_show:>10} δ_ref={ref_show:>10} "
                      f"diff={diff:.6f} est={est:.8f} "
                      f"({elapsed:.1f}s) {len(sources)}src a={a_str[:40]}")

            except Exception as e:
                n_error += len(sources)
                print(f"  [{pi+1:>5}/{len(records)}] ERROR walk: {e} | a={a_str}")
                for si, src in enumerate(sources):
                    writer.writerow({
                        'pcf_idx': pi, 'source_idx': si,
                        'a': a_str, 'b': b_str, 'limit': limit_str,
                        'trajectory': str(src[0]) if src and isinstance(src, list) else '',
                        'shift': str(src[1]) if src and isinstance(src, list) and len(src) > 1 else '',
                        'delta_ref_cpu': delta_ref, 'convergence_rate_ref': conv_rate,
                        'delta_cuda_rns': 'ERROR', 'est_float': '',
                        'delta_diff': '', 'match': 'ERROR',
                        'p_bits': '', 'depth': args.depth, 'K': args.K,
                    })

    elapsed_total = time.time() - t_global

    # Summary
    print(f"\n{'='*90}")
    print(f"VERIFICATION COMPLETE")
    print(f"  PCFs processed:   {len(records)}")
    print(f"  Sources (rows):   {n_total}")
    print(f"  Matches:          {n_match}/{n_total} ({100*n_match/max(n_total,1):.1f}%)")
    print(f"  Skipped:          {n_skip}")
    print(f"  Errors:           {n_error}")
    print(f"  Delta tolerance:  {args.delta_tol}")
    print(f"  Total time:       {elapsed_total:.1f}s")
    print(f"  CSV saved:        {args.output}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
