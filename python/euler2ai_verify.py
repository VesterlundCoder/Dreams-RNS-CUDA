#!/usr/bin/env python3
"""
Euler2AI PCF Verification — correct RNS pipeline.

Loads PCFs from pcfs.json or cmf_pcfs.json (JSONL), runs the correct
companion-matrix walk, and verifies against stated limits.

Usage:
    python euler2ai_verify.py --input pcfs.json --depth 2000 --K 32 --max-tasks 20
    python euler2ai_verify.py --input cmf_pcfs.json --depth 2000 --K 32 --max-tasks 50
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))
from dreams_rns.runner import (
    compile_pcf, pcf_initial_values, run_pcf_walk,
    crt_reconstruct, centered,
    compute_dreams_delta_float, compute_dreams_delta_exact,
    verify_pcf,
)


def parse_limit(limit_str: str, dps: int = 200):
    """Parse limit string to mpmath high-precision float."""
    import sympy as sp
    import mpmath as mp
    mp.mp.dps = dps
    expr = sp.sympify(limit_str, locals={"pi": sp.pi, "E": sp.E,
                                          "EulerGamma": sp.EulerGamma})
    return mp.mpf(str(sp.N(expr, dps)))


def load_pcfs_json(path: str):
    """Load pcfs.json (Euler2AI format) — one JSON object per line."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({
                'a': rec['a'],
                'b': rec['b'],
                'limit': rec.get('limit', ''),
                'ref_delta': rec.get('delta', None),
                'ref_cr': rec.get('convergence_rate', None),
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Euler2AI PCF Verification")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to pcfs.json or cmf_pcfs.json")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="0 = all")
    parser.add_argument("--output", type=str, default="verify_report.csv")
    parser.add_argument("--dps", type=int, default=200,
                        help="mpmath decimal precision")
    args = parser.parse_args()

    records = load_pcfs_json(args.input)
    if args.max_tasks > 0:
        records = records[:args.max_tasks]

    print(f"Loaded {len(records)} PCFs from {args.input}")
    print(f"Walk depth={args.depth}, K={args.K} primes ({args.K * 31} bits)")
    print()

    results = []
    n_ok = 0
    n_fail = 0
    n_skip = 0
    t0 = time.time()

    for idx, rec in enumerate(tqdm(records, desc="PCF walks")):
        a_str = rec['a']
        b_str = rec['b']
        limit_str = rec['limit']

        if not limit_str:
            n_skip += 1
            continue

        try:
            res = verify_pcf(a_str, b_str, limit_str,
                             depth=args.depth, K=args.K, dps=args.dps)
        except Exception as e:
            tqdm.write(f"  SKIP [{idx}] a={a_str[:40]}: {type(e).__name__}: {e}")
            n_skip += 1
            continue

        if res is None:
            n_skip += 1
            continue

        ref_delta = rec.get('ref_delta')
        delta_exact = res['delta_exact']
        delta_float = res['delta_float']

        row = {
            'idx': idx,
            'a': a_str,
            'b': b_str,
            'limit': limit_str,
            'target': res['target'],
            'est_float': res['est_float'],
            'delta_exact': delta_exact,
            'delta_float': delta_float,
            'ref_delta': ref_delta,
            'delta_diff': (delta_exact - ref_delta) if ref_delta is not None else None,
            'depth': args.depth,
            'K': args.K,
            'p_bits': res['p_bits'],
        }
        results.append(row)

        # Check if limit matches (est should be close to target)
        limit_match = abs(res['est_float'] - res['target']) < 0.01 if math.isfinite(res['est_float']) else False
        if limit_match:
            n_ok += 1
        else:
            n_fail += 1

        if idx < 10 or (idx + 1) % 50 == 0:
            tqdm.write(
                f"  [{idx}] PCF({a_str[:20]}, {b_str[:20]}) "
                f"δ_exact={delta_exact:.6f}  δ_float={delta_float:.6f}  "
                f"ref_δ={ref_delta}  "
                f"limit_match={'✓' if limit_match else '✗'}"
            )

    elapsed = time.time() - t0

    # Write CSV
    if results:
        import csv
        keys = results[0].keys()
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nWrote {len(results)} rows to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total PCFs:      {len(records)}")
    print(f"  Completed:       {len(results)}")
    print(f"  Skipped:         {n_skip}")
    print(f"  Limit matches:   {n_ok}/{len(results)} ({100*n_ok/max(len(results),1):.1f}%)")
    print(f"  Limit failures:  {n_fail}/{len(results)}")
    print(f"  Elapsed:         {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/PCF)")

    if results:
        deltas = [r['delta_exact'] for r in results if r['delta_exact'] is not None
                  and math.isfinite(r['delta_exact'])]
        if deltas:
            print(f"  Delta range:     [{min(deltas):.4f}, {max(deltas):.4f}]")
            print(f"  Delta mean:      {sum(deltas)/len(deltas):.4f}")

        diffs = [r['delta_diff'] for r in results if r['delta_diff'] is not None
                 and math.isfinite(r['delta_diff'])]
        if diffs:
            print(f"  |δ - ref| max:   {max(abs(d) for d in diffs):.6f}")
            print(f"  |δ - ref| mean:  {sum(abs(d) for d in diffs)/len(diffs):.6f}")


if __name__ == "__main__":
    main()
