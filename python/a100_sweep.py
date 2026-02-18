#!/usr/bin/env python3
"""
A100 Sweep: Euler2AI Pi PCFs + pFq CMF exhaust with multi-constant matching.

Two modes:
  1. Euler2AI: Load pcfs.json, verify Pi-related PCFs
  2. CMF sweep: Load pFq CMF specs, run all trajectories × shifts,
     test against 15 mathematical constants, save hits

Saves results where:
  - limit is within 1e-3 of any known constant, OR
  - delta > 0 (strong convergence signal)

Usage:
    # Euler2AI Pi PCFs
    python a100_sweep.py --mode euler2ai --input pcfs.json --depth 2000 --K 32

    # 3F2 CMF sweep (1000 CMFs, all traj+shifts)
    python a100_sweep.py --mode cmf --input sweep_data/3F2/3F2_part00.jsonl \\
        --traj sweep_data/trajectories/dim5_trajectories.json \\
        --shifts sweep_data/shifts/dim5_shifts.json \\
        --depth 2000 --K 32 --output results/3F2_part00/
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dreams_rns import compile_pcf, pcf_initial_values, run_pcf_walk
from dreams_rns import crt_reconstruct, centered
from dreams_rns.constants import (
    load_constants, match_against_constants, compute_delta_against_constant,
)

# Lazy GPU imports — only loaded when CMF mode actually runs
def _lazy_gpu_imports():
    from dreams_rns.gpu_walk import gpu_available, run_pcf_walk_batch_gpu, run_cmf_walk_batch_gpu
    return gpu_available, run_pcf_walk_batch_gpu, run_cmf_walk_batch_gpu

def _lazy_cmf_imports():
    from dreams_rns.cmf_walk import compile_cmf_spec, run_cmf_walk, shift_to_axis_offsets
    return compile_cmf_spec, run_cmf_walk, shift_to_axis_offsets


# ── Euler2AI mode ────────────────────────────────────────────────────────

def load_pcfs_json(path: str) -> List[Dict]:
    """Load PCFs from Euler2AI pcfs.json (JSONL: one JSON object per line)."""
    records = []
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # Single JSON array
            items = json.load(f)
        else:
            # JSONL: one JSON object per line
            items = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    for rec in items:
        if 'a' in rec and 'b' in rec:
            records.append({
                'a': str(rec['a']),
                'b': str(rec['b']),
                'limit': rec.get('limit', None),
            })
    return records


def run_euler2ai_mode(args, constants):
    """Run Euler2AI PCF verification with multi-constant matching."""
    records = load_pcfs_json(args.input)
    if args.max_tasks > 0:
        records = records[:args.max_tasks]

    print(f"Loaded {len(records)} PCFs from {args.input}")
    print(f"Testing against {len(constants)} mathematical constants")
    print(f"Depth={args.depth}, K={args.K}")
    print(f"{'='*80}")

    hits = []
    all_results = []

    for i, rec in enumerate(records):
        a_str, b_str = rec['a'], rec['b']
        limit_str = rec.get('limit', None)
        t0 = time.time()

        try:
            program = compile_pcf(a_str, b_str)
            if program is None:
                print(f"  [{i+1}/{len(records)}] SKIP (compile fail): a={a_str}, b={b_str}")
                continue

            a0 = pcf_initial_values(a_str)

            # Auto-K: estimate primes needed for exact CRT
            if args.K > 0:
                K_use = args.K
            else:
                from euler2ai_full_verify import estimate_K
                K_use = estimate_K(a_str, b_str, args.depth)

            res = run_pcf_walk(program, a0, args.depth, K_use)

            # CRT reconstruction
            primes = [int(p) for p in res['primes']]
            p_big, Mp = crt_reconstruct([int(r) for r in res['p_residues']], primes)
            q_big, _ = crt_reconstruct([int(r) for r in res['q_residues']], primes)
            p_big = centered(p_big, Mp)
            q_big = centered(q_big, Mp)

            # Float estimate
            est = res['p_float'] / res['q_float'] if abs(res['q_float']) > 1e-300 else float('nan')
            log_scale = res.get('log_scale', 0.0)
            q_float = res['q_float']

            elapsed = time.time() - t0

            # Detect CRT overflow: compare CRT ratio with float shadow
            crt_valid = True
            if q_big == 0 or not math.isfinite(est):
                crt_valid = False
            else:
                try:
                    import mpmath as _mp
                    crt_ratio = float(_mp.mpf(p_big) / _mp.mpf(q_big))
                    if not math.isfinite(crt_ratio):
                        crt_valid = False
                    elif abs(est) > 1e-10:
                        crt_valid = abs(crt_ratio - est) / abs(est) < 0.01
                    else:
                        crt_valid = abs(crt_ratio - est) < 0.01
                except Exception:
                    crt_valid = False

            # 1. Delta against stated limit (each Euler2AI PCF has its own target)
            delta_stated = float('-inf')
            limit_float = None
            if limit_str:
                import sympy as sp
                import mpmath as mp
                mp.mp.dps = args.dps
                try:
                    target_expr = sp.sympify(limit_str, locals={
                        "pi": sp.pi, "E": sp.E, "EulerGamma": sp.EulerGamma})
                    target_mp = mp.mpf(str(sp.N(target_expr, args.dps)))
                    limit_float = float(target_mp)
                    if crt_valid:
                        delta_stated = float(compute_delta_against_constant(
                            p_big, q_big, target_mp, args.dps))
                    else:
                        # CRT overflow — use float shadow delta
                        err = abs(est - limit_float)
                        if err > 0:
                            log_err = math.log(err)
                            log_q = log_scale + (math.log(abs(q_float)) if abs(q_float) > 1e-300 else 0)
                            if log_q > 0:
                                delta_stated = -(1.0 + log_err / log_q)
                except Exception:
                    pass

            # 2. Delta against all 15 bank constants
            matches = match_against_constants(est, constants, args.proximity)
            best_delta = float('-inf')
            best_const = None
            for c in constants:
                if crt_valid:
                    delta = compute_delta_against_constant(p_big, q_big, c['value'], args.dps)
                else:
                    # Float delta for bank constants too
                    try:
                        err = abs(est - c['value_float'])
                        if err > 0:
                            log_err = math.log(err)
                            log_q = log_scale + (math.log(abs(q_float)) if abs(q_float) > 1e-300 else 0)
                            delta = -(1.0 + log_err / log_q) if log_q > 0 else float('-inf')
                        else:
                            delta = float('inf')
                    except Exception:
                        delta = float('-inf')
                if delta > best_delta:
                    best_delta = delta
                    best_const = c['name']

            result = {
                'idx': i,
                'a': a_str,
                'b': b_str,
                'limit_stated': limit_str or '',
                'limit_float': limit_float,
                'delta_stated': delta_stated,
                'est_float': est,
                'best_const': best_const,
                'best_delta': best_delta,
                'p_bits': Mp.bit_length(),
                'elapsed_s': elapsed,
                'matches': [m['name'] for m in matches],
            }
            all_results.append(result)

            # Hit if: stated delta > 0, or bank delta > 0, or proximity match
            is_hit = delta_stated > 0 or best_delta > 0 or len(matches) > 0
            status = "HIT" if is_hit else "---"
            if is_hit:
                hits.append(result)

            match_str = ', '.join(m['name'] for m in matches[:3]) if matches else '-'
            delta_show = delta_stated if delta_stated > best_delta else best_delta
            const_show = 'stated' if delta_stated > best_delta else best_const
            print(f"  [{i+1:>4}/{len(records)}] {status} δ={delta_show:>8.4f} "
                  f"→{const_show:<12} est={est:.8f} "
                  f"matches=[{match_str}] ({elapsed:.1f}s) a={a_str}")

        except Exception as e:
            print(f"  [{i+1}/{len(records)}] ERROR: {e} | a={a_str}, b={b_str}")

    return all_results, hits


# ── CMF sweep mode ───────────────────────────────────────────────────────

def load_cmf_specs(path: str) -> List[Dict]:
    specs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                specs.append(json.loads(line))
    return specs


def load_trajectories(path: str) -> List[List[int]]:
    with open(path) as f:
        data = json.load(f)
    return data['trajectories']


def load_shifts(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data['shifts']


def compile_cmf_matrix_to_pcf(spec: Dict) -> Optional[tuple]:
    """Convert a pFq CMF spec to a 2×2 PCF for the runner.

    For pFq CMFs, the companion matrix is r×r where r = max(p,q)+1.
    We extract the effective a(n) and b(n) from the last column
    for 2×2 reduction when applicable, or fall back to direct
    matrix walk for higher rank.

    For rank-2 (2×2) CMFs: directly use M[0,1] as b(n), M[1,1] as a(n).
    For higher rank: return the full matrix expressions.
    """
    rank = spec.get('rank', 2)
    matrix = spec.get('matrix', {})

    if rank == 2:
        # Direct 2×2: a(n) = matrix["1,1"], b(n) = matrix["0,1"]
        a_str = matrix.get("1,1", "0")
        b_str = matrix.get("0,1", "0")
        return a_str, b_str

    # For higher rank, extract last-column entries
    # The walk convergent comes from the (rank)×(rank) companion
    # For now, use the top-right and bottom-right as effective a,b
    # (This is a simplification — full rank walk needs generalized runner)
    r = rank
    a_str = matrix.get(f"{r-1},{r-1}", "0")
    b_str = matrix.get(f"0,{r-1}", "0")
    return a_str, b_str


def run_cmf_sweep_mode(args, constants):
    """Run CMF sweep: all specs × trajectories × shifts.

    Uses the full r×r companion matrix walk with per-axis variables.
    Each (CMF, trajectory) pair is compiled once, then ALL shifts are
    walked in a single GPU batch for maximum throughput.

    For 3F2 (rank=4, dim=5): 1000 CMFs × 8161 traj × 512 shifts = 4.18B walks
    For a quick first pass: use --max-traj to limit trajectories.
    """
    # Lazy imports — only loaded when CMF mode is actually used
    gpu_available, _, run_cmf_walk_batch_gpu = _lazy_gpu_imports()
    compile_cmf_spec, run_cmf_walk, shift_to_axis_offsets = _lazy_cmf_imports()

    specs = load_cmf_specs(args.input)
    if args.max_tasks > 0:
        specs = specs[:args.max_tasks]

    trajs = load_trajectories(args.traj)
    shifts = load_shifts(args.shifts)

    max_traj = getattr(args, 'max_traj', 0)
    if max_traj > 0:
        trajs = trajs[:max_traj]

    n_cmfs = len(specs)
    n_traj = len(trajs)
    n_shifts = len(shifts)
    total_walks = n_cmfs * n_traj * n_shifts

    use_gpu = gpu_available()
    backend = "GPU (CuPy)" if use_gpu else "CPU (numpy)"

    print(f"CMF Sweep Configuration:")
    print(f"  CMFs:          {n_cmfs}")
    print(f"  Trajectories:  {n_traj}")
    print(f"  Shifts:        {n_shifts}")
    print(f"  Total walks:   {total_walks:,}")
    print(f"  Constants:     {len(constants)}")
    # CMF mode: use explicit K or default to 1000 for general coverage
    cmf_K = args.K if args.K > 0 else 1000
    print(f"  Depth={args.depth}, K={cmf_K}")
    print(f"  Backend:       {backend}")
    print(f"  Proximity:     {args.proximity}")
    print(f"{'='*80}")

    hits = []
    run_count = 0
    hit_count = 0
    t_global = time.time()

    for ci, spec in enumerate(specs):
        name = spec.get('name', f'CMF_{ci}')
        rank = spec.get('rank', 2)
        spec_hash = spec.get('spec_hash', '')
        dim = spec.get('dim', spec['p'] + spec['q'])
        cmf_hits = 0
        t_cmf = time.time()

        # Pre-compute all shift offset vectors once
        all_shift_offsets = []
        for shift in shifts:
            so = shift_to_axis_offsets(shift, spec['p'])
            while len(so) < dim:
                so.append(1)
            all_shift_offsets.append(so[:dim])

        for ti, traj in enumerate(trajs):
            # Compile once per (CMF, trajectory) pair
            try:
                program = compile_cmf_spec(spec, trajectory=list(traj))
                if program is None:
                    run_count += n_shifts
                    continue
            except Exception:
                run_count += n_shifts
                continue

            # Batch ALL shifts through GPU walker at once
            try:
                results = run_cmf_walk_batch_gpu(
                    program, args.depth, cmf_K,
                    shift_val_list=all_shift_offsets,
                    batch_size=getattr(args, 'batch_size', 128),
                )
            except Exception as e:
                print(f"    WARN: batch walk failed for {name} traj {ti}: {e}")
                run_count += n_shifts
                continue

            # Post-process each result
            for si, res in enumerate(results):
                run_count += 1
                try:
                    primes = [int(p) for p in res['primes']]
                    p_big, Mp = crt_reconstruct(
                        [int(r) for r in res['p_residues']], primes)
                    q_big, _ = crt_reconstruct(
                        [int(r) for r in res['q_residues']], primes)
                    p_big = centered(p_big, Mp)
                    q_big = centered(q_big, Mp)

                    est = (res['p_float'] / res['q_float']
                           if abs(res['q_float']) > 1e-300 else float('nan'))
                    log_scale = res.get('log_scale', 0.0)
                    q_float = res['q_float']

                    if not math.isfinite(est):
                        continue

                    # Detect CRT overflow
                    crt_ok = True
                    if q_big == 0:
                        crt_ok = False
                    else:
                        try:
                            cr = float(p_big) / float(q_big)
                            if not math.isfinite(cr):
                                crt_ok = False
                            elif abs(est) > 1e-10:
                                crt_ok = abs(cr - est) / abs(est) < 0.01
                            else:
                                crt_ok = abs(cr - est) < 0.01
                        except Exception:
                            crt_ok = False

                    # Quick proximity check (cheap, float-only)
                    proximity_matches = match_against_constants(
                        est, constants, args.proximity)

                    # Delta computation (CRT or float fallback)
                    best_delta = float('-inf')
                    best_const = None
                    log_q = log_scale + (math.log(abs(q_float)) if abs(q_float) > 1e-300 else 0)

                    candidates = ([next(c for c in constants if c['name'] == m['name'])
                                   for m in proximity_matches]
                                  if proximity_matches
                                  else [c for c in constants if abs(est - c['value_float']) < 1.0])

                    for c in candidates:
                        if crt_ok:
                            delta = compute_delta_against_constant(
                                p_big, q_big, c['value'], args.dps)
                        else:
                            try:
                                err = abs(est - c['value_float'])
                                delta = (-(1.0 + math.log(err) / log_q)
                                         if err > 0 and log_q > 0 else float('-inf'))
                            except Exception:
                                delta = float('-inf')
                        if delta > best_delta:
                            best_delta = delta
                            best_const = c['name']

                    is_hit = best_delta > 0 or len(proximity_matches) > 0
                    if is_hit:
                        result = {
                            'cmf_name': name,
                            'spec_hash': spec_hash,
                            'rank': rank,
                            'traj_idx': ti,
                            'traj': list(traj),
                            'shift_idx': si,
                            'shift_offsets': all_shift_offsets[si],
                            'est_float': est,
                            'best_const': best_const,
                            'best_delta': best_delta,
                            'p_bits': Mp.bit_length(),
                            'proximity_matches': [m['name'] for m in proximity_matches],
                            'delta_positive': best_delta > 0,
                        }
                        hits.append(result)
                        cmf_hits += 1
                        hit_count += 1

                except Exception:
                    continue

        elapsed_cmf = time.time() - t_cmf
        walks_this_cmf = n_traj * n_shifts
        rate = walks_this_cmf / elapsed_cmf if elapsed_cmf > 0 else 0
        print(f"  [{ci+1:>4}/{n_cmfs}] {name:<40} "
              f"hits={cmf_hits:>4} "
              f"({elapsed_cmf:.1f}s, {rate:.0f} walks/s)")

    elapsed_total = time.time() - t_global

    return [], hits, {
        'total_runs': run_count,
        'total_hits': hit_count,
        'elapsed_s': elapsed_total,
    }


# ── Output ───────────────────────────────────────────────────────────────

def save_hits(hits: List[Dict], output_dir: str, prefix: str = "hits"):
    """Save hits to both JSONL and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # JSONL (full data)
    jsonl_path = os.path.join(output_dir, f"{prefix}.jsonl")
    with open(jsonl_path, 'w') as f:
        for h in hits:
            f.write(json.dumps(h, default=str) + "\n")

    # CSV (summary)
    if hits:
        csv_path = os.path.join(output_dir, f"{prefix}.csv")
        keys = list(hits[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for h in hits:
                row = {k: str(v) if isinstance(v, list) else v for k, v in h.items()}
                writer.writerow(row)

    return jsonl_path


def save_all_results(results: List[Dict], output_dir: str, prefix: str = "all"):
    """Save all results to JSONL."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{prefix}.jsonl")
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    return path


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A100 Sweep: Euler2AI + CMF exhaust with multi-constant matching"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["euler2ai", "cmf"],
                        help="euler2ai: verify PCFs from pcfs.json; cmf: sweep pFq CMFs")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file (pcfs.json or CMF spec JSONL)")
    parser.add_argument("--traj", type=str, default=None,
                        help="Trajectory JSON file (CMF mode only)")
    parser.add_argument("--shifts", type=str, default=None,
                        help="Shifts JSON file (CMF mode only)")
    parser.add_argument("--depth", type=int, default=2000)
    parser.add_argument("--K", type=int, default=0,
                        help="RNS primes (0=auto per PCF for euler2ai mode)")
    parser.add_argument("--dps", type=int, default=200,
                        help="mpmath decimal precision")
    parser.add_argument("--proximity", type=float, default=1e-3,
                        help="Proximity threshold for constant matching")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Max CMFs/PCFs to process (0=all)")
    parser.add_argument("--max-traj", type=int, default=0,
                        help="Max trajectories per CMF (0=all, CMF mode only)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="GPU batch size (walks per GPU kernel)")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory")
    args = parser.parse_args()

    # Load constants bank
    print("Loading mathematical constants bank...")
    constants = load_constants(args.dps)
    print(f"  {len(constants)} constants loaded:")
    for c in constants:
        print(f"    {c['name']:<15} = {c['value_float']:.12f}  ({c['description']})")
    print()

    if args.mode == "euler2ai":
        all_results, hits = run_euler2ai_mode(args, constants)

        print(f"\n{'='*80}")
        print(f"EULER2AI RESULTS")
        print(f"  Total PCFs:     {len(all_results)}")
        print(f"  Hits (δ>0 or proximity): {len(hits)}")

        if all_results:
            save_all_results(all_results, args.output, "euler2ai_all")
        if hits:
            p = save_hits(hits, args.output, "euler2ai_hits")
            print(f"  Hits saved:     {p}")

            # Print top hits
            top = sorted(hits, key=lambda x: x.get('best_delta', -999), reverse=True)[:20]
            print(f"\n  Top {len(top)} by delta:")
            for h in top:
                print(f"    δ={h['best_delta']:>8.4f} → {h['best_const']:<12} "
                      f"a={h['a']}, b={h['b']}")

    elif args.mode == "cmf":
        if not args.traj or not args.shifts:
            parser.error("CMF mode requires --traj and --shifts")

        _, hits, stats = run_cmf_sweep_mode(args, constants)

        print(f"\n{'='*80}")
        print(f"CMF SWEEP RESULTS")
        print(f"  Total runs:     {stats['total_runs']:,}")
        print(f"  Total hits:     {stats['total_hits']:,}")
        print(f"  Elapsed:        {stats['elapsed_s']:.1f}s")
        if stats['elapsed_s'] > 0:
            print(f"  Rate:           {stats['total_runs']/stats['elapsed_s']:.0f} runs/s")

        if hits:
            p = save_hits(hits, args.output, "cmf_hits")
            print(f"  Hits saved:     {p}")

            # Summary by constant
            from collections import Counter
            const_counts = Counter(h['best_const'] for h in hits if h.get('delta_positive'))
            if const_counts:
                print(f"\n  Hits with δ>0 by constant:")
                for name, count in const_counts.most_common():
                    print(f"    {name:<15}: {count}")

            # Top hits
            top = sorted(hits, key=lambda x: x.get('best_delta', -999), reverse=True)[:20]
            print(f"\n  Top {len(top)} by delta:")
            for h in top:
                print(f"    δ={h['best_delta']:>8.4f} → {h['best_const']:<12} "
                      f"cmf={h['cmf_name']}")


if __name__ == "__main__":
    main()
