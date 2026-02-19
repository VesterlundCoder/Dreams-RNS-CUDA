#!/usr/bin/env python3
"""
Systematic axis search for ζ(5) and ζ(7) CMFs.

Tests convergence against ζ(2)–ζ(30) (even AND odd) using:
  - "Global shift" equivalent: all axes set to k=s for s=1..15
  - Single-axis trajectory scaling: ×1/2, ×-1/2, ×2, ×-1, ×0
  - Perpendicular single-axis trajectories
  - Half-axis shift perturbations

Depth: 2000, K=16 (float shadow primary metric).

Usage:
    python zeta_axis_search.py                    # both ζ(5) and ζ(7)
    python zeta_axis_search.py --cmf zeta_5       # only ζ(5)
    python zeta_axis_search.py --cmf zeta_7       # only ζ(7)
    python zeta_axis_search.py --cmf zeta_9       # only ζ(9)
"""

import argparse
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odd_zeta_cmf_generator import build_cmf_matrix_2n
from dreams_rns.compiler import compile_cmf_from_dict
from dreams_rns.cmf_walk import run_cmf_walk_vec

import mpmath as mp


# ── Extended constants bank: ζ(2)..ζ(30) + classical ──────────────────

def build_constants_bank(dps=50):
    """Build constants for ζ(2)..ζ(30) plus classical constants."""
    mp.mp.dps = dps
    constants = []

    # Zeta values: ζ(2) through ζ(30)
    for n in range(2, 31):
        val = mp.zeta(n)
        constants.append({
            'name': f'zeta({n})',
            'value': float(val),
            'mp_value': val,
        })

    # Products/ratios of zeta: ζ(a)*ζ(b) for small a,b
    for a in [2, 3, 4, 5]:
        for b in range(a, min(a + 6, 16)):
            if a == b:
                continue
            val = mp.zeta(a) * mp.zeta(b)
            constants.append({
                'name': f'zeta({a})*zeta({b})',
                'value': float(val),
                'mp_value': val,
            })

    # Powers of pi
    for label, expr in [
        ('pi', mp.pi),
        ('pi^2/6', mp.pi**2 / 6),
        ('pi^4/90', mp.pi**4 / 90),
        ('pi^6/945', mp.pi**6 / 945),
        ('pi^2', mp.pi**2),
        ('1/pi', 1 / mp.pi),
    ]:
        constants.append({'name': label, 'value': float(expr), 'mp_value': expr})

    # Classical
    for label, expr in [
        ('ln2', mp.log(2)),
        ('catalan', mp.catalan),
        ('euler_gamma', mp.euler),
    ]:
        constants.append({'name': label, 'value': float(expr), 'mp_value': expr})

    return constants


def match_estimate(est, constants, threshold_digits=3.0):
    """Match estimate against all constants, return best match."""
    best_name = "none"
    best_digits = -1.0

    for c in constants:
        err = abs(est - c['value'])
        if err == 0:
            return c['name'], 16.0
        if c['value'] != 0:
            digits = -math.log10(err / max(abs(c['value']), 1e-300))
        else:
            digits = 0.0
        if digits > best_digits:
            best_digits = digits
            best_name = c['name']

    return best_name, max(best_digits, 0.0)


# ── CMF builder ───────────────────────────────────────────────────────

def build_cmf(n: int):
    """Build and compile a ζ(2n+1) CMF, return (program, v0, acc_idx, const_idx, dim, dirs, shifts)."""
    matrix_dict, dim, axis_names, dirs, def_shifts, target = build_cmf_matrix_2n(n, multiaxis=True)

    # matrix_dict already has {(row, col): "expr"} format
    program = compile_cmf_from_dict(
        matrix_dict=matrix_dict,
        m=2 * n,
        dim=dim,
        axis_names=axis_names,
        directions=dirs,
    )

    # Initial state
    from odd_zeta_sweep import compute_initial_state
    v0 = compute_initial_state(n)

    acc_idx = n
    const_idx = n + 1

    return program, v0, acc_idx, const_idx, dim, dirs, def_shifts, axis_names


def global_shift_for_k(n: int, k_start: int, dim: int, dirs: List[int], def_shifts: List[int]):
    """Compute multi-axis shift vector equivalent to starting walk at k=k_start.

    For ζ(2n+1) with n rows:
      Numerator axes (dir=1, def_shift=2): value at k = k+1 → shift = k_start+1
      Denominator axes (dir=2, def_shift=3): value at k = 2k+1 → shift = 2*k_start+1
      Coupling axes (dir=1, def_shift=1): value at k = k → shift = k_start
      Accumulator (dir=1, def_shift=2): value at k = k+1 → shift = k_start+1
    """
    shift = []
    for i in range(dim):
        d = dirs[i]
        s0 = def_shifts[i]
        # At k=1: value = s0. At k=k_start: value = s0 + d*(k_start - 1)
        shift.append(Fraction(s0 + d * (k_start - 1)))
    return shift


# ── Trajectory/Shift generators ──────────────────────────────────────

def generate_search_combos(dim, dirs, def_shifts, n_val):
    """Generate (label, shift_vec, traj_vec) tuples for systematic search."""
    combos = []
    base_dirs = [Fraction(d) for d in dirs]
    base_shifts = [Fraction(s) for s in def_shifts]

    # ── GROUP 1: Global k-shift with standard trajectory ──
    # Equivalent to single-axis "shift=s" from earlier experiments
    for k_start in range(1, 16):
        shift = global_shift_for_k(n_val, k_start, dim, dirs, def_shifts)
        combos.append((f"k={k_start}", shift, list(base_dirs)))

    # ── GROUP 2: Standard shift, single-axis trajectory scaling ──
    # For each axis, multiply by 1/2, -1/2, 2, -1, 0, 3, 1/3
    mults = [
        (Fraction(1, 2), "×1/2"),
        (Fraction(-1, 2), "×-1/2"),
        (Fraction(2), "×2"),
        (Fraction(-1), "×-1"),
        (Fraction(0), "×0"),
        (Fraction(3), "×3"),
        (Fraction(1, 3), "×1/3"),
        (Fraction(-1, 3), "×-1/3"),
        (Fraction(3, 2), "×3/2"),
        (Fraction(2, 3), "×2/3"),
    ]
    for axis in range(dim):
        for mult, label in mults:
            traj = list(base_dirs)
            traj[axis] = base_dirs[axis] * mult
            combos.append(
                (f"traj x{axis}{label}", list(base_shifts), traj))

    # ── GROUP 3: Perpendicular single-axis trajectories ──
    # Only one axis nonzero (the rest = 0)
    for axis in range(dim):
        traj = [Fraction(0)] * dim
        traj[axis] = base_dirs[axis]
        combos.append((f"perp x{axis}", list(base_shifts), traj))
        # Also try half
        traj2 = [Fraction(0)] * dim
        traj2[axis] = base_dirs[axis] * Fraction(1, 2)
        combos.append((f"perp x{axis}×1/2", list(base_shifts), traj2))

    # ── GROUP 4: Two-axis perpendicular pairs ──
    for a1 in range(dim):
        for a2 in range(a1 + 1, dim):
            traj = [Fraction(0)] * dim
            traj[a1] = base_dirs[a1]
            traj[a2] = base_dirs[a2]
            combos.append((f"perp x{a1}+x{a2}", list(base_shifts), traj))

    # ── GROUP 5: Standard shift + single-axis shift ±1/2 ──
    for axis in range(dim):
        for offset, label in [(Fraction(1, 2), "+1/2"), (Fraction(-1, 2), "-1/2")]:
            shift = list(base_shifts)
            shift[axis] = base_shifts[axis] + offset
            combos.append(
                (f"shift x{axis}{label}", shift, list(base_dirs)))

    # ── GROUP 6: Global k-shift with half-axis trajectories ──
    # Combine k-shift with single-axis ×1/2 and ×-1/2
    for k_start in [1, 2, 3, 4, 5]:
        shift = global_shift_for_k(n_val, k_start, dim, dirs, def_shifts)
        for axis in range(dim):
            for mult, ml in [(Fraction(1, 2), "×1/2"), (Fraction(-1, 2), "×-1/2")]:
                traj = list(base_dirs)
                traj[axis] = base_dirs[axis] * mult
                combos.append(
                    (f"k={k_start} traj x{axis}{ml}", shift, traj))

    # ── GROUP 7: Uniform trajectory scaling (all axes) ──
    for mult, label in [(Fraction(1, 2), "×1/2"), (Fraction(2), "×2"),
                         (Fraction(-1), "×-1"), (Fraction(3), "×3")]:
        traj = [d * mult for d in base_dirs]
        combos.append((f"all traj {label}", list(base_shifts), traj))

    return combos


# ── Main ──────────────────────────────────────────────────────────────

def run_search(n_val: int, depth: int, K: int, constants, output_dir: str):
    """Run axis search for a single ζ(2n+1) CMF."""
    zeta_target = 2 * n_val + 1
    cmf_name = f"zeta_{zeta_target}"
    print(f"\n{'='*80}")
    print(f"  {cmf_name}: ζ({zeta_target}), n={n_val}, {2*n_val}×{2*n_val} matrix")
    print(f"{'='*80}")

    program, v0, acc_idx, const_idx, dim, dirs, def_shifts, axis_names = build_cmf(n_val)
    print(f"  Axes: {axis_names}")
    print(f"  Default dirs:   {dirs}")
    print(f"  Default shifts: {def_shifts}")
    print(f"  Initial state:  {[str(x) for x in v0]}")
    print(f"  Acc idx: {acc_idx}, Const idx: {const_idx}")

    combos = generate_search_combos(dim, dirs, def_shifts, n_val)
    print(f"  Total combos: {len(combos)}")
    print(f"  Depth: {depth}, K: {K}")
    print(f"{'─'*80}")

    results = []
    hits = []
    t0 = time.time()

    for ci, (label, shift_vec, traj_vec) in enumerate(combos):
        try:
            res = run_cmf_walk_vec(
                program, depth, K,
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

            best_name, best_digits = match_estimate(est, constants)

            result = {
                'cmf': cmf_name,
                'label': label,
                'shift': [str(s) for s in shift_vec],
                'trajectory': [str(t) for t in traj_vec],
                'estimate': est,
                'best_match': best_name,
                'match_digits': round(best_digits, 1),
                'depth': depth,
            }
            results.append(result)

            if best_digits >= 4.0:
                hits.append(result)
                marker = "★★★ HIT" if best_digits >= 6 else "★ near"
                print(f"  [{ci+1}/{len(combos)}] {marker} {best_digits:.1f}d → "
                      f"{best_name}  est={est:.15g}  [{label}]")

        except Exception as e:
            if ci < 3:
                print(f"  [{ci+1}] ERROR: {e}")
            continue

        # Progress
        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            eta = (len(combos) - ci - 1) / rate
            print(f"  ... {ci+1}/{len(combos)} ({rate:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(results)} valid results in {elapsed:.1f}s")
    print(f"  Hits (≥4d): {len([h for h in hits if h['match_digits'] >= 4])}")
    print(f"  Hits (≥6d): {len([h for h in hits if h['match_digits'] >= 6])}")

    # Sort hits by digits
    hits.sort(key=lambda x: -x['match_digits'])

    # Print hit summary
    if hits:
        print(f"\n  {'─'*70}")
        print(f"  {'DIGITS':>6}  {'CONSTANT':<20}  {'LABEL':<30}  ESTIMATE")
        print(f"  {'─'*70}")
        for h in hits[:40]:
            print(f"  {h['match_digits']:>6.1f}  {h['best_match']:<20}  "
                  f"{h['label']:<30}  {h['estimate']:.15g}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{cmf_name}_axis_search.jsonl")
    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  Results saved to {out_path}")

    # Save hits separately
    if hits:
        hits_path = os.path.join(output_dir, f"{cmf_name}_axis_hits.json")
        with open(hits_path, 'w') as f:
            json.dump(hits, f, indent=2, default=str)
        print(f"  Hits saved to {hits_path}")

    return results, hits


def main():
    parser = argparse.ArgumentParser(
        description="Systematic axis search for odd-zeta CMFs"
    )
    parser.add_argument("--cmf", type=str, default="",
                        help="Which CMF: zeta_5, zeta_7, zeta_9 (empty=5+7)")
    parser.add_argument("--depth", type=int, default=2000,
                        help="Walk depth (default 2000)")
    parser.add_argument("--K", type=int, default=16,
                        help="RNS primes (default 16)")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent / "axis_search_results"),
                        help="Output directory")
    args = parser.parse_args()

    print("Building extended constants bank (ζ(2)..ζ(30) + classical)...")
    constants = build_constants_bank(dps=50)
    print(f"  {len(constants)} constants loaded")

    # Map CMF name to n value
    cmf_map = {
        'zeta_5': 2,
        'zeta_7': 3,
        'zeta_9': 4,
    }

    if args.cmf:
        if args.cmf not in cmf_map:
            print(f"Unknown CMF: {args.cmf}. Options: {list(cmf_map.keys())}")
            sys.exit(1)
        targets = [(args.cmf, cmf_map[args.cmf])]
    else:
        targets = [('zeta_5', 2), ('zeta_7', 3)]

    all_hits = {}
    for cmf_name, n_val in targets:
        results, hits = run_search(n_val, args.depth, args.K, constants, args.output)
        all_hits[cmf_name] = hits

    # Final summary
    print(f"\n{'='*80}")
    print(f"SEARCH COMPLETE")
    for cmf_name, hits in all_hits.items():
        n_exact = len([h for h in hits if h['match_digits'] >= 6])
        n_near = len([h for h in hits if 4 <= h['match_digits'] < 6])
        print(f"  {cmf_name}: {n_exact} exact hits, {n_near} near-misses")


if __name__ == "__main__":
    main()
