#!/usr/bin/env python3
"""
Generate CMF specs for ζ(2n+1) using the HPHP08-based general formula.

From General_CMF_Odd_Zeta_Draft.docx:
  - For ζ(2n+1), the CMF is a 2n×2n matrix K(k) over Q(k)
  - Active states: U0..U_{n-1}, S (accumulator), 1 (constant)
  - Zero-padded to 2n×2n
  - Walk variable: k (single axis, direction=1)

Output: JSONL file with compiled CMF specs ready for sweep.

Usage:
    python odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl
    python odd_zeta_cmf_generator.py --n-min 2 --n-max 50 --output odd_zeta_specs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from fractions import Fraction

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_cmf_matrix_2n(n: int):
    """Build the 2n×2n CMF matrix for ζ(2n+1) as a sympy Matrix in k.

    From the HPHP08 construction:
      - r_k = -(k+1) / (2*(2k+1))
      - U0(k+1) = r_k * U0(k)
      - Uj(k+1) = r_k * Uj(k) + (r_k / k²) * U_{j-1}(k)
      - 1(k+1) = 1(k)
      - S(k+1) = S(k) + (1/2)/(k+1)³ * Σ_j c_{n,j}(k+1) * Uj(k+1)
      - Zero rows for padding

    Returns: (matrix_dict, dim, axis_names, target_zeta)
    """
    k = sp.Symbol('k')
    d = 2 * n

    # Index layout: [U0 U1 ... U_{n-1} S 1 Z0 Z1 ... Z_{n-3}]
    idxU = list(range(0, n))
    idxS = n
    idx1 = n + 1

    # Basic ratio
    rk = -(k + 1) / (2 * (2*k + 1))

    # Build matrix entries
    M = sp.zeros(d, d)

    # U0 row: U0_{k+1} = r_k * U0_k
    M[idxU[0], idxU[0]] = rk

    # Uj rows (j >= 1): Uj_{k+1} = r_k * Uj_k + (r_k / k²) * U_{j-1}_k
    for j in range(1, n):
        M[idxU[j], idxU[j]] = rk
        M[idxU[j], idxU[j-1]] = rk / (k**2)

    # Constant row: 1(k+1) = 1(k)
    M[idx1, idx1] = 1

    # Zero-padding rows: already zero

    # Accumulator row: S(k+1) = S(k) + factor * Σ_j c_{n,j}(k+1) * [row for Uj]
    # where factor = (1/2) / (k+1)³
    # and c_{n,j}(k+1) are the kernel coefficients at k+1

    kp1 = k + 1

    # Kernel coefficients c_{n,j} at k+1
    c_syms = [sp.Rational(0)] * n
    c_syms[n-1] = sp.Rational(5) * (sp.Rational(-1)**(n-1))
    for j in range(0, n-1):
        t = (n-1) - j
        c_syms[j] += sp.Rational(4) * (sp.Rational(-1)**j) / (kp1**(2*t))

    # Build accumulator row by composing with U rows
    # S(k+1) = S(k) + factor * Σ_j c_{n,j} * [Uj row applied to state]
    # The Uj row gives: Uj(k+1) = M[j, :] · state
    # So S(k+1) = S(k) + factor * Σ_j c_{n,j} * (Σ_col M[j, col] * state[col])

    factor = sp.Rational(1, 2) / (kp1**3)

    # Start with S(k+1) = S(k)
    M[idxS, idxS] = 1

    # Add the contribution from each Uj
    for j in range(n):
        coeff = factor * c_syms[j]
        for col in range(d):
            if M[idxU[j], col] != 0:
                M[idxS, col] += coeff * M[idxU[j], col]

    # Convert to dictionary format for our compiler
    matrix_dict = {}
    for i in range(d):
        for j in range(d):
            entry = sp.simplify(M[i, j])
            if entry != 0:
                matrix_dict[(i, j)] = str(entry)

    return matrix_dict, d, ['k'], 2*n + 1


def generate_shifts(n_shifts: int = 512):
    """Generate shift values for the walk variable k.

    k starts at shift_val, walks k = shift_val, shift_val+1, shift_val+2, ...
    Different shifts explore different entry points into the recurrence.
    """
    shifts = []
    for s in range(1, n_shifts + 1):
        shifts.append([s])
    return shifts


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMF specs for ζ(2n+1) odd zeta values"
    )
    parser.add_argument("--n-min", type=int, default=2,
                        help="Minimum n (ζ(2n+1), so n=2 → ζ(5))")
    parser.add_argument("--n-max", type=int, default=10,
                        help="Maximum n (n=10 → ζ(21), 20×20 matrix)")
    parser.add_argument("--output", type=str, default="odd_zeta_specs.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--shifts", type=int, default=512,
                        help="Number of shifts to generate")
    parser.add_argument("--shifts-output", type=str, default="odd_zeta_shifts.json",
                        help="Output shifts JSON file")
    parser.add_argument("--verify", action="store_true",
                        help="Quick verification of each CMF (sympy, slow)")
    args = parser.parse_args()

    print(f"Generating CMF specs for ζ(2n+1), n={args.n_min}..{args.n_max}")
    print(f"Output: {args.output}")
    print(f"{'='*70}")

    specs = []

    for n in range(args.n_min, args.n_max + 1):
        zeta_val = 2*n + 1
        d = 2*n

        print(f"  ζ({zeta_val}): n={n}, {d}×{d} matrix ... ", end="", flush=True)

        try:
            matrix_dict, dim_mat, axis_names, target = build_cmf_matrix_2n(n)
            n_nonzero = len(matrix_dict)

            spec = {
                "name": f"zeta_{zeta_val}",
                "target": f"zeta({zeta_val})",
                "n": n,
                "zeta_val": zeta_val,
                "rank": d,
                "dim": 1,
                "matrix": {f"{r},{c}": expr for (r, c), expr in matrix_dict.items()},
                "axis_names": axis_names,
                "directions": [1],
                "accumulator_idx": n,
                "constant_idx": n + 1,
            }
            specs.append(spec)

            print(f"{n_nonzero} nonzero entries ✓")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Write specs
    with open(args.output, 'w') as f:
        for spec in specs:
            f.write(json.dumps(spec) + "\n")

    print(f"\nWrote {len(specs)} CMF specs to {args.output}")

    # Write shifts
    shifts = generate_shifts(args.shifts)
    with open(args.shifts_output, 'w') as f:
        json.dump(shifts, f)
    print(f"Wrote {len(shifts)} shifts to {args.shifts_output}")

    # Summary table
    print(f"\n{'n':>4} {'ζ':>8} {'dim':>6} {'nonzero':>8}")
    print("-" * 30)
    for spec in specs:
        n = spec['n']
        d = spec['rank']
        nz = len(spec['matrix'])
        print(f"{n:>4} ζ({spec['zeta_val']}) {d:>3}×{d:<3} {nz:>6}")

    # Practical guidance
    print(f"\nPractical limits for GPU sweep:")
    print(f"  4×4  (ζ(5)):   fast, ~1000 shifts trivial")
    print(f"  10×10 (ζ(11)): moderate, 512 shifts OK")
    print(f"  20×20 (ζ(21)): ~25x slower per step than 4×4")
    print(f"  40×40 (ζ(41)): heavy, reduce shifts or depth")


if __name__ == "__main__":
    main()
