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


def build_cmf_matrix_2n(n: int, multiaxis: bool = True):
    """Build the 2n×2n CMF matrix for ζ(2n+1) as a sympy Matrix.

    From the HPHP08 construction:
      - r_k = -(k+1) / (2*(2k+1))
      - U0(k+1) = r_k * U0(k)
      - Uj(k+1) = r_k * Uj(k) + (r_k / k²) * U_{j-1}(k)
      - 1(k+1) = 1(k)
      - S(k+1) = S(k) + (1/2)/(k+1)³ * Σ_j c_{n,j}(k+1) * Uj(k+1)
      - Zero rows for padding

    If multiaxis=True, decomposes into per-row independent axes.
    Each U-row j gets its own numerator and denominator axes, plus a
    coupling axis if j >= 1.  The accumulator gets its own axis.

    Axis layout per U-row j:
      x_{3j}   = (k+1) numerator of r_k in row j   (direction=1, shift=2)
      x_{3j+1} = (2k+1) denominator of r_k in row j (direction=2, shift=3)
      x_{3j+2} = k coupling in row j (j>=1 only)    (direction=1, shift=1)
    Final axis:
      x_{3n-n+1..} = (k+1) in accumulator/kernel     (direction=1, shift=2)

    Total dimensions = 2*n (num+den per row) + (n-1) (coupling) + 1 (acc) = 3n
    Standard directions reproduce original k=1,2,3,...

    Returns: (matrix_dict, dim, axis_names, directions, default_shifts, target_zeta)
    """
    d = 2 * n

    # Index layout: [U0 U1 ... U_{n-1} S 1 Z0 Z1 ... Z_{n-3}]
    idxU = list(range(0, n))
    idxS = n
    idx1 = n + 1

    if multiaxis:
        # Build per-row axis symbols
        # Layout: [num_0, den_0, num_1, den_1, sq_1, num_2, den_2, sq_2, ..., acc]
        axis_names = []
        directions = []
        default_shifts = []

        # Per-row symbols and their axis indices
        row_num_sym = []   # x for (k+1) numerator per row
        row_den_sym = []   # x for (2k+1) denominator per row
        row_sq_sym = []    # x for k coupling per row (None for j=0)
        axis_idx = 0

        for j in range(n):
            # Numerator axis: replaces (k+1) in r_k for row j
            name_n = f'x{axis_idx}'
            sym_n = sp.Symbol(name_n)
            row_num_sym.append(sym_n)
            axis_names.append(name_n)
            directions.append(1)       # (k+1) advances by 1
            default_shifts.append(2)   # k+1 = 2 at k=1
            axis_idx += 1

            # Denominator axis: replaces (2k+1) in r_k for row j
            name_d = f'x{axis_idx}'
            sym_d = sp.Symbol(name_d)
            row_den_sym.append(sym_d)
            axis_names.append(name_d)
            directions.append(2)       # (2k+1) advances by 2
            default_shifts.append(3)   # 2k+1 = 3 at k=1
            axis_idx += 1

            # Coupling axis: replaces k in k² for rows j >= 1
            if j >= 1:
                name_s = f'x{axis_idx}'
                sym_s = sp.Symbol(name_s)
                row_sq_sym.append(sym_s)
                axis_names.append(name_s)
                directions.append(1)       # k advances by 1
                default_shifts.append(1)   # k = 1 at k=1
                axis_idx += 1
            else:
                row_sq_sym.append(None)

        # Accumulator axis: replaces (k+1) in factor and kernel coefficients
        acc_name = f'x{axis_idx}'
        kp1_acc = sp.Symbol(acc_name)
        axis_names.append(acc_name)
        directions.append(1)       # (k+1) advances by 1
        default_shifts.append(2)   # k+1 = 2 at k=1
        axis_idx += 1

        dim = axis_idx

        # Build per-row r_k ratios
        rk_per_row = []
        for j in range(n):
            rk_per_row.append(-row_num_sym[j] / (2 * row_den_sym[j]))

    else:
        k = sp.Symbol('k')
        kp1_acc = k + 1

        axis_names = ['k']
        directions = [1]
        default_shifts = [1]
        dim = 1

        rk_per_row = [-(k + 1) / (2 * (2*k + 1))] * n
        row_sq_sym = [k] * n

    # Build matrix entries
    M = sp.zeros(d, d)

    # U0 row: U0_{k+1} = r_k * U0_k
    M[idxU[0], idxU[0]] = rk_per_row[0]

    # Uj rows (j >= 1): Uj_{k+1} = r_k * Uj_k + (r_k / k²) * U_{j-1}_k
    for j in range(1, n):
        M[idxU[j], idxU[j]] = rk_per_row[j]
        sq = row_sq_sym[j] if multiaxis else row_sq_sym[j]
        M[idxU[j], idxU[j-1]] = rk_per_row[j] / (sq**2)

    # Constant row: 1(k+1) = 1(k)
    M[idx1, idx1] = 1

    # Kernel coefficients c_{n,j} at k+1 (using accumulator axis)
    c_syms = [sp.Rational(0)] * n
    c_syms[n-1] = sp.Rational(5) * (sp.Rational(-1)**(n-1))
    for j in range(0, n-1):
        t = (n-1) - j
        c_syms[j] += sp.Rational(4) * (sp.Rational(-1)**j) / (kp1_acc**(2*t))

    # Accumulator factor (using accumulator axis)
    factor = sp.Rational(1, 2) / (kp1_acc**3)

    # S(k+1) = S(k) + factor * Σ_j c_{n,j} * [Uj row applied to state]
    M[idxS, idxS] = 1

    for j in range(n):
        coeff = factor * c_syms[j]
        for col in range(d):
            if M[idxU[j], col] != 0:
                M[idxS, col] += coeff * M[idxU[j], col]

    # Convert to dictionary format for compiler
    matrix_dict = {}
    for i in range(d):
        for j in range(d):
            entry = sp.simplify(M[i, j])
            if entry != 0:
                matrix_dict[(i, j)] = str(entry)

    return matrix_dict, dim, axis_names, directions, default_shifts, 2*n + 1


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
            matrix_dict, dim_val, axis_names, dirs, def_shifts, target = build_cmf_matrix_2n(n)
            n_nonzero = len(matrix_dict)

            spec = {
                "name": f"zeta_{zeta_val}",
                "target": f"zeta({zeta_val})",
                "n": n,
                "zeta_val": zeta_val,
                "rank": d,
                "dim": dim_val,
                "matrix": {f"{r},{c}": expr for (r, c), expr in matrix_dict.items()},
                "axis_names": axis_names,
                "directions": dirs,
                "default_shifts": def_shifts,
                "accumulator_idx": n,
                "constant_idx": n + 1,
            }
            specs.append(spec)

            print(f"{n_nonzero} nonzero entries, {dim_val} axes ✓")

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
    print(f"\n{'n':>4} {'ζ':>8} {'matrix':>8} {'axes':>6} {'nonzero':>8}")
    print("-" * 40)
    for spec in specs:
        n = spec['n']
        d = spec['rank']
        nz = len(spec['matrix'])
        dim = spec['dim']
        print(f"{n:>4} ζ({spec['zeta_val']}) {d:>3}×{d:<3}  {dim:>4}    {nz:>6}")


if __name__ == "__main__":
    main()
