"""
Mathematical constants bank for PCF/CMF limit matching.

Each constant has:
  - name: short identifier
  - value: high-precision mpmath value (computed at runtime)
  - sympy_expr: symbolic expression for exact representation
  - description: human-readable name

Covers: zeta values (ζ(2)..ζ(30), even and odd), pi variants,
Catalan's G, Euler-Mascheroni, ln(2), golden ratio, sqrt(2), e.

IMPORTANT: For matching, use compute_match_digits() which applies
ratio matching for estimates near 1.0 to avoid false positives.
Absolute matching near 1.0 will falsely flag high ζ(n) values.
"""

import math
from typing import List, Dict, Any, Tuple

CONSTANTS_REGISTRY = [
    # Zeta values: ζ(2)..ζ(30), even and odd
    {"name": "zeta2",   "sympy_expr": "zeta(2)",   "description": "ζ(2) = π²/6"},
    {"name": "zeta3",   "sympy_expr": "zeta(3)",   "description": "Apéry's constant ζ(3)"},
    {"name": "zeta4",   "sympy_expr": "zeta(4)",   "description": "ζ(4) = π⁴/90"},
    {"name": "zeta5",   "sympy_expr": "zeta(5)",   "description": "ζ(5)"},
    {"name": "zeta6",   "sympy_expr": "zeta(6)",   "description": "ζ(6) = π⁶/945"},
    {"name": "zeta7",   "sympy_expr": "zeta(7)",   "description": "ζ(7)"},
    {"name": "zeta8",   "sympy_expr": "zeta(8)",   "description": "ζ(8)"},
    {"name": "zeta9",   "sympy_expr": "zeta(9)",   "description": "ζ(9)"},
    {"name": "zeta10",  "sympy_expr": "zeta(10)",  "description": "ζ(10)"},
    {"name": "zeta11",  "sympy_expr": "zeta(11)",  "description": "ζ(11)"},
    {"name": "zeta12",  "sympy_expr": "zeta(12)",  "description": "ζ(12)"},
    {"name": "zeta13",  "sympy_expr": "zeta(13)",  "description": "ζ(13)"},
    {"name": "zeta14",  "sympy_expr": "zeta(14)",  "description": "ζ(14)"},
    {"name": "zeta15",  "sympy_expr": "zeta(15)",  "description": "ζ(15)"},
    {"name": "zeta16",  "sympy_expr": "zeta(16)",  "description": "ζ(16)"},
    {"name": "zeta17",  "sympy_expr": "zeta(17)",  "description": "ζ(17)"},
    {"name": "zeta18",  "sympy_expr": "zeta(18)",  "description": "ζ(18)"},
    {"name": "zeta19",  "sympy_expr": "zeta(19)",  "description": "ζ(19)"},
    {"name": "zeta20",  "sympy_expr": "zeta(20)",  "description": "ζ(20)"},
    {"name": "zeta21",  "sympy_expr": "zeta(21)",  "description": "ζ(21)"},
    {"name": "zeta22",  "sympy_expr": "zeta(22)",  "description": "ζ(22)"},
    {"name": "zeta23",  "sympy_expr": "zeta(23)",  "description": "ζ(23)"},
    {"name": "zeta24",  "sympy_expr": "zeta(24)",  "description": "ζ(24)"},
    {"name": "zeta25",  "sympy_expr": "zeta(25)",  "description": "ζ(25)"},
    {"name": "zeta26",  "sympy_expr": "zeta(26)",  "description": "ζ(26)"},
    {"name": "zeta27",  "sympy_expr": "zeta(27)",  "description": "ζ(27)"},
    {"name": "zeta28",  "sympy_expr": "zeta(28)",  "description": "ζ(28)"},
    {"name": "zeta29",  "sympy_expr": "zeta(29)",  "description": "ζ(29)"},
    {"name": "zeta30",  "sympy_expr": "zeta(30)",  "description": "ζ(30)"},

    # Pi variants
    {"name": "pi",      "sympy_expr": "pi",        "description": "π"},
    {"name": "pi2",     "sympy_expr": "pi**2",     "description": "π²"},
    {"name": "pi_sq6",  "sympy_expr": "pi**2/6",   "description": "π²/6 = ζ(2)"},
    {"name": "1/pi",    "sympy_expr": "1/pi",      "description": "1/π"},
    {"name": "4/pi",    "sympy_expr": "4/pi",      "description": "4/π"},

    # Classical constants
    {"name": "catalan", "sympy_expr": "catalan",   "description": "Catalan's constant G"},
    {"name": "euler_gamma", "sympy_expr": "EulerGamma", "description": "Euler-Mascheroni γ"},
    {"name": "ln2",     "sympy_expr": "log(2)",    "description": "ln(2)"},
    {"name": "e",       "sympy_expr": "E",         "description": "Euler's number e"},
    {"name": "sqrt2",   "sympy_expr": "sqrt(2)",   "description": "√2"},
    {"name": "phi",     "sympy_expr": "(1+sqrt(5))/2", "description": "Golden ratio φ"},
]


def load_constants(dps: int = 200) -> List[Dict[str, Any]]:
    """Evaluate all constants to high precision.

    Returns list of dicts with 'name', 'value' (mpmath mpf), 'value_float',
    'sympy_expr', 'description'.
    """
    import sys
    # Python 3.11+ limits int→str conversion; high-dps mpmath needs large ints
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(max(sys.get_int_max_str_digits(), dps * 4))

    import sympy as sp
    import mpmath as mp
    mp.mp.dps = dps

    results = []
    ns = {
        "pi": sp.pi, "E": sp.E, "zeta": sp.zeta,
        "catalan": sp.Catalan, "EulerGamma": sp.EulerGamma,
        "log": sp.log, "sqrt": sp.sqrt,
    }

    for entry in CONSTANTS_REGISTRY:
        try:
            expr = sp.sympify(entry["sympy_expr"], locals=ns)
            val_mp = mp.mpf(str(sp.N(expr, dps + 10)))
            results.append({
                "name": entry["name"],
                "value": val_mp,
                "value_float": float(val_mp),
                "sympy_expr": entry["sympy_expr"],
                "description": entry["description"],
            })
        except Exception as e:
            print(f"WARNING: Could not evaluate constant '{entry['name']}': {e}")

    return results


def compute_match_digits(
    estimate: float,
    constants: List[Dict[str, Any]],
) -> Tuple[str, float]:
    """Match estimate against constants using ratio-aware matching.

    For estimates near 1.0 (within 0.5), uses ratio matching:
      digits = -log10(|est-1 - (c-1)| / |c-1|)
    This prevents false positives where est≈1.0 matches high ζ(n)
    by absolute proximity (since ζ(n)→1 for large n).

    For estimates far from 1.0, uses standard relative error:
      digits = -log10(|est - c| / |c|)

    Returns:
        (best_name, best_digits) — name and digit count of best match.
    """
    near_one = abs(estimate - 1.0) < 0.5
    best_name = "none"
    best_digits = -1.0

    for c in constants:
        cv = c['value_float']
        if near_one and abs(cv - 1.0) < 0.5:
            # Ratio matching: compare (est-1) vs (cv-1)
            tail_est = estimate - 1.0
            tail_cv = cv - 1.0
            if abs(tail_cv) < 1e-300:
                continue
            err = abs(tail_est - tail_cv)
            if err == 0:
                digits = 16.0
            else:
                digits = -math.log10(err / max(abs(tail_cv), 1e-300))
        else:
            # Standard relative error
            err = abs(estimate - cv)
            if err == 0:
                digits = 16.0
            elif abs(cv) > 1e-300:
                digits = -math.log10(err / max(abs(cv), 1e-300))
            else:
                digits = 0.0
        if digits > best_digits:
            best_digits = digits
            best_name = c['name']

    return best_name, max(best_digits, 0.0)


def match_against_constants(
    estimate: float,
    constants: List[Dict[str, Any]],
    proximity_threshold: float = 1e-3,
) -> List[Dict[str, Any]]:
    """Find constants close to the given estimate.

    Returns list of matches with 'name', 'distance', 'value_float'.
    """
    matches = []
    for c in constants:
        dist = abs(estimate - c["value_float"])
        if dist < proximity_threshold:
            matches.append({
                "name": c["name"],
                "distance": dist,
                "value_float": c["value_float"],
                "description": c["description"],
            })
    matches.sort(key=lambda x: x["distance"])
    return matches


def compute_delta_against_constant(
    p_big: int,
    q_big: int,
    constant_mp,
    dps: int = 200,
) -> float:
    """Compute exact delta against a specific constant using mpmath."""
    import mpmath as mp
    mp.mp.dps = dps

    if q_big == 0:
        return float('-inf')

    ratio = mp.mpf(p_big) / mp.mpf(q_big)
    err = abs(ratio - constant_mp)

    if err == 0:
        return float('inf')

    log_err = float(mp.log(err))
    log_q = float(mp.log(abs(mp.mpf(q_big))))

    if log_q <= 0:
        return float('-inf')

    return -(1.0 + log_err / log_q)
