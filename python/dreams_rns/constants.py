"""
Mathematical constants bank for PCF/CMF limit matching.

Each constant has:
  - name: short identifier
  - value: high-precision mpmath value (computed at runtime)
  - sympy_expr: symbolic expression for exact representation
  - description: human-readable name

Covers: zeta values, pi variants, Catalan's G, Euler-Mascheroni,
ln(2), Apery-related, golden ratio, sqrt(2), e.
"""

from typing import List, Dict, Any

CONSTANTS_REGISTRY = [
    # Zeta values
    {"name": "zeta3",   "sympy_expr": "zeta(3)",   "description": "Apéry's constant ζ(3)"},
    {"name": "zeta5",   "sympy_expr": "zeta(5)",   "description": "ζ(5)"},
    {"name": "zeta7",   "sympy_expr": "zeta(7)",   "description": "ζ(7)"},
    {"name": "zeta9",   "sympy_expr": "zeta(9)",   "description": "ζ(9)"},

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
