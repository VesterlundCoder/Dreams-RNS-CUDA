"""
Dreams-RNS-CUDA: GPU-accelerated Ramanujan Dreams pipeline using RNS arithmetic.

Correct PCF walk convention (matching ramanujantools):
  M(n) = [[0, b(n)], [1, a(n)]]          companion form
  P(N) = A · M(1) · M(2) · ... · M(N)    A = [[1, a(0)], [0, 1]]
  p = P[0, m-1],  q = P[1, m-1]          last column extraction
  delta = -(1 + log|p/q - L| / log|q|)
"""

from .compiler import compile_cmf, compile_cmf_from_dict, CmfCompiler, CmfProgram, Opcode
from .constants import load_constants, match_against_constants, compute_delta_against_constant
from .runner import (
    WalkConfig,
    generate_rns_primes,
    crt_reconstruct,
    centered,
    compute_dreams_delta_float,
    compute_dreams_delta_exact,
    compile_pcf,
    pcf_initial_values,
    run_pcf_walk,
    verify_pcf,
)

__version__ = "0.2.0"
__all__ = [
    "compile_cmf",
    "compile_cmf_from_dict",
    "CmfCompiler",
    "CmfProgram",
    "Opcode",
    "WalkConfig",
    "generate_rns_primes",
    "crt_reconstruct",
    "centered",
    "compute_dreams_delta_float",
    "compute_dreams_delta_exact",
    "compile_pcf",
    "pcf_initial_values",
    "run_pcf_walk",
    "verify_pcf",
]
