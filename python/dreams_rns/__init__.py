"""
Dreams-RNS-CUDA: GPU-accelerated Ramanujan Dreams pipeline using RNS arithmetic.
"""

from .compiler import compile_cmf, CmfCompiler, Opcode
from .runner import DreamsRunner, WalkConfig
from .analysis import analyze_hits, verify_hit

__version__ = "0.1.0"
__all__ = [
    "compile_cmf",
    "CmfCompiler", 
    "Opcode",
    "DreamsRunner",
    "WalkConfig",
    "analyze_hits",
    "verify_hit",
]
