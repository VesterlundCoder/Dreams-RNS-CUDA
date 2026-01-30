"""
Dreams Runner: GPU execution wrapper for the RNS walk pipeline.

This module handles GPU memory management, kernel execution, and result collection.
Designed for use in Google Colab with A100 GPU.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import math

from .compiler import CmfProgram, Opcode


@dataclass
class WalkConfig:
    """Configuration for walk execution."""
    K: int = 64                     # Number of primes
    B: int = 1000                   # Batch size (shifts per CMF)
    depth: int = 2000               # Walk depth
    topk: int = 100                 # Top-K hits to keep
    target: float = math.pi         # Target constant
    snapshot_depths: Tuple[int, int] = (200, 2000)
    delta_threshold: float = 1e-6   # Hit threshold
    use_float_shadow: bool = True   # Use float64 shadow for quick delta


@dataclass
class Hit:
    """A hit result from the walk."""
    cmf_idx: int
    shift: List[int]
    depth: int
    delta: float
    log_q: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cmf_idx': self.cmf_idx,
            'shift': self.shift,
            'depth': self.depth,
            'delta': self.delta,
            'log_q': self.log_q
        }


def generate_rns_primes(K: int) -> np.ndarray:
    """Generate K 31-bit primes for RNS representation."""
    PRIME_MAX = (1 << 31) - 1
    PRIME_MIN = 1 << 30
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = []
    candidate = PRIME_MAX
    while len(primes) < K and candidate >= PRIME_MIN:
        if is_prime(candidate):
            primes.append(candidate)
        candidate -= 2
    
    return np.array(primes, dtype=np.uint32)


def compute_barrett_mu(primes: np.ndarray) -> np.ndarray:
    """Compute Barrett reduction constants."""
    return np.array([(1 << 63) // p * 2 for p in primes], dtype=np.uint64)


class DreamsRunner:
    """
    GPU runner for Dreams pipeline.
    
    Usage:
        runner = DreamsRunner(programs, target=math.pi)
        hits = runner.run(shifts_per_cmf=1000, depth=2000)
    """
    
    def __init__(self, programs: List[CmfProgram], target: float = math.pi,
                 config: Optional[WalkConfig] = None):
        """
        Initialize the runner.
        
        Args:
            programs: List of compiled CMF programs
            target: Target constant to search for
            config: Walk configuration
        """
        self.programs = programs
        self.target = target
        self.config = config or WalkConfig(target=target)
        
        # Will be set when GPU is initialized
        self._gpu_initialized = False
        self._primes = None
        self._mus = None
    
    def _init_gpu(self):
        """Initialize GPU resources."""
        if self._gpu_initialized:
            return
        
        try:
            import cupy as cp
            self._cp = cp
            self._gpu_available = True
        except ImportError:
            self._gpu_available = False
            print("CuPy not available, using CPU fallback")
        
        # Generate primes
        self._primes = generate_rns_primes(self.config.K)
        self._mus = compute_barrett_mu(self._primes)
        
        if self._gpu_available:
            self._d_primes = self._cp.asarray(self._primes)
            self._d_mus = self._cp.asarray(self._mus)
        
        self._gpu_initialized = True
    
    def generate_shifts(self, cmf_idx: int, n_shifts: int, 
                        dim: int, method: str = 'random',
                        bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Generate shift values for exploration.
        
        Args:
            cmf_idx: CMF index (for deterministic seeding)
            n_shifts: Number of shifts to generate
            dim: Number of dimensions
            method: 'random', 'grid', or 'sphere'
            bounds: (min, max) bounds for shifts
        
        Returns:
            Array of shape (n_shifts, dim)
        """
        bounds = bounds or (-1000, 1000)
        rng = np.random.default_rng(seed=42 + cmf_idx)
        
        if method == 'random':
            return rng.integers(bounds[0], bounds[1], size=(n_shifts, dim), dtype=np.int32)
        
        elif method == 'grid':
            # Uniform grid in each dimension
            side = int(np.ceil(n_shifts ** (1/dim)))
            grids = [np.linspace(bounds[0], bounds[1], side, dtype=np.int32) for _ in range(dim)]
            mesh = np.meshgrid(*grids, indexing='ij')
            shifts = np.stack([m.flatten() for m in mesh], axis=1)
            return shifts[:n_shifts]
        
        elif method == 'sphere':
            # Random points on hypersphere surface
            radius = (bounds[1] - bounds[0]) / 2
            center = (bounds[0] + bounds[1]) / 2
            points = rng.standard_normal((n_shifts, dim))
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            radii = rng.uniform(0, 1, (n_shifts, 1)) ** (1/dim)
            points = center + points * radii * radius
            return points.astype(np.int32)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_walk_cpu(self, program: CmfProgram, shifts: np.ndarray) -> List[Hit]:
        """CPU fallback for walk computation."""
        m = program.m
        dim = program.dim
        B = len(shifts)
        K = self.config.K
        
        # Initialize P to identity for each (prime, shift)
        P = np.zeros((K, B, m, m), dtype=np.uint64)
        for k in range(K):
            for b in range(B):
                for i in range(m):
                    P[k, b, i, i] = 1
        
        # Also track float shadow
        P_float = np.zeros((B, m, m), dtype=np.float64)
        for b in range(B):
            for i in range(m):
                P_float[b, i, i] = 1.0
        log_scale = np.zeros(B, dtype=np.float64)
        
        hits = []
        
        # Walk loop
        for step in range(self.config.depth):
            # Evaluate step matrix for each shift
            for b in range(B):
                M = np.zeros((m, m), dtype=np.float64)
                M_rns = np.zeros((K, m, m), dtype=np.uint64)
                
                # Simple evaluation (placeholder - real implementation uses bytecode)
                # For now, just use identity + small perturbation based on shift
                for i in range(m):
                    M[i, i] = 1.0
                    for k in range(K):
                        M_rns[k, i, i] = 1
                
                # Update P = P @ M
                for k in range(K):
                    p = int(self._primes[k])
                    new_P = np.zeros((m, m), dtype=np.uint64)
                    for i in range(m):
                        for j in range(m):
                            acc = 0
                            for l in range(m):
                                acc += int(P[k, b, i, l]) * int(M_rns[k, l, j])
                            new_P[i, j] = acc % p
                    P[k, b] = new_P
                
                # Update float shadow
                new_P_float = P_float[b] @ M
                max_val = np.max(np.abs(new_P_float))
                if max_val > 1e10:
                    new_P_float /= max_val
                    log_scale[b] += np.log(max_val)
                P_float[b] = new_P_float
            
            # Check for hits at snapshot depth
            if step + 1 in self.config.snapshot_depths:
                for b in range(B):
                    # Extract p, q from trajectory
                    p_val = P_float[b, 0, -1]
                    q_val = P_float[b, 1, -1]
                    
                    if abs(q_val) > 1e-10:
                        ratio = p_val / q_val
                        delta = abs(ratio - self.target)
                        
                        if delta < self.config.delta_threshold:
                            hits.append(Hit(
                                cmf_idx=0,
                                shift=list(shifts[b]),
                                depth=step + 1,
                                delta=delta,
                                log_q=log_scale[b] + np.log(abs(q_val))
                            ))
        
        return hits
    
    def _run_walk_gpu(self, program: CmfProgram, shifts: np.ndarray, 
                      cmf_idx: int) -> List[Hit]:
        """GPU implementation using CuPy."""
        cp = self._cp
        m = program.m
        dim = program.dim
        B = len(shifts)
        K = self.config.K
        
        # Upload shifts to GPU
        d_shifts = cp.asarray(shifts)
        
        # Initialize P to identity
        P_rns = cp.zeros((K, B, m, m), dtype=cp.uint32)
        for i in range(m):
            P_rns[:, :, i, i] = 1
        
        # Float shadow
        P_float = cp.zeros((B, m, m), dtype=cp.float64)
        for i in range(m):
            P_float[:, i, i] = 1.0
        log_scale = cp.zeros(B, dtype=cp.float64)
        
        # Convert constants to RNS
        const_rns = cp.zeros((K, len(program.constants)), dtype=cp.uint32)
        for k in range(K):
            p = int(self._primes[k])
            for c_idx, c_val in enumerate(program.constants):
                const_rns[k, c_idx] = c_val % p if c_val >= 0 else (p - (-c_val % p)) % p
        
        hits = []
        
        # Main walk loop (simplified - real impl uses CUDA kernels)
        for step in range(self.config.depth):
            # Evaluate step matrices (simplified placeholder)
            M_rns = cp.zeros((K, B, m, m), dtype=cp.uint32)
            M_float = cp.zeros((B, m, m), dtype=cp.float64)
            
            # Identity + step-dependent values (placeholder)
            for i in range(m):
                M_rns[:, :, i, i] = 1
                M_float[:, i, i] = 1.0
            
            # Add axis-dependent terms (simplified)
            for axis in range(dim):
                axis_vals = d_shifts[:, axis] + step * program.directions[axis]
                # Real implementation would evaluate full bytecode here
            
            # Batched modular matmul P = P @ M
            for k in range(K):
                p = int(self._primes[k])
                mu = int(self._mus[k])
                
                # Simple matmul (real impl uses optimized CUDA kernel)
                new_P = cp.zeros((B, m, m), dtype=cp.uint64)
                for i in range(m):
                    for j in range(m):
                        for l in range(m):
                            new_P[:, i, j] += P_rns[k, :, i, l].astype(cp.uint64) * M_rns[k, :, l, j].astype(cp.uint64)
                P_rns[k] = (new_P % p).astype(cp.uint32)
            
            # Float shadow update with rescaling
            for b_start in range(0, B, 256):
                b_end = min(b_start + 256, B)
                batch = P_float[b_start:b_end] @ M_float[b_start:b_end]
                max_vals = cp.max(cp.abs(batch), axis=(1, 2))
                needs_rescale = max_vals > 1e10
                if cp.any(needs_rescale):
                    scale_factors = cp.where(needs_rescale, max_vals, 1.0)
                    batch = batch / scale_factors[:, None, None]
                    log_scale[b_start:b_end] += cp.log(scale_factors)
                P_float[b_start:b_end] = batch
            
            # Check for hits at snapshot depths
            if step + 1 in self.config.snapshot_depths:
                p_vals = P_float[:, 0, -1]
                q_vals = P_float[:, 1, -1]
                
                valid = cp.abs(q_vals) > 1e-10
                ratios = cp.where(valid, p_vals / q_vals, 0.0)
                deltas = cp.abs(ratios - self.target)
                
                hit_mask = (deltas < self.config.delta_threshold) & valid
                hit_indices = cp.where(hit_mask)[0]
                
                for idx in hit_indices.get():
                    hits.append(Hit(
                        cmf_idx=cmf_idx,
                        shift=list(shifts[idx]),
                        depth=step + 1,
                        delta=float(deltas[idx].get()),
                        log_q=float(log_scale[idx].get() + cp.log(cp.abs(q_vals[idx])).get())
                    ))
        
        return hits
    
    def run(self, shifts_per_cmf: Optional[int] = None,
            depth: Optional[int] = None,
            shift_method: str = 'random',
            shift_bounds: Optional[Tuple[int, int]] = None) -> List[Hit]:
        """
        Run the Dreams pipeline on all CMF programs.
        
        Args:
            shifts_per_cmf: Number of shifts to try per CMF
            depth: Walk depth (overrides config)
            shift_method: Method for generating shifts
            shift_bounds: Bounds for shift generation
        
        Returns:
            List of Hit objects
        """
        self._init_gpu()
        
        if shifts_per_cmf is not None:
            self.config.B = shifts_per_cmf
        if depth is not None:
            self.config.depth = depth
        
        all_hits = []
        
        for cmf_idx, program in enumerate(self.programs):
            print(f"Processing CMF {cmf_idx + 1}/{len(self.programs)}...")
            
            # Generate shifts for this CMF
            shifts = self.generate_shifts(
                cmf_idx, 
                self.config.B, 
                program.dim,
                method=shift_method,
                bounds=shift_bounds
            )
            
            # Run walk
            if self._gpu_available:
                hits = self._run_walk_gpu(program, shifts, cmf_idx)
            else:
                hits = self._run_walk_cpu(program, shifts)
            
            all_hits.extend(hits)
            print(f"  Found {len(hits)} hits")
        
        # Sort by delta and keep top-k
        all_hits.sort(key=lambda h: h.delta)
        return all_hits[:self.config.topk]
    
    def run_single(self, program: CmfProgram, shifts: np.ndarray,
                   cmf_idx: int = 0) -> List[Hit]:
        """Run walk on a single CMF with specified shifts."""
        self._init_gpu()
        
        if self._gpu_available:
            return self._run_walk_gpu(program, shifts, cmf_idx)
        else:
            return self._run_walk_cpu(program, shifts)
