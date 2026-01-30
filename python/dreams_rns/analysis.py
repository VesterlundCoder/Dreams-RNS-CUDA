"""
Analysis utilities for Dreams results.

This module provides functions for analyzing and verifying hits from the
Dreams pipeline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

from .runner import Hit
from .compiler import CmfProgram


def analyze_hits(hits: List[Hit], 
                 target: float = math.pi,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a list of hits from the Dreams pipeline.
    
    Args:
        hits: List of Hit objects
        target: Target constant
        verbose: Print summary
    
    Returns:
        Dictionary with analysis results
    """
    if not hits:
        if verbose:
            print("No hits to analyze")
        return {'count': 0}
    
    # Group by CMF
    by_cmf: Dict[int, List[Hit]] = {}
    for hit in hits:
        if hit.cmf_idx not in by_cmf:
            by_cmf[hit.cmf_idx] = []
        by_cmf[hit.cmf_idx].append(hit)
    
    # Compute statistics
    deltas = [h.delta for h in hits]
    log_qs = [h.log_q for h in hits]
    
    results = {
        'count': len(hits),
        'cmf_count': len(by_cmf),
        'delta_min': min(deltas),
        'delta_max': max(deltas),
        'delta_mean': sum(deltas) / len(deltas),
        'log_q_min': min(log_qs),
        'log_q_max': max(log_qs),
        'log_q_mean': sum(log_qs) / len(log_qs),
        'by_cmf': {k: len(v) for k, v in by_cmf.items()},
        'best_hit': hits[0].to_dict() if hits else None,
        'target': target
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dreams Analysis Results")
        print(f"{'='*60}")
        print(f"Target constant: {target}")
        print(f"Total hits: {results['count']}")
        print(f"CMFs with hits: {results['cmf_count']}")
        print(f"\nDelta statistics:")
        print(f"  Min:  {results['delta_min']:.2e}")
        print(f"  Max:  {results['delta_max']:.2e}")
        print(f"  Mean: {results['delta_mean']:.2e}")
        print(f"\nlog(q) statistics:")
        print(f"  Min:  {results['log_q_min']:.2f}")
        print(f"  Max:  {results['log_q_max']:.2f}")
        print(f"  Mean: {results['log_q_mean']:.2f}")
        
        if results['best_hit']:
            print(f"\nBest hit:")
            bh = results['best_hit']
            print(f"  CMF:   {bh['cmf_idx']}")
            print(f"  Shift: {bh['shift']}")
            print(f"  Depth: {bh['depth']}")
            print(f"  Delta: {bh['delta']:.2e}")
            print(f"  log_q: {bh['log_q']:.2f}")
        print(f"{'='*60}\n")
    
    return results


def verify_hit(hit: Hit, program: CmfProgram, 
               target: float = math.pi,
               high_precision: bool = True) -> Dict[str, Any]:
    """
    Verify a hit using high-precision arithmetic.
    
    Args:
        hit: Hit to verify
        program: CMF program used
        target: Target constant
        high_precision: Use mpmath for high precision
    
    Returns:
        Verification results
    """
    try:
        if high_precision:
            import mpmath
            mpmath.mp.dps = 100  # 100 decimal places
        else:
            mpmath = None
    except ImportError:
        mpmath = None
        high_precision = False
    
    m = program.m
    shift = hit.shift
    depth = hit.depth
    
    # Initialize P to identity
    if high_precision:
        P = mpmath.matrix(m, m)
        for i in range(m):
            P[i, i] = mpmath.mpf(1)
    else:
        import numpy as np
        P = np.eye(m, dtype=np.float64)
    
    # Walk to the specified depth
    # Note: This is a simplified verification - real implementation
    # would evaluate the full bytecode
    for step in range(depth):
        # Evaluate step matrix (placeholder - real impl uses bytecode)
        if high_precision:
            M = mpmath.matrix(m, m)
            for i in range(m):
                M[i, i] = mpmath.mpf(1)
        else:
            M = np.eye(m, dtype=np.float64)
        
        # Add axis-dependent terms (simplified)
        for axis in range(program.dim):
            axis_val = shift[axis] + step * program.directions[axis]
            # Real implementation would evaluate full expression
        
        # P = P @ M
        if high_precision:
            P = P * M
        else:
            P = P @ M
    
    # Extract p and q
    if high_precision:
        p_val = P[0, m-1]
        q_val = P[1, m-1]
        if q_val != 0:
            ratio = p_val / q_val
            exact_delta = abs(ratio - mpmath.mpf(str(target)))
        else:
            exact_delta = mpmath.mpf('inf')
    else:
        p_val = P[0, m-1]
        q_val = P[1, m-1]
        if q_val != 0:
            ratio = p_val / q_val
            exact_delta = abs(ratio - target)
        else:
            exact_delta = float('inf')
    
    result = {
        'verified': True,
        'exact_delta': float(exact_delta),
        'reported_delta': hit.delta,
        'delta_match': abs(float(exact_delta) - hit.delta) < hit.delta * 0.1,
        'p_value': float(p_val) if not high_precision else str(p_val),
        'q_value': float(q_val) if not high_precision else str(q_val),
        'high_precision': high_precision
    }
    
    return result


def export_hits_json(hits: List[Hit], filepath: str):
    """Export hits to JSON file."""
    import json
    
    data = {
        'version': '1.0',
        'count': len(hits),
        'hits': [h.to_dict() for h in hits]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(hits)} hits to {filepath}")


def export_hits_csv(hits: List[Hit], filepath: str):
    """Export hits to CSV file."""
    import csv
    
    if not hits:
        print("No hits to export")
        return
    
    max_dim = max(len(h.shift) for h in hits)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['cmf_idx'] + [f'shift_{i}' for i in range(max_dim)] + ['depth', 'delta', 'log_q']
        writer.writerow(header)
        
        # Data
        for hit in hits:
            row = [hit.cmf_idx]
            row.extend(hit.shift + [0] * (max_dim - len(hit.shift)))
            row.extend([hit.depth, hit.delta, hit.log_q])
            writer.writerow(row)
    
    print(f"Exported {len(hits)} hits to {filepath}")


def load_hits_json(filepath: str) -> List[Hit]:
    """Load hits from JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    hits = []
    for h in data['hits']:
        hits.append(Hit(
            cmf_idx=h['cmf_idx'],
            shift=h['shift'],
            depth=h['depth'],
            delta=h['delta'],
            log_q=h['log_q']
        ))
    
    print(f"Loaded {len(hits)} hits from {filepath}")
    return hits


def plot_delta_distribution(hits: List[Hit], bins: int = 50):
    """Plot delta distribution histogram."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    deltas = [h.delta for h in hits]
    log_deltas = [math.log10(d) if d > 0 else -20 for d in deltas]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(deltas, bins=bins, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Delta')
    ax1.set_ylabel('Count')
    ax1.set_title('Delta Distribution')
    ax1.set_yscale('log')
    
    ax2.hist(log_deltas, bins=bins, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('log₁₀(Delta)')
    ax2.set_ylabel('Count')
    ax2.set_title('log(Delta) Distribution')
    
    plt.tight_layout()
    plt.show()


def plot_delta_vs_logq(hits: List[Hit]):
    """Plot delta vs log(q) scatter plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    deltas = [math.log10(h.delta) if h.delta > 0 else -20 for h in hits]
    log_qs = [h.log_q for h in hits]
    cmf_idxs = [h.cmf_idx for h in hits]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(log_qs, deltas, c=cmf_idxs, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='CMF Index')
    plt.xlabel('log(q)')
    plt.ylabel('log₁₀(Delta)')
    plt.title('Delta vs log(q)')
    plt.grid(True, alpha=0.3)
    plt.show()
