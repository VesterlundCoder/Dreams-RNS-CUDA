"""
Example CMF definitions for testing the Dreams-RNS-CUDA pipeline.

These are simplified CMF examples. For real research, use CMFs from
ramanujantools or your own database.
"""

import math
from typing import List, Dict, Tuple


def get_example_cmfs() -> List[Dict]:
    """
    Get a list of example CMF definitions.
    
    Each CMF is defined as a dictionary with:
    - 'name': Human-readable name
    - 'matrix': Dictionary of (row, col) -> expression string
    - 'm': Matrix dimension
    - 'dim': Number of axes
    - 'axis_names': Names of axis symbols
    - 'directions': Walk directions for each axis
    - 'target': Target constant (optional)
    """
    
    cmfs = []
    
    # Example 1: Simple 2x2 PCF-like CMF for pi
    # Based on the classic: pi = 4 * sum((-1)^n / (2n+1))
    cmfs.append({
        'name': 'Simple_Pi_2x2',
        'matrix': {
            (0, 0): 'x',
            (0, 1): '1',
            (1, 0): 'x**2',
            (1, 1): '2*x + 1',
        },
        'm': 2,
        'dim': 1,
        'axis_names': ['x'],
        'directions': [1],
        'target': math.pi
    })
    
    # Example 2: Apéry-like CMF for zeta(3)
    cmfs.append({
        'name': 'Apery_Zeta3_2x2',
        'matrix': {
            (0, 0): 'n**3',
            (0, 1): '1',
            (1, 0): '-(n+1)**6',
            (1, 1): '34*n**3 + 51*n**2 + 27*n + 5',
        },
        'm': 2,
        'dim': 1,
        'axis_names': ['n'],
        'directions': [1],
        'target': 1.2020569031595942  # zeta(3)
    })
    
    # Example 3: 2D CMF (multi-axis)
    cmfs.append({
        'name': 'TwoAxis_4x4',
        'matrix': {
            (0, 0): 'x + y',
            (0, 1): '1',
            (0, 2): '0',
            (0, 3): '0',
            (1, 0): 'x*y',
            (1, 1): 'x - y',
            (1, 2): '1',
            (1, 3): '0',
            (2, 0): 'x**2',
            (2, 1): 'y**2',
            (2, 2): 'x + 1',
            (2, 3): '1',
            (3, 0): '1',
            (3, 1): 'x',
            (3, 2): 'y',
            (3, 3): 'x*y + 1',
        },
        'm': 4,
        'dim': 2,
        'axis_names': ['x', 'y'],
        'directions': [1, 1],
        'target': math.pi
    })
    
    # Example 4: Euler's number e
    cmfs.append({
        'name': 'Euler_E_2x2',
        'matrix': {
            (0, 0): 'n + 1',
            (0, 1): '1',
            (1, 0): 'n + 1',
            (1, 1): 'n + 2',
        },
        'm': 2,
        'dim': 1,
        'axis_names': ['n'],
        'directions': [1],
        'target': math.e
    })
    
    # Example 5: Golden ratio phi
    cmfs.append({
        'name': 'Golden_Phi_2x2',
        'matrix': {
            (0, 0): '1',
            (0, 1): '1',
            (1, 0): '1',
            (1, 1): '0',
        },
        'm': 2,
        'dim': 1,
        'axis_names': ['n'],
        'directions': [1],
        'target': (1 + math.sqrt(5)) / 2  # phi
    })
    
    return cmfs


def compile_example_cmfs():
    """Compile example CMFs to bytecode programs."""
    from dreams_rns import compile_cmf_from_dict
    
    cmfs = get_example_cmfs()
    programs = []
    
    for cmf_def in cmfs:
        print(f"Compiling: {cmf_def['name']}")
        program = compile_cmf_from_dict(
            matrix_dict=cmf_def['matrix'],
            m=cmf_def['m'],
            dim=cmf_def['dim'],
            axis_names=cmf_def['axis_names'],
            directions=cmf_def['directions']
        )
        programs.append(program)
        print(f"  Opcodes: {len(program.opcodes)}")
        print(f"  Constants: {len(program.constants)}")
    
    return programs, cmfs


def demo_single_walk():
    """Demo: run a single walk on the simple pi CMF."""
    import numpy as np
    from dreams_rns import DreamsRunner, WalkConfig
    from dreams_rns.compiler import compile_cmf_from_dict
    
    # Compile the simple pi CMF
    cmf_def = get_example_cmfs()[0]
    program = compile_cmf_from_dict(
        matrix_dict=cmf_def['matrix'],
        m=cmf_def['m'],
        dim=cmf_def['dim'],
        axis_names=cmf_def['axis_names'],
        directions=cmf_def['directions']
    )
    
    print(f"Running walk on: {cmf_def['name']}")
    print(f"Target: {cmf_def['target']}")
    
    # Configure and run
    config = WalkConfig(
        K=32,           # 32 primes
        B=100,          # 100 shifts
        depth=500,      # 500 steps
        topk=10,        # Keep top 10
        target=cmf_def['target']
    )
    
    runner = DreamsRunner([program], target=cmf_def['target'], config=config)
    
    # Generate some shifts
    shifts = np.random.randint(-100, 100, size=(100, 1), dtype=np.int32)
    
    # Run
    hits = runner.run_single(program, shifts, cmf_idx=0)
    
    print(f"\nFound {len(hits)} hits:")
    for hit in hits[:5]:
        print(f"  shift={hit.shift}, depth={hit.depth}, delta={hit.delta:.2e}")
    
    return hits


if __name__ == '__main__':
    print("Dreams-RNS-CUDA Example CMFs")
    print("=" * 40)
    
    # List available CMFs
    cmfs = get_example_cmfs()
    print(f"\nAvailable CMFs: {len(cmfs)}")
    for i, cmf in enumerate(cmfs):
        print(f"  {i}: {cmf['name']} ({cmf['m']}x{cmf['m']}, {cmf['dim']}D)")
    
    print("\nTo compile and run, use:")
    print("  from examples.example_cmfs import compile_example_cmfs, demo_single_walk")
    print("  programs, cmfs = compile_example_cmfs()")
    print("  demo_single_walk()")
