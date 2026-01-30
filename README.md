# Dreams-RNS-CUDA

A GPU-accelerated Ramanujan Dreams pipeline using RNS (Residue Number System) arithmetic for exact integer computation on NVIDIA GPUs. Designed for Google Colab A100 execution.

## Architecture Overview

This pipeline implements a **persistent kernel** approach where the entire CMF walk runs on GPU without CPU round-trips:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPU PERSISTENT KERNEL                               │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   COMPILE    │───▶│    EVAL      │───▶│    WALK      │───▶│   TOPK    │ │
│  │  CMF→Bytecode│    │ Axis Matrices│    │ P = P @ M_t  │    │  Delta    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌─────────┐  │
│   │ Opcodes  │        │ A_rns    │        │ P_rns    │        │ Hits    │  │
│   │ Constants│        │ [K,B,m,m]│        │ [K,B,m,m]│        │ TopK    │  │
│   └──────────┘        └──────────┘        └──────────┘        └─────────┘  │
│                                                                             │
│  Memory Layout: SoA over K primes, batched over B shifts                   │
│  All computation: modular arithmetic in RNS representation                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ (only small results)
                            ┌───────────────┐
                            │   CPU HOST    │
                            │  - Logging    │
                            │  - Final CRT  │
                            │  - Verify     │
                            └───────────────┘
```

## Pipeline Stages

### 1. CMF Compile (Offline, CPU)
- Input: Symbolic CMF expression from `ramanujantools`
- Output: GPU bytecode program (opcodes + constants)
- Converts symbolic matrix expressions to evaluatable DAG

### 2. Axis Matrix Evaluation (GPU)
- Evaluates each axis matrix A_j(x) for all shifts and primes in parallel
- Uses bytecode interpreter with modular arithmetic
- Output: `A_rns[K, B, m, m]` per axis

### 3. Step Matrix Composition (GPU)
- Computes M_t = Π_j A_j via batched modular matrix multiplication
- All operations in RNS representation

### 4. Trajectory Walk (GPU)
- Updates P = P @ M_t for T steps (typically 2000)
- Maintains full precision via RNS (K primes × 31 bits each)
- Optional: parallel float64 shadow run for quick delta estimation

### 5. Delta Proxy & TopK (GPU)
- At snapshot depths: extract p, q from trajectory
- Compute delta = |p/q - target| approximation
- Maintain TopK candidates per CMF
- Only TopK hits sent to CPU

## Files Structure

```
Dreams-RNS-CUDA/
├── README.md                    # This file
├── CMakeLists.txt              # Build system
├── include/
│   └── dreams/
│       ├── config.h            # Configuration and types
│       ├── cmf_program.h       # CMF bytecode program structure
│       ├── eval_kernel.h       # Expression evaluator declarations
│       ├── walk_kernel.h       # Walk kernel declarations
│       └── topk_kernel.h       # TopK/scoring declarations
├── src/
│   └── cuda/
│       ├── persistent_kernel.cu # Main fused GPU kernel
│       ├── eval_kernel.cu      # Expression evaluation
│       ├── walk_kernel.cu      # Matrix walk
│       └── topk_kernel.cu      # Scoring and TopK
├── python/
│   ├── dreams_rns/
│   │   ├── __init__.py
│   │   ├── compiler.py         # CMF → bytecode compiler
│   │   ├── runner.py           # GPU execution wrapper
│   │   └── analysis.py         # Results analysis
│   └── setup.py
├── notebooks/
│   └── dreams_colab.ipynb      # Google Colab notebook
└── examples/
    └── example_cmfs.py         # Example CMF definitions
```

## RNS Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| K (primes) | 32-64 | Number of 31-bit primes |
| B (batch) | 100-1000 | Shifts per CMF |
| m (matrix) | 4-8 | Matrix dimension |
| T (depth) | 2000 | Walk steps |

Bit capacity: K × 31 bits (e.g., 64 primes = 1984 bits)

## Bytecode Opcodes

| Opcode | Args | Description |
|--------|------|-------------|
| `LOAD_X` | axis_idx | Load axis value x_j |
| `LOAD_C` | const_idx | Load constant |
| `ADD` | r1, r2 | r1 + r2 mod p |
| `SUB` | r1, r2 | r1 - r2 mod p |
| `MUL` | r1, r2 | r1 × r2 mod p |
| `NEG` | r1 | -r1 mod p |
| `POW2` | r1 | r1² mod p |
| `POW3` | r1 | r1³ mod p |
| `INV` | r1 | r1⁻¹ mod p |
| `STORE` | r1, out_idx | Write to output matrix entry |

## Dependencies

- CUDA Toolkit 11.0+
- RNS-CUDA library (https://github.com/VesterlundCoder/RNS-CUDA)
- Python 3.8+
- ramanujantools (for CMF definitions)
- NumPy, CuPy (for Python bindings)

## Usage (Colab)

```python
# 1. Compile CMFs to bytecode (offline)
from dreams_rns import compile_cmf
programs = [compile_cmf(cmf) for cmf in cmfs]

# 2. Run GPU pipeline
from dreams_rns import DreamsRunner
runner = DreamsRunner(programs, target=math.pi)
hits = runner.run(
    shifts_per_cmf=1000,
    depth=2000,
    topk=100
)

# 3. Analyze results
from dreams_rns import analyze_hits
analyze_hits(hits)
```

## Performance Targets

| Metric | Target (A100) |
|--------|---------------|
| Shifts/sec | 100K+ |
| Walk steps/sec | 10M+ |
| Memory | < 20GB |
| Kernel launches | 1 (persistent) |

## Algorithm Details

### Garner's CRT Reconstruction
Used for converting RNS representation back to BigInt when needed (only for final verification):

```
x = r_0
for i = 1..K-1:
    x_i = ((r_i - x) * y_i) mod p_i
    x = x + x_i * (p_0 * p_1 * ... * p_{i-1})
```

### Delta Proxy Computation
Approximate delta without full CRT:
1. Use first K_small primes for rough BigInt (128-256 bits)
2. Convert to float64 for p/q ratio
3. Compute |p/q - target|

### TopK Selection
GPU-based parallel reduction:
1. Each block maintains local TopK
2. Final reduction across blocks
3. Only TopK hits transferred to CPU

## License

MIT License
