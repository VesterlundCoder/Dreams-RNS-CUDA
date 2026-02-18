#!/bin/bash
# Lambda Labs A100 setup for Dreams-RNS-CUDA full sweep
#
# Runs:
#   1. Euler2AI Pi PCFs (149 PCFs against 15 constants)
#   2. 3F2 CMF sweep (1000 CMFs × 512 shifts, multi-constant matching)
#
# Upload:
#   scp -i key.pem scripts/lambda_a100_setup.sh pcfs.json ubuntu@<IP>:~/
#   ssh -i key.pem ubuntu@<IP> 'bash lambda_a100_setup.sh'
#
# Results saved where: limit within 1e-3 of a constant OR delta > 0

set -euo pipefail
echo "============================================================"
echo "  Dreams-RNS-CUDA A100 Sweep"
echo "  K=32 (992-bit), depth=2000, 15 mathematical constants"
echo "============================================================"

# ── 1. Dependencies ──────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git
pip3 install --quiet numpy sympy mpmath scipy
pip3 install --quiet cupy-cuda12x

# ── 2. Clone / update repo ──────────────────────────────────────────
cd ~
if [ -d "Dreams-RNS-CUDA" ]; then
    cd Dreams-RNS-CUDA && git pull origin main && cd ~
else
    git clone https://github.com/VesterlundCoder/Dreams-RNS-CUDA.git
fi

PYDIR=~/Dreams-RNS-CUDA/python
SWEEPDIR=~/Dreams-RNS-CUDA/sweep_data
RESULTS=~/results
mkdir -p "${RESULTS}"

# ── 3. Smoke test (5 PCFs, shallow) ─────────────────────────────────
echo ""
echo "=== Smoke test (5 PCFs, depth=500, K=16) ==="
cd "${PYDIR}"
python3 a100_sweep.py \
    --mode euler2ai \
    --input ~/pcfs.json \
    --depth 500 --K 16 --max-tasks 5 \
    --output "${RESULTS}/smoke/"

# ── 4. Euler2AI Pi PCFs (full, 149 PCFs) ────────────────────────────
echo ""
echo "=== Euler2AI: 149 PCFs × 15 constants (depth=2000, K=32) ==="
python3 a100_sweep.py \
    --mode euler2ai \
    --input ~/pcfs.json \
    --depth 2000 --K 32 \
    --proximity 1e-3 \
    --output "${RESULTS}/euler2ai/"

# ── 5. 3F2 CMF sweep: first pass (10 traj × 512 shifts × 1000 CMFs) ─
echo ""
echo "=== 3F2 CMF sweep (first pass): 1000 CMFs × 10 traj × 512 shifts ==="
echo "    Full r×r walk with per-axis trajectories + shifts"
echo "    (depth=2000, K=32, proximity=1e-3)"
python3 a100_sweep.py \
    --mode cmf \
    --input "${SWEEPDIR}/3F2/3F2_part00.jsonl" \
    --traj "${SWEEPDIR}/trajectories/dim5_trajectories.json" \
    --shifts "${SWEEPDIR}/shifts/dim5_shifts.json" \
    --depth 2000 --K 32 \
    --proximity 1e-3 \
    --max-traj 10 \
    --output "${RESULTS}/3F2_first_pass/"

# ── 6. Full 3F2 sweep (all 8161 traj — run manually if first pass looks good)
# Uncomment to run the full sweep (estimated ~24-48h on CPU, faster on GPU):
# python3 a100_sweep.py \
#     --mode cmf \
#     --input "${SWEEPDIR}/3F2/3F2_part00.jsonl" \
#     --traj "${SWEEPDIR}/trajectories/dim5_trajectories.json" \
#     --shifts "${SWEEPDIR}/shifts/dim5_shifts.json" \
#     --depth 2000 --K 32 \
#     --proximity 1e-3 \
#     --output "${RESULTS}/3F2_full/"

# ── 6. Summary ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE — Results in ${RESULTS}/"
echo "============================================================"
echo ""
echo "Euler2AI hits:"
if [ -f "${RESULTS}/euler2ai/euler2ai_hits.csv" ]; then
    wc -l "${RESULTS}/euler2ai/euler2ai_hits.csv"
    head -5 "${RESULTS}/euler2ai/euler2ai_hits.csv"
else
    echo "  (no hits)"
fi
echo ""
echo "3F2 CMF hits:"
if [ -f "${RESULTS}/3F2_part00/cmf_hits.csv" ]; then
    wc -l "${RESULTS}/3F2_part00/cmf_hits.csv"
    head -5 "${RESULTS}/3F2_part00/cmf_hits.csv"
else
    echo "  (no hits)"
fi
echo ""
find "${RESULTS}" -name "*.jsonl" -o -name "*.csv" | sort
