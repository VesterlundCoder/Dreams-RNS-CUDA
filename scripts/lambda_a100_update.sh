#!/bin/bash
# Lambda Labs A100 — pull latest Dreams-RNS-CUDA and run full pipeline
#
# What's new:
#   - euler2ai: 5192 source tasks (not 1693 PCFs), auto-K, CRT overflow detection
#   - odd_zeta_sweep: RNS walk depth=5000, trajectory support, auto-K
#   - odd_zeta_exact: CPU big-integer deep analysis (run locally, not here)
#   - constants.py: fixed int_max_str_digits for high-dps mpmath
#
# Usage:
#   scp -i euler2ai.pem scripts/lambda_a100_update.sh ubuntu@<IP>:~/
#   ssh -i euler2ai.pem ubuntu@<IP> 'bash lambda_a100_update.sh'

set -euo pipefail
echo "============================================================"
echo "  Dreams-RNS-CUDA A100 — Latest Update"
echo "============================================================"

# ── 1. Pull latest code ──────────────────────────────────────────
cd ~
if [ -d "Dreams-RNS-CUDA" ]; then
    echo "Pulling latest..."
    cd Dreams-RNS-CUDA && git pull origin main && cd ~
else
    echo "Cloning..."
    git clone https://github.com/VesterlundCoder/Dreams-RNS-CUDA.git
fi

# ── 2. Dependencies ──────────────────────────────────────────────
pip3 install --quiet numpy sympy mpmath scipy 2>/dev/null || true

PYDIR=~/Dreams-RNS-CUDA/python
RESULTS=~/results_$(date +%Y%m%d_%H%M)
mkdir -p "${RESULTS}"

echo "Results dir: ${RESULTS}"
echo ""

# ── 3. Euler2AI: all 5192 source tasks (auto-K) ─────────────────
echo "============================================================"
echo "  Euler2AI: 5192 source tasks, depth=2000, auto-K"
echo "============================================================"
cd "${PYDIR}"

if [ -f ~/cmf_pcfs.json ]; then
    python3 euler2ai_full_verify.py \
        --input ~/cmf_pcfs.json \
        --depth 2000 --K 0 \
        --output "${RESULTS}/euler2ai_5192.csv"
    echo "Done. Results: ${RESULTS}/euler2ai_5192.csv"
else
    echo "SKIP: ~/cmf_pcfs.json not found"
fi

# ── 4. Odd-Zeta RNS Sweep: ζ(5)..ζ(21) ─────────────────────────
echo ""
echo "============================================================"
echo "  Odd-Zeta RNS Sweep: ζ(5)..ζ(21), depth=5000"
echo "  512 shifts × trajectories 1,2,3"
echo "============================================================"

# Generate specs
python3 odd_zeta_cmf_generator.py \
    --n-min 2 --n-max 10 \
    --output odd_zeta_specs.jsonl

# Run sweep
python3 odd_zeta_sweep.py \
    --specs odd_zeta_specs.jsonl \
    --depth 5000 --shifts 512 \
    --trajectories 1 2 3 \
    --output "${RESULTS}/odd_zeta_sweep/"

# ── 5. a100_sweep euler2ai mode (if pcfs.json exists) ───────────
echo ""
echo "============================================================"
echo "  a100_sweep euler2ai mode (legacy pcfs.json)"
echo "============================================================"

if [ -f ~/pcfs.json ]; then
    python3 a100_sweep.py \
        --mode euler2ai \
        --input ~/pcfs.json \
        --depth 2000 --K 0 \
        --output "${RESULTS}/euler2ai_legacy/"
    echo "Done."
else
    echo "SKIP: ~/pcfs.json not found"
fi

# ── 6. Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  DONE — All results in ${RESULTS}/"
echo "============================================================"
echo ""
echo "Files:"
find "${RESULTS}" -type f \( -name "*.jsonl" -o -name "*.csv" \) | sort
echo ""
echo "Odd-zeta summary:"
if [ -f "${RESULTS}/odd_zeta_sweep/odd_zeta_summary.csv" ]; then
    cat "${RESULTS}/odd_zeta_sweep/odd_zeta_summary.csv"
fi
echo ""
echo "To download results:"
echo "  scp -i euler2ai.pem -r ubuntu@\$(hostname -I | awk '{print \$1}'):${RESULTS} ."
