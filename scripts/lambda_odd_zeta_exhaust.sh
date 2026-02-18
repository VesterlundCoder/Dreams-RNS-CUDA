#!/bin/bash
# Lambda Labs A100 — Exhaustive odd-zeta CMF sweep
#
# Uses compiled RNS bytecode walk (no sympy in hot loop).
# Rational shifts (±1/d) via modular inverse, 1000 trajectories.
# Float shadow (~15 digit precision) for fast filtering.
# Results are resumable: re-run to extend without repeating.
#
# Each CMF gets 50 rational shifts × 1000 trajectories = 50,000 walks.
# Processed ONE CMF at a time to keep memory bounded.
#
# Usage:
#   bash lambda_odd_zeta_exhaust.sh          # all CMFs
#   bash lambda_odd_zeta_exhaust.sh zeta_5   # single CMF
#
# Interesting hits → run locally:
#   python3 odd_zeta_exact.py --specs odd_zeta_specs.jsonl \
#       --cmf-name zeta_5 --depth 5000 --shifts 1 --trajectories 1

set -euo pipefail

PYDIR=~/Dreams-RNS-CUDA/python
RESULTS=~/results_odd_zeta
DEPTH=1000
K=16
N_SHIFTS=50
N_TRAJ=1000

echo "============================================================"
echo "  Odd-Zeta Exhaustive Sweep — A100"
echo "  Depth=${DEPTH}, K=${K}"
echo "  ${N_SHIFTS} multi-dim shift vectors"
echo "  ${N_TRAJ} multi-dim trajectory vectors"
echo "  Results: ${RESULTS}"
echo "============================================================"

# Pull latest code
cd ~/Dreams-RNS-CUDA && git pull origin main 2>/dev/null || true

# Dependencies
pip3 install --quiet numpy sympy mpmath 2>/dev/null || true

# Generate specs if needed
cd "${PYDIR}"
if [ ! -f odd_zeta_specs.jsonl ]; then
    echo "Generating CMF specs (ζ(5)..ζ(21))..."
    python3 odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl
fi

# CMF names
CMFS=$(python3 -c "
import json
with open('odd_zeta_specs.jsonl') as f:
    for line in f:
        print(json.loads(line.strip())['name'])")

# Single CMF if requested
if [ "${1:-}" != "" ]; then
    CMFS="${1}"
    echo "Running single CMF: ${CMFS}"
fi

TOTAL_CMFS=$(echo "${CMFS}" | wc -l)
CMF_NUM=0

for CMF_NAME in ${CMFS}; do
    CMF_NUM=$((CMF_NUM + 1))
    OUT="${RESULTS}/${CMF_NAME}"
    mkdir -p "${OUT}"

    echo ""
    echo "============================================================"
    echo "  [${CMF_NUM}/${TOTAL_CMFS}] ${CMF_NAME}"
    echo "  ${N_SHIFTS} shift vecs × ${N_TRAJ} traj vecs = $((N_SHIFTS * N_TRAJ)) walks"
    echo "  Output: ${OUT}/"
    echo "============================================================"

    python3 odd_zeta_sweep.py \
        --specs odd_zeta_specs.jsonl \
        --cmf-name "${CMF_NAME}" \
        --depth ${DEPTH} \
        --K ${K} \
        --n-shifts ${N_SHIFTS} \
        --n-traj ${N_TRAJ} \
        --output "${OUT}/" \
        --resume

    echo "${CMF_NAME} DONE"
done

# Summary: all hits with ≥6 matching digits
echo ""
echo "============================================================"
echo "  ALL DONE — Summary of hits (≥6 matching digits)"
echo "============================================================"

python3 -c "
import json, os, glob
hits = []
for f in sorted(glob.glob('${RESULTS}/*/odd_zeta_results.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r.get('match_digits', 0) >= 6:
            hits.append(r)
hits.sort(key=lambda x: -x['match_digits'])
print(f'  {len(hits)} hits with ≥6 digits')
for h in hits[:30]:
    print(f'  {h[\"match_digits\"]:>5.1f}d  {h[\"cmf\"]:>10}  shift={h[\"shift\"]:>5}  traj={h[\"trajectory\"]:>4}  → {h[\"best_const\"]}  est={h[\"est\"]:.12f}')
if not hits:
    print('  (none)')
" 2>/dev/null || echo "  (parse error)"

echo ""
echo "Download results:"
echo "  scp -i euler2ai.pem -r ubuntu@\$(hostname -I | awk '{print \$1}'):${RESULTS} ."
echo ""
echo "Deep analysis of hits:"
echo "  python3 odd_zeta_exact.py --specs odd_zeta_specs.jsonl --depth 5000 --shifts 1 --trajectories 1 --cmf-name <NAME>"
