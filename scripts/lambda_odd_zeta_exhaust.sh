#!/bin/bash
# Lambda Labs A100 — Exhaustive odd-zeta CMF sweep
#
# Processes ONE CMF at a time to avoid memory overload.
# Each CMF gets 1024 shifts × 50 trajectories = 51,200 walks.
# Uses depth=1000 + K=64 for fast filtering (float delta).
# Interesting hits → run locally with odd_zeta_exact.py at depth=5000-10000.
#
# Usage (Jupyter terminal):
#   bash lambda_odd_zeta_exhaust.sh
#
# Or run a single CMF:
#   bash lambda_odd_zeta_exhaust.sh zeta_5

set -euo pipefail

PYDIR=~/Dreams-RNS-CUDA/python
RESULTS=~/results_odd_zeta_$(date +%Y%m%d_%H%M)
DEPTH=1000
K=64
SHIFTS=1024
TRAJS="1 2 3 4 5 6 7 8 9 10 12 15 20 25 30 35 40 45 50"

echo "============================================================"
echo "  Odd-Zeta Exhaustive Sweep — A100"
echo "  Depth=${DEPTH}, K=${K}, Shifts=${SHIFTS}"
echo "  Trajectories: ${TRAJS}"
echo "  Results: ${RESULTS}"
echo "============================================================"

# Pull latest code
cd ~/Dreams-RNS-CUDA && git pull origin main 2>/dev/null || true

# Generate specs if needed
cd "${PYDIR}"
if [ ! -f odd_zeta_specs.jsonl ]; then
    echo "Generating CMF specs..."
    python3 odd_zeta_cmf_generator.py --n-min 2 --n-max 10 --output odd_zeta_specs.jsonl
fi

mkdir -p "${RESULTS}"

# CMF names: zeta_5, zeta_7, ..., zeta_21
CMFS=$(python3 -c "
import json
with open('odd_zeta_specs.jsonl') as f:
    for line in f:
        spec = json.loads(line.strip())
        print(spec['name'])
")

# If a specific CMF is requested, filter
if [ "${1:-}" != "" ]; then
    CMFS="${1}"
    echo "Running single CMF: ${CMFS}"
fi

TOTAL_CMFS=$(echo "${CMFS}" | wc -w | tr -d ' ')
CMF_NUM=0

for CMF_NAME in ${CMFS}; do
    CMF_NUM=$((CMF_NUM + 1))
    echo ""
    echo "============================================================"
    echo "  [${CMF_NUM}/${TOTAL_CMFS}] ${CMF_NAME}"
    echo "  ${SHIFTS} shifts × $(echo ${TRAJS} | wc -w | tr -d ' ') trajectories"
    echo "============================================================"

    OUTPUT_DIR="${RESULTS}/${CMF_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    python3 odd_zeta_sweep.py \
        --specs odd_zeta_specs.jsonl \
        --cmf-name "${CMF_NAME}" \
        --depth ${DEPTH} \
        --K ${K} \
        --shifts ${SHIFTS} \
        --trajectories ${TRAJS} \
        --output "${OUTPUT_DIR}/"

    # Quick summary
    if [ -f "${OUTPUT_DIR}/odd_zeta_summary.csv" ]; then
        echo ""
        echo "--- ${CMF_NAME} summary ---"
        cat "${OUTPUT_DIR}/odd_zeta_summary.csv"
    fi

    echo ""
    echo "${CMF_NAME} DONE"
    echo "============================================================"
done

# Final summary
echo ""
echo "============================================================"
echo "  ALL DONE — Results in ${RESULTS}/"
echo "============================================================"
echo ""
echo "Files:"
find "${RESULTS}" -name "*.csv" -o -name "*.jsonl" | sort
echo ""
echo "Hits with δ > 0:"
grep -h '"best_delta"' "${RESULTS}"/*/odd_zeta_results.jsonl 2>/dev/null | python3 -c "
import sys, json
hits = []
for line in sys.stdin:
    r = json.loads(line)
    if r.get('best_delta', -999) > 0:
        hits.append(r)
hits.sort(key=lambda x: -x['best_delta'])
print(f'  {len(hits)} hits with δ > 0')
for h in hits[:20]:
    print(f'  δ={h[\"best_delta\"]:.6f} {h[\"cmf\"]} shift={h[\"shift\"]} traj={h[\"trajectory\"]} → {h[\"best_const\"]}')
if not hits:
    print('  (none)')
" 2>/dev/null || echo "  (no hits or parse error)"

echo ""
echo "Download results:"
echo "  scp -i euler2ai.pem -r ubuntu@\$(hostname -I | awk '{print \$1}'):${RESULTS} ."
echo ""
echo "Deep analysis of interesting hits:"
echo "  python3 odd_zeta_exact.py --specs odd_zeta_specs.jsonl --depth 5000 --shifts <S> --trajectories <T> --cmf-name <NAME>"
