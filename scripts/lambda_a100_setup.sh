#!/bin/bash
# Lambda Labs A100 setup for Dreams-RNS-CUDA verification
# Usage: scp this script + data to the instance, then run it
#
#   scp -i euler2ai.pem lambda_a100_setup.sh pcfs.json cmf_pcfs.json ubuntu@<IP>:~/
#   ssh -i euler2ai.pem ubuntu@<IP> 'bash lambda_a100_setup.sh'

set -euo pipefail
echo "=== Dreams-RNS-CUDA A100 Setup ==="

# 1. System packages
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git

# 2. Python dependencies
pip3 install --quiet numpy sympy mpmath tqdm

# 3. Clone the repo
cd ~
if [ -d "Dreams-RNS-CUDA" ]; then
    cd Dreams-RNS-CUDA && git pull origin main && cd ~
else
    git clone https://github.com/VesterlundCoder/Dreams-RNS-CUDA.git
fi

# 4. Quick smoke test (5 PCFs, low depth)
echo ""
echo "=== Smoke test (5 PCFs, depth=500, K=16) ==="
cd ~/Dreams-RNS-CUDA/python
python3 euler2ai_verify.py \
    --input ~/pcfs.json \
    --depth 500 --K 16 --max-tasks 5 \
    --output ~/smoke_report.csv

# 5. Full pcfs.json run
echo ""
echo "=== Full pcfs.json (149 PCFs, depth=2000, K=32) ==="
python3 euler2ai_verify.py \
    --input ~/pcfs.json \
    --depth 2000 --K 32 --max-tasks 0 \
    --output ~/pcfs_report.csv

# 6. Full cmf_pcfs.json run (if file exists)
if [ -f ~/cmf_pcfs.json ]; then
    echo ""
    echo "=== Full cmf_pcfs.json (1693 PCFs, depth=2000, K=32) ==="
    python3 euler2ai_verify.py \
        --input ~/cmf_pcfs.json \
        --depth 2000 --K 32 --max-tasks 0 \
        --output ~/cmf_pcfs_report.csv
fi

echo ""
echo "=== Done! Reports in ~/  ==="
ls -la ~/*report*.csv
