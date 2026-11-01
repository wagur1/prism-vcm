#!/usr/bin/env bash
set -euo pipefail
cd /kaggle/working
if [[ ! -d prism-vcm/.git ]]; then
    git clone https://github.com/wagur1/prism-vcm.git prism-vcm
fi
cd prism-vcm
python3 -u tier1a/run_tier1a_step1.py "$@"
