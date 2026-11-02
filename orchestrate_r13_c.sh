#!/bin/sh
# Round-13 batch 3 scheduler: wait for the batch-1 selection, then 50-Hz runs for top-1, top-2.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
log() { echo "$(date '+%F %T') [batch3] $*" >> r13_orchestrator.log; }
until [ -f r13_batch1_selection.json ]; do sleep 60; done
TOP=$($PY -c "import json;print(' '.join(json.load(open('r13_batch1_selection.json'))['top2']))")
for fam in $TOP; do
  log "start r13c_${fam}_50hz"
  $PY run_r13_50hz.py --family $fam >> r13_batch3.log 2>&1
  log "end r13c_${fam}_50hz (exit $?)"
done
