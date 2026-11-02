#!/bin/sh
# Round-14 EGG batch: after batch 3 (50 Hz) has ended for both top-2 families, run
# WavLM-L16 + EGG features and EGG-only for the top-2 families (one factor vs batch 1).
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
log() { echo "$(date '+%F %T') [r14-egg] $*" >> r13_orchestrator.log; }
until [ -f r13_batch1_selection.json ] && [ -f r14_egg_features.npz ]; do sleep 60; done
TOP=$($PY -c "import json;print(' '.join(json.load(open('r13_batch1_selection.json'))['top2']))")
for fam in $TOP; do until grep -q "end r13c_${fam}_50hz" r13_orchestrator.log; do sleep 60; done; done
for fam in $TOP; do
  for inp in wavlm16_egg egg_only; do
    log "start r13b_${fam}_${inp}_joint"
    $PY run_r13_followup.py --family $fam --input $inp >> r14_egg.log 2>&1
    log "end r13b_${fam}_${inp}_joint (exit $?)"
  done
done
