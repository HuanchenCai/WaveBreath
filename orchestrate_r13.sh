#!/bin/sh
# Round-13 unattended scheduler: batch 1 -> selection -> batch 1b + batch 2 (two queues).
# Log: r13_orchestrator.log. Safe to re-run: runners resume from saved folds.
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONIOENCODING=utf-8
log() { echo "$(date '+%F %T') $*" >> r13_orchestrator.log; }

done_fam() { $PY -c "import json,sys;d=json.load(open('family_runs/r13_$1/results.json',encoding='utf-8'));sys.exit(0 if d.get('completed') else 1)" 2>/dev/null; }
done_run() { $PY -c "import json,sys;d=json.load(open('family_runs/$1/results.json',encoding='utf-8'));sys.exit(0 if d.get('completed') else 1)" 2>/dev/null; }

log "waiting for batch 1"
for f in tcn bigru convlstm transformer unet; do
  until done_fam $f; do sleep 60; done
  log "batch 1 complete: $f"
done
$PY select_r13_top.py >> r13_orchestrator.log 2>&1
TOP=$($PY -c "import json;print(' '.join(json.load(open('r13_batch1_selection.json'))['top2']))")
log "top2 = $TOP"

queue() {  # $1 = family
  fam=$1
  for spec in "--scheme separate" "--scheme tasklabel" "--input wavlm_all" "--input mert_all"; do
    case "$spec" in
      "--input wavlm_all") name=r13b_${fam}_wavlm_all_joint ;;
      "--input mert_all") name=r13b_${fam}_mert_all_joint ;;
      "--scheme separate") name=r13b_${fam}_wavlm16_separate ;;
      "--scheme tasklabel") name=r13b_${fam}_wavlm16_tasklabel ;;
    esac
    if [ "$fam" = convlstm ]; then  # early runs happen in r13_p3b.log; never run the same name twice at once
      until [ -f r13_p3_done.flag ]; do sleep 60; done
    fi
    if done_run $name; then log "skip $name (already complete)"; continue; fi
    log "start $name"
    $PY run_r13_followup.py --family $fam $spec >> r13_queue_$fam.log 2>&1
    log "end $name (exit $?)"
  done
}

for fam in $TOP; do queue $fam & done
wait
log "batch 1b + 2 finished"
