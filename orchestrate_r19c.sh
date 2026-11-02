#!/bin/sh
# Start round-19 C (LoRA) after all round-19 A/B runs have completed or failed.
cd "$(dirname "$0")"
for f in r19_win_convlstm.log r19_win_bigru.log r19_win_unet.log r19_pers_convlstm.log r19_pers_unet.log; do
  until grep -q "COMPLETE\|Traceback" $f; do sleep 60; done
done
PYTHONIOENCODING=utf-8 ${PYTHON:-python} run_r19c_lora.py --family convlstm > r19_lora_convlstm.log 2>&1
