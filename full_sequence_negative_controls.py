"""Apply timing perturbations to every held-out fold's fitted TCN."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_breath_pilot import metrics
from sequence_model_screen import TCN, predict, recordings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--cv-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    records=recordings(args.cache)
    cv=json.loads(args.cv_results.read_text(encoding="utf-8"))
    rows=[]
    for fold in cv["folds"]:
        state=torch.load(fold["checkpoint"],map_location="cpu",weights_only=False)
        model=TCN(32); model.load_state_dict(state["model"]); model.eval()
        test=set(fold["test_singers"])
        for meta,x,y in records:
            if meta["singer"] not in test:
                continue
            conditions={"aligned":x,"shift_5s":np.roll(x,25,axis=0),"shift_10s":np.roll(x,50,axis=0)}
            scores={name:metrics(y,predict(model,xx,state["mean"],state["std"],"cpu"))["r"] for name,xx in conditions.items()}
            rows.append({"fold":fold["fold"],"singer":meta["singer"],"number":meta["number"],**scores})
        print(f"fold {fold['fold']} controls complete",flush=True)
    summary={k:float(np.median([r[k] for r in rows])) for k in ("aligned","shift_5s","shift_10s")}
    output={"kind":"exploratory_full_cv_test_input_shift","summary_median_segment_r":summary,
            "segments":rows,"note":"Circular time shift is a diagnostic perturbation and creates an artificial boundary."}
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary,flush=True)


if __name__=="__main__":
    main()
