"""Check whether TCN predictions depend on correctly timed audio."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from evaluate_breath_pilot import metrics
from sequence_model_screen import TCN, predict, recordings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    records=recordings(args.cache)
    selected=[]; counts={}
    for i in sorted(range(len(records)),key=lambda j:(records[j][0]["singer"],int(records[j][0]["number"]))):
        name=records[i][0]["singer"]
        if counts.get(name,0)<2:
            selected.append(i); counts[name]=counts.get(name,0)+1
    groups=np.array([records[i][0]["singer"] for i in selected])
    _,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260930).split(np.arange(len(selected)),groups=groups))
    test=np.array(selected)[te]
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=TCN(32); model.load_state_dict(state["model"]); model.eval()
    rows=[]
    for i in test:
        meta,x,y=records[i]
        conditions={
            "aligned":x,
            "shift_5s":np.roll(x,25,axis=0),
            "shift_10s":np.roll(x,50,axis=0),
            "mask_rms":np.column_stack([np.full(len(x),state["mean"][0]),x[:,1:]]),
            "mask_low_band":np.column_stack([x[:,:2],np.full(len(x),state["mean"][2]),x[:,3:]]),
        }
        results={name:metrics(y,predict(model,xx,state["mean"],state["std"],"cpu")) for name,xx in conditions.items()}
        rows.append({"singer":meta["singer"],"number":meta["number"],"results":results})
    summary={name:{"median_r":float(np.median([row["results"][name]["r"] for row in rows])),
                   "median_nrmse":float(np.median([row["results"][name]["nrmse"] for row in rows]))}
             for name in conditions}
    result={"kind":"exploratory_test_input_perturbation","summary":summary,"segments":rows,
            "warning":"Masking is an input perturbation, not a retrained feature ablation; circular shifts create an artificial boundary."}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
