"""Show separate A/B predictions for a recording with opposing belt motion."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dual_belt_cv import DualTCN, predict
from sequence_model_screen import recordings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--cv-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    records=recordings(args.cache); cv=json.loads(args.cv_results.read_text(encoding="utf-8"))
    i=next(i for i,(m,_,_) in enumerate(records) if m["singer"]=="P24" and m["number"]=="12")
    meta,x,y=records[i]
    fold=next(f for f in cv["folds"] if meta["singer"] in f["test_singers"])
    state=torch.load(args.cv_results.parent/fold["checkpoint"],map_location="cpu",weights_only=False)
    model=DualTCN(); model.load_state_dict(state["model"]); model.eval()
    p=predict(model,x,state["mean"],state["std"],"cpu")
    t=np.arange(len(y))/5
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,constrained_layout=True)
    for ax,j,label in zip(axes,(0,1),("Belt A","Belt B")):
        r=np.corrcoef(y[:,j],p[:,j])[0,1]
        ax.plot(t,y[:,j],color="#245da8",lw=1.5,label="Measured")
        ax.plot(t,p[:,j],color="#159364",lw=1.3,label="Predicted")
        ax.set_title(f"{label}: prediction r={r:.2f}")
        ax.set_ylabel("Relative units")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"P24 #12: measured belts oppose each other (A/B r={np.corrcoef(y[:,0],y[:,1])[0,1]:.2f})",fontsize=15,weight="bold")
    args.out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.out,dpi=180)
    print(args.out)


if __name__=="__main__": main()
