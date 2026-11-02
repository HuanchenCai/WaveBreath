"""Plot paired five-fold model results and the worst held-out waveform."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sequence_model_screen import TCN, predict, recordings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--tree-results",type=Path,required=True)
    ap.add_argument("--tcn-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    tcn=json.loads(args.tcn_results.read_text(encoding="utf-8"))
    tree=json.loads(args.tree_results.read_text(encoding="utf-8"))
    tr={(r["singer"],str(r["number"])):r for f in tree["folds"] for r in f["segments"]}
    ts=[r for f in tcn["folds"] for r in f["test_segments"]]
    assert len(ts)==len(tr)==197
    names=sorted({r["singer"] for r in ts})
    styles={r["singer"]:r["style"] for r in ts}
    colors={"BelCanto":"#5c65b9","FolkSongs":"#199b70","PopularMusic":"#dc742c"}
    fig,ax=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    p=ax[0,0]
    for style in colors:
        subset=[n for n in names if styles[n]==style]
        x=[np.median([tr[(r["singer"],str(r["number"]))]["hgb"]["r"] for r in ts if r["singer"]==n]) for n in subset]
        y=[np.median([r["r"] for r in ts if r["singer"]==n]) for n in subset]
        p.scatter(x,y,color=colors[style],label=style,alpha=.9)
    p.plot([0,1],[0,1],"--",color="#777777",lw=1)
    p.set(xlabel="Tree: median r per singer",ylabel="TCN: median r per singer",xlim=(0,1),ylim=(0,1),title="32 singers: paired waveform correlation")
    p.legend(loc="lower right")
    p=ax[0,1]
    p.boxplot([[tr[(r["singer"],str(r["number"]))]["hgb"]["r"] for r in ts],[r["r"] for r in ts]],tick_labels=["Tree","TCN"],showfliers=False)
    p.set(ylabel="Signed Pearson r",ylim=(-.1,1),title="197 held-out recordings")
    p.grid(axis="y",alpha=.2)
    p=ax[1,0]
    event_values=[[tr[(r["singer"],str(r["number"]))]["hgb"]["positive_maxima"]["f1"] for r in ts],
                  [r["peak_f1"] for r in ts],
                  [tr[(r["singer"],str(r["number"]))]["hgb"]["negative_minima"]["f1"] for r in ts],
                  [r["trough_f1"] for r in ts]]
    p.boxplot(event_values,tick_labels=["Tree peak","TCN peak","Tree trough","TCN trough"],showfliers=False)
    p.set(ylabel="Event F1 (0.4 s tolerance)",ylim=(0,1),title="Timing improves; troughs remain difficult")
    p.grid(axis="y",alpha=.2)
    records=recordings(args.cache)
    worst=min(ts,key=lambda r:r["r"])
    i=next(i for i,(m,_,_) in enumerate(records) if m["singer"]==worst["singer"] and m["number"]==worst["number"])
    fold=next(f for f in tcn["folds"] if worst["singer"] in f["test_singers"])
    state=torch.load(fold["checkpoint"],map_location="cpu",weights_only=False)
    model=TCN(32); model.load_state_dict(state["model"]); model.eval()
    _,x,y=records[i]; pred=predict(model,x,state["mean"],state["std"],"cpu")
    p=ax[1,1]; n=min(len(y),200); time=np.arange(n)/5
    p.plot(time,y[:n],color="#245da8",lw=1.5,label="Measured")
    p.plot(time,pred[:n],color="#159364",lw=1.2,label="TCN")
    p.set(xlabel="Time (s)",ylabel="Relative belt units",title=f"Worst recording: {worst['singer']} #{worst['number']}, r={worst['r']:.2f}")
    p.legend()
    fig.suptitle("Second-round TCN versus first-round tree: singer-held-out five-fold results",fontsize=15,weight="bold")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180)
    print(args.out)


if __name__=="__main__":
    main()
