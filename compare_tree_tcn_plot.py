"""Compare the selected TCN with the previous tree on identical held-out clips."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from breath_model_pilot import context_features, fit_predict
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
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260930).split(np.arange(len(selected)),groups=groups))
    train=np.array(selected)[tr]; test=np.array(selected)[te]
    train_x=np.concatenate([context_features(records[i][1],5) for i in train]); train_y=np.concatenate([records[i][2] for i in train])
    state=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=TCN(32)
    model.load_state_dict(state["model"]); model.eval()
    examples=[("P09","1","High original score"),("P29","3","Near original median"),("P30","5","Original failure")]
    fig,axes=plt.subplots(3,1,figsize=(15,9),constrained_layout=True)
    for ax,(singer,number,label) in zip(axes,examples):
        i=next(i for i in test if records[i][0]["singer"]==singer and records[i][0]["number"]==number)
        _,x,y=records[i]
        tree=fit_predict("hgb",{"learning_rate":.03,"max_leaf_nodes":31},train_x,train_y,context_features(x,5))
        tcn=predict(model,x,state["mean"],state["std"],"cpu")
        n=min(len(y),150); t=np.arange(n)/5
        ax.plot(t,y[:n],color="#1c4f9c",lw=1.8,label="Measured")
        ax.plot(t,tree[:n],color="#e27627",lw=1.2,alpha=.8,label="Tree")
        ax.plot(t,tcn[:n],color="#15875b",lw=1.5,label="TCN")
        ax.set_title(f"{label}: {singer} #{number} | full-clip r: tree {np.corrcoef(y,tree)[0,1]:.2f}, TCN {np.corrcoef(y,tcn)[0,1]:.2f}")
        ax.set_ylabel("Relative units")
        ax.set_xlabel("Time (s)")
    axes[0].legend(ncol=3,loc="upper right")
    fig.suptitle("Same held-out singers: tree model versus sequence TCN",fontsize=15,weight="bold")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180)
    print(args.out)


if __name__=="__main__":
    main()
