"""Compare A/B TCN predictions with separate per-belt tree baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from breath_model_pilot import context_features, fit_predict
from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score
from sequence_model_screen import recordings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    records=recordings(args.cache)
    singers=np.array([m["singer"] for m,_,_ in records]); styles=np.array([m["style"] for m,_,_ in records])
    contexts=[context_features(x,5) for _,x,_ in records]
    folds=[]
    for fold,(train,test) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260930).split(np.arange(len(records)),styles,singers),1):
        assert not set(singers[train])&set(singers[test])
        xtr=np.concatenate([contexts[i] for i in train]); xte=np.concatenate([contexts[i] for i in test])
        cut=np.cumsum([len(records[i][2]) for i in test])[:-1]
        pred={}
        for c in (0,1):
            ytr=np.concatenate([records[i][2][:,c] for i in train])
            flat=fit_predict("hgb",{"learning_rate":.03,"max_leaf_nodes":31},xtr,ytr,xte)
            pred[c]=dict(zip(test,np.split(flat,cut)))
        segments=[]
        for i in test:
            meta,_,y=records[i]; row={**meta}
            for c,label in ((0,"A"),(1,"B")):
                p=pred[c][i]
                row[label]={**metrics(y[:,c],p),"peak_f1":extrema_score(y[:,c],p,+1)["f1"],
                            "trough_f1":extrema_score(y[:,c],p,-1)["f1"]}
            segments.append(row)
        folds.append({"fold":fold,"test_singers":sorted(set(singers[test])),"test_segments":segments})
        print(f"fold {fold}/5 complete",flush=True)
    all_rows=[r for f in folds for r in f["test_segments"]]
    summary={c:{k:float(np.median([r[c][k] for r in all_rows])) for k in ("r","nrmse","peak_f1","trough_f1")} for c in ("A","B")}
    args.out.write_text(json.dumps({"kind":"exploratory_dual_tree_baseline","summary":summary,"folds":folds},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
