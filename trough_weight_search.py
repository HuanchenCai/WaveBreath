"""Exploratory TCN loss grid to improve negative-belt excursions."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from sequence_model_screen import TCN, Windows, evaluate, recordings, singer_median_r


def weighted_loss(pred, target, mask, alpha):
    weights=1+alpha*torch.relu(-target)
    return (((pred-target)**2)*weights*mask).sum()/(weights*mask).sum()


def singer_metric(rows,key):
    names=sorted({r["singer"] for r in rows})
    return float(np.median([np.median([r[key] for r in rows if r["singer"]==name]) for name in names]))


def train(records,indices,mean,std,alpha,epochs,device):
    random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
    model=TCN(32).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(Windows(records,indices,mean,std),batch_size=32,shuffle=True,num_workers=0)
    for _ in range(epochs):
        model.train()
        for x,y,mask in loader:
            x=x.to(device); y=y.to(device); mask=mask.to(device)
            opt.zero_grad(set_to_none=True); loss=weighted_loss(model(x),y,mask,alpha)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4); device="cuda" if torch.cuda.is_available() else "cpu"
    records=recordings(args.cache)
    selected=[]; counts={}
    for i in sorted(range(len(records)),key=lambda j:(records[j][0]["singer"],int(records[j][0]["number"]))):
        name=records[i][0]["singer"]
        if counts.get(name,0)<2:
            selected.append(i); counts[name]=counts.get(name,0)+1
    groups=np.array([records[i][0]["singer"] for i in selected])
    otr,ote=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260930).split(np.arange(len(selected)),groups=groups))
    pool=np.array(selected)[otr]; test=np.array(selected)[ote]
    pgroups=np.array([records[i][0]["singer"] for i in pool])
    itr,iva=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260931).split(pool,groups=pgroups))
    train_i=pool[itr]; val=pool[iva]
    train_x=np.concatenate([records[i][1] for i in train_i]); mean=train_x.mean(axis=0); std=np.maximum(train_x.std(axis=0),1e-5)
    results=[]
    for alpha in (0.0,1.0,2.0):
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=TCN(32).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(Windows(records,train_i,mean,std),batch_size=32,shuffle=True,num_workers=0)
        best=float("inf"); best_epoch=1; history=[]
        for epoch in range(1,31):
            model.train()
            for x,y,mask in loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device)
                opt.zero_grad(set_to_none=True); loss=weighted_loss(model(x),y,mask,alpha)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            rows=evaluate(model,records,val,mean,std,device)
            nrmse=singer_metric(rows,"nrmse")
            history.append({"epoch":epoch,"val_singer_median_nrmse":nrmse,
                            "val_singer_median_r":singer_median_r(rows),
                            "val_trough_f1":singer_metric(rows,"trough_f1")})
            if nrmse<best-.002:
                best=nrmse; best_epoch=epoch
        full_x=np.concatenate([records[i][1] for i in pool]); full_mean=full_x.mean(axis=0); full_std=np.maximum(full_x.std(axis=0),1e-5)
        final=train(records,pool,full_mean,full_std,alpha,best_epoch,device)
        test_rows=evaluate(final,records,test,full_mean,full_std,device)
        result={"alpha":alpha,"best_epoch_by_val_nrmse":best_epoch,"best_val_singer_median_nrmse":best,
                "test_segment_median_r":float(np.median([r["r"] for r in test_rows])),
                "test_segment_median_nrmse":float(np.median([r["nrmse"] for r in test_rows])),
                "test_segment_median_peak_f1":float(np.median([r["peak_f1"] for r in test_rows])),
                "test_segment_median_trough_f1":float(np.median([r["trough_f1"] for r in test_rows])),
                "history":history,"test_segments":test_rows}
        results.append(result)
        print("alpha",alpha,"val_nrmse",best,"test",result["test_segment_median_r"],result["test_segment_median_trough_f1"],flush=True)
        args.out.write_text(json.dumps({"kind":"exploratory_trough_weight_grid_running","results":results},ensure_ascii=False,indent=2),encoding="utf-8")
    args.out.write_text(json.dumps({"kind":"exploratory_trough_weight_grid_not_final","results":results},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
