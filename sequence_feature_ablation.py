"""Retrain the selected TCN with RMS-only and no-RMS inputs."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from sequence_model_screen import TCN, Windows, evaluate, loss_fn, recordings, singer_median_r


def train_model(records, indices, mean, std, epochs, device):
    random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
    model=TCN(32).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(Windows(records,indices,mean,std),batch_size=32,shuffle=True,num_workers=0)
    for epoch in range(epochs):
        model.train()
        for x,y,mask in loader:
            x=x.to(device); y=y.to(device); mask=mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y,mask,0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
    return model


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4); device="cuda" if torch.cuda.is_available() else "cpu"
    original=recordings(args.cache)
    selected=[]; counts={}
    for i in sorted(range(len(original)),key=lambda j:(original[j][0]["singer"],int(original[j][0]["number"]))):
        name=original[i][0]["singer"]
        if counts.get(name,0)<2:
            selected.append(i); counts[name]=counts.get(name,0)+1
    groups=np.array([original[i][0]["singer"] for i in selected])
    otr,ote=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260930).split(np.arange(len(selected)),groups=groups))
    pool=np.array(selected)[otr]; test=np.array(selected)[ote]
    pgroups=np.array([original[i][0]["singer"] for i in pool])
    itr,iva=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20260931).split(pool,groups=pgroups))
    train=pool[itr]; val=pool[iva]
    raw_mean=np.concatenate([original[i][1] for i in train]).mean(axis=0)
    results={}
    for label, keep in (("rms_only",[0]),("without_rms",[1,2,3,4,5])):
        masked=[]
        absent=[j for j in range(6) if j not in keep]
        for meta,x,y in original:
            xx=x.copy(); xx[:,absent]=raw_mean[absent]
            masked.append((meta,xx,y))
        train_x=np.concatenate([masked[i][1] for i in train]); mean=train_x.mean(axis=0); std=np.maximum(train_x.std(axis=0),1e-5)
        best=-1e9; best_epoch=1; history=[]
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=TCN(32).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(Windows(masked,train,mean,std),batch_size=32,shuffle=True,num_workers=0)
        for epoch in range(1,31):
            model.train()
            for x,y,mask in loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device)
                opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y,mask,0); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            vr=singer_median_r(evaluate(model,masked,val,mean,std,device)); history.append(vr)
            if vr>best+0.002:
                best=vr; best_epoch=epoch
            print(label,epoch,round(vr,3),flush=True)
        pool_x=np.concatenate([masked[i][1] for i in pool]); pmean=pool_x.mean(axis=0); pstd=np.maximum(pool_x.std(axis=0),1e-5)
        final=train_model(masked,pool,pmean,pstd,best_epoch,device)
        test_rows=evaluate(final,masked,test,pmean,pstd,device)
        results[label]={"best_val_singer_median_r":best,"best_epoch":best_epoch,
                        "test_segment_median_r":float(np.median([r["r"] for r in test_rows])),
                        "test_segment_median_nrmse":float(np.median([r["nrmse"] for r in test_rows])),
                        "test_segment_median_peak_f1":float(np.median([r["peak_f1"] for r in test_rows])),
                        "test_segment_median_trough_f1":float(np.median([r["trough_f1"] for r in test_rows])),
                        "history":history,"test_segments":test_rows}
        print(label,"TEST",results[label]["test_segment_median_r"],flush=True)
    args.out.write_text(json.dumps({"kind":"exploratory_retrained_feature_ablation","results":results},ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
