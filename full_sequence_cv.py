"""Exploratory nested singer-fold evaluation of the selected TCN architecture."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from sequence_model_screen import TCN, Windows, evaluate, recordings, singer_median_r
from sequence_model_screen import loss_fn


def fit_epochs(records, indices, mean, std, epochs, device, checkpoint=None):
    random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
    model=TCN(32).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loader=DataLoader(Windows(records,indices,mean,std),batch_size=32,shuffle=True,num_workers=0)
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for x,y,mask in loader:
            x=x.to(device); y=y.to(device); mask=mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y,mask,0)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    if checkpoint:
        torch.save({"model":model.state_dict(),"mean":mean,"std":std,"epochs":epochs},checkpoint)
    return model,history


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--tree-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint-dir",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4); device="cuda" if torch.cuda.is_available() else "cpu"
    records=recordings(args.cache)
    singers=np.array([m["singer"] for m,_,_ in records]); styles=np.array([m["style"] for m,_,_ in records])
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260930).split(np.arange(len(records)),styles,singers))
    old=json.loads(args.tree_results.read_text(encoding="utf-8"))
    for fold,(_,te) in enumerate(splits):
        assert set(singers[te])==set(old["folds"][fold]["test_singers"]),"Tree/TCN fold mismatch"
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    fold_results=[]
    for fold,(outer_train,outer_test) in enumerate(splits,1):
        assert not set(singers[outer_train])&set(singers[outer_test])
        tg=singers[outer_train]
        inner_train,inner_val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20260930+fold).split(outer_train,groups=tg))
        train=outer_train[inner_train]; val=outer_train[inner_val]
        assert not set(singers[train])&set(singers[val])
        xtrain=np.concatenate([records[i][1] for i in train]); mean=xtrain.mean(axis=0); std=np.maximum(xtrain.std(axis=0),1e-5)
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=TCN(32).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(Windows(records,train,mean,std),batch_size=32,shuffle=True,num_workers=0)
        best=-1e9; best_epoch=1; history=[]; bad=0
        for epoch in range(1,31):
            model.train(); losses=[]
            for x,y,mask in loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(x),y,mask,0)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step(); losses.append(float(loss.detach().cpu()))
            val_rows=evaluate(model,records,val,mean,std,device)
            val_r=singer_median_r(val_rows)
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_singer_median_r":val_r})
            if val_r>best+.002:
                best=val_r; best_epoch=epoch; bad=0
            else:
                bad+=1
            if bad>=5:
                break
        xfull=np.concatenate([records[i][1] for i in outer_train]); full_mean=xfull.mean(axis=0); full_std=np.maximum(xfull.std(axis=0),1e-5)
        ckpt=args.checkpoint_dir/f"tcn_fold{fold}.pt"
        final,_=fit_epochs(records,outer_train,full_mean,full_std,best_epoch,device,ckpt)
        test_rows=evaluate(final,records,outer_test,full_mean,full_std,device)
        fold_result={"fold":fold,"best_epoch":best_epoch,"best_val_singer_median_r":best,
                     "train_singers":sorted(set(singers[outer_train])),"test_singers":sorted(set(singers[outer_test])),
                     "history":history,"test_segments":test_rows,"checkpoint":str(ckpt)}
        fold_results.append(fold_result)
        print(f"fold {fold}/5 epoch={best_epoch} test_singers={len(set(singers[outer_test]))} median_r={singer_median_r(test_rows):.3f}",flush=True)
        args.out.write_text(json.dumps({"kind":"exploratory_full_tcn_cv_running","completed_folds":fold,"folds":fold_results},ensure_ascii=False,indent=2),encoding="utf-8")
    all_rows=[r for fold in fold_results for r in fold["test_segments"]]
    summary={"median_segment_r":float(np.median([r["r"] for r in all_rows])),
             "median_singer_r":singer_median_r(all_rows),
             "median_segment_nrmse":float(np.median([r["nrmse"] for r in all_rows])),
             "median_peak_f1":float(np.median([r["peak_f1"] for r in all_rows])),
             "median_trough_f1":float(np.median([r["trough_f1"] for r in all_rows]))}
    result={"kind":"exploratory_full_tcn_cv_not_final","n_segments":len(records),"n_singers":len(set(singers)),
            "summary":summary,"folds":fold_results}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
