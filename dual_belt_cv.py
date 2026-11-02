"""Singer-held-out TCN predicting the two uncalibrated belts separately."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score
from sequence_model_screen import recordings


class DualWindows(Dataset):
    def __init__(self,records,indices,mean,std,size=128,step=64):
        self.records=records; self.mean=mean; self.std=std; self.size=size; self.locations=[]
        for i in indices:
            n=len(records[i][2]); starts=list(range(0,max(n-size+1,1),step))
            if not starts or starts[-1]!=max(0,n-size):
                starts.append(max(0,n-size))
            self.locations.extend((i,s) for s in starts)
    def __len__(self): return len(self.locations)
    def __getitem__(self,j):
        i,start=self.locations[j]; _,x,y=self.records[i]
        xx=((x[start:start+self.size]-self.mean)/self.std).astype("float32")
        yy=y[start:start+self.size].astype("float32")
        mask=np.ones(len(yy),dtype="float32")
        if len(yy)<self.size:
            pad=self.size-len(yy); xx=np.pad(xx,((0,pad),(0,0))); yy=np.pad(yy,((0,pad),(0,0))); mask=np.pad(mask,(0,pad))
        return torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(mask)


class DualTCN(nn.Module):
    def __init__(self,width=32):
        super().__init__(); self.input=nn.Conv1d(6,width,1); self.blocks=nn.ModuleList()
        for dilation in (1,2,4,8):
            self.blocks.append(nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation),
                                             nn.GELU(),nn.Dropout(.1),nn.Conv1d(width,width,1)))
        self.output=nn.Conv1d(width,2,1)
    def forward(self,x):
        h=self.input(x.transpose(1,2))
        for block in self.blocks:
            h=torch.nn.functional.gelu(h+block(h))
        return self.output(h).transpose(1,2)


def train_epoch(model,loader,opt,device):
    model.train(); losses=[]
    for x,y,mask in loader:
        x=x.to(device); y=y.to(device); mask=mask.to(device)
        opt.zero_grad(set_to_none=True)
        pred=model(x); loss=((((pred-y)**2)*mask[:,:,None]).sum()/(mask.sum()*2))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model,x,mean,std,device):
    model.eval(); xx=((x-mean)/std).astype("float32"); size=128; step=64
    starts=list(range(0,max(len(x)-size+1,1),step))
    if not starts or starts[-1]!=max(0,len(x)-size): starts.append(max(0,len(x)-size))
    sums=np.zeros((len(x),2),dtype="float64"); counts=np.zeros(len(x),dtype="float64")
    for start in starts:
        chunk=xx[start:start+size]; n=len(chunk)
        if n<size: chunk=np.pad(chunk,((0,size-n),(0,0)))
        p=model(torch.from_numpy(chunk[None]).to(device))[0,:n].cpu().numpy()
        sums[start:start+n]+=p; counts[start:start+n]+=1
    return (sums/counts[:,None]).astype("float32")


def evaluate(model,records,indices,mean,std,device):
    result=[]
    for i in indices:
        meta,x,y=records[i]; p=predict(model,x,mean,std,device)
        row={**meta}
        for channel in (0,1):
            label="A" if channel==0 else "B"
            row[label]={**metrics(y[:,channel],p[:,channel]),
                        "peak_f1":extrema_score(y[:,channel],p[:,channel],+1)["f1"],
                        "trough_f1":extrema_score(y[:,channel],p[:,channel],-1)["f1"]}
        result.append(row)
    return result


def singer_median(rows,key):
    names=sorted({r["singer"] for r in rows})
    return float(np.median([np.median([np.mean([r[c][key] for c in ("A","B")]) for r in rows if r["singer"]==name]) for name in names]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--tcn-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint-dir",type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4); device="cuda" if torch.cuda.is_available() else "cpu"
    records=recordings(args.cache); singers=np.array([m["singer"] for m,_,_ in records]); styles=np.array([m["style"] for m,_,_ in records])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260930).split(np.arange(len(records)),styles,singers))
    old=json.loads(args.tcn_results.read_text(encoding="utf-8"))
    for k,(_,te) in enumerate(folds): assert set(singers[te])==set(old["folds"][k]["test_singers"])
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True); results=[]
    for fold,(outer_train,outer_test) in enumerate(folds,1):
        it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20260930+fold).split(outer_train,groups=singers[outer_train]))
        train=outer_train[it]; val=outer_train[iv]
        xtr=np.concatenate([records[i][1] for i in train]); mean=xtr.mean(axis=0); std=np.maximum(xtr.std(axis=0),1e-5)
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=DualTCN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(DualWindows(records,train,mean,std),batch_size=32,shuffle=True,num_workers=0)
        best=-1e9; best_epoch=1; bad=0; history=[]
        for epoch in range(1,31):
            loss=train_epoch(model,loader,opt,device)
            vr=singer_median(evaluate(model,records,val,mean,std,device),"r")
            history.append({"epoch":epoch,"train_loss":loss,"val_singer_median_AB_r":vr})
            if vr>best+.002: best=vr; best_epoch=epoch; bad=0
            else: bad+=1
            if bad>=5: break
        xf=np.concatenate([records[i][1] for i in outer_train]); fm=xf.mean(axis=0); fs=np.maximum(xf.std(axis=0),1e-5)
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=DualTCN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(DualWindows(records,outer_train,fm,fs),batch_size=32,shuffle=True,num_workers=0)
        for _ in range(best_epoch): train_epoch(model,loader,opt,device)
        ckpt=args.checkpoint_dir/f"dual_tcn_fold{fold}.pt"
        torch.save({"model":model.state_dict(),"mean":fm,"std":fs,"epochs":best_epoch},ckpt)
        test_rows=evaluate(model,records,outer_test,fm,fs,device)
        result={"fold":fold,"best_epoch":best_epoch,"best_val_singer_median_AB_r":best,
                "test_singers":sorted(set(singers[outer_test])),"history":history,
                "test_segments":test_rows,"checkpoint":os.path.relpath(ckpt,args.out.parent)}
        results.append(result)
        print(f"fold {fold}/5 epoch={best_epoch} test_A/B_singer_r={singer_median(test_rows,'r'):.3f}",flush=True)
        args.out.write_text(json.dumps({"kind":"dual_belt_cv_running","completed_folds":fold,"folds":results},ensure_ascii=False,indent=2),encoding="utf-8")
    all_rows=[r for f in results for r in f["test_segments"]]
    summary={}
    for c in ("A","B"):
        summary[c]={k:float(np.median([r[c][k] for r in all_rows])) for k in ("r","nrmse","peak_f1","trough_f1")}
    args.out.write_text(json.dumps({"kind":"exploratory_dual_belt_cv_not_final","summary":summary,"folds":results},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
