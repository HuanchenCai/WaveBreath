"""Singer-held-out screen of waveform sequence models on cached 5-Hz features."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score


def recordings(cache: Path):
    data = np.load(cache, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))
    offsets = data["offsets"]
    return [(meta[i], data["x"][offsets[i]:offsets[i+1]], data["y"][offsets[i]:offsets[i+1]]) for i in range(len(meta))]


class Windows(Dataset):
    def __init__(self, records, indices, mean, std, size=128, step=64):
        self.records = records; self.indices = indices; self.mean = mean; self.std = std; self.size = size
        self.locations = []
        for i in indices:
            n = len(records[i][2])
            starts = list(range(0, max(n-size+1, 1), step))
            if not starts or starts[-1] != max(0, n-size):
                starts.append(max(0, n-size))
            self.locations.extend((i, start) for start in starts)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, j):
        i, start = self.locations[j]
        _, x, y = self.records[i]
        xx = ((x[start:start+self.size] - self.mean) / self.std).astype("float32")
        yy = y[start:start+self.size].astype("float32")
        mask = np.ones(len(yy), dtype="float32")
        if len(yy) < self.size:
            pad = self.size-len(yy)
            xx = np.pad(xx, ((0,pad),(0,0))); yy = np.pad(yy, (0,pad)); mask = np.pad(mask, (0,pad))
        return torch.from_numpy(xx), torch.from_numpy(yy), torch.from_numpy(mask)


class TCN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Conv1d(6, width, 1)
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4, 8):
            self.blocks.append(nn.Sequential(nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation),
                                             nn.GELU(), nn.Dropout(0.1), nn.Conv1d(width, width, 1)))
        self.output = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        h = self.input(x.transpose(1, 2))
        for block in self.blocks:
            h = torch.nn.functional.gelu(h + block(h))
        return self.output(h).squeeze(1)


class BiLSTM(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = nn.LSTM(6, width, batch_first=True, bidirectional=True)
        self.output = nn.Linear(width * 2, 1)

    def forward(self, x):
        return self.output(self.rnn(x)[0]).squeeze(-1)


def loss_fn(pred, target, mask, derivative_weight):
    base = (((pred-target)**2)*mask).sum()/mask.sum()
    dy = pred[:,1:]-pred[:,:-1]
    ty = target[:,1:]-target[:,:-1]
    pair_mask = mask[:,1:]*mask[:,:-1]
    derivative = (((dy-ty)**2)*pair_mask).sum()/pair_mask.sum()
    return base + derivative_weight*derivative


@torch.no_grad()
def predict(model, x, mean, std, device):
    model.eval()
    normalized = ((x-mean)/std).astype("float32")
    # Average overlapping 128-frame windows; edge values have one prediction.
    size, stride = 128, 64
    starts = list(range(0, max(len(x)-size+1,1), stride))
    if not starts or starts[-1] != max(0,len(x)-size):
        starts.append(max(0,len(x)-size))
    sums = np.zeros(len(x), dtype="float64"); counts = np.zeros(len(x), dtype="float64")
    for start in starts:
        chunk = normalized[start:start+size]
        length = len(chunk)
        if length < size:
            chunk = np.pad(chunk, ((0,size-length),(0,0)))
        tensor = torch.from_numpy(chunk[None]).to(device)
        result = model(tensor)[0,:length].cpu().numpy()
        sums[start:start+length] += result; counts[start:start+length] += 1
    return (sums/counts).astype("float32")


def evaluate(model, records, indices, mean, std, device):
    rows = []
    for i in indices:
        meta, x, y = records[i]
        pred = predict(model, x, mean, std, device)
        row = {**meta, **metrics(y,pred), "peak_f1": extrema_score(y,pred,+1)["f1"],
               "trough_f1": extrema_score(y,pred,-1)["f1"]}
        rows.append(row)
    return rows


def singer_median_r(rows):
    names = sorted({r["singer"] for r in rows})
    return float(np.median([np.median([r["r"] for r in rows if r["singer"]==name]) for name in names]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    records=recordings(args.cache)
    selected=[]; counts={}
    for i in sorted(range(len(records)),key=lambda j:(records[j][0]["singer"],int(records[j][0]["number"]))):
        name=records[i][0]["singer"]
        if counts.get(name,0)<2:
            selected.append(i); counts[name]=counts.get(name,0)+1
    groups=np.array([records[i][0]["singer"] for i in selected])
    outer_train, outer_test=next(GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=20260930).split(np.arange(len(selected)),groups=groups))
    train_pool=np.array(selected)[outer_train]; test=np.array(selected)[outer_test]
    pool_groups=np.array([records[i][0]["singer"] for i in train_pool])
    inner_train, inner_val=next(GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=20260931).split(train_pool,groups=pool_groups))
    train=train_pool[inner_train]; val=train_pool[inner_val]
    assert not ({records[i][0]["singer"] for i in train}&{records[i][0]["singer"] for i in val})
    assert not ({records[i][0]["singer"] for i in test}&{records[i][0]["singer"] for i in train_pool})
    train_x=np.concatenate([records[i][1] for i in train])
    mean=train_x.mean(axis=0); std=np.maximum(train_x.std(axis=0),1e-5)
    configs=[("tcn32",TCN,32,0.0),("tcn64_d",TCN,64,0.25),
             ("bilstm64",BiLSTM,64,0.0),("bilstm128_d",BiLSTM,128,0.25)]
    runs=[]; args.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    for label,kind,width,dweight in configs:
        random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
        model=kind(width).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        loader=DataLoader(Windows(records,train,mean,std),batch_size=32,shuffle=True,num_workers=0)
        best=-1e9; bad=0; history=[]; checkpoint=args.checkpoint_dir/f"{label}.pt"
        for epoch in range(1,31):
            model.train(); losses=[]
            for x,y,mask in loader:
                x=x.to(device); y=y.to(device); mask=mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred=model(x)
                loss=loss_fn(pred,y,mask,dweight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step(); losses.append(float(loss.detach().cpu()))
            val_rows=evaluate(model,records,val,mean,std,device)
            val_score=singer_median_r(val_rows)
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_singer_median_r":val_score})
            print(f"{label} epoch={epoch} train_loss={np.mean(losses):.4f} val_r={val_score:.3f}",flush=True)
            if val_score>best+0.002:
                best=val_score; bad=0
                torch.save({"model":model.state_dict(),"mean":mean,"std":std},checkpoint)
            else:
                bad+=1
            if bad>=5:
                break
        runs.append({"label":label,"kind":kind.__name__,"width":width,"derivative_weight":dweight,
                     "best_val_singer_median_r":best,"best_epoch":max(history,key=lambda q:q["val_singer_median_r"])["epoch"],
                     "history":history,"checkpoint":str(checkpoint)})
    selected_run=max(runs,key=lambda q:q["best_val_singer_median_r"])
    kind=TCN if selected_run["kind"]=="TCN" else BiLSTM
    all_train_x=np.concatenate([records[i][1] for i in train_pool])
    full_mean=all_train_x.mean(axis=0); full_std=np.maximum(all_train_x.std(axis=0),1e-5)
    random.seed(20260930); np.random.seed(20260930); torch.manual_seed(20260930); torch.cuda.manual_seed_all(20260930)
    model=kind(selected_run["width"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    full_loader=DataLoader(Windows(records,train_pool,full_mean,full_std),batch_size=32,shuffle=True,num_workers=0)
    for epoch in range(selected_run["best_epoch"]):
        model.train()
        for x,y,mask in full_loader:
            x=x.to(device); y=y.to(device); mask=mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y,mask,selected_run["derivative_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
        print(f"full retrain epoch {epoch+1}/{selected_run['best_epoch']}",flush=True)
    full_checkpoint=args.checkpoint_dir/f"{selected_run['label']}_full.pt"
    torch.save({"model":model.state_dict(),"mean":full_mean,"std":full_std},full_checkpoint)
    test_rows=evaluate(model,records,test,full_mean,full_std,device)
    result={"kind":"exploratory_sequence_screen_not_final","n_train":len(train),"n_val":len(val),"n_test":len(test),
            "train_singers":sorted({records[i][0]["singer"] for i in train}),
            "val_singers":sorted({records[i][0]["singer"] for i in val}),
            "test_singers":sorted({records[i][0]["singer"] for i in test}),
            "runs":runs,"selected":selected_run["label"],"full_checkpoint":str(full_checkpoint),
            "test_singer_median_r":singer_median_r(test_rows),
            "test_segment_median_r":float(np.median([r["r"] for r in test_rows])),
            "test_segment_median_nrmse":float(np.median([r["nrmse"] for r in test_rows])),
            "test_segment_median_peak_f1":float(np.median([r["peak_f1"] for r in test_rows])),
            "test_segment_median_trough_f1":float(np.median([r["trough_f1"] for r in test_rows])),
            "test_segments":test_rows}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("SELECTED",selected_run["label"],"TEST",result["test_segment_median_r"],result["test_segment_median_nrmse"],flush=True)


if __name__=="__main__":
    main()
