"""Plot only the two measured and predicted belt waveforms."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sequence_model_screen import recordings
from latent_belt_autoencoder_cv import BeltAE
from local_attention_voice_belt_cv import AudioMapper, predict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--cv-results",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    records=recordings(args.cache); cv=json.loads(args.cv_results.read_text(encoding="utf-8"))
    examples=[("P29","3","Typical held-out recording"),("P24","12","Difficult held-out recording")]
    fig,axes=plt.subplots(2,2,figsize=(15,8),constrained_layout=True)
    for row,(singer,number,description) in enumerate(examples):
        i=next(i for i,(m,_,_) in enumerate(records) if m["singer"]==singer and m["number"]==number)
        meta,x,y=records[i]
        fold=next(f for f in cv["folds"] if singer in f["test_singers"])
        state=torch.load(args.cv_results.parent/fold["checkpoint"],map_location="cpu",weights_only=False)
        ae_state=torch.load(args.cv_results.parent/fold["ae_checkpoint"],map_location="cpu",weights_only=False)
        mapper=AudioMapper(input_dim=x.shape[1],latent_dim=state["latent_dim"])
        mapper.load_state_dict(state["mapper"]); mapper.eval()
        ae=BeltAE(state["latent_dim"]); ae.load_state_dict(ae_state["model"]); ae.eval()
        pred=predict(mapper,ae,x,state["mean"],state["std"],"cpu")
        n=min(len(y),150 if row==0 else len(y)); t=np.arange(n)/5
        for col,label in enumerate(("A","B")):
            ax=axes[row,col]; r=np.corrcoef(y[:,col],pred[:,col])[0,1]
            ax.plot(t,y[:n,col],color="#235fa4",lw=1.6,label="Measured")
            ax.plot(t,pred[:n,col],color="#14946c",lw=1.3,label="Predicted")
            ax.set_title(f"{description}: {singer} #{number} — Belt {label}, r={r:.2f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Relative belt units")
            if row==0 and col==0: ax.legend()
    fig.suptitle("Local cross-attention: measured vs predicted belt A and B",fontsize=16,weight="bold")
    args.out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.out,dpi=180)
    print(args.out)


if __name__=="__main__": main()
