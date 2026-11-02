"""Create separate A/B belt targets without re-extracting acoustic features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--voice-cache",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig",newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["suspect"]!="1" and float(r["duration_s"])>=10]
    voice=np.load(args.voice_cache,allow_pickle=False)
    meta=json.loads(str(voice["metadata"]))
    offsets=voice["offsets"]
    assert len(meta)==len(rows)
    targets=[]
    filt=signal.butter(4,1.5,fs=50,output="sos")
    for i,row in enumerate(rows):
        assert meta[i]["singer"]==row["singer_en"] and meta[i]["number"]==row["number"]
        belt,sr=sf.read(row["belt"],dtype="float32",always_2d=True)
        if sr!=44100 or belt.shape[1]!=2:
            raise ValueError(row["belt"])
        down=signal.resample_poly(belt,1,882,axis=0)
        down=signal.sosfiltfilt(filt,down,axis=0)
        n=int(offsets[i+1]-offsets[i])
        y=down[:n*10].reshape(n,10,2).mean(axis=1)
        y=(y-np.median(y,axis=0))/np.maximum(np.std(y,axis=0),1e-6)
        targets.append(y.astype("float32"))
        if (i+1)%20==0:
            print(f"processed {i+1}/{len(rows)}",flush=True)
    np.savez_compressed(args.out,x=voice["x"],y=np.concatenate(targets),offsets=offsets,metadata=voice["metadata"])
    print(f"wrote {args.out}: {len(rows)} recordings, two belt targets",flush=True)


if __name__=="__main__":
    main()
