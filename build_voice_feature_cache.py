"""Extract 5-Hz acoustic features from Voice only, with no belt labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def acoustic_features(audio_path: str) -> np.ndarray:
    audio,sr=sf.read(audio_path,dtype="float32",always_2d=True)
    if sr!=44100 or audio.shape[1]<1:
        raise ValueError(audio_path)
    voice=signal.resample_poly(audio[:,0],1,20)
    n=len(voice)//441
    if n<50: raise ValueError(f"Short audio: {audio_path}")
    frames=voice[:n*441].reshape(n,441)
    power=np.abs(np.fft.rfft(frames*np.hanning(441),axis=1))**2
    freqs=np.fft.rfftfreq(441,1/2205)
    bands=[(0,150),(150,350),(350,700),(700,1100)]
    energies=[np.log1p(power[:,(freqs>=lo)&(freqs<hi)].sum(axis=1)) for lo,hi in bands]
    rms=np.log1p(1000*np.sqrt((frames*frames).mean(axis=1)))
    zcr=(np.signbit(frames[:,1:])!=np.signbit(frames[:,:-1])).mean(axis=1)
    return np.column_stack([rms,zcr,*energies]).astype("float32")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig",newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["suspect"]!="1" and float(r["duration_s"])>=10]
    xs=[]; offsets=[0]; meta=[]
    for i,row in enumerate(rows):
        x=acoustic_features(row["audio"]); xs.append(x); offsets.append(offsets[-1]+len(x))
        meta.append({"singer":row["singer_en"],"style":row["style"],"number":row["number"],"seconds":len(x)/5})
        if (i+1)%20==0: print(f"processed {i+1}/{len(rows)}",flush=True)
    np.savez_compressed(args.out,x=np.concatenate(xs),offsets=np.array(offsets),metadata=np.array(json.dumps(meta,ensure_ascii=False)))
    print(f"wrote Voice-only features: {len(rows)} files, {offsets[-1]} frames",flush=True)


if __name__=="__main__": main()
