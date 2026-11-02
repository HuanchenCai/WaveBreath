"""Concatenate frozen Whisper Voice embeddings and six handcrafted Voice features."""
import argparse
import hashlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", type=Path, required=True)
    ap.add_argument("--handcrafted", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    w = np.load(args.whisper, allow_pickle=False)
    h = np.load(args.handcrafted, allow_pickle=False)
    assert np.array_equal(w["offsets"], h["offsets"])
    assert np.array_equal(w["y"], h["y"])
    assert str(w["metadata"]) == str(h["metadata"])
    x = np.concatenate((w["x"], h["x"].astype("float16")), axis=1)
    assert np.isfinite(x).all() and x.shape[1] == 518
    np.savez_compressed(args.out, x=x, y=w["y"], offsets=w["offsets"],
                        metadata=w["metadata"],
                        feature_source=np.array("Whisper Base 512 + handcrafted Voice 6"))
    print(f"wrote {args.out}: {x.shape} sha256={hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
