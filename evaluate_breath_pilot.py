"""Compare pilot winner with held-out baselines on the identical singers."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit

from breath_model_pilot import FS, context_features, fit_predict, read_pair


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    r = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 1e-8 else None
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    nrmse = rmse / max(float(np.std(y)), 1e-8)
    return {"r": r, "nrmse": nrmse}


def vad_reconstruction(x: np.ndarray) -> np.ndarray:
    # x[:,0] is log(1 + 1000*rms): invert for a segment-level VAD threshold.
    rms = np.expm1(x[:, 0])
    voiced = rms > np.percentile(rms, 95) * 0.1
    silent = (~voiced).sum()
    flow = np.where(voiced, -1.0, voiced.sum() / max(silent, 1))
    reconstructed = signal.detrend(np.cumsum(flow)).astype("float32")
    return reconstructed / max(float(np.std(reconstructed)), 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected, counts = [], {}
    for row in sorted(rows, key=lambda r: (r["singer_en"], int(r["number"]))):
        singer = row["singer_en"]
        if row["suspect"] == "1" or counts.get(singer, 0) >= 2:
            continue
        selected.append(row)
        counts[singer] = counts.get(singer, 0) + 1
    pieces = [(row, *read_pair(row)) for row in selected]
    groups = np.array([p[0]["singer_en"] for p in pieces])
    train_seg, test_seg = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=20260930).split(np.arange(len(pieces)), groups=groups))
    assert not set(groups[train_seg]) & set(groups[test_seg])
    vad_train = np.concatenate([vad_reconstruction(pieces[i][1]) for i in train_seg])
    y_vad_train = np.concatenate([pieces[i][2] for i in train_seg])
    vad_gain = float(np.dot(vad_train, y_vad_train) / np.dot(vad_train, vad_train))
    models = {
        "ridge": {"name": "ridge", "params": {"context": 5, "alpha": 100.0}},
        "hgb": {"name": "hgb", "params": {"context": 5, "learning_rate": 0.03, "max_leaf_nodes": 31}},
    }
    predictions = {}
    for label, spec in models.items():
        xx = [context_features(p[1], spec["params"]["context"]) for p in pieces]
        xtr = np.concatenate([xx[i] for i in train_seg]); ytr = np.concatenate([pieces[i][2] for i in train_seg])
        xte = np.concatenate([xx[i] for i in test_seg])
        flat = fit_predict(spec["name"], spec["params"], xtr, ytr, xte)
        cut = np.cumsum([len(pieces[i][2]) for i in test_seg])[:-1]
        predictions[label] = dict(zip(test_seg, np.split(flat, cut)))
    per_segment = []
    for i in test_seg:
        row, x, y = pieces[i]
        one = {"singer": groups[i], "style": row["style"], "number": row["number"], "seconds": round(len(y) / FS, 1)}
        one["zero"] = metrics(y, np.zeros_like(y))
        one["vad_integral"] = metrics(y, vad_gain * vad_reconstruction(x))
        for label in models:
            one[label] = metrics(y, predictions[label][i])
        per_segment.append(one)
    summary = {}
    for label in ("zero", "vad_integral", "ridge", "hgb"):
        rs = [r[label]["r"] for r in per_segment if r[label]["r"] is not None]
        errs = [r[label]["nrmse"] for r in per_segment]
        summary[label] = {"median_r": float(np.median(rs)) if rs else None, "median_nrmse": float(np.median(errs))}
    output = {"kind": "pilot_baseline_comparison", "vad_train_gain": vad_gain, "summary": summary, "segments": per_segment}
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
