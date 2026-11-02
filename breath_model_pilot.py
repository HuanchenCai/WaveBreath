"""Small singer-held-out screening experiment; never treat its score as final."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterGrid
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

FS = 5


def read_pair(row: dict) -> tuple[np.ndarray, np.ndarray]:
    audio, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
    belt, br = sf.read(row["belt"], dtype="float32", always_2d=True)
    if sr != br or len(audio) != len(belt) or belt.shape[1] != 2:
        raise ValueError(f"Unaligned pair: {row['belt']}")
    # The first audio channel is Voice; the second is EGG and is excluded.
    voice = signal.resample_poly(audio[:, 0], 1, 20)
    belt50 = signal.resample_poly(belt, 1, 882, axis=0)
    filt = signal.butter(4, 1.5, fs=50, output="sos")
    belt50 = signal.sosfiltfilt(filt, belt50, axis=0)
    samples = min(len(voice) // 441, len(belt50) // 10)
    if samples < 50:
        raise ValueError("Short segment")
    v = voice[: samples * 441].reshape(samples, 441)
    target = belt50[: samples * 10].reshape(samples, 10, 2).mean(axis=1)
    # Relative waveform shape. This is a label definition, not amplitude prediction.
    target = (target - np.median(target, axis=0)) / np.maximum(np.std(target, axis=0), 1e-6)
    y = target.mean(axis=1)
    # 200-ms acoustic frames: energy, zero crossing, and broad-band spectrum.
    power = np.abs(np.fft.rfft(v * np.hanning(441), axis=1)) ** 2
    freqs = np.fft.rfftfreq(441, 1 / 2205)
    bands = [(0, 150), (150, 350), (350, 700), (700, 1100)]
    e = [np.log1p(power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)) for lo, hi in bands]
    rms = np.log1p(1000 * np.sqrt((v * v).mean(axis=1)))
    zcr = (np.signbit(v[:, 1:]) != np.signbit(v[:, :-1])).mean(axis=1)
    features = np.column_stack([rms, zcr, *e]).astype("float32")
    return features, y.astype("float32")


def context_features(features: np.ndarray, seconds: int) -> np.ndarray:
    offsets = range(-seconds * FS, seconds * FS + 1, FS)
    padded = np.pad(features, ((seconds * FS, seconds * FS), (0, 0)), mode="edge")
    return np.concatenate([padded[seconds * FS + i : seconds * FS + i + len(features)] for i in offsets], axis=1)


def score(y: np.ndarray, pred: np.ndarray, segment_ids: np.ndarray) -> float:
    vals = []
    for sid in np.unique(segment_ids):
        mask = segment_ids == sid
        if mask.sum() > 20 and np.std(pred[mask]) > 1e-8:
            vals.append(float(np.corrcoef(y[mask], pred[mask])[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def fit_predict(model_name: str, params: dict, xtr, ytr, xte) -> np.ndarray:
    if model_name == "ridge":
        model = Ridge(alpha=params["alpha"], solver="lsqr")
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=100, max_iter=150, l2_regularization=1.0,
            early_stopping=False, random_state=20260930,
        )
    model.fit(xtr, ytr)
    return model.predict(xte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-singer", type=int, default=2)
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected, counts = [], {}
    for row in sorted(rows, key=lambda r: (r["singer_en"], int(r["number"]))):
        singer = row["singer_en"]
        if row["suspect"] == "1" or counts.get(singer, 0) >= args.per_singer:
            continue
        selected.append(row)
        counts[singer] = counts.get(singer, 0) + 1
    pieces = []
    for idx, row in enumerate(selected):
        x, y = read_pair(row)
        pieces.append((row, x, y))
        print(f"loaded {idx+1}/{len(selected)} {row['singer_en']} #{row['number']} {len(y)/FS:.1f}s", flush=True)
    assert len(counts) == 32
    groups = np.array([p[0]["singer_en"] for p in pieces])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=20260930)
    train_seg, test_seg = next(splitter.split(np.arange(len(pieces)), groups=groups))
    train_singers = set(groups[train_seg]); test_singers = set(groups[test_seg])
    assert not train_singers.intersection(test_singers)
    data = {}
    for context in (2, 5, 10):
        xx = [context_features(p[1], context) for p in pieces]
        data[context] = xx
    candidates = [("ridge", p) for p in ParameterGrid({"context": [2, 5, 10], "alpha": [1.0, 10.0, 100.0]})]
    candidates += [("hgb", p) for p in ParameterGrid({"context": [2, 5], "learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31]})]
    inner = GroupKFold(n_splits=3)
    runs = []
    for model_name, params in candidates:
        xpieces = data[params["context"]]
        fold_scores = []
        for itr, iva in inner.split(train_seg, groups=groups[train_seg]):
            trseg, vaseg = train_seg[itr], train_seg[iva]
            xtr = np.concatenate([xpieces[i] for i in trseg]); ytr = np.concatenate([pieces[i][2] for i in trseg])
            xva = np.concatenate([xpieces[i] for i in vaseg]); yva = np.concatenate([pieces[i][2] for i in vaseg])
            ids = np.concatenate([np.full(len(pieces[i][2]), i) for i in vaseg])
            pred = fit_predict(model_name, params, xtr, ytr, xva)
            fold_scores.append(score(yva, pred, ids))
        result = {"model": model_name, "params": params, "inner_scores": fold_scores, "inner_mean": float(np.mean(fold_scores))}
        runs.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    best = max(runs, key=lambda r: r["inner_mean"])
    xpieces = data[best["params"]["context"]]
    xtr = np.concatenate([xpieces[i] for i in train_seg]); ytr = np.concatenate([pieces[i][2] for i in train_seg])
    xte = np.concatenate([xpieces[i] for i in test_seg]); yte = np.concatenate([pieces[i][2] for i in test_seg])
    ids = np.concatenate([np.full(len(pieces[i][2]), i) for i in test_seg])
    pred = fit_predict(best["model"], best["params"], xtr, ytr, xte)
    per_segment = []
    for i in test_seg:
        mask = ids == i
        per_segment.append({"singer": groups[i], "style": pieces[i][0]["style"], "number": pieces[i][0]["number"],
                            "r": score(yte[mask], pred[mask], ids[mask])})
    result = {"kind": "pilot_not_final", "n_segments": len(pieces), "n_singers": len(counts),
              "train_singers": sorted(train_singers), "test_singers": sorted(test_singers),
              "grid": runs, "selected": best, "test_median_segment_r": score(yte, pred, ids), "test_segments": per_segment}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PILOT TEST", result["test_median_segment_r"], flush=True)


if __name__ == "__main__":
    main()
