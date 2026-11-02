"""Exploratory five-fold singer-independent robustness check on all good pairs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.model_selection import StratifiedGroupKFold

from breath_model_pilot import context_features, fit_predict, read_pair
from evaluate_breath_pilot import metrics, vad_reconstruction


def extrema_score(y: np.ndarray, pred: np.ndarray, sign: int) -> dict:
    """Score corresponding positive maxima or negative minima within 0.5 s."""
    def locate(a):
        peaks, _ = signal.find_peaks(sign * a, distance=5, prominence=max(0.25, 0.5 * float(np.std(a))))
        return peaks
    true, estimated = locate(y), locate(pred)
    remaining = set(map(int, estimated))
    errors = []
    for t in true:
        eligible = [e for e in remaining if abs(e - int(t)) <= 2]
        if eligible:
            closest = min(eligible, key=lambda e: abs(e - int(t)))
            remaining.remove(closest)
            errors.append(abs(closest - int(t)) / 5)
    precision = len(errors) / max(len(estimated), 1)
    recall = len(errors) / max(len(true), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"f1": f1, "time_mae_s": float(np.mean(errors)) if errors else None,
            "true_count": len(true), "pred_count": len(estimated)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["suspect"] != "1" and float(r["duration_s"]) >= 10.0]
    pieces = []
    for j, row in enumerate(rows):
        x, y = read_pair(row)
        pieces.append((row, x, y, context_features(x, 5)))
        if (j + 1) % 20 == 0:
            print(f"loaded {j+1}/{len(rows)}", flush=True)
    singers = np.array([p[0]["singer_en"] for p in pieces])
    styles = np.array([p[0]["style"] for p in pieces])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
    fold_results = []
    for fold, (tr, te) in enumerate(cv.split(np.arange(len(pieces)), styles, singers), 1):
        assert not set(singers[tr]) & set(singers[te])
        xtr = np.concatenate([pieces[i][3] for i in tr]); ytr = np.concatenate([pieces[i][2] for i in tr])
        xte = np.concatenate([pieces[i][3] for i in te])
        vad_tr = np.concatenate([vad_reconstruction(pieces[i][1]) for i in tr])
        vad_gain = float(np.dot(vad_tr, ytr) / np.dot(vad_tr, vad_tr))
        candidates = {
            "ridge": {"name": "ridge", "params": {"alpha": 100.0}},
            "hgb": {"name": "hgb", "params": {"learning_rate": 0.03, "max_leaf_nodes": 31}},
        }
        predictions = {}
        for label, spec in candidates.items():
            flat = fit_predict(spec["name"], spec["params"], xtr, ytr, xte)
            cut = np.cumsum([len(pieces[i][2]) for i in te])[:-1]
            predictions[label] = dict(zip(te, np.split(flat, cut)))
        segment_results = []
        for i in te:
            row, x, y, _ = pieces[i]
            one = {"singer": singers[i], "style": styles[i], "number": row["number"], "seconds": round(len(y) / 5, 1)}
            one["zero"] = metrics(y, np.zeros_like(y))
            one["vad_integral"] = metrics(y, vad_gain * vad_reconstruction(x))
            for label in candidates:
                one[label] = metrics(y, predictions[label][i])
                one[label]["positive_maxima"] = extrema_score(y, predictions[label][i], +1)
                one[label]["negative_minima"] = extrema_score(y, predictions[label][i], -1)
            segment_results.append(one)
        fold_result = {"fold": fold, "train_singers": sorted(set(singers[tr])), "test_singers": sorted(set(singers[te])),
                       "vad_train_gain": vad_gain, "segments": segment_results}
        fold_results.append(fold_result)
        hgb_r = [s["hgb"]["r"] for s in segment_results]
        print(f"fold {fold}/5 test_singers={len(set(singers[te]))} median_segment_hgb_r={np.median(hgb_r):.3f}", flush=True)
    all_segments = [s for fold in fold_results for s in fold["segments"]]
    summary = {}
    for label in ("zero", "vad_integral", "ridge", "hgb"):
        values = [s[label]["r"] for s in all_segments if s[label]["r"] is not None]
        singer_values = []
        for singer in sorted(set(singers)):
            cur = [s[label]["r"] for s in all_segments if s["singer"] == singer and s[label]["r"] is not None]
            if cur:
                singer_values.append(float(np.median(cur)))
        summary[label] = {"median_segment_r": float(np.median(values)) if values else None,
                          "median_singer_r": float(np.median(singer_values)) if singer_values else None,
                          "median_segment_nrmse": float(np.median([s[label]["nrmse"] for s in all_segments]))}
        if label in ("ridge", "hgb"):
            for key in ("positive_maxima", "negative_minima"):
                summary[label][f"median_{key}_f1"] = float(np.median([s[label][key]["f1"] for s in all_segments]))
    result = {"kind": "exploratory_5fold_cv_not_final", "n_segments": len(pieces), "n_singers": len(set(singers)),
              "exclusions": ["1 prior SUSPECT belt", "1 segment shorter than 10 seconds (P14 #7, 9.002 s)"],
              "summary": summary, "folds": fold_results}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
