"""Plot preregistered high/low pilot examples with aligned predicted waveforms."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from breath_model_pilot import FS, context_features, fit_predict, read_pair


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
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=20260930).split(np.arange(len(pieces)), groups=groups))
    xx = [context_features(p[1], 5) for p in pieces]
    train_x = np.concatenate([xx[i] for i in tr]); train_y = np.concatenate([pieces[i][2] for i in tr])
    examples = [("P09", "1", "higher-scoring example"), ("P30", "5", "failure example")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False, constrained_layout=True)
    for ax, (singer, number, label) in zip(axes, examples):
        i = next(i for i in te if groups[i] == singer and pieces[i][0]["number"] == number)
        y = pieces[i][2]
        pred = fit_predict("hgb", {"learning_rate": 0.03, "max_leaf_nodes": 31}, train_x, train_y, xx[i])
        n = min(len(y), 30 * FS)
        t = np.arange(n) / FS
        ax.plot(t, y[:n], label="measured belt shape", linewidth=1.5)
        ax.plot(t, pred[:n], label="audio prediction", linewidth=1.3)
        ax.set_title(f"{label}: {singer} #{number}; full-segment r={np.corrcoef(y,pred)[0,1]:.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative belt units")
        ax.legend(loc="upper right")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(args.out)


if __name__ == "__main__":
    main()
