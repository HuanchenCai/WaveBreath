"""Show the current framewise tree model and representative held-out failures."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.model_selection import GroupShuffleSplit

from breath_model_pilot import FS, context_features, fit_predict, read_pair
from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score


def box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                facecolor=color, edgecolor="#425466", linewidth=1.1))
    ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + w/2, y + h*0.30, detail, ha="center", va="center", fontsize=9)


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
    assert not set(groups[tr]) & set(groups[te])
    contexts = [context_features(p[1], 5) for p in pieces]
    train_x = np.concatenate([contexts[i] for i in tr])
    train_y = np.concatenate([pieces[i][2] for i in tr])

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 1.4, 1.4, 1.4])
    flow = fig.add_subplot(gs[0]); flow.set_xlim(0, 1); flow.set_ylim(0, 1); flow.axis("off")
    parts = [
        (0.01, "Voice channel", "44.1 kHz audio\nEGG excluded", "#e8f3fb"),
        (0.21, "Acoustic frames", "200 ms; 6 features\nRMS, ZCR, 4 bands", "#e8f3fb"),
        (0.42, "Time context", "-5 to +5 s, 1 s steps\n66 values / frame", "#e8f3fb"),
        (0.63, "Tree ensemble", "150 boosting rounds\n31 leaves, lr 0.03", "#fff0d9"),
        (0.83, "Output", "A/B relative mean\n5 samples / second", "#e6f6e9"),
    ]
    for x, title, detail, color in parts:
        box(flow, x, 0.24, 0.16, 0.54, title, detail, color)
    for a, b in zip(parts[:-1], parts[1:]):
        flow.add_patch(FancyArrowPatch((a[0]+0.17, 0.51), (b[0]-0.01, 0.51), arrowstyle="-|>", mutation_scale=15, linewidth=1.2))
    flow.text(0.5, 0.95, "Current model: framewise histogram gradient boosting (offline)", ha="center", va="top", fontsize=14, weight="bold")
    flow.text(0.5, 0.08, "Each time point is predicted separately; the model has no explicit breathing-cycle or waveform-continuity constraint.",
              ha="center", va="center", fontsize=10, color="#8c3b1e")

    examples = [
        ("P09", "1", "High-scoring example"),
        ("P29", "3", "Near-median example"),
        ("P30", "5", "Failure example"),
    ]
    for j, (singer, number, label) in enumerate(examples, start=1):
        ax = fig.add_subplot(gs[j])
        i = next(i for i in te if groups[i] == singer and pieces[i][0]["number"] == number)
        y = pieces[i][2]
        pred = fit_predict("hgb", {"learning_rate": 0.03, "max_leaf_nodes": 31}, train_x, train_y, contexts[i])
        m = metrics(y, pred)
        peak = extrema_score(y, pred, +1)["f1"]
        trough = extrema_score(y, pred, -1)["f1"]
        n = min(len(y), 30*FS)
        t = np.arange(n)/FS
        ax.plot(t, y[:n], color="#245da8", lw=1.8, label="Measured belt shape")
        ax.plot(t, pred[:n], color="#eb6a24", lw=1.4, label="Predicted from voice")
        ax.axhline(0, color="#8b8b8b", lw=0.5)
        ax.set_title(f"{label}: {singer} #{number}   full-segment r={m['r']:.2f}, NRMSE={m['nrmse']:.2f}, peak F1={peak:.2f}, trough F1={trough:.2f}", fontsize=11)
        ax.set_ylabel("Relative units")
        ax.set_xlabel("Time (seconds)")
        if j == 1:
            ax.legend(loc="upper right", ncol=2)
    fig.suptitle("What the current model learns—and where waveform matching breaks", fontsize=16, weight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print(args.out)


if __name__ == "__main__":
    main()
