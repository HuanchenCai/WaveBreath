"""Singer-held-out residual temporal-attention TCN for belt A/B."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from dual_belt_cv import DualWindows, evaluate, singer_median, train_epoch
from sequence_model_screen import recordings


class FeatureTCN(nn.Module):
    def __init__(self, input_dim=512, width=32):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.ModuleList()
        for dilation in (1, 2, 4, 8):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
                nn.GELU(), nn.Dropout(.1), nn.Conv1d(width, width, 1)))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=.1)
        self.norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                         nn.GELU(), nn.Linear(2 * width, width))
        self.attention_gain = nn.Parameter(torch.tensor(.1))
        self.output = nn.Conv1d(width, 2, 1)

    @staticmethod
    def positions(length, width, device):
        t = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32)
                         * (-math.log(10000) / width))
        p = torch.zeros(length, width, device=device)
        p[:, 0::2] = torch.sin(t * freq)
        p[:, 1::2] = torch.cos(t * freq)
        return p

    def forward(self, x):
        h = self.input(x.transpose(1, 2))
        for block in self.blocks:
            h = torch.nn.functional.gelu(h + block(h))
        small = torch.nn.functional.avg_pool1d(h, kernel_size=4, stride=4).transpose(1, 2)
        small = small + self.positions(small.shape[1], small.shape[2], x.device)
        attended, _ = self.attention(small, small, small, need_weights=False)
        attended = self.norm(small + attended)
        attended = attended + self.feedforward(attended)
        context = torch.nn.functional.interpolate(attended.transpose(1, 2), size=h.shape[-1], mode="linear", align_corners=False)
        h = h + torch.tanh(self.attention_gain) * context
        return self.output(h).transpose(1, 2)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = recordings(args.cache)
    singers = np.array([m["singer"] for m, _, _ in records])
    styles = np.array([m["style"] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
                 .split(np.arange(len(records)), styles, singers))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold, (outer_train, outer_test) in enumerate(folds, 1):
        inner_train, inner_val = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                   random_state=20260930 + fold).split(
                                       outer_train, groups=singers[outer_train]))
        train, val = outer_train[inner_train], outer_train[inner_val]
        xtr = np.concatenate([records[i][1] for i in train]).astype("float32")
        mean, std = xtr.mean(axis=0), np.maximum(xtr.std(axis=0), 1e-5)
        seed_all(20260930)
        model = FeatureTCN(input_dim=records[0][1].shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, train, mean, std), batch_size=32,
                            shuffle=True, num_workers=0)
        best, best_epoch, bad, history = -1e9, 1, 0, []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, opt, device)
            score = singer_median(evaluate(model, records, val, mean, std, device), "r")
            history.append({"epoch": epoch, "train_loss": loss, "val_singer_median_AB_r": score})
            if score > best + .002:
                best, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
            if bad >= 5:
                break
        xf = np.concatenate([records[i][1] for i in outer_train]).astype("float32")
        fm, fs = xf.mean(axis=0), np.maximum(xf.std(axis=0), 1e-5)
        seed_all(20260930)
        model = FeatureTCN(input_dim=records[0][1].shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, outer_train, fm, fs), batch_size=32,
                            shuffle=True, num_workers=0)
        for _ in range(best_epoch):
            train_epoch(model, loader, opt, device)
        ckpt = args.checkpoint_dir / f"residual_attention_tcn_fold{fold}.pt"
        torch.save({"model": model.state_dict(), "mean": fm, "std": fs,
                    "epochs": best_epoch}, ckpt)
        test_rows = evaluate(model, records, outer_test, fm, fs, device)
        results.append({"fold": fold, "best_epoch": best_epoch,
                        "best_val_singer_median_AB_r": best,
                        "test_singers": sorted(set(singers[outer_test])),
                        "history": history, "test_segments": test_rows,
                        "checkpoint": str(ckpt.relative_to(args.out.parent))})
        args.out.write_text(json.dumps({"kind": "residual_attention_tcn_running",
                                        "completed_folds": fold, "folds": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fold {fold}/5 epoch={best_epoch} test AB singer r={singer_median(test_rows, 'r'):.3f}", flush=True)
    rows = [row for fold in results for row in fold["test_segments"]]
    summary = {c: {k: float(np.median([r[c][k] for r in rows]))
                   for k in ("r", "nrmse", "peak_f1", "trough_f1")}
               for c in ("A", "B")}
    args.out.write_text(json.dumps({"kind": "exploratory_residual_attention_tcn_cv_not_final",
                                    "summary": summary, "folds": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
