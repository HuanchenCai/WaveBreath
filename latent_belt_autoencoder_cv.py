"""Train-only respiratory belt autoencoder; held-out reconstruction ceiling."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score
from sequence_model_screen import recordings


SIZE = 128
STEP = 64


class BeltWindows(Dataset):
    def __init__(self, records, indices):
        self.records = records
        self.locations = []
        for i in indices:
            n = len(records[i][2])
            starts = list(range(0, max(n - SIZE + 1, 1), STEP))
            if not starts or starts[-1] != max(0, n - SIZE):
                starts.append(max(0, n - SIZE))
            self.locations.extend((i, s) for s in starts)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, j):
        i, start = self.locations[j]
        y = self.records[i][2][start:start + SIZE].astype("float32")
        mask = np.ones(len(y), dtype="float32")
        if len(y) < SIZE:
            pad = SIZE - len(y)
            y = np.pad(y, ((0, pad), (0, 0)))
            mask = np.pad(mask, (0, pad))
        return torch.from_numpy(y), torch.from_numpy(mask)


class BeltAE(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.GELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, latent_dim, 5, stride=2, padding=2))
        self.up = nn.ModuleList([nn.Sequential(nn.Conv1d(32, 32, 5, padding=2),
                                                nn.GELU()) for _ in range(3)])
        self.start = nn.Sequential(nn.Conv1d(latent_dim, 32, 3, padding=1), nn.GELU())
        self.end = nn.Conv1d(32, 2, 7, padding=3)

    def encode(self, y):
        return self.encoder(y.transpose(1, 2)).transpose(1, 2)

    def decode(self, z):
        h = self.start(z.transpose(1, 2))
        for block in self.up:
            h = block(nn.functional.interpolate(h, scale_factor=2, mode="linear",
                                                 align_corners=False))
        return self.end(h).transpose(1, 2)

    def forward(self, y):
        return self.decode(self.encode(y))


def loss_fn(pred, y, mask):
    point = (((pred - y) ** 2) * mask[:, :, None]).sum() / (mask.sum() * 2)
    valid_diff = mask[:, 1:] * mask[:, :-1]
    derivative = (((pred[:, 1:] - pred[:, :-1] - (y[:, 1:] - y[:, :-1])) ** 2)
                  * valid_diff[:, :, None]).sum() / (valid_diff.sum() * 2)
    return point + .1 * derivative


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, opt, device):
    model.train()
    losses = []
    for y, mask in loader:
        y, mask = y.to(device), mask.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(y), y, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    values = []
    for y, mask in loader:
        y, mask = y.to(device), mask.to(device)
        values.append(float(loss_fn(model(y), y, mask).cpu()))
    return float(np.mean(values))


@torch.no_grad()
def reconstruct(model, y, device):
    model.eval()
    starts = list(range(0, max(len(y) - SIZE + 1, 1), STEP))
    if not starts or starts[-1] != max(0, len(y) - SIZE):
        starts.append(max(0, len(y) - SIZE))
    sums = np.zeros_like(y, dtype="float64")
    weights = np.zeros(len(y), dtype="float64")
    for start in starts:
        chunk = y[start:start + SIZE]
        n = len(chunk)
        if n < SIZE:
            chunk = np.pad(chunk, ((0, SIZE - n), (0, 0)))
        pred = model(torch.from_numpy(chunk[None].astype("float32")).to(device))[0, :n].cpu().numpy()
        taper = np.maximum(np.hanning(n), .05)
        sums[start:start + n] += pred * taper[:, None]
        weights[start:start + n] += taper
    return (sums / weights[:, None]).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--latent-dim", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
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
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                       random_state=20260930 + fold).split(
                           outer_train, groups=singers[outer_train]))
        train, val = outer_train[it], outer_train[iv]
        seed_all(20260930)
        model = BeltAE(args.latent_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train_loader = DataLoader(BeltWindows(records, train), batch_size=32,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(BeltWindows(records, val), batch_size=32,
                                shuffle=False, num_workers=0)
        best, best_epoch, bad, history = float("inf"), 1, 0, []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, opt, device)
            val_loss = validation_loss(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best - .001:
                best, best_epoch, bad = val_loss, epoch, 0
            else:
                bad += 1
            if bad >= 8:
                break
        seed_all(20260930)
        model = BeltAE(args.latent_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        full_loader = DataLoader(BeltWindows(records, outer_train), batch_size=32,
                                 shuffle=True, num_workers=0)
        for _ in range(best_epoch):
            train_epoch(model, full_loader, opt, device)
        checkpoint = args.checkpoint_dir / f"belt_ae_fold{fold}.pt"
        torch.save({"model": model.state_dict(), "latent_dim": args.latent_dim,
                    "epochs": best_epoch}, checkpoint)
        test_rows = []
        for i in outer_test:
            meta, _, y = records[i]
            pred = reconstruct(model, y, device)
            row = {**meta}
            for col, channel in enumerate(("A", "B")):
                row[channel] = {**metrics(y[:, col], pred[:, col]),
                                "peak_f1": extrema_score(y[:, col], pred[:, col], +1)["f1"],
                                "trough_f1": extrema_score(y[:, col], pred[:, col], -1)["f1"],
                                "std_ratio": float(np.std(pred[:, col]) / np.std(y[:, col]))}
            test_rows.append(row)
        results.append({"fold": fold, "best_epoch": best_epoch, "history": history,
                        "test_singers": sorted(set(singers[outer_test])),
                        "test_segments": test_rows,
                        "checkpoint": str(checkpoint.relative_to(args.out.parent))})
        args.out.write_text(json.dumps({"kind": "belt_autoencoder_running",
                                        "latent_dim": args.latent_dim,
                                        "completed_folds": fold, "folds": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fold {fold}/5 AE epoch={best_epoch}", flush=True)
    rows = [r for fold in results for r in fold["test_segments"]]
    summary = {c: {k: float(np.median([r[c][k] for r in rows]))
                   for k in ("r", "nrmse", "peak_f1", "trough_f1", "std_ratio")}
               for c in ("A", "B")}
    args.out.write_text(json.dumps({"kind": "exploratory_belt_autoencoder_oracle_ceiling",
                                    "latent_dim": args.latent_dim, "summary": summary,
                                    "folds": results}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
