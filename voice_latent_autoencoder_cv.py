"""Independently train a parallel Voice autoencoder on training singers only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from latent_belt_autoencoder_cv import SIZE, STEP, seed_all
from sequence_model_screen import recordings


class VoiceWindows(Dataset):
    def __init__(self, records, indices, mean, std):
        self.records, self.mean, self.std = records, mean, std
        self.locations = []
        for i in indices:
            n = len(records[i][1])
            starts = list(range(0, max(n - SIZE + 1, 1), STEP))
            if not starts or starts[-1] != max(0, n - SIZE):
                starts.append(max(0, n - SIZE))
            self.locations.extend((i, s) for s in starts)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, j):
        i, start = self.locations[j]
        x = ((self.records[i][1][start:start + SIZE] - self.mean) / self.std).astype("float32")
        mask = np.ones(len(x), dtype="float32")
        if len(x) < SIZE:
            pad = SIZE - len(x)
            x = np.pad(x, ((0, pad), (0, 0)))
            mask = np.pad(mask, (0, pad))
        return torch.from_numpy(x), torch.from_numpy(mask)


class VoiceAE(nn.Module):
    def __init__(self, input_dim=512, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1), nn.GELU(),
            nn.Conv1d(128, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(128, 96, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(96, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, latent_dim, 1))
        self.start = nn.Sequential(nn.Conv1d(latent_dim, 64, 3, padding=1), nn.GELU())
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv1d(64, 96, 5, padding=2), nn.GELU()),
            nn.Sequential(nn.Conv1d(96, 128, 5, padding=2), nn.GELU()),
            nn.Sequential(nn.Conv1d(128, 128, 5, padding=2), nn.GELU())])
        self.end = nn.Conv1d(128, input_dim, 1)

    def encode(self, x):
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)

    def decode(self, z):
        h = self.start(z.transpose(1, 2))
        for block in self.up:
            h = block(nn.functional.interpolate(h, scale_factor=2,
                                                 mode="linear", align_corners=False))
        return self.end(h).transpose(1, 2)

    def forward(self, x):
        return self.decode(self.encode(x))


def loss_fn(pred, x, mask):
    return (((pred - x) ** 2) * mask[:, :, None]).sum() / (mask.sum() * x.shape[2])


def train_epoch(model, loader, opt, device):
    model.train()
    losses = []
    for x, mask in loader:
        x, mask = x.to(device), mask.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), x, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    return float(np.mean([float(loss_fn(model(x.to(device)), x.to(device), mask.to(device)).cpu())
                          for x, mask in loader]))


@torch.no_grad()
def reconstruct(model, x, mean, std, device):
    model.eval()
    xx = ((x - mean) / std).astype("float32")
    starts = list(range(0, max(len(xx) - SIZE + 1, 1), STEP))
    if not starts or starts[-1] != max(0, len(xx) - SIZE):
        starts.append(max(0, len(xx) - SIZE))
    sums = np.zeros_like(xx, dtype="float64")
    weights = np.zeros(len(xx), dtype="float64")
    for start in starts:
        chunk = xx[start:start + SIZE]
        n = len(chunk)
        if n < SIZE:
            chunk = np.pad(chunk, ((0, SIZE - n), (0, 0)))
        pred = model(torch.from_numpy(chunk[None]).to(device))[0, :n].cpu().numpy()
        taper = np.maximum(np.hanning(n), .05)
        sums[start:start + n] += pred * taper[:, None]
        weights[start:start + n] += taper
    return (sums / weights[:, None]).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
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
        xtr = np.concatenate([records[i][1] for i in train]).astype("float32")
        mean, std = xtr.mean(axis=0), np.maximum(xtr.std(axis=0), 1e-5)
        seed_all(20260930)
        model = VoiceAE(records[0][1].shape[1], args.latent_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        train_loader = DataLoader(VoiceWindows(records, train, mean, std), batch_size=32,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(VoiceWindows(records, val, mean, std), batch_size=32,
                                shuffle=False, num_workers=0)
        best, best_epoch, bad, history = float("inf"), 1, 0, []
        for epoch in range(1, args.epochs + 1):
            tr = train_epoch(model, train_loader, opt, device)
            va = validation_loss(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
            if va < best - .001:
                best, best_epoch, bad = va, epoch, 0
            else:
                bad += 1
            if bad >= 6:
                break
        xf = np.concatenate([records[i][1] for i in outer_train]).astype("float32")
        fm, fs = xf.mean(axis=0), np.maximum(xf.std(axis=0), 1e-5)
        seed_all(20260930)
        model = VoiceAE(records[0][1].shape[1], args.latent_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        full_loader = DataLoader(VoiceWindows(records, outer_train, fm, fs), batch_size=32,
                                 shuffle=True, num_workers=0)
        for _ in range(best_epoch):
            train_epoch(model, full_loader, opt, device)
        checkpoint = args.checkpoint_dir / f"voice_ae_fold{fold}.pt"
        torch.save({"model": model.state_dict(), "mean": fm, "std": fs,
                    "latent_dim": args.latent_dim, "epochs": best_epoch}, checkpoint)
        test_rows = []
        for i in outer_test:
            meta, x, _ = records[i]
            pred = reconstruct(model, x, fm, fs, device)
            target = ((x - fm) / fs).astype("float32")
            mse = float(np.mean((pred - target) ** 2))
            variance = float(np.mean(target ** 2))
            cosine = float(np.mean(np.sum(pred * target, axis=1)
                                   / np.maximum(np.linalg.norm(pred, axis=1)
                                                * np.linalg.norm(target, axis=1), 1e-8)))
            test_rows.append({**meta, "normalized_mse": mse,
                              "normalized_r2": 1 - mse / max(variance, 1e-8),
                              "frame_cosine": cosine})
        results.append({"fold": fold, "best_epoch": best_epoch,
                        "test_singers": sorted(set(singers[outer_test])),
                        "history": history, "test_segments": test_rows,
                        "checkpoint": str(checkpoint.relative_to(args.out.parent))})
        args.out.write_text(json.dumps({"kind": "voice_autoencoder_running",
                                        "latent_dim": args.latent_dim,
                                        "completed_folds": fold, "folds": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fold {fold}/5 Voice AE epoch={best_epoch}", flush=True)
    rows = [r for f in results for r in f["test_segments"]]
    summary = {k: float(np.median([r[k] for r in rows]))
               for k in ("normalized_mse", "normalized_r2", "frame_cosine")}
    args.out.write_text(json.dumps({"kind": "exploratory_voice_autoencoder_oracle_ceiling",
                                    "latent_dim": args.latent_dim,
                                    "summary": summary, "folds": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
