"""Parallel Voice-latent to belt-latent map with train-only belt decoder."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from dual_belt_cv import DualWindows, singer_median
from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score
from latent_belt_autoencoder_cv import (BeltAE, BeltWindows, SIZE, STEP,
                                        loss_fn, seed_all, train_epoch as train_ae_epoch)
from sequence_model_screen import recordings


class AudioMapper(nn.Module):
    def __init__(self, input_dim=512, latent_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, latent_dim, 3, padding=1))

    def forward(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2)


def freeze_ae(ae):
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


def train_epoch(mapper, ae, loader, opt, device, latent_loss_weight):
    mapper.train()
    losses = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if latent_loss_weight:
            with torch.no_grad():
                target_z = ae.encode(y)
        opt.zero_grad(set_to_none=True)
        z = mapper(x)
        pred = ae.decode(z)
        loss = loss_fn(pred, y, mask)
        if latent_loss_weight:
            loss = loss + latent_loss_weight * nn.functional.mse_loss(z, target_z)
        loss.backward()
        nn.utils.clip_grad_norm_(mapper.parameters(), 1)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(mapper, ae, x, mean, std, device):
    mapper.eval()
    ae.eval()
    xx = ((x - mean) / std).astype("float32")
    starts = list(range(0, max(len(xx) - SIZE + 1, 1), STEP))
    if not starts or starts[-1] != max(0, len(xx) - SIZE):
        starts.append(max(0, len(xx) - SIZE))
    sums = np.zeros((len(xx), 2), dtype="float64")
    weights = np.zeros(len(xx), dtype="float64")
    for start in starts:
        chunk = xx[start:start + SIZE]
        n = len(chunk)
        if n < SIZE:
            chunk = np.pad(chunk, ((0, SIZE - n), (0, 0)))
        z = mapper(torch.from_numpy(chunk[None]).to(device))
        pred = ae.decode(z)[0, :n].cpu().numpy()
        taper = np.maximum(np.hanning(n), .05)
        sums[start:start + n] += pred * taper[:, None]
        weights[start:start + n] += taper
    return (sums / weights[:, None]).astype("float32")


def evaluate(mapper, ae, records, indices, mean, std, device):
    result = []
    for i in indices:
        meta, x, y = records[i]
        pred = predict(mapper, ae, x, mean, std, device)
        row = {**meta}
        for col, channel in enumerate(("A", "B")):
            row[channel] = {**metrics(y[:, col], pred[:, col]),
                            "peak_f1": extrema_score(y[:, col], pred[:, col], +1)["f1"],
                            "trough_f1": extrema_score(y[:, col], pred[:, col], -1)["f1"],
                            "std_ratio": float(np.std(pred[:, col]) / np.std(y[:, col]))}
        result.append(row)
    return result


def train_inner_ae(records, train, epochs, latent_dim, device):
    seed_all(20260930)
    ae = BeltAE(latent_dim).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(BeltWindows(records, train), batch_size=32,
                        shuffle=True, num_workers=0)
    for _ in range(epochs):
        train_ae_epoch(ae, loader, opt, device)
    return freeze_ae(ae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--ae-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent-loss-weight", type=float, default=.05)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = recordings(args.cache)
    ae_results = json.loads(args.ae_results.read_text(encoding="utf-8"))
    latent_dim = ae_results["latent_dim"]
    singers = np.array([m["singer"] for m, _, _ in records])
    styles = np.array([m["style"] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
                 .split(np.arange(len(records)), styles, singers))
    assert len(ae_results["folds"]) == 5
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold, (outer_train, outer_test) in enumerate(folds, 1):
        ae_fold = ae_results["folds"][fold - 1]
        assert sorted(set(singers[outer_test])) == ae_fold["test_singers"]
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                       random_state=20260930 + fold).split(
                           outer_train, groups=singers[outer_train]))
        train, val = outer_train[it], outer_train[iv]
        inner_ae = train_inner_ae(records, train, ae_fold["best_epoch"], latent_dim, device)
        xtr = np.concatenate([records[i][1] for i in train]).astype("float32")
        mean, std = xtr.mean(axis=0), np.maximum(xtr.std(axis=0), 1e-5)
        seed_all(20260930)
        mapper = AudioMapper(records[0][1].shape[1], latent_dim).to(device)
        opt = torch.optim.AdamW(mapper.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, train, mean, std), batch_size=32,
                            shuffle=True, num_workers=0)
        best, best_epoch, bad, history = -1e9, 1, 0, []
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(mapper, inner_ae, loader, opt, device,
                                     args.latent_loss_weight)
            val_rows = evaluate(mapper, inner_ae, records, val, mean, std, device)
            val_score = singer_median(val_rows, "r")
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_singer_median_AB_r": val_score})
            if val_score > best + .002:
                best, best_epoch, bad = val_score, epoch, 0
            else:
                bad += 1
            if bad >= 5:
                break
        xf = np.concatenate([records[i][1] for i in outer_train]).astype("float32")
        fm, fs = xf.mean(axis=0), np.maximum(xf.std(axis=0), 1e-5)
        state = torch.load(args.ae_results.parent / ae_fold["checkpoint"],
                           map_location=device, weights_only=False)
        final_ae = BeltAE(latent_dim).to(device)
        final_ae.load_state_dict(state["model"])
        freeze_ae(final_ae)
        seed_all(20260930)
        mapper = AudioMapper(records[0][1].shape[1], latent_dim).to(device)
        opt = torch.optim.AdamW(mapper.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, outer_train, fm, fs), batch_size=32,
                            shuffle=True, num_workers=0)
        for _ in range(best_epoch):
            train_epoch(mapper, final_ae, loader, opt, device, args.latent_loss_weight)
        checkpoint = args.checkpoint_dir / f"voice_to_belt_latent_fold{fold}.pt"
        torch.save({"mapper": mapper.state_dict(), "mean": fm, "std": fs,
                    "latent_dim": latent_dim, "epochs": best_epoch,
                    "latent_loss_weight": args.latent_loss_weight}, checkpoint)
        test_rows = evaluate(mapper, final_ae, records, outer_test, fm, fs, device)
        results.append({"fold": fold, "best_epoch": best_epoch,
                        "best_val_singer_median_AB_r": best,
                        "test_singers": sorted(set(singers[outer_test])),
                        "history": history, "test_segments": test_rows,
                        "checkpoint": str(checkpoint.relative_to(args.out.parent)),
                        "ae_checkpoint": ae_fold["checkpoint"]})
        args.out.write_text(json.dumps({"kind": "latent_voice_mapper_running",
                                        "latent_dim": latent_dim,
                                        "latent_loss_weight": args.latent_loss_weight,
                                        "completed_folds": fold,
                                        "folds": results}, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"fold {fold}/5 mapper epoch={best_epoch} test singer r={singer_median(test_rows, 'r'):.3f}", flush=True)
    rows = [r for f in results for r in f["test_segments"]]
    summary = {c: {k: float(np.median([r[c][k] for r in rows]))
                   for k in ("r", "nrmse", "peak_f1", "trough_f1", "std_ratio")}
               for c in ("A", "B")}
    args.out.write_text(json.dumps({"kind": "exploratory_parallel_voice_latent_belt_latent_cv",
                                    "latent_dim": latent_dim,
                                    "latent_loss_weight": args.latent_loss_weight,
                                    "summary": summary,
                                    "folds": results}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
