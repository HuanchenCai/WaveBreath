"""Two independently pretrained AEs and bidirectional low-rate latent bridges."""
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
                                        loss_fn as belt_loss, seed_all,
                                        train_epoch as train_belt_ae_epoch)
from sequence_model_screen import recordings
from voice_latent_autoencoder_cv import (VoiceAE, VoiceWindows,
                                         loss_fn as voice_loss,
                                         train_epoch as train_voice_ae_epoch)


class LatentBridge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(64, out_dim, 1))

    def forward(self, z):
        return self.network(z.transpose(1, 2)).transpose(1, 2)


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_inner_aes(records, train, voice_epochs, belt_epochs, voice_dim, belt_dim, device):
    xtr = np.concatenate([records[i][1] for i in train]).astype("float32")
    mean, std = xtr.mean(axis=0), np.maximum(xtr.std(axis=0), 1e-5)
    seed_all(20260930)
    voice_ae = VoiceAE(records[0][1].shape[1], voice_dim).to(device)
    opt = torch.optim.AdamW(voice_ae.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(VoiceWindows(records, train, mean, std), batch_size=32,
                        shuffle=True, num_workers=0)
    for _ in range(voice_epochs):
        train_voice_ae_epoch(voice_ae, loader, opt, device)
    seed_all(20260930)
    belt_ae = BeltAE(belt_dim).to(device)
    opt = torch.optim.AdamW(belt_ae.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(BeltWindows(records, train), batch_size=32,
                        shuffle=True, num_workers=0)
    for _ in range(belt_epochs):
        train_belt_ae_epoch(belt_ae, loader, opt, device)
    return freeze(voice_ae), freeze(belt_ae), mean, std


def latent_error(pred, target, mode):
    error = nn.functional.mse_loss(pred, target)
    if mode == "relative":
        error = error / torch.clamp(torch.mean(target.detach() ** 2), min=1e-5)
    return error


def train_epoch(v_to_b, b_to_v, voice_ae, belt_ae, loader, opt, device,
                latent_loss_mode):
    v_to_b.train()
    b_to_v.train()
    losses = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        with torch.no_grad():
            zv = voice_ae.encode(x)
            zb = belt_ae.encode(y)
        pred_zb = v_to_b(zv)
        pred_zv = b_to_v(zb)
        pred_y = belt_ae.decode(pred_zb)
        pred_x = voice_ae.decode(pred_zv)
        loss = (belt_loss(pred_y, y, mask)
                + .1 * voice_loss(pred_x, x, mask)
                + .05 * latent_error(pred_zb, zb, latent_loss_mode)
                + .02 * latent_error(pred_zv, zv, latent_loss_mode)
                + .02 * latent_error(v_to_b(pred_zv), zb, latent_loss_mode)
                + .005 * latent_error(b_to_v(pred_zb), zv, latent_loss_mode))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(v_to_b.parameters()) + list(b_to_v.parameters()), 1)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(v_to_b, voice_ae, belt_ae, x, mean, std, device):
    v_to_b.eval()
    voice_ae.eval()
    belt_ae.eval()
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
        x_tensor = torch.from_numpy(chunk[None]).to(device)
        pred = belt_ae.decode(v_to_b(voice_ae.encode(x_tensor)))[0, :n].cpu().numpy()
        taper = np.maximum(np.hanning(n), .05)
        sums[start:start + n] += pred * taper[:, None]
        weights[start:start + n] += taper
    return (sums / weights[:, None]).astype("float32")


def evaluate(v_to_b, voice_ae, belt_ae, records, indices, mean, std, device):
    result = []
    for i in indices:
        meta, x, y = records[i]
        pred = predict(v_to_b, voice_ae, belt_ae, x, mean, std, device)
        row = {**meta}
        for col, channel in enumerate(("A", "B")):
            row[channel] = {**metrics(y[:, col], pred[:, col]),
                            "peak_f1": extrema_score(y[:, col], pred[:, col], +1)["f1"],
                            "trough_f1": extrema_score(y[:, col], pred[:, col], -1)["f1"],
                            "std_ratio": float(np.std(pred[:, col]) / np.std(y[:, col]))}
        result.append(row)
    return result


@torch.no_grad()
def reverse_audit(b_to_v, voice_ae, belt_ae, records, indices, mean, std, device):
    b_to_v.eval()
    latent_mse, feature_mse, oracle_feature_mse, feature_power = [], [], [], []
    for i in indices:
        _, x, y = records[i]
        xx = ((x - mean) / std).astype("float32")
        starts = list(range(0, max(len(xx) - SIZE + 1, 1), STEP))
        if not starts or starts[-1] != max(0, len(xx) - SIZE):
            starts.append(max(0, len(xx) - SIZE))
        for start in starts:
            xc, yc = xx[start:start + SIZE], y[start:start + SIZE]
            n = len(xc)
            if n < SIZE:
                xc = np.pad(xc, ((0, SIZE - n), (0, 0)))
                yc = np.pad(yc, ((0, SIZE - n), (0, 0)))
            xt = torch.from_numpy(xc[None]).to(device)
            yt = torch.from_numpy(yc[None].astype("float32")).to(device)
            zv = voice_ae.encode(xt)
            pred_zv = b_to_v(belt_ae.encode(yt))
            pred_x = voice_ae.decode(pred_zv)[:, :n]
            oracle_x = voice_ae.decode(zv)[:, :n]
            latent_mse.append(float(nn.functional.mse_loss(pred_zv, zv).cpu()))
            feature_mse.append(float(nn.functional.mse_loss(pred_x, xt[:, :n]).cpu()))
            oracle_feature_mse.append(float(nn.functional.mse_loss(oracle_x, xt[:, :n]).cpu()))
            feature_power.append(float(torch.mean(xt[:, :n] ** 2).cpu()))
    return {"latent_mse_window_median": float(np.median(latent_mse)),
            "voice_feature_mse_window_median": float(np.median(feature_mse)),
            "voice_feature_r2_vs_zero_window_median": float(np.median(
                1 - np.array(feature_mse) / np.maximum(np.array(feature_power), 1e-8))),
            "voice_ae_oracle_feature_r2_window_median": float(np.median(
                1 - np.array(oracle_feature_mse) / np.maximum(np.array(feature_power), 1e-8)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--voice-ae-results", type=Path, required=True)
    ap.add_argument("--belt-ae-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent-loss-mode", choices=("raw", "relative"), default="raw")
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = recordings(args.cache)
    voice_results = json.loads(args.voice_ae_results.read_text(encoding="utf-8"))
    belt_results = json.loads(args.belt_ae_results.read_text(encoding="utf-8"))
    voice_dim, belt_dim = voice_results["latent_dim"], belt_results["latent_dim"]
    singers = np.array([m["singer"] for m, _, _ in records])
    styles = np.array([m["style"] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
                 .split(np.arange(len(records)), styles, singers))
    assert len(voice_results["folds"]) == len(belt_results["folds"]) == 5
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold, (outer_train, outer_test) in enumerate(folds, 1):
        vf, bf = voice_results["folds"][fold - 1], belt_results["folds"][fold - 1]
        assert sorted(set(singers[outer_test])) == vf["test_singers"] == bf["test_singers"]
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                       random_state=20260930 + fold).split(
                           outer_train, groups=singers[outer_train]))
        train, val = outer_train[it], outer_train[iv]
        vae, bae, mean, std = make_inner_aes(records, train, vf["best_epoch"],
                                               bf["best_epoch"], voice_dim, belt_dim, device)
        seed_all(20260930)
        v_to_b = LatentBridge(voice_dim, belt_dim).to(device)
        b_to_v = LatentBridge(belt_dim, voice_dim).to(device)
        opt = torch.optim.AdamW(list(v_to_b.parameters()) + list(b_to_v.parameters()),
                                lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, train, mean, std), batch_size=32,
                            shuffle=True, num_workers=0)
        best, best_epoch, bad, history = -1e9, 1, 0, []
        for epoch in range(1, args.epochs + 1):
            tr = train_epoch(v_to_b, b_to_v, vae, bae, loader, opt, device,
                             args.latent_loss_mode)
            val_rows = evaluate(v_to_b, vae, bae, records, val, mean, std, device)
            score = singer_median(val_rows, "r")
            history.append({"epoch": epoch, "train_loss": tr,
                            "val_singer_median_AB_r": score})
            if score > best + .002:
                best, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
            if bad >= 5:
                break
        xf = np.concatenate([records[i][1] for i in outer_train]).astype("float32")
        fm, fs = xf.mean(axis=0), np.maximum(xf.std(axis=0), 1e-5)
        vst = torch.load(args.voice_ae_results.parent / vf["checkpoint"],
                         map_location=device, weights_only=False)
        bst = torch.load(args.belt_ae_results.parent / bf["checkpoint"],
                         map_location=device, weights_only=False)
        vae = VoiceAE(records[0][1].shape[1], voice_dim).to(device)
        vae.load_state_dict(vst["model"])
        bae = BeltAE(belt_dim).to(device)
        bae.load_state_dict(bst["model"])
        freeze(vae)
        freeze(bae)
        seed_all(20260930)
        v_to_b = LatentBridge(voice_dim, belt_dim).to(device)
        b_to_v = LatentBridge(belt_dim, voice_dim).to(device)
        opt = torch.optim.AdamW(list(v_to_b.parameters()) + list(b_to_v.parameters()),
                                lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, outer_train, fm, fs), batch_size=32,
                            shuffle=True, num_workers=0)
        for _ in range(best_epoch):
            train_epoch(v_to_b, b_to_v, vae, bae, loader, opt, device,
                        args.latent_loss_mode)
        checkpoint = args.checkpoint_dir / f"dual_latent_bridge_fold{fold}.pt"
        torch.save({"v_to_b": v_to_b.state_dict(), "b_to_v": b_to_v.state_dict(),
                    "voice_dim": voice_dim, "belt_dim": belt_dim,
                    "mean": fm, "std": fs, "epochs": best_epoch,
                    "latent_loss_mode": args.latent_loss_mode}, checkpoint)
        test_rows = evaluate(v_to_b, vae, bae, records, outer_test, fm, fs, device)
        reverse_stats = reverse_audit(b_to_v, vae, bae, records, outer_test, fm, fs, device)
        results.append({"fold": fold, "best_epoch": best_epoch,
                        "test_singers": sorted(set(singers[outer_test])),
                        "best_val_singer_median_AB_r": best,
                        "history": history, "test_segments": test_rows,
                        "reverse_belt_to_voice": reverse_stats,
                        "checkpoint": str(checkpoint.relative_to(args.out.parent)),
                        "voice_ae_checkpoint": vf["checkpoint"],
                        "belt_ae_checkpoint": bf["checkpoint"]})
        args.out.write_text(json.dumps({"kind": "dual_latent_bridge_running",
                                        "voice_dim": voice_dim, "belt_dim": belt_dim,
                                        "latent_loss_mode": args.latent_loss_mode,
                                        "completed_folds": fold, "folds": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fold {fold}/5 bridge epoch={best_epoch} test singer r={singer_median(test_rows, 'r'):.3f}", flush=True)
    rows = [r for f in results for r in f["test_segments"]]
    summary = {c: {k: float(np.median([r[c][k] for r in rows]))
                   for k in ("r", "nrmse", "peak_f1", "trough_f1", "std_ratio")}
               for c in ("A", "B")}
    summary["reverse_belt_to_voice"] = {
        key: float(np.median([f["reverse_belt_to_voice"][key] for f in results]))
        for key in ("latent_mse_window_median", "voice_feature_mse_window_median",
                    "voice_feature_r2_vs_zero_window_median",
                    "voice_ae_oracle_feature_r2_window_median")}
    args.out.write_text(json.dumps({"kind": "exploratory_independent_bidirectional_latent_bridge_cv",
                                    "voice_dim": voice_dim, "belt_dim": belt_dim,
                                    "latent_loss_mode": args.latent_loss_mode,
                                    "summary": summary, "folds": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
