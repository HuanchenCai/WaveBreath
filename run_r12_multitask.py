"""Round-12 stage C: waveform + breath-event multi-task TCN.

Frozen design (see the round-10 parameter-evaluation results and decision):
- Backbone = FeatureTCN (WavLM-L16 features), output conv widened to 6 channels:
  waveform A/B, "inhaling" state A/B (logits), inhalation-onset A/B (logits).
- Event labels come from the frozen extractor (breath_parameters.inhalations) applied to the
  measured target of each TRAINING record; state = frames trough..peak; onset = Gaussian
  soft label sigma=1 frame at each trough (max 1).
- Loss = MSE + 0.2 slope MSE (original) + lambda * (BCE state + BCE onset), masked.
- lambda in {0, 0.3, 1}; lambda=0 uses the same runner, selection rule and seeds, so the
  comparison isolates the auxiliary supervision.
- Inner early stopping on inner-singer median of mean(onset F1 +-0.4 s from the waveform,
  CCC), averaged over A/B; max 20 epochs, patience 5, min 8 epochs; outer refit for the
  selected epoch count, exactly as the original runner. Seeds 20260930, +1, +2.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from breath_parameters import inhalations, record_parameters
from dual_belt_cv import DualWindows
from run_r11_stage_b import load_records
from waveform_family_models import waveform_loss
from whisper_feature_tcn_cv import FeatureTCN

ROOT = Path(__file__).resolve().parent
SIZE = 128


def event_labels(y):
    lab = np.zeros((len(y), 4), dtype='float32')
    t = np.arange(len(y))
    for k in range(2):
        for tr, pk, _ in inhalations(y[:, k]):
            lab[int(tr):int(pk) + 1, k] = 1
            lab[:, 2 + k] = np.maximum(lab[:, 2 + k], np.exp(-.5 * (t - tr) ** 2))
    return lab


class MultiTaskTCN(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = FeatureTCN(inputs)
        width = self.net.output.in_channels
        self.net.output = nn.Conv1d(width, 6, 1)

    def forward(self, x):
        return self.net(x)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_fn(out, y, mask, lam):
    loss = waveform_loss(out[:, :, :2], y[:, :, :2], mask)
    if lam:
        w = mask[:, :, None].expand(-1, -1, 4)
        bce = F.binary_cross_entropy_with_logits(out[:, :, 2:], y[:, :, 2:], reduction='none')
        loss = loss + lam * 2 * (bce * w).sum() / w.sum().clamp_min(1)
    return loss


@torch.no_grad()
def predict(model, records, indices, mean, std, device):
    model.eval()
    ds = DualWindows(records, indices, mean, std)
    sums = {int(i): np.zeros((len(records[i][2]), 6)) for i in indices}
    counts = {int(i): np.zeros(len(records[i][2])) for i in indices}
    off = 0
    for x, _, mask in DataLoader(ds, batch_size=64, shuffle=False):
        out = model(x.to(device)).float().cpu().numpy()
        out[:, :, 2:] = 1 / (1 + np.exp(-out[:, :, 2:]))
        for j, p in enumerate(out):
            i, s = ds.locations[off + j]; n = int(mask[j].sum())
            sums[i][s:s + n] += p[:n]; counts[i][s:s + n] += 1
        off += len(out)
    return {i: (sums[i] / counts[i][:, None]).astype('float32') for i in sums}


def score(records, preds):
    rows = []
    for i, p in preds.items():
        m, _, y = records[i]
        yw, pw = y[:, :2], p[:, :2]
        row = {**m, 'params': record_parameters(yw, pw)}
        for k, c in enumerate('AB'):
            a, b = yw[:, k], pw[:, k]
            ccc = 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8)
            row[c] = {'r': float(np.corrcoef(a, b)[0, 1]), 'ccc': float(ccc),
                      'nrmse': float(np.sqrt(np.mean((a - b) ** 2)) / (a.std() + 1e-8)),
                      'f1': row['params'][c]['f1_0.4s']}
        rows.append(row)
    return rows


def composite(rows):
    singers = sorted({r['singer'] for r in rows})
    return float(np.median([np.median([np.mean([(r[c]['f1'] + r[c]['ccc']) / 2 for c in 'AB'])
                                       for r in rows if r['singer'] == s]) for s in singers]))


def train_epoch(model, loader, opt, device, lam):
    model.train(); losses = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y, mask, lam)
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def normalize(records, idx):
    v = np.concatenate([records[i][1] for i in idx]).astype('float32')
    return v.mean(axis=0), np.maximum(v.std(axis=0), 1e-5)


def run(args, lam, seed, device):
    name = f'r12_{args.target}_lam{lam:g}_s{seed - 20260930}'
    out = ROOT / 'family_runs' / name
    out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    data = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {
        'experiment': name, 'target': args.target, 'lambda': lam, 'seed': seed, 'epochs_max': 20,
        'selection': 'inner singer median of mean(onset F1 0.4s, CCC)', 'folds': []}
    base = load_records(ROOT / f'wavlm_large_layer16_voice_AB_5hz.npz__{args.target}' if args.target != 'orig'
                        else ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz')
    records = [(m, x, np.c_[y, event_labels(y)].astype('float32')) for m, x, y in base]
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
                 .split(np.arange(len(records)), styles, singers))
    done = {f['fold'] for f in data['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic()
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold)
                      .split(otr, groups=singers[otr]))
        tr, va = otr[it], otr[iv]
        assert not set(singers[tr]) & set(singers[va]) and not set(singers[otr]) & set(singers[ote])
        mean, std = normalize(records, tr); seed_all(seed)
        model = MultiTaskTCN(records[0][1].shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, tr, mean, std), batch_size=32, shuffle=True)
        best, sel, bad, hist = -1e9, 1, 0, []
        for ep in range(1, 21):
            loss = train_epoch(model, loader, opt, device, lam)
            sc = composite(score(records, predict(model, records, va, mean, std, device)))
            hist.append({'epoch': ep, 'loss': loss, 'inner_composite': sc})
            if sc > best + .002:
                best, sel, bad = sc, ep, 0
            else:
                bad += 1
            if bad >= 5 and ep >= 8:
                break
        mean, std = normalize(records, otr); seed_all(seed)
        model = MultiTaskTCN(records[0][1].shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        loader = DataLoader(DualWindows(records, otr, mean, std), batch_size=32, shuffle=True)
        for _ in range(sel):
            train_epoch(model, loader, opt, device, lam)
        preds = predict(model, records, ote, mean, std, device)
        rows = score(records, preds)
        torch.save({'model': model.state_dict(), 'mean': mean, 'std': std, 'selected_epoch': sel,
                    'train_singers': sorted(set(singers[otr])), 'test_singers': sorted(set(singers[ote]))}, out / f'fold{fold}.pt')
        order = [int(i) for i in ote]
        np.savez_compressed(out / f'fold{fold}_waveforms.npz', prediction=np.concatenate([preds[i] for i in order]),
                            target=np.concatenate([records[i][2] for i in order]),
                            metadata=np.array(json.dumps([records[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        data['folds'].append({'fold': fold, 'selected_epoch': sel, 'best_inner': best, 'history': hist,
                              'test_singers': sorted(set(singers[ote])), 'test_segments': rows,
                              'elapsed_s': time.monotonic() - t0})
        rp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'{name} fold {fold} epoch={sel} composite={composite(rows):.4f}', flush=True)
    data['completed'] = True
    rp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='tgt_dc', choices=['orig', 'tgt_dc', 'tgt_dc_singer'])
    ap.add_argument('--lams', nargs='+', type=float, default=[0, .3, 1])
    ap.add_argument('--seeds', nargs='+', type=int, default=[20260930, 20260931, 20260932])
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for seed in args.seeds:
        for lam in args.lams:
            run(args, lam, seed, device)


if __name__ == '__main__':
    main()
