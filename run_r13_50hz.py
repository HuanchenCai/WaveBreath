"""Round-13 batch 3: 50-Hz output with a two-scale model (only the time resolution changes).

- Input: WavLM-Large layer 16 at its native 50 Hz (one encoder frame per 20-ms bin). The
  coarse branch sees the 10-frame mean = exactly the batch-1 5-Hz input.
- Coarse branch: batch-1 winner family/config of that fold, on 5-Hz pooled features.
- Fine branch: 1x1 conv (1024+2 -> 64) + 3 residual dilated convs (k=5, d=1,2,4) at 50 Hz,
  adding a residual to the linearly upsampled coarse output.
- Loss: batch-1 loss on the 10-frame mean of prediction vs target (MSE + 0.2 slope MSE at 5 Hz)
  + MSE at 50 Hz. Early stopping: inner singer-median CCC on the 5-Hz mean (as batch 1).
- Evaluation: 5-Hz pooled metrics (comparable to batch 1) and 50-Hz metrics including
  breath parameters with the same rules expressed in seconds (breath_parameters_fs).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from breath_parameters_fs import record_parameters_fs
from r13_heads import build
from run_r11_stage_b import load_records
from run_r13_search import SEEDS, ccc, seed_all
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent
CACHE = Path.home() / 'breath_r13_cache'
UP = 10


class Win50(Dataset):
    def __init__(self, recs, idx, mean, size5):
        self.r, self.mean, self.size = recs, mean.astype(np.float16), size5 * UP
        self.loc = []
        for i in idx:
            n = len(recs[i][2]); s = list(range(0, max(n - self.size + 1, 1), self.size // 2))
            if not s or s[-1] != max(0, n - self.size):
                s.append(max(0, n - self.size))
            self.loc += [(i, t) for t in s]

    def __len__(self):
        return len(self.loc)

    def __getitem__(self, j):
        i, s = self.loc[j]; _, x, y = self.r[i]
        xx = np.asarray(x[s:s + self.size], dtype=np.float16); yy = y[s:s + self.size].astype('float32')
        m = np.ones(len(yy), 'float32')
        if len(yy) < self.size:
            p = self.size - len(yy)
            xx = np.concatenate([xx, np.broadcast_to(self.mean, (p, xx.shape[1]))])
            yy = np.pad(yy, ((0, p), (0, 0))); m = np.pad(m, (0, p))
        return torch.from_numpy(np.ascontiguousarray(xx)), torch.from_numpy(yy), torch.from_numpy(m)


class TwoScale(nn.Module):
    def __init__(self, family, cfg, inputs=1024, width=64):
        super().__init__()
        self.coarse = build(family, inputs, {**cfg['arch'], 'dropout': cfg['dropout']})
        self.inp = nn.Conv1d(inputs + 2, width, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * d, dilation=d), nn.GELU(),
                                                   nn.Dropout(cfg['dropout']), nn.Conv1d(width, width, 1)) for d in (1, 2, 4)])
        self.out = nn.Conv1d(width, 2, 1)

    def forward(self, x50):
        b, t, d = x50.shape
        x5 = x50.reshape(b, t // UP, UP, d).mean(2)
        c5 = self.coarse(x5)                                              # (b, t/10, 2)
        c50 = F.interpolate(c5.transpose(1, 2), size=t, mode='linear', align_corners=False)
        h = self.inp(torch.cat([x50.transpose(1, 2), c50], 1))
        for blk in self.blocks:
            h = F.gelu(h + blk(h))
        return (c50 + self.out(h)).transpose(1, 2)


def pool5(a):
    return a.reshape(a.shape[0], -1, UP, a.shape[-1]).mean(2)


def loss_fn(p, y, m):
    m5 = (m.reshape(m.shape[0], -1, UP).mean(2) > .999).float()
    l5 = waveform_loss(pool5(p), pool5(y), m5)
    l50 = (((p - y) ** 2) * m[:, :, None]).sum() / (2 * m.sum().clamp_min(1))
    return l5 + l50


def normalize(recs, idx):
    s = np.zeros(recs[idx[0]][1].shape[1]); ss = np.zeros_like(s); n = 0
    for i in idx:
        v = np.asarray(recs[i][1], dtype='float64'); s += v.sum(0); ss += (v ** 2).sum(0); n += len(v)
    mean = s / n
    return mean.astype('float32'), np.maximum(np.sqrt(np.maximum(ss / n - mean ** 2, 0)), 1e-5).astype('float32')


@torch.no_grad()
def predict(model, recs, idx, mean, std, size5, device):
    model.eval(); ds = Win50(recs, idx, mean, size5)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    sums = {int(i): np.zeros((len(recs[i][2]), 2)) for i in idx}; cnt = {int(i): np.zeros(len(recs[i][2])) for i in idx}
    off = 0
    for x, _, m in DataLoader(ds, batch_size=16):
        out = model((x.to(device).float() - mt) / st).float().cpu().numpy()
        assert np.isfinite(out).all()
        for j, p in enumerate(out):
            i, s = ds.loc[off + j]; n = int(m[j].sum()); sums[i][s:s + n] += p[:n]; cnt[i][s:s + n] += 1
        off += len(out)
    return {i: (sums[i] / cnt[i][:, None]).astype('float32') for i in sums}


def score5(recs, preds):
    by = {}
    for i, p in preds.items():
        m, _, y = recs[i]; y5 = y.reshape(-1, UP, 2).mean(1); p5 = p.reshape(-1, UP, 2).mean(1)
        by.setdefault(m['singer'], []).append(np.mean([ccc(y5[:, k], p5[:, k]) for k in range(2)]))
    return float(np.median([np.median(v) for v in by.values()]))


def train(recs, tr, va, family, cfg, seed, device, epochs=None):
    mean, std = normalize(recs, tr); seed_all(seed)
    model = TwoScale(family, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(Win50(recs, tr, mean, cfg['window']), batch_size=32, shuffle=True)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    best, sel, bad = -1e9, 1, 0
    for ep in range(1, (epochs or 60) + 1):
        model.train()
        for x, y, m in loader:
            x = (x.to(device).float() - mt) / st; y, m = y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y, m); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if va is None:
            continue
        sc = score5(recs, predict(model, recs, va, mean, std, cfg['window'], device))
        if sc > best + .002:
            best, sel, bad = sc, ep, 0
        else:
            bad += 1
        if bad >= 10 and ep >= 8:
            break
    return (model, mean, std) if va is None else (best, sel)


def metrics(y, p, fs):
    row = {'params': record_parameters_fs(y, p, fs)}
    for k, c in enumerate('AB'):
        a, b = y[:, k], p[:, k]
        row[c] = {'r': float(np.corrcoef(a, b)[0, 1]), 'ccc': ccc(a, b), 'nrmse': float(np.sqrt(np.mean((a - b) ** 2)) / (a.std() + 1e-8))}
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True); args = ap.parse_args()
    torch.set_num_threads(2); device = 'cuda'
    base = load_records(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz__tgt_dc')
    x50 = np.load(CACHE / 'wavlm_large_L16_50hz.f16.npy')[:, 0]
    with np.load(ROOT / 'r13_tgt_dc_50hz_y.npz') as z:
        y50, off = z['y'], z['offsets']
    recs = [(m, x50[off[i]:off[i + 1]], y50[off[i]:off[i + 1]]) for i, (m, _, _) in enumerate(base)]
    batch1 = json.loads((ROOT / 'family_runs' / f'r13_{args.family}' / 'results.json').read_text(encoding='utf-8'))
    winners = {f['fold']: f['winner_config'] for f in batch1['folds']}; assert len(winners) == 5
    singers = np.array([m['singer'] for m, _, _ in recs]); styles = np.array([m['style'] for m, _, _ in recs])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(recs)), styles, singers))
    name = f'r13c_{args.family}_50hz'; out = ROOT / 'family_runs' / name; out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {'name': name, 'family': args.family, 'folds': []}
    done = {f['fold'] for f in res['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg = winners[fold]
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold).split(otr, groups=singers[otr]))
        inner = [train(recs, otr[it], otr[iv], args.family, cfg, s, device) for s in SEEDS]
        epochs = int(np.median([e for _, e in inner])); seeds = []
        for s in SEEDS:
            model, mean, std = train(recs, otr, None, args.family, cfg, s, device, epochs=epochs)
            seeds.append(predict(model, recs, ote, mean, std, cfg['window'], device)); del model; torch.cuda.empty_cache()
        order = [int(i) for i in ote]; ens = {i: np.mean([p[i] for p in seeds], axis=0) for i in order}
        rows = []
        for i in order:
            y = recs[i][2]; p = ens[i]
            rows.append({**recs[i][0], 'hz50': metrics(y, p, 50),
                         'hz5': metrics(y.reshape(-1, UP, 2).mean(1), p.reshape(-1, UP, 2).mean(1), 5)})
        np.savez_compressed(out / f'fold{fold}_waveforms_50hz.npz', prediction=np.concatenate([ens[i] for i in order]),
                            target=np.concatenate([recs[i][2] for i in order]),
                            metadata=np.array(json.dumps([recs[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(recs[i][2]) for i in order])])
        # 5-Hz pooled copy for the shared comparison scripts
        np.savez_compressed(out / f'fold{fold}_waveforms.npz',
                            prediction=np.concatenate([ens[i].reshape(-1, UP, 2).mean(1) for i in order]),
                            target=np.concatenate([recs[i][2].reshape(-1, UP, 2).mean(1) for i in order]),
                            metadata=np.array(json.dumps([recs[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(recs[i][2]) // UP for i in order])])
        res['folds'].append({'fold': fold, 'config': cfg, 'inner_ccc5': [b for b, _ in inner], 'epochs': epochs,
                             'test_segments': rows, 'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== {name} fold{fold} 5Hz A/B ccc={np.median([r["hz5"]["A"]["ccc"] for r in rows]):.4f}/'
              f'{np.median([r["hz5"]["B"]["ccc"] for r in rows]):.4f} {time.monotonic() - t0:.0f}s', flush=True)
    res['completed'] = True; rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'COMPLETE {name}', flush=True)


if __name__ == '__main__':
    main()
