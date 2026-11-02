"""Round 21: recover large inhalations (peak attenuation) with
  (a) conditional adversarial training (generator = batch-1 Conv-LSTM, discriminator sees belts +
      projected voice features; least-squares GAN), lambda_adv in {0 (control), 0.05, 0.2};
  (b) range matching: + lambda_rng * mean_band ((max-min)_pred - (max-min)_true)^2 per window.
All variants: batch-1 winner config per fold, original waveform loss, 3 seeds, outer 3-seed
ensemble. Inner early stopping for ALL variants (incl. control) on
  S = singer-median mean_band CCC - 0.5 * |1 - singer-median within-record depth slope|
(depth slope = regression of predicted on measured inhalation depth, matched onsets <=0.4 s).
See the round-21 large-inhalation pre-registration."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from breath_parameters import inhalations, match
from r13_heads import build
from run_r13_followup import Windows, load_input, normalize, predict, prep
from run_r13_search import SEEDS, ccc, seed_all, seg_metrics
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent


class Disc(nn.Module):
    """Multi-scale conditional 1-D discriminator (patch outputs)."""
    def __init__(self, cond=1024, ch=64):
        super().__init__()
        self.proj = nn.Conv1d(cond, 16, 1)

        def stack():
            return nn.Sequential(nn.Conv1d(18, ch, 5, 2, 2), nn.LeakyReLU(.2), nn.Conv1d(ch, ch, 5, 2, 2), nn.LeakyReLU(.2),
                                 nn.Conv1d(ch, ch, 5, 2, 2), nn.LeakyReLU(.2), nn.Conv1d(ch, 1, 3, 1, 1))
        self.d1, self.d2 = stack(), stack()

    def forward(self, belt, cond):
        h = torch.cat([belt.transpose(1, 2), self.proj(cond.transpose(1, 2))], 1)
        return [self.d1(h), self.d2(F.avg_pool1d(h, 2))]


def depth_slope(y, p):
    out = []
    for k in range(2):
        sd = y[:, k].std() + 1e-9; ey, ep = inhalations(y[:, k]), inhalations(p[:, k])
        m = match(ey[:, 0], ep[:, 0], 2)
        if len(m) < 4:
            continue
        dm = np.array([(y[int(ey[i, 1]), k] - y[int(ey[i, 0]), k]) / sd for i, _ in m])
        dp = np.array([(p[int(ep[j, 1]), k] - p[int(ep[j, 0]), k]) / sd for _, j in m])
        if dm.std() > 0:
            out.append(np.polyfit(dm, dp, 1)[0])
    return float(np.mean(out)) if out else np.nan


def inner_score(records, preds):
    c, s = {}, {}
    for i, p in preds.items():
        m, _, y = records[i]
        c.setdefault(m['singer'], []).append(np.mean([ccc(y[:, k], p[:, k]) for k in range(2)]))
        v = depth_slope(y, p)
        if np.isfinite(v):
            s.setdefault(m['singer'], []).append(v)
    cm = np.median([np.median(v) for v in c.values()]); sm = np.median([np.median(v) for v in s.values()]) if s else 0.
    return float(cm - .5 * abs(1 - sm))


def train(records, tr, va, cfg, seed, lam_adv, lam_rng, device, epochs=None):
    mean, std = normalize(records, tr); seed_all(seed)
    G = build('convlstm', records[0][1].shape[1], {**cfg['arch'], 'dropout': cfg['dropout']}).to(device)
    D = Disc(records[0][1].shape[1]).to(device) if lam_adv else None
    og = torch.optim.AdamW(G.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    od = torch.optim.AdamW(D.parameters(), lr=1e-4, betas=(.5, .9)) if D else None
    loader = DataLoader(Windows(records, tr, mean, std, cfg['window'], cfg['window'] // 2), batch_size=32, shuffle=True)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    best, sel, bad = -1e9, 1, 0
    for ep in range(1, (epochs or 60) + 1):
        G.train()
        for x, y, m in loader:
            x, y, m = prep(x, mt, st, device), y.to(device), m.to(device)
            fake = G(x); mm = m[:, :, None]
            if D is not None:
                od.zero_grad(set_to_none=True)
                dr, df = D(y * mm, x), D(fake.detach() * mm, x)
                ld = sum(((a - 1) ** 2).mean() + (b ** 2).mean() for a, b in zip(dr, df))
                ld.backward(); od.step()
            og.zero_grad(set_to_none=True)
            loss = waveform_loss(fake, y, m)
            if D is not None:
                loss = loss + lam_adv * sum(((a - 1) ** 2).mean() for a in D(fake * mm, x))
            if lam_rng:
                big = 1e3 * (1 - mm)
                rng_p = (fake - big).amax(1) - (fake + big).amin(1); rng_y = (y - big).amax(1) - (y + big).amin(1)
                loss = loss + lam_rng * ((rng_p - rng_y) ** 2).mean()
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 1); og.step()
        if va is None:
            continue
        sc = inner_score(records, predict(G, records, va, mean, std, cfg['window'], device))
        if sc > best + .002:
            best, sel, bad = sc, ep, 0
        else:
            bad += 1
        if bad >= 10 and ep >= 8:
            break
    return (G, mean, std) if va is None else (best, sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adv', type=float, default=0.); ap.add_argument('--rng', type=float, default=0.)
    args = ap.parse_args()
    torch.set_num_threads(2); device = 'cuda'
    records = load_input('wavlm16', None)
    b1 = json.loads((ROOT / 'family_runs' / 'r13_convlstm' / 'results.json').read_text(encoding='utf-8'))
    winners = {f['fold']: f['winner_config'] for f in b1['folds']}
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(records)), styles, singers))
    name = f'r21_convlstm_adv{args.adv:g}_rng{args.rng:g}'
    out = ROOT / 'family_runs' / name; out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {'name': name, 'adv': args.adv, 'rng': args.rng, 'folds': []}
    done = {f['fold'] for f in res['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg = winners[fold]
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold).split(otr, groups=singers[otr]))
        inner = [train(records, otr[it], otr[iv], cfg, s, args.adv, args.rng, device) for s in SEEDS]
        epochs = int(np.median([e for _, e in inner])); seeds = []
        for s in SEEDS:
            G, mean, std = train(records, otr, None, cfg, s, args.adv, args.rng, device, epochs=epochs)
            seeds.append(predict(G, records, ote, mean, std, cfg['window'], device)); del G; torch.cuda.empty_cache()
        order = [int(i) for i in ote]; ens = {i: np.mean([p[i] for p in seeds], axis=0) for i in order}
        rows = [{**records[i][0], **seg_metrics(records[i][2], ens[i])} for i in order]
        np.savez_compressed(out / f'fold{fold}_waveforms.npz', prediction=np.concatenate([ens[i] for i in order]),
                            prediction_seeds=np.stack([np.concatenate([p[i] for i in order]) for p in seeds]),
                            target=np.concatenate([records[i][2] for i in order]), metadata=np.array(json.dumps([records[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        res['folds'].append({'fold': fold, 'config': cfg, 'inner': [b for b, _ in inner], 'epochs': epochs, 'test_segments': rows,
                             'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== {name} fold{fold} ep={epochs} A/B ccc {np.median([r["A"]["ccc"] for r in rows]):.4f}/{np.median([r["B"]["ccc"] for r in rows]):.4f} {time.monotonic() - t0:.0f}s', flush=True)
    res['completed'] = True; rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
