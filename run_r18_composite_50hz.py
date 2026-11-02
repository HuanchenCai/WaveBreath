"""Round 18: 50-Hz two-scale Conv-LSTM trained with a composite loss that combines
per-band waveform fidelity and chest-abdomen coordination (fixed before running; see
round-18 composite 50 Hz loss pre-registration).

Loss on each 50-Hz window (masked):
  L = [batch-1 loss on 10-frame means: MSE + 0.2 slope MSE]  + MSE_50            (as round 13 batch 3)
    + 0.3 * sum_band (1 - r_band)                                             (per-band shape)
    + 1.0 * [same waveform loss on the B-A difference signal]                 (relation)
    + w_ab * (r_AB(pred) - r_AB(true))^2                                      (synchrony level)
    + 1.0 * mean_lag (xcorr_AB(pred, lag) - xcorr_AB(true, lag))^2, lags -0.5..0.5 s step 0.1 s
                                                                              (who leads, by how much)
w_ab in {1, 3} chosen per outer fold on the inner singer split with the score
  S = singer-median[mean_band CCC_5Hz] - singer-median[|r_AB(pred) - r_AB(true)|]  (50 Hz)
Epochs: inner early stopping on S (patience 10, min 8, max 60); 3 seeds; outer 3-seed ensemble.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from run_r11_stage_b import load_records
from run_r13_50hz import CACHE, UP, TwoScale, Win50, metrics, normalize, pool5, predict
from run_r13_search import SEEDS, ccc, seed_all
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent
LAGS = list(range(-25, 26, 5))  # frames at 50 Hz = -0.5..0.5 s


def mcorr(a, b, w):
    a = a - (a * w).sum(1, keepdim=True); b = b - (b * w).sum(1, keepdim=True)
    return (a * b * w).sum(1) / torch.sqrt((a * a * w).sum(1) * (b * b * w).sum(1) + 1e-6)


def xcorr_curve(x, m):
    out = []
    t = x.shape[1]
    for L in LAGS:
        if L >= 0:
            a, b, w = x[:, :t - L, 0], x[:, L:, 1], m[:, :t - L] * m[:, L:]
        else:
            a, b, w = x[:, -L:, 0], x[:, :t + L, 1], m[:, -L:] * m[:, :t + L]
        w = w / w.sum(1, keepdim=True).clamp_min(1)
        out.append(mcorr(a, b, w))
    return torch.stack(out, 1)


def composite_loss(p, y, m, w_ab):
    m5 = (m.reshape(m.shape[0], -1, UP).mean(2) > .999).float()
    loss = waveform_loss(pool5(p), pool5(y), m5) + (((p - y) ** 2) * m[:, :, None]).sum() / (2 * m.sum().clamp_min(1))
    w = m / m.sum(1, keepdim=True).clamp_min(1)
    loss = loss + .3 * sum((1 - mcorr(p[:, :, k], y[:, :, k], w)).mean() for k in range(2))
    dp = (p[:, :, 1] - p[:, :, 0])[:, :, None].expand(-1, -1, 2); dy = (y[:, :, 1] - y[:, :, 0])[:, :, None].expand(-1, -1, 2)
    loss = loss + waveform_loss(pool5(dp), pool5(dy), m5) + (((dp - dy) ** 2) * m[:, :, None]).sum() / (2 * m.sum().clamp_min(1))
    loss = loss + w_ab * ((mcorr(p[:, :, 0], p[:, :, 1], w) - mcorr(y[:, :, 0], y[:, :, 1], w)) ** 2).mean()
    loss = loss + ((xcorr_curve(p, m) - xcorr_curve(y, m)) ** 2).mean()
    return loss


def inner_score(recs, preds):
    by_c, by_d = {}, {}
    for i, p in preds.items():
        meta, _, y = recs[i]
        y5, p5 = y.reshape(-1, UP, 2).mean(1), p.reshape(-1, UP, 2).mean(1)
        by_c.setdefault(meta['singer'], []).append(np.mean([ccc(y5[:, k], p5[:, k]) for k in range(2)]))
        by_d.setdefault(meta['singer'], []).append(abs(np.corrcoef(p.T)[0, 1] - np.corrcoef(y.T)[0, 1]))
    return float(np.median([np.median(v) for v in by_c.values()]) - np.median([np.median(v) for v in by_d.values()]))


def train(recs, tr, va, cfg, seed, w_ab, device, epochs=None):
    mean, std = normalize(recs, tr); seed_all(seed)
    model = TwoScale('convlstm', cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(Win50(recs, tr, mean, cfg['window']), batch_size=32, shuffle=True)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    best, sel, bad = -1e9, 1, 0
    for ep in range(1, (epochs or 60) + 1):
        model.train()
        for x, y, m in loader:
            x = (x.to(device).float() - mt) / st; y, m = y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            loss = composite_loss(model(x), y, m, w_ab); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if va is None:
            continue
        sc = inner_score(recs, predict(model, recs, va, mean, std, cfg['window'], device))
        if sc > best + .002:
            best, sel, bad = sc, ep, 0
        else:
            bad += 1
        if bad >= 10 and ep >= 8:
            break
    return (model, mean, std) if va is None else (best, sel)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    torch.set_num_threads(2); device = 'cuda'
    base = load_records(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz__tgt_dc')
    x50 = np.load(CACHE / 'wavlm_large_L16_50hz.f16.npy')[:, 0]
    with np.load(ROOT / 'r13_tgt_dc_50hz_y.npz') as z:
        y50, off = z['y'], z['offsets']
    recs = [(m, x50[off[i]:off[i + 1]], y50[off[i]:off[i + 1]]) for i, (m, _, _) in enumerate(base)]
    winners = {f['fold']: f['winner_config'] for f in json.loads((ROOT / 'family_runs' / 'r13_convlstm' / 'results.json').read_text(encoding='utf-8'))['folds']}
    if args.smoke:
        import torch as T
        p = T.randn(2, 1280, 2, requires_grad=True); y = T.randn(2, 1280, 2); m = T.ones(2, 1280); m[1, 1000:] = 0
        for w in (1, 3):
            l = composite_loss(p, y, m, w); l.backward(); print('w', w, float(l), bool(T.isfinite(p.grad).all()))
        print('perfect', float(composite_loss(y.clone(), y, m, 3)))
        return
    singers = np.array([m['singer'] for m, _, _ in recs]); styles = np.array([m['style'] for m, _, _ in recs])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(recs)), styles, singers))
    out = ROOT / 'family_runs' / 'r18_convlstm_composite_50hz'; out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {'name': out.name, 'folds': []}
    done = {f['fold'] for f in res['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg = winners[fold]
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold).split(otr, groups=singers[otr]))
        inner = {w: [train(recs, otr[it], otr[iv], cfg, s, w, device) for s in SEEDS] for w in (1, 3)}
        w_ab = max(inner, key=lambda w: np.mean([b for b, _ in inner[w]]))
        epochs = int(np.median([e for _, e in inner[w_ab]])); seeds = []
        for s in SEEDS:
            model, mean, std = train(recs, otr, None, cfg, s, w_ab, device, epochs=epochs)
            seeds.append(predict(model, recs, ote, mean, std, cfg['window'], device)); del model; torch.cuda.empty_cache()
        order = [int(i) for i in ote]; ens = {i: np.mean([p[i] for p in seeds], axis=0) for i in order}
        rows = [{**recs[i][0], 'hz50': metrics(recs[i][2], ens[i], 50),
                 'hz5': metrics(recs[i][2].reshape(-1, UP, 2).mean(1), ens[i].reshape(-1, UP, 2).mean(1), 5)} for i in order]
        np.savez_compressed(out / f'fold{fold}_waveforms_50hz.npz', prediction=np.concatenate([ens[i] for i in order]),
                            target=np.concatenate([recs[i][2] for i in order]), metadata=np.array(json.dumps([recs[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(recs[i][2]) for i in order])])
        np.savez_compressed(out / f'fold{fold}_waveforms.npz', prediction=np.concatenate([ens[i].reshape(-1, UP, 2).mean(1) for i in order]),
                            target=np.concatenate([recs[i][2].reshape(-1, UP, 2).mean(1) for i in order]),
                            metadata=np.array(json.dumps([recs[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(recs[i][2]) // UP for i in order])])
        res['folds'].append({'fold': fold, 'config': cfg, 'inner_scores': {str(w): [b for b, _ in v] for w, v in inner.items()},
                             'w_ab': w_ab, 'epochs': epochs, 'test_segments': rows, 'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== fold{fold} w_ab={w_ab} 5Hz A/B ccc={np.median([r["hz5"]["A"]["ccc"] for r in rows]):.4f}/'
              f'{np.median([r["hz5"]["B"]["ccc"] for r in rows]):.4f} {time.monotonic() - t0:.0f}s', flush=True)
    res['completed'] = True; rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
