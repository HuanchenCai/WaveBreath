"""Round-13 follow-up batches that change ONE factor relative to batch 1.

For each outer fold the architecture/optimiser config is the batch-1 winner of that family
and fold (family_runs/r13_<family>/results.json). Epoch count is re-selected on the inner
singer split (3 seeds, median of selected epochs, same early-stopping rule), then 3 seeds
are refit on the outer training set and ensembled. Only the factor below changes.

--input  wavlm16 (batch-1 input) | wavlm_all | mert_all | wavlm16_egg | egg_only (round 14)
         *_all = all 25 hidden layers of the frozen encoder, combined by a learnable softmax
         layer weighting (SUPERB-style); learned weights are saved.
--scheme joint (batch-1 training) | separate (one model per task: song / scale / reading,
         trained and tested only on that task) | tasklabel (joint + one-hot task input).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from r13_heads import build
from run_r11_stage_b import load_records
from run_r13_search import SEEDS, ccc, seed_all, seg_metrics
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent
CACHE = Path.home() / 'breath_r13_cache'
TASKS = ('song', 'scale', 'reading')


class Windows(Dataset):
    """Like dual_belt_cv.DualWindows but for 2-D (T,D) or 3-D (T,L,D) features."""
    def __init__(self, records, indices, mean, std, size, step):
        self.r, self.mean, self.std, self.size = records, mean, std, size
        self.loc = []
        for i in indices:
            n = len(records[i][2]); s = list(range(0, max(n - size + 1, 1), step))
            if not s or s[-1] != max(0, n - size):
                s.append(max(0, n - size))
            self.loc += [(i, t) for t in s]

    def __len__(self):
        return len(self.loc)

    def __getitem__(self, j):
        # Returns RAW features (normalization is applied on the GPU by `prep`); padded frames
        # are filled with the mean so they normalize to exactly 0, as in DualWindows.
        i, s = self.loc[j]; _, x, y = self.r[i]
        xx = np.asarray(x[s:s + self.size], dtype=np.float16)
        yy = y[s:s + self.size].astype('float32'); m = np.ones(len(yy), 'float32')
        if len(yy) < self.size:
            p = self.size - len(yy)
            xx = np.concatenate([xx, np.broadcast_to(self.mean.astype(np.float16), (p,) + xx.shape[1:])])
            yy = np.pad(yy, ((0, p), (0, 0))); m = np.pad(m, (0, p))
        return torch.from_numpy(np.ascontiguousarray(xx)), torch.from_numpy(yy), torch.from_numpy(m)


def prep(x, mean, std, device):
    return (x.to(device, non_blocking=True).float() - mean) / std


class LayerMix(nn.Module):
    def __init__(self, layers, head):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(layers)); self.head = head

    def forward(self, x):
        return self.head((x * torch.softmax(self.w, 0)[None, None, :, None]).sum(2))


_NORM = {}


def normalize(records, idx):
    key = (id(records), tuple(int(i) for i in idx))
    if key not in _NORM:
        _NORM[key] = _normalize(records, idx)
    return _NORM[key]


def _normalize(records, idx):
    x0 = records[idx[0]][1]
    s = np.zeros(x0.shape[1:], 'float64'); ss = np.zeros_like(s); n = 0
    for i in idx:
        v = np.asarray(records[i][1], dtype='float32')
        s += v.sum(0); ss += (v.astype('float64') ** 2).sum(0); n += len(v)
    mean = s / n
    return mean.astype('float32'), np.maximum(np.sqrt(np.maximum(ss / n - mean ** 2, 0)), 1e-5).astype('float32')


def make_model(family, cfg, x_shape):
    head_in = x_shape[-1]
    head = build(family, head_in, {**cfg['arch'], 'dropout': cfg['dropout']})
    return LayerMix(x_shape[0], head) if len(x_shape) == 2 else head


@torch.no_grad()
def predict(model, records, idx, mean, std, window, device):
    model.eval()
    ds = Windows(records, idx, mean, std, window, window // 2)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    sums = {int(i): np.zeros((len(records[i][2]), 2)) for i in idx}; cnt = {int(i): np.zeros(len(records[i][2])) for i in idx}
    off = 0
    for x, _, m in DataLoader(ds, batch_size=32, shuffle=False):
        out = model(prep(x, mt, st, device)).float().cpu().numpy()
        assert np.isfinite(out).all()
        for j, p in enumerate(out):
            i, s = ds.loc[off + j]; n = int(m[j].sum())
            sums[i][s:s + n] += p[:n]; cnt[i][s:s + n] += 1
        off += len(out)
    return {i: (sums[i] / cnt[i][:, None]).astype('float32') for i in sums}


def score(records, preds):
    by = {}
    for i, p in preds.items():
        m, _, y = records[i]; ks = range(2) if BAND is None else [BAND]
        by.setdefault(m['singer'], []).append(np.mean([ccc(y[:, k], p[:, k]) for k in ks]))
    return float(np.median([np.median(v) for v in by.values()]))


LOSS = 'default'
BAND = None  # round 20: train on one belt only (0 = A, 1 = B)
CORR_W = 1.0


def coupling_loss(pred, y, m, mode):
    """Round 17. 'diff': + waveform loss on the B-A difference signal (penalises chest-abdomen
    relation errors). 'diffcorr': 'diff' + squared error between predicted and measured A-B
    correlation in each window (masked)."""
    if BAND is not None:  # single-belt model: the loss sees only that belt (other output unused)
        sel = [BAND, BAND]
        return waveform_loss(pred[:, :, sel], y[:, :, sel], m)
    base = waveform_loss(pred, y, m)
    if mode == 'default':
        return base
    d_p = (pred[:, :, 1] - pred[:, :, 0])[:, :, None].expand(-1, -1, 2)
    d_y = (y[:, :, 1] - y[:, :, 0])[:, :, None].expand(-1, -1, 2)
    loss = base + waveform_loss(d_p, d_y, m)
    if mode == 'diffcorr':
        w = m / m.sum(1, keepdim=True).clamp_min(1)
        def corr(a, b):
            a = a - (a * w).sum(1, keepdim=True); b = b - (b * w).sum(1, keepdim=True)
            return (a * b * w).sum(1) / torch.sqrt((a * a * w).sum(1) * (b * b * w).sum(1) + 1e-6)
        loss = loss + CORR_W * ((corr(pred[:, :, 0], pred[:, :, 1]) - corr(y[:, :, 0], y[:, :, 1])) ** 2).mean()
    return loss


def train(records, tr, va, cfg, family, seed, device, epochs=None):
    mean, std = normalize(records, tr); seed_all(seed)
    model = make_model(family, cfg, records[0][1].shape[1:]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(Windows(records, tr, mean, std, cfg['window'], cfg['window'] // 2), batch_size=32, shuffle=True)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    best, sel, bad = -1e9, 1, 0
    for ep in range(1, (epochs or 60) + 1):
        model.train()
        for x, y, m in loader:
            x, y, m = prep(x, mt, st, device), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            loss = coupling_loss(model(x), y, m, LOSS)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
        if va is None:
            continue
        sc = score(records, predict(model, records, va, mean, std, cfg['window'], device))
        if sc > best + .002:
            best, sel, bad = sc, ep, 0
        else:
            bad += 1
        if bad >= 10 and ep >= 8:
            break
    return (model, mean, std) if va is None else (best, sel)


def load_input(kind, tasks):
    base = load_records(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz__tgt_dc')
    if kind == 'wavlm16':
        return base
    if kind in ('wavlm16_egg', 'egg_only'):  # round 14: 5 EGG glottal features (build_r14_egg_features.py)
        with np.load(ROOT / 'r14_egg_features.npz') as z:
            e, off = z['x5'].astype('float32'), z['offsets']
        if kind == 'egg_only':
            return [(m, e[off[i]:off[i + 1]], y) for i, (m, _, y) in enumerate(base)]
        return [(m, np.c_[x, e[off[i]:off[i + 1]]], y) for i, (m, x, y) in enumerate(base)]
    arr = np.load(CACHE / {'wavlm_all': 'wavlm_large_all_layers_5hz.f16.npy', 'mert_all': 'mert330m_all_layers_5hz.f16.npy'}[kind])
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        off = z['offsets']
    return [(m, arr[off[i]:off[i + 1]], y) for i, (m, _, y) in enumerate(base)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--input', default='wavlm16', choices=['wavlm16', 'wavlm_all', 'mert_all', 'wavlm16_egg', 'egg_only'])
    ap.add_argument('--scheme', default='joint', choices=['joint', 'separate', 'tasklabel'])
    ap.add_argument('--loss', default='default', choices=['default', 'diff', 'diffcorr'])
    ap.add_argument('--corr-weight', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=0, help='round 19: override window (frames), 0 = batch-1 winner')
    ap.add_argument('--band', default='', choices=['', 'A', 'B'])
    ap.add_argument('--train-frac', type=float, default=1.0, help='learning curve: fraction of outer-train singers')
    ap.add_argument('--draw', type=int, default=0, help='learning curve: subsample draw index')
    args = ap.parse_args()
    global LOSS, CORR_W, BAND
    LOSS = args.loss; CORR_W = args.corr_weight; BAND = None if not args.band else 'AB'.index(args.band)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    tasks = np.array([a['task'] for a in info])
    records = load_input(args.input, tasks)
    for (m, _, _), a in zip(records, info):
        assert m['singer'] == a['singer'] and str(m['number']) == str(a['number'])
    if args.scheme == 'tasklabel':
        assert records[0][1].ndim == 2, 'task label needs 2-D features'
        records = [(m, np.c_[x, np.tile(np.eye(3, dtype='float32')[TASKS.index(t)], (len(x), 1))], y)
                   for (m, x, y), t in zip(records, tasks)]
    batch1 = json.loads((ROOT / 'family_runs' / f'r13_{args.family}' / 'results.json').read_text(encoding='utf-8'))
    winners = {f['fold']: f['winner_config'] for f in batch1['folds']}
    assert len(winners) == 5, 'batch 1 must be complete for this family'
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(records)), styles, singers))
    name = f'r13b_{args.family}_{args.input}_{args.scheme}' + ('' if args.loss == 'default' else f'_loss{args.loss}') + ('' if args.corr_weight == 1.0 else f'_w{args.corr_weight:g}') + ('' if not args.window else f'_win{args.window}') + ('' if not args.band else f'_band{args.band}') + ('' if args.train_frac == 1.0 and not args.draw else f'_frac{args.train_frac:g}_d{args.draw}')
    out = ROOT / 'family_runs' / name; out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {
        'name': name, 'family': args.family, 'input': args.input, 'scheme': args.scheme,
        'rule': 'batch-1 winner config per fold; epochs re-selected on inner split (3 seeds); 3-seed outer ensemble', 'folds': []}
    done = {f['fold'] for f in res['folds']}
    groups = TASKS if args.scheme == 'separate' else ('all',)
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg = winners[fold] if not args.window else {**winners[fold], 'window': args.window}
        if args.train_frac < 1.0:  # learning curve: keep a random subset of outer-train singers (fixed per fold/draw)
            us = sorted(set(singers[otr])); rs = np.random.default_rng(1000 * fold + args.draw)
            keep = set(rs.choice(us, max(2, int(round(args.train_frac * len(us)))), replace=False))
            otr = np.array([i for i in otr if singers[i] in keep])
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold).split(otr, groups=singers[otr]))
        preds, detail, layer_w = {}, {}, []
        for g in groups:
            keep = (lambda idx: idx) if g == 'all' else (lambda idx, g=g: idx[tasks[idx] == g])
            tr, va, o_tr, o_te = keep(otr[it]), keep(otr[iv]), keep(otr), keep(ote)
            if len(o_te) == 0:
                continue
            inner = [train(records, tr, va, cfg, args.family, s, device) for s in SEEDS]
            epochs = int(np.median([e for _, e in inner]))
            seeds = []
            for s in SEEDS:
                model, mean, std = train(records, o_tr, None, cfg, args.family, s, device, epochs=epochs)
                seeds.append(predict(model, records, o_te, mean, std, cfg['window'], device))
                if isinstance(model, LayerMix):
                    layer_w.append(torch.softmax(model.w.detach(), 0).cpu().numpy().tolist())
                del model; torch.cuda.empty_cache()
            for i in o_te:
                preds[int(i)] = (np.mean([p[int(i)] for p in seeds], axis=0), [p[int(i)] for p in seeds])
            detail[g] = {'inner_ccc': [b for b, _ in inner], 'epochs': epochs, 'n_train': len(o_tr), 'n_test': len(o_te)}
        order = [int(i) for i in ote]
        rows = [{**records[i][0], 'task': tasks[i], **seg_metrics(records[i][2], preds[i][0]),
                 'single_seed': [seg_metrics(records[i][2], p) for p in preds[i][1]]} for i in order]
        np.savez_compressed(out / f'fold{fold}_waveforms.npz', prediction=np.concatenate([preds[i][0] for i in order]),
                            target=np.concatenate([records[i][2] for i in order]),
                            metadata=np.array(json.dumps([records[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        res['folds'].append({'fold': fold, 'config': cfg, 'groups': detail, 'layer_weights': layer_w,
                             'test_segments': rows, 'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== {name} fold{fold} A/B ccc median={np.median([r["A"]["ccc"] for r in rows]):.4f}/'
              f'{np.median([r["B"]["ccc"] for r in rows]):.4f} {time.monotonic() - t0:.0f}s', flush=True)
    res['completed'] = True
    rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'COMPLETE {name}', flush=True)


if __name__ == '__main__':
    main()
