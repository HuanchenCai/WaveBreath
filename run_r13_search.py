"""Round-13 batch 1: nested hyper-parameter search per waveform-head family.

Fixed before any run (also written to r13_protocol.json on first start):
- Data: WavLM-Large layer-16 features (5 Hz), target tgt_dc (DC removed, per-record standardized).
- Outer: the same 5 singer-grouped folds as all earlier rounds. Inner: 20% of outer-train
  singers held out (same GroupShuffleSplit seeds as before).
- Per family 12 configs: config 0 = reference setting (current main model for TCN,
  literature setting for the others); configs 1..11 sampled once with seed 20261001.
- Screening: every config, 1 seed, inner early stopping on inner singer-median CCC (A/B mean),
  max 60 epochs, patience 10, min 8, improvement threshold 0.002.
- Confirmation: top-3 screened configs get 2 more seeds; winner = highest mean inner CCC.
- Outer: winner retrained on the full outer-train for 3 seeds with the median selected epoch
  count; the reported prediction is the mean of the 3 seed models (ensemble). Single-seed
  predictions are saved too.
- Loss unchanged (MSE + 0.2 slope MSE). Breath parameters are evaluation only.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from breath_parameters import record_parameters
from dual_belt_cv import DualWindows
from r13_heads import build
from run_r11_stage_b import load_records
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent
SEEDS = (20260930, 20260931, 20260932)
COMMON = {'lr': [1e-4, 3e-4, 1e-3], 'dropout': [.1, .2, .3], 'weight_decay': [1e-4, 1e-3], 'window': [128, 256]}
SPACE = {'tcn': {'width': [32, 64, 128], 'blocks': [4, 5, 6, 7], 'kernel': [3, 5]},
         'bigru': {'hidden': [64, 128, 256], 'layers': [1, 2, 3]},
         'convlstm': {'channels': [64, 128], 'conv_layers': [1, 2], 'hidden': [64, 128], 'layers': [1, 2]},
         'transformer': {'width': [64, 128, 256], 'layers': [1, 2, 3, 4]},
         'unet': {'base': [16, 32, 64], 'depth': [2, 3, 4]}}
REFERENCE = {'tcn': {'width': 32, 'blocks': 4, 'kernel': 5},
             'bigru': {'hidden': 64, 'layers': 3},
             'convlstm': {'channels': 128, 'conv_layers': 2, 'hidden': 128, 'layers': 2},
             'transformer': {'width': 64, 'layers': 3},
             'unet': {'base': 32, 'depth': 3}}
REF_COMMON = {'lr': 3e-4, 'dropout': .1, 'weight_decay': 1e-4, 'window': 128}


def make_protocol():
    rng = random.Random(20261001)
    configs = {}
    for fam, space in SPACE.items():
        seen, cfgs = set(), [{**REF_COMMON, 'arch': REFERENCE[fam]}]
        seen.add(json.dumps(cfgs[0], sort_keys=True))
        while len(cfgs) < 12:
            c = {k: rng.choice(v) for k, v in COMMON.items()}
            c['arch'] = {k: rng.choice(v) for k, v in space.items()}
            key = json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key); cfgs.append(c)
        configs[fam] = cfgs
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in ('run_r13_search.py', 'r13_heads.py')}
    return {'round': 13, 'batch': 1, 'features': 'wavlm_large_layer16_voice_AB_5hz.npz', 'target': 'tgt_dc',
            'seeds': SEEDS, 'max_epochs': 60, 'patience': 10, 'min_epochs': 8, 'batch_size': 32,
            'selection': 'inner singer-median CCC (mean of A,B)', 'configs': configs, 'code_sha256_at_start': code}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def normalize(records, idx):
    v = np.concatenate([records[i][1] for i in idx]).astype('float32')
    return v.mean(axis=0), np.maximum(v.std(axis=0), 1e-5)


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


@torch.no_grad()
def predict(model, records, idx, mean, std, window, device):
    model.eval()
    ds = DualWindows(records, idx, mean, std, size=window, step=window // 2)
    sums = {int(i): np.zeros((len(records[i][2]), 2)) for i in idx}
    cnt = {int(i): np.zeros(len(records[i][2])) for i in idx}
    off = 0
    for x, _, mask in DataLoader(ds, batch_size=64, shuffle=False):
        out = model(x.to(device)).float().cpu().numpy()
        if not np.isfinite(out).all():
            raise RuntimeError('non-finite output')
        for j, p in enumerate(out):
            i, s = ds.locations[off + j]; n = int(mask[j].sum())
            sums[i][s:s + n] += p[:n]; cnt[i][s:s + n] += 1
        off += len(out)
    return {i: (sums[i] / cnt[i][:, None]).astype('float32') for i in sums}


def inner_score(records, preds):
    by = {}
    for i, p in preds.items():
        m, _, y = records[i]
        by.setdefault(m['singer'], []).append(np.mean([ccc(y[:, k], p[:, k]) for k in range(2)]))
    return float(np.median([np.median(v) for v in by.values()]))


def train(records, tr, va, cfg, family, seed, device, epochs=None):
    """If va is given: early stopping, returns (best_score, selected_epoch, history).
    Else: train exactly `epochs` epochs and return the model and normalization."""
    mean, std = normalize(records, tr)
    seed_all(seed)
    model = build(family, records[0][1].shape[1], {**cfg['arch'], 'dropout': cfg['dropout']}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    loader = DataLoader(DualWindows(records, tr, mean, std, size=cfg['window'], step=cfg['window'] // 2),
                        batch_size=32, shuffle=True)
    best, sel, bad, hist = -1e9, 1, 0, []
    for ep in range(1, (epochs or 60) + 1):
        model.train(); losses = []
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = waveform_loss(model(x), y, mask)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            losses.append(float(loss.detach()))
        if va is None:
            continue
        sc = inner_score(records, predict(model, records, va, mean, std, cfg['window'], device))
        hist.append({'epoch': ep, 'loss': float(np.mean(losses)), 'inner_ccc': sc})
        if sc > best + .002:
            best, sel, bad = sc, ep, 0
        else:
            bad += 1
        if bad >= 10 and ep >= 8:
            break
    if va is None:
        return model, mean, std
    return best, sel, hist


def seg_metrics(y, p):
    row = {'params': record_parameters(y, p)}
    for k, c in enumerate('AB'):
        a, b = y[:, k], p[:, k]
        row[c] = {'r': float(np.corrcoef(a, b)[0, 1]), 'ccc': ccc(a, b),
                  'nrmse': float(np.sqrt(np.mean((a - b) ** 2)) / (a.std() + 1e-8))}
    return row


def run_family(family, protocol, records, folds, singers, device, tag):
    out = ROOT / 'family_runs' / f'r13_{tag}{family}'
    out.mkdir(parents=True, exist_ok=True)
    cfgs = protocol['configs'][family]
    log = out / 'inner_runs.json'
    runs = json.loads(log.read_text(encoding='utf-8')) if log.exists() else {}
    resp = out / 'results.json'
    res = json.loads(resp.read_text(encoding='utf-8')) if resp.exists() else {'family': family, 'folds': []}
    done = {f['fold'] for f in res['folds']}

    def inner(fold, tr, va, c, seed):
        key = f'{fold}|{c}|{seed}'
        if key not in runs:
            t0 = time.monotonic()
            best, sel, hist = train(records, tr, va, cfgs[c], family, seed, device)
            runs[key] = {'fold': fold, 'config': c, 'seed': seed, 'inner_ccc': best, 'epoch': sel,
                         'history': hist, 'seconds': time.monotonic() - t0}
            log.write_text(json.dumps(runs, indent=1), encoding='utf-8')
            print(f'{family} fold{fold} cfg{c} seed{seed} inner_ccc={best:.4f} ep={sel} {runs[key]["seconds"]:.0f}s', flush=True)
        return runs[key]

    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold).split(otr, groups=singers[otr]))
        tr, va = otr[it], otr[iv]
        assert not set(singers[tr]) & set(singers[va]) and not set(singers[otr]) & set(singers[ote])
        screen = [inner(fold, tr, va, c, SEEDS[0]) for c in range(len(cfgs))]
        top = sorted(range(len(cfgs)), key=lambda c: -screen[c]['inner_ccc'])[:3]
        conf = {c: [inner(fold, tr, va, c, s) for s in SEEDS] for c in top}
        winner = max(top, key=lambda c: np.mean([r['inner_ccc'] for r in conf[c]]))
        epochs = int(np.median([r['epoch'] for r in conf[winner]]))
        preds_seed = []
        for s in SEEDS:
            model, mean, std = train(records, otr, None, cfgs[winner], family, s, device, epochs=epochs)
            preds_seed.append(predict(model, records, ote, mean, std, cfgs[winner]['window'], device))
            del model; torch.cuda.empty_cache()
        order = [int(i) for i in ote]
        ens = {i: np.mean([p[i] for p in preds_seed], axis=0) for i in order}
        rows = [{**records[i][0], **seg_metrics(records[i][2], ens[i]),
                 'single_seed': [seg_metrics(records[i][2], p[i]) for p in preds_seed]} for i in order]
        np.savez_compressed(out / f'fold{fold}_waveforms.npz',
                            prediction=np.concatenate([ens[i] for i in order]),
                            prediction_seeds=np.stack([np.concatenate([p[i] for i in order]) for p in preds_seed]),
                            target=np.concatenate([records[i][2] for i in order]),
                            metadata=np.array(json.dumps([records[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        res['folds'].append({'fold': fold, 'screen_inner_ccc': [r['inner_ccc'] for r in screen], 'top3': top,
                             'confirm_inner_ccc': {str(c): [r['inner_ccc'] for r in conf[c]] for c in top},
                             'winner': winner, 'winner_config': cfgs[winner], 'refit_epochs': epochs,
                             'test_singers': sorted(set(singers[ote])), 'test_segments': rows})
        resp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== {family} fold{fold} winner cfg{winner} {cfgs[winner]} epochs={epochs} '
              f'test A/B ccc median={np.median([r["A"]["ccc"] for r in rows]):.4f}/{np.median([r["B"]["ccc"] for r in rows]):.4f}', flush=True)
    res['completed'] = True
    resp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'COMPLETE {family}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pp = ROOT / 'r13_protocol.json'
    protocol = make_protocol()
    if pp.exists():
        saved = json.loads(pp.read_text(encoding='utf-8'))
        assert saved['configs'] == json.loads(json.dumps(protocol['configs'])), 'protocol changed'
        protocol = saved
    elif not args.smoke:
        pp.write_text(json.dumps(protocol, indent=1), encoding='utf-8')
    records = load_records(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz__tgt_dc')
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(records)), styles, singers))
    if args.smoke:
        for fam in args.families:
            for c in (0, 5):
                cfg = protocol['configs'][fam][c]
                t0 = time.monotonic()
                otr = folds[0][0]
                m, _, _ = train(records, otr[:40], None, cfg, fam, SEEDS[0], device, epochs=2)
                print(fam, c, cfg, f'2 epochs on 40 recs: {time.monotonic() - t0:.1f}s', flush=True)
        return
    for fam in args.families:
        run_family(fam, protocol, records, folds, singers, device, '')


if __name__ == '__main__':
    main()
