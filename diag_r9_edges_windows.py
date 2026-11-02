"""Round-9 diagnostics with frozen outer-fold weights (no training, no label change).

1. Reproduce saved OOF predictions from fold checkpoints (equal-weight, step 64).
2. Decompose squared error by distance to record start/end and by position inside
   the 128-frame inference window (each window scored separately before stitching).
3. Inference-only stitching ablations, fixed before viewing their results:
   eq64  equal weights, step 64 (original)
   eq16  equal weights, step 16
   hann16  Hann-like weights sin^2(pi*(p+.5)/128), step 16; record ends are still
           covered because the weight at p=0 is small but positive and the per-frame
           denominator is the sum of the weights actually applied (never zero).
   core16  weight 1 inside window positions [16,112), 1e-3 elsewhere, step 16.
   Metrics are always computed on the full record with no windowing of the targets.
4. Error vs distance to the 30-s (150-frame) WavLM feature-extraction chunk boundary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_waveform_families import load_records
from waveform_family_models import make_model

SIZE = 128
VARIANTS = {'eq64': (64, 'equal'), 'eq16': (16, 'equal'), 'hann16': (16, 'hann'), 'core16': (16, 'core')}


def weights(kind):
    p = np.arange(SIZE)
    if kind == 'equal':
        return np.ones(SIZE)
    if kind == 'hann':
        return np.sin(np.pi * (p + .5) / SIZE) ** 2
    if kind == 'core':
        return np.where((p >= 16) & (p < 112), 1., 1e-3)
    raise ValueError(kind)


def starts_for(n, step):
    s = list(range(0, max(n - SIZE + 1, 1), step))
    if not s or s[-1] != max(0, n - SIZE):
        s.append(max(0, n - SIZE))
    return s


@torch.no_grad()
def window_outputs(model, x, mean, std, step, device):
    xx = ((x - mean) / std).astype('float32')
    out = []
    for s in starts_for(len(x), step):
        chunk = xx[s:s + SIZE]
        n = len(chunk)
        if n < SIZE:
            chunk = np.pad(chunk, ((0, SIZE - n), (0, 0)))
        p = model(torch.from_numpy(chunk[None]).to(device))[0, :n].float().cpu().numpy()
        out.append((s, p))
    return out


def stitch(outs, n, kind):
    w = weights(kind)
    num = np.zeros((n, 2))
    den = np.zeros(n)
    for s, p in outs:
        k = len(p)
        num[s:s + k] += p * w[:k, None]
        den[s:s + k] += w[:k]
    assert (den > 0).all()
    return (num / den[:, None]).astype('float32')


def score(y, p):
    r = float(np.corrcoef(y, p)[0, 1])
    ccc = float(2 * np.mean((y - y.mean()) * (p - p.mean())) / (y.var() + p.var() + (y.mean() - p.mean()) ** 2 + 1e-8))
    return {'r': r, 'ccc': ccc, 'rmse': float(np.sqrt(np.mean((y - p) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', default='wavlm_large16_tcn')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = args.root / 'family_runs' / args.config
    info = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    records = load_records(args.root / info['cache'])
    key = {(m['singer'], str(m['number']), m['style']): i for i, (m, _, _) in enumerate(records)}
    seg_rows, repro = [], []
    edge_se = {'start': np.zeros((150, 2)), 'end': np.zeros((150, 2)), 'n_start': np.zeros(150), 'n_end': np.zeros(150)}
    pos_se = np.zeros((SIZE, 2)); pos_n = np.zeros(SIZE)
    pos_se_interior = np.zeros((SIZE, 2)); pos_n_interior = np.zeros(SIZE)
    chunk_se = np.zeros((75, 2)); chunk_n = np.zeros(75)
    disagreement = []
    for fold in range(1, 6):
        ck = torch.load(run / f'fold{fold}.pt', map_location=device, weights_only=False)
        model = make_model(ck['family'], ck['input_dim']).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        with np.load(run / f'fold{fold}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']; saved = z['prediction']; saved_t = z['target']
        for j, m in enumerate(meta):
            i = key[(m['singer'], str(m['number']), m['style'])]
            _, x, y = records[i]
            assert np.allclose(y, saved_t[off[j]:off[j + 1]])
            assert m['singer'] in ck['test_singers'] and m['singer'] not in ck['train_singers']
            n = len(y)
            outs64 = window_outputs(model, x, ck['mean'], ck['std'], 64, device)
            outs16 = window_outputs(model, x, ck['mean'], ck['std'], 16, device)
            preds = {'eq64': stitch(outs64, n, 'equal'), 'eq16': stitch(outs16, n, 'equal'),
                     'hann16': stitch(outs16, n, 'hann'), 'core16': stitch(outs16, n, 'core')}
            repro.append(float(np.abs(preds['eq64'] - saved[off[j]:off[j + 1]]).max()))
            row = {**m, 'fold': fold, 'frames': n}
            for v, p in preds.items():
                row[v] = {c: score(y[:, k], p[:, k]) for k, c in enumerate('AB')}
            seg_rows.append(row)
            se = (preds['eq64'] - y) ** 2
            d = np.arange(n)
            for t in range(min(150, n)):
                edge_se['start'][t] += se[t]; edge_se['n_start'][t] += 1
                edge_se['end'][t] += se[n - 1 - t]; edge_se['n_end'][t] += 1
            # per-window position error, windows scored before stitching (step 16 grid)
            for s, p in outs16:
                k = len(p)
                e = (p - y[s:s + k]) ** 2
                pos_se[:k] += e; pos_n[:k] += 1
                # interior: frame at least 150 frames (30 s) from both record ends
                interior = (d[s:s + k] >= 150) & (d[s:s + k] < n - 150)
                pos_se_interior[:k][interior] += e[interior]; pos_n_interior[:k][interior] += 1
            # disagreement between overlapping step-16 windows, interior frames only
            stack = np.full((len(outs16), n, 2), np.nan)
            for q, (s, p) in enumerate(outs16):
                stack[q, s:s + len(p)] = p
            sd = np.nanstd(stack, axis=0)
            disagreement.append(float(np.nanmean(sd[150:n - 150])) if n > 300 else np.nan)
            # distance to 150-frame feature chunk boundary (interior frames, >=150 from ends)
            dist = np.minimum(d % 150, 150 - d % 150)
            for t in np.where((d >= 150) & (d < n - 150))[0]:
                if dist[t] < 75:
                    chunk_se[dist[t]] += se[t]; chunk_n[dist[t]] += 1
        del model
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    def med(v, c, k):
        return float(np.median([r[v][c][k] for r in seg_rows]))

    def singer_paired(v, c, k):
        singers = sorted({r['singer'] for r in seg_rows})
        diff = [np.median([r[v][c][k] - r['eq64'][c][k] for r in seg_rows if r['singer'] == s]) for s in singers]
        better = sum((d > 0) if k != 'rmse' else (d < 0) for d in diff)
        return {'median_diff': float(np.median(diff)), 'better_singers': int(better), 'n_singers': len(singers)}

    summary = {'config': args.config, 'n_segments': len(seg_rows), 'max_abs_repro_diff': float(max(repro)),
               'variants': {v: {c: {k: med(v, c, k) for k in ('r', 'ccc', 'rmse')} for c in 'AB'} for v in VARIANTS},
               'paired_vs_eq64': {v: {c: {k: singer_paired(v, c, k) for k in ('r', 'ccc', 'rmse')} for c in 'AB'}
                                  for v in VARIANTS if v != 'eq64'},
               'B_r_below_0.5': {v: sum(r[v]['B']['r'] < .5 for r in seg_rows) for v in VARIANTS},
               'mse_by_seconds_from_start': [(edge_se['start'][t] / edge_se['n_start'][t]).tolist() for t in range(0, 150, 5)],
               'mse_by_seconds_from_end': [(edge_se['end'][t] / edge_se['n_end'][t]).tolist() for t in range(0, 150, 5)],
               'mse_by_window_position_all': (pos_se / pos_n[:, None]).tolist(),
               'mse_by_window_position_interior': (pos_se_interior / pos_n_interior[:, None]).tolist(),
               'mse_by_distance_to_feature_chunk_boundary': (chunk_se / chunk_n[:, None]).tolist(),
               'median_window_disagreement_sd_interior': float(np.nanmedian(disagreement))}
    (out / f'{args.config}_edges_windows.json').write_text(json.dumps({'summary': summary, 'segments': seg_rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('n_segments', 'max_abs_repro_diff', 'variants', 'paired_vs_eq64', 'B_r_below_0.5', 'median_window_disagreement_sd_interior')}, indent=1))


if __name__ == '__main__':
    main()
