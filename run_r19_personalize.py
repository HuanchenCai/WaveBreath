"""Round 19-B: per-singer calibration with one belt-recorded segment (see the round-19 pre-registration).

For each outer fold: retrain the batch-1 outer models of the family (same winner config and refit
epoch count as batch 1, 3 seeds). For each test singer the LONGEST recording is the calibration
segment; evaluation uses only that singer's remaining recordings, under three conditions:
  none     : base 3-seed ensemble
  linear   : per singer and band, least squares  y_band ~ a*predA + b*predB + c  fitted on the
             calibration segment (ensemble prediction), applied to the remaining recordings
  finetune : each seed model fine-tuned on the calibration segment (lr 1e-4, 20 epochs, default
             loss, all parameters), then ensembled
Outputs: family_runs/r19b_<family>_<condition>/fold*_waveforms.npz (remaining recordings only) and
r19b_<family>_results.json.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from run_r13_followup import Windows, load_input, predict, prep, train
from run_r13_search import SEEDS, seed_all, seg_metrics
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent


def finetune(model, records, idx, mean, std, cfg, device, seed):
    m = copy.deepcopy(model); seed_all(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=cfg['weight_decay'])
    loader = DataLoader(Windows(records, idx, mean, std, cfg['window'], max(cfg['window'] // 4, 8)), batch_size=8, shuffle=True)
    mt, st = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    for _ in range(20):
        m.train()
        for x, y, mask in loader:
            x, y, mask = prep(x, mt, st, device), y.to(device), mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss = waveform_loss(m(x), y, mask); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1); opt.step()
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', default='convlstm'); args = ap.parse_args()
    torch.set_num_threads(2); device = 'cuda'
    records = load_input('wavlm16', None)
    b1 = json.loads((ROOT / 'family_runs' / f'r13_{args.family}' / 'results.json').read_text(encoding='utf-8'))
    winners = {f['fold']: (f['winner_config'], f['refit_epochs']) for f in b1['folds']}
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(records)), styles, singers))
    conds = ('none', 'linear', 'finetune')
    outs = {c: ROOT / 'family_runs' / f'r19b_{args.family}_{c}' for c in conds}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    rp = ROOT / f'r19b_{args.family}_results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {'family': args.family, 'folds': []}
    done = {f['fold'] for f in res['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg, epochs = winners[fold]
        models = [train(records, otr, None, cfg, args.family, s, device, epochs=epochs) for s in SEEDS]
        calib = {s: max([int(i) for i in ote if singers[i] == s], key=lambda i: len(records[i][2])) for s in set(singers[ote])}
        rest = [int(i) for i in ote if int(i) not in calib.values()]
        preds = {c: {} for c in conds}; coef = {}
        for s, ci in calib.items():
            r_idx = [i for i in rest if singers[i] == s]
            if not r_idx:
                continue
            base = [predict(m, records, [ci] + r_idx, mean, std, cfg['window'], device) for m, mean, std in models]
            ens = {i: np.mean([b[i] for b in base], axis=0) for i in [ci] + r_idx}
            X = np.c_[ens[ci], np.ones(len(ens[ci]))]; W = np.linalg.lstsq(X, records[ci][2], rcond=None)[0]; coef[s] = W.tolist()
            fts = []
            for k, (m, mean, std) in enumerate(models):
                fm = finetune(m, records, [ci], mean, std, cfg, device, SEEDS[k])
                fts.append(predict(fm, records, r_idx, mean, std, cfg['window'], device)); del fm
            for i in r_idx:
                preds['none'][i] = ens[i]
                preds['linear'][i] = (np.c_[ens[i], np.ones(len(ens[i]))] @ W).astype('float32')
                preds['finetune'][i] = np.mean([f[i] for f in fts], axis=0)
        order = sorted(preds['none'])
        rows = []
        for i in order:
            rows.append({**records[i][0], **{c: seg_metrics(records[i][2], preds[c][i]) for c in conds}})
        for c in conds:
            np.savez_compressed(outs[c] / f'fold{fold}_waveforms.npz', prediction=np.concatenate([preds[c][i] for i in order]),
                                target=np.concatenate([records[i][2] for i in order]),
                                metadata=np.array(json.dumps([records[i][0] for i in order])),
                                offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        res['folds'].append({'fold': fold, 'calibration': {s: records[ci][0] for s, ci in calib.items()}, 'linear_coef': coef,
                             'rows': rows, 'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        msg = ' '.join(f'{c} {np.median([r[c]["A"]["ccc"] for r in rows]):.3f}/{np.median([r[c]["B"]["ccc"] for r in rows]):.3f}' for c in conds)
        print(f'== fold{fold} {msg} {time.monotonic() - t0:.0f}s', flush=True)
        del models; torch.cuda.empty_cache()
    res['completed'] = True; rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
