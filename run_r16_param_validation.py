"""Round-16: parameter-level validation of EVERY model (evaluation only, no training).

For each model with complete 197-record outer-fold predictions:
  1. per record: measured and predicted parameter sets (breath_param_set.py) ->
     family_runs/<model>/parameter_set.csv
  2. agreement per parameter: CCC (Concordance Correlation Coefficient) across records,
     median bias (pred - meas), singer-level median absolute error with 2000-sample singer
     bootstrap 95% CI; pairwise event metrics as singer-level medians
  3. physiological relations R1-R3 per singer (as round 15) measured vs predicted
Round-12 seeds of one lambda are averaged into one model.
Outputs: r16_param_validation.json, round16_parameter_validation_report.md, r16_figures/*.png
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from breath_param_set import BANDS, pair_params, phrases
from compare_r11_r12 import load
from diag_r10_parameter_eval import boot_ci

ROOT = Path(__file__).resolve().parent
SIGNAL = ['rate_per_min', 'interval_s', 'insp_dur_s', 'depth_sd', 'exp_slope_sd_s']
DUAL = ['ab_corr', 'ab_lag_s', 'paradox_frac']
PAIR = ['f1_0.4s', 'f1_1.0s', 'onset_err_s', 'depth_rank_rho']
NAMES = {'rate_per_min': 'Breathing rate (breaths/min)', 'interval_s': 'Breath-group duration (s)', 'insp_dur_s': 'Inspiration duration (s)',
         'depth_sd': 'Inhalation depth (SD units)', 'exp_slope_sd_s': 'Within-phrase expiratory slope (SD/s)',
         'ab_corr': 'Chest-abdomen correlation', 'ab_lag_s': 'Chest-abdomen inhalation lag B-A (s)', 'paradox_frac': 'Paradoxical-motion fraction',
         'f1_0.4s': 'Onset F1 ±0.4 s', 'f1_1.0s': 'Onset F1 ±1.0 s', 'onset_err_s': 'Onset time error (s)',
         'depth_rank_rho': 'Within-segment inhalation-depth ranking (Spearman)'}


def model_list():
    out = []
    for d in sorted((ROOT / 'family_runs').iterdir()):
        if d.is_dir() and all((d / f'fold{k}_waveforms.npz').exists() for k in range(1, 6)) and not d.name.startswith('r12_'):
            out.append((d.name, [d.name]))
    for lam in ('0', '0.3', '1'):
        out.append((f'r12_tgt_dc_lam{lam}_seedmean', [f'r12_tgt_dc_lam{lam}_s{s}' for s in range(3)]))
    return out


def load_group(members):
    sets = [load(m) for m in members]
    return {k: (sets[0][k][0], np.mean([s[k][1] for s in sets], axis=0)) for k in sets[0]}


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-12))


def singer_level(rows, fn):
    by = {}
    for r in rows:
        v = fn(r)
        if v is not None and np.isfinite(v):
            by.setdefault(r['singer'], []).append(v)
    return {s: float(np.median(v)) for s, v in by.items()}


def relations(rows):
    """R1 depth~phrase dur (partial on pause), R2 slope~SPL, R3 depth~SPL: per-singer Spearman."""
    out = {}
    by = {}
    for r in rows:
        for ph in r['phrases']:
            by.setdefault(r['singer'], []).append(ph)
    for c in BANDS:
        for rel in ('R1', 'R2', 'R3'):
            cm, cp = [], []
            for s, phs in by.items():
                if len(phs) < 15:
                    continue
                def coef(src):
                    if rel == 'R1':
                        a, b, z = [p[f'{src}_{c}_depth'] for p in phs], [p['dur'] for p in phs], [p['pause'] for p in phs]
                        ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
                        res = lambda q: q - np.polyval(np.polyfit(rz, q, 1), rz)
                        return np.corrcoef(res(ra), res(rb))[0, 1]
                    if rel == 'R2':
                        return spearmanr([p[f'{src}_{c}_slope'] for p in phs], [p['spl'] for p in phs]).statistic
                    return spearmanr([p[f'{src}_{c}_depth'] for p in phs], [p['spl'] for p in phs]).statistic
                cm.append(coef('meas')); cp.append(coef('pred'))
            cm, cp = np.array(cm), np.array(cp); ok = np.isfinite(cm) & np.isfinite(cp); cm, cp = cm[ok], cp[ok]
            out[f'{rel}_{c}'] = {'meas_median': float(np.median(cm)), 'pred_median': float(np.median(cp)),
                                 'meas_pos': float((cm > 0).mean()), 'pred_pos': float((cp > 0).mean()),
                                 'across_singer_r': float(np.corrcoef(cm, cp)[0, 1]),
                                 'sign_agree': float((np.sign(cm) == np.sign(cp)).mean()), 'n_singers': int(len(cm))}
    return out


def evaluate(name, members, info, voiced_all, db_all, off):
    data = load_group(members)
    rows = []
    for i, a in enumerate(info):
        key = (a['singer'], str(a['number']))
        y, p = data[key]; v = voiced_all[off[i]:off[i + 1]]; db = db_all[off[i]:off[i + 1]]
        sy, sp, pair = pair_params(y, p, v)
        ref = np.median(db[v]) if v.any() else 0
        phs = []
        prev = None
        for s, e in phrases(v):
            if prev is not None and s - prev >= 1:
                ph = {'dur': (e - s) / 5, 'pause': (s - prev) / 5, 'spl': float(db[s:e].mean() - ref)}
                for src, w in (('meas', y), ('pred', p)):
                    for k, c in enumerate(BANDS):
                        sd = w[:, k].std() + 1e-9
                        ph[f'{src}_{c}_depth'] = float((w[s, k] - w[prev:s + 1, k].min()) / sd)
                        ph[f'{src}_{c}_slope'] = float(np.polyfit(np.arange(e - s) / 5, w[s:e, k] / sd, 1)[0])
                phs.append(ph)
            prev = e
        rows.append({'singer': a['singer'], 'number': str(a['number']), 'task': a['task'], 'meas': sy, 'pred': sp,
                     'pair': pair, 'phrases': phs})
    # parameter_set.csv
    out_dir = ROOT / 'family_runs' / name; out_dir.mkdir(exist_ok=True)
    with open(out_dir / 'parameter_set.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        head = ['singer', 'number', 'task']
        for c in BANDS:
            head += [f'{src}_{c}_{k}' for k in SIGNAL for src in ('meas', 'pred')] + [f'{c}_{k}' for k in PAIR]
        head += [f'{src}_{k}' for k in DUAL for src in ('meas', 'pred')]
        w.writerow(head)
        for r in rows:
            line = [r['singer'], r['number'], r['task']]
            for c in BANDS:
                line += [r[src][c][k] for k in SIGNAL for src in ('meas', 'pred')] + [r['pair'][c][k] for k in PAIR]
            line += [r[src]['dual'][k] for k in DUAL for src in ('meas', 'pred')]
            w.writerow([f'{x:.5g}' if isinstance(x, float) else x for x in line])
    # agreement
    summ = {'signal': {}, 'pair': {}, 'dual': {}}
    for c in BANDS:
        for k in SIGNAL:
            m = [r['meas'][c][k] for r in rows]; p = [r['pred'][c][k] for r in rows]
            err = singer_level(rows, lambda r: abs(r['pred'][c][k] - r['meas'][c][k]))
            med, lo, hi = boot_ci(list(err.values()))
            summ['signal'][f'{c}_{k}'] = {'ccc': ccc(m, p), 'bias': float(np.nanmedian(np.subtract(p, m))),
                                          'mae': med, 'mae_ci': [lo, hi], 'meas_median': float(np.nanmedian(m)),
                                          'pred_median': float(np.nanmedian(p))}
        for k in PAIR:
            s = singer_level(rows, lambda r: r['pair'][c][k])
            med, lo, hi = boot_ci(list(s.values()))
            summ['pair'][f'{c}_{k}'] = {'median': med, 'ci': [lo, hi]}
    for k in DUAL:
        m = [r['meas']['dual'][k] for r in rows]; p = [r['pred']['dual'][k] for r in rows]
        err = singer_level(rows, lambda r: abs(r['pred']['dual'][k] - r['meas']['dual'][k]))
        med, lo, hi = boot_ci(list(err.values()))
        summ['dual'][k] = {'ccc': ccc(m, p), 'bias': float(np.nanmedian(np.subtract(p, m))), 'mae': med, 'mae_ci': [lo, hi],
                           'meas_median': float(np.nanmedian(m)), 'pred_median': float(np.nanmedian(p))}
    summ['relations'] = relations(rows)
    # task-stratified CCC for the main record-level parameters
    summ['by_task'] = {}
    for t in ('song', 'scale', 'reading'):
        sub = [r for r in rows if r['task'] == t]
        summ['by_task'][t] = {f'{c}_{k}': ccc([r['meas'][c][k] for r in sub], [r['pred'][c][k] for r in sub])
                              for c in BANDS for k in SIGNAL}
        summ['by_task'][t].update({f'{c}_f1_0.4s': float(np.median(list(singer_level(sub, lambda r: r['pair'][c]['f1_0.4s']).values())))
                                   for c in BANDS})
    return summ, rows


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        voiced_all, off = z['voiced'], z['offsets']
    db_all = np.load(ROOT / 'diagnostics_r10' / 'mic_db_5hz.npy')
    res_path = ROOT / 'r16_param_validation.json'
    results = json.loads(res_path.read_text(encoding='utf-8')) if res_path.exists() else {}
    for name, members in model_list():
        if name in results:
            continue
        results[name], rows = evaluate(name, members, info, voiced_all, db_all, off)
        if name == 'r15_ens_convlstm_unet':
            (ROOT / 'r16_recommended_rows.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        if name == 'r11_tgt_dc':
            (ROOT / 'r16_baseline_rows.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
        res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
        print(name, 'done', flush=True)


if __name__ == '__main__':
    main()
