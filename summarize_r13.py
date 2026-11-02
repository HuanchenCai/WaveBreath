"""Summarize round 13 into round13_results_summary.md (re-runnable while jobs are still running).
Reference = r11_tgt_dc (original main model: WavLM-L16 + TCN, one fixed config, one seed).
All comparisons are singer-level medians with 2000-sample singer bootstrap 95% CIs."""
import json
from pathlib import Path

import numpy as np

from compare_r11_r12 import load, seg_metrics, singer_med
from diag_r10_parameter_eval import boot_ci

ROOT = Path(__file__).resolve().parent
REF = 'r11_tgt_dc'
LABEL = {'r13_tcn': 'Batch 1 TCN (tuned + 3-seed ensemble)', 'r13_bigru': 'Batch 1 BiGRU', 'r13_convlstm': 'Batch 1 Conv-LSTM',
         'r13_transformer': 'Batch 1 Transformer', 'r13_unet': 'Batch 1 U-Net'}
TASK = {(a['singer'], str(a['number'])): a['task'] for a in
        json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))}
METRICS = [('r', 'r'), ('ccc', 'CCC'), ('nrmse', 'NRMSE'), ('f1_0.4s', 'Onset F1±0.4s'),
           ('f1_1.0s', 'Onset F1±1.0s'), ('interval_abs_err_s', 'Breath-group duration error s')]


def complete(name):
    p = ROOT / 'family_runs' / name / 'results.json'
    return p.exists() and json.loads(p.read_text(encoding='utf-8')).get('completed', False)


def table(name):
    rows = load(name)
    flat = {}
    for key, (y, p) in rows.items():
        m = seg_metrics(y, p)
        flat[key] = {f'{c}_{k}': (m[c][k] if k in m[c] else m['params'][c][k]) for c in 'AB' for k, _ in METRICS}
    return flat


def med(flat, metric, keys=None):
    sub = {k: v for k, v in flat.items() if keys is None or k in keys}
    s = singer_med(sub, metric)
    return float(np.median(list(s.values()))) if s else float('nan')


def main():
    names = sorted(p.name for p in (ROOT / 'family_runs').iterdir()
                   if p.is_dir() and p.name.startswith(('r13_', 'r13b_', 'r13c_')) and complete(p.name))
    ref = table(REF)
    lines = ['# Round 13: model re-try results summary (auto-generated, re-runnable)', '',
             f'Reference: `{REF}` = baseline model (WavLM layer 16 + TCN, one fixed hyper-parameter set, single seed). '
             'All numbers are within segment, then singer median, then median across 32 singers; paired differences are singer-level median differences with singer-bootstrap 95% CIs in brackets. '
             'r: Pearson correlation; CCC: concordance correlation coefficient; NRMSE: normalized root-mean-square error (lower is better); F1: harmonic mean of precision and recall. '
             'Outer singers were inspected repeatedly in earlier rounds; this table is exploratory.', '',
             '## Overall table (5 Hz)', '', '|Configuration|' + '|'.join(f'A {l}|B {l}' for _, l in METRICS) + '|',
             '|---|' + '---|---|' * len(METRICS)]
    tables = {REF: ref}
    for n in names:
        tables[n] = table(n)
    for n, flat in tables.items():
        lines.append(f'|{LABEL.get(n, n)}|' + '|'.join(f'{med(flat, f"A_{k}"):.3f}|{med(flat, f"B_{k}"):.3f}' for k, _ in METRICS) + '|')
    lines += ['', '## Paired differences vs the baseline model', '', '|Configuration|Metric|A diff [95% CI]|A singers improved|B diff [95% CI]|B singers improved|', '|---|---|---|---|---|---|']
    for n in names:
        for k, lab in METRICS[:4]:
            cells = []
            for c in 'AB':
                a = singer_med(tables[n], f'{c}_{k}'); b = singer_med(ref, f'{c}_{k}')
                common = sorted(set(a) & set(b)); d = [a[s] - b[s] for s in common]
                m, lo, hi = boot_ci(d)
                better = sum((x < 0) if k in ('nrmse', 'interval_abs_err_s') else (x > 0) for x in d)
                cells.append(f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]|{better}/{len(d)}')
            lines.append(f'|{LABEL.get(n, n)}|{lab}|' + '|'.join(cells) + '|')
    lines += ['', '## By task (CCC, A/B)', '', '|Configuration|Song|Scale|Reading|', '|---|---|---|---|']
    for n, flat in tables.items():
        cells = []
        for t in ('song', 'scale', 'reading'):
            keys = {k for k in flat if TASK[k] == t}
            cells.append(f'{med(flat, "A_ccc", keys):.3f}/{med(flat, "B_ccc", keys):.3f}')
        lines.append(f'|{LABEL.get(n, n)}|' + '|'.join(cells) + '|')
    # learned layer weights
    lw = [n for n in names if n.startswith('r13b_') and ('_all_' in n)]
    if lw:
        lines += ['', '## Learned layer weights (mean over folds and seeds; layer 0 is the convolutional front-end output)', '']
        for n in lw:
            d = json.loads((ROOT / 'family_runs' / n / 'results.json').read_text(encoding='utf-8'))
            w = np.mean([x for f in d['folds'] for x in f['layer_weights']], axis=0)
            top = np.argsort(-w)[:5]
            lines.append(f'- `{n}`: highest-weighted layers ' + ', '.join(f'layer {i} {w[i]:.3f}' for i in top) +
                         f'; uniform weight is {1 / len(w):.3f}')
    # 50 Hz specifics
    c50 = [n for n in names if n.startswith('r13c_')]
    if c50:
        lines += ['', '## Batch 3: 50 Hz specific metrics (singer median)', '',
                  '|Configuration|A F1±0.4s|B F1±0.4s|A onset error s|B onset error s|A insp. duration error s|B insp. duration error s|A-B lag error s|Measured B-A lag s|',
                  '|---|---|---|---|---|---|---|---|---|']
        for n in c50:
            d = json.loads((ROOT / 'family_runs' / n / 'results.json').read_text(encoding='utf-8'))
            segs = [s for f in d['folds'] for s in f['test_segments']]

            def sm(fn):
                by = {}
                for s in segs:
                    v = fn(s['hz50']['params'])
                    if np.isfinite(v):
                        by.setdefault(s['singer'], []).append(v)
                return float(np.median([np.median(v) for v in by.values()]))
            lines.append(f'|{n}|' + '|'.join(f'{sm(f):.3f}' for f in (
                lambda p: p['A']['f1_0.4s'], lambda p: p['B']['f1_0.4s'], lambda p: p['A']['onset_abs_err_s'],
                lambda p: p['B']['onset_abs_err_s'], lambda p: p['A']['insp_dur_abs_err_s'], lambda p: p['B']['insp_dur_abs_err_s'],
                lambda p: p['AB_lag_abs_err_s'], lambda p: p['AB_lag_ref_s'])) + '|')
    # tuning vs ensembling: single-seed CCC (mean over the 3 seeds) vs 3-seed ensemble CCC
    b1 = [n for n in names if n.startswith('r13_')]
    if b1:
        lines += ['', '## Separate contributions of tuning and ensembling (CCC, A/B, singer median)', '',
                  '|Configuration|Baseline model|Tuned single seed (mean of 3 seeds)|Tuned 3-seed ensemble|', '|---|---|---|---|']
        for n in b1:
            d = json.loads((ROOT / 'family_runs' / n / 'results.json').read_text(encoding='utf-8'))
            single, ens = {}, {}
            for f in d['folds']:
                for s in f['test_segments']:
                    single.setdefault(s['singer'], []).append([np.mean([x[c]['ccc'] for x in s['single_seed']]) for c in 'AB'])
                    ens.setdefault(s['singer'], []).append([s[c]['ccc'] for c in 'AB'])
            sm = lambda by: np.median([np.median(v, axis=0) for v in by.values()], axis=0)
            a, b = sm(single), sm(ens)
            lines.append(f'|{LABEL.get(n, n)}|{med(ref, "A_ccc"):.3f}/{med(ref, "B_ccc"):.3f}|{a[0]:.3f}/{a[1]:.3f}|{b[0]:.3f}/{b[1]:.3f}|')
    running = sorted(p.name for p in (ROOT / 'family_runs').iterdir()
                     if p.is_dir() and p.name.startswith(('r13_', 'r13b_', 'r13c_')) and not complete(p.name))
    lines += ['', f'Not finished yet: {", ".join(running) if running else "none"}']
    (ROOT / 'round13_results_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
