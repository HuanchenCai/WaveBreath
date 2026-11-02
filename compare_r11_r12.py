"""Compare configs on waveform metrics and breath parameters (each against its own target).
Usage: python compare_r11_r12.py OUT.md REF CONFIG [CONFIG ...]
Groups whose names differ only by _s<k> (seed) are averaged per segment before summary."""
import json
import re
import sys
from pathlib import Path

import numpy as np

from breath_parameters import record_parameters
from diag_r10_parameter_eval import boot_ci

ROOT = Path(__file__).resolve().parent


def load(name):
    rows = {}
    for fold in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{fold}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                y = z['target'][off[j]:off[j + 1], :2]; p = z['prediction'][off[j]:off[j + 1], :2]
                rows[(m['singer'], str(m['number']))] = (y, p)
    assert len(rows) == 197
    return rows


def seg_metrics(y, p):
    out = {'params': record_parameters(y, p)}
    for k, c in enumerate('AB'):
        a, b = y[:, k], p[:, k]
        out[c] = {'r': float(np.corrcoef(a, b)[0, 1]),
                  'ccc': float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8)),
                  'nrmse': float(np.sqrt(np.mean((a - b) ** 2)) / (a.std() + 1e-8))}
    return out


def metric_table(configs):
    """configs: list of names or seed groups; returns {group: {seg_key: {metric: value}}}."""
    groups = {}
    for c in configs:
        groups.setdefault(re.sub(r'_s\d+$', '', c), []).append(c)
    table = {}
    for g, members in groups.items():
        per = [{k: seg_metrics(*v) for k, v in load(m).items()} for m in members]
        flat = {}
        for key in per[0]:
            vals = {}
            for c in 'AB':
                for mm in ('r', 'ccc', 'nrmse'):
                    vals[f'{c}_{mm}'] = float(np.mean([p[key][c][mm] for p in per]))
                for mm in ('f1_0.4s', 'f1_1.0s', 'interval_abs_err_s', 'depth_spearman'):
                    vals[f'{c}_{mm}'] = float(np.nanmean([p[key]['params'][c][mm] for p in per])) if any(
                        np.isfinite(p[key]['params'][c][mm]) for p in per) else float('nan')
            flat[key] = vals
        table[g] = (flat, len(members))
    return table


def singer_med(flat, metric):
    singers = sorted({k[0] for k in flat})
    out = {}
    for s in singers:
        v = [flat[k][metric] for k in flat if k[0] == s and np.isfinite(flat[k][metric])]
        if v:
            out[s] = float(np.median(v))
    return out


def main():
    out_md, ref, *configs = sys.argv[1:]
    table = metric_table([ref, *configs])
    ref_g = re.sub(r'_s\d+$', '', ref)
    metrics = [f'{c}_{m}' for m in ('r', 'ccc', 'nrmse', 'f1_0.4s', 'f1_1.0s', 'interval_abs_err_s', 'depth_spearman') for c in 'AB']
    lines = [f'# Configuration comparison (reference {ref_g})', '', 'Singer-level medians; multi-seed groups are first averaged over seeds per segment. CCC/NRMSE are comparable only within the same target definition.', '',
             '|Configuration|Seeds|' + '|'.join(metrics) + '|', '|---|---|' + '---|' * len(metrics)]
    for g, (flat, n) in table.items():
        lines.append(f'|{g}|{n}|' + '|'.join(f'{np.median(list(singer_med(flat, m).values())):.3f}' for m in metrics) + '|')
    lines += ['', '## Singer-level paired difference vs reference (median [95% bootstrap CI], singers improved)', '', '|Configuration|Metric|Diff|CI|Improved/total|', '|---|---|---|---|---|']
    rflat = table[ref_g][0]
    for g, (flat, _) in table.items():
        if g == ref_g:
            continue
        for m in metrics:
            a = singer_med(flat, m); b = singer_med(rflat, m); common = sorted(set(a) & set(b))
            d = [a[s] - b[s] for s in common]
            med, lo, hi = boot_ci(d)
            better = sum((x < 0) if ('nrmse' in m or 'err' in m) else (x > 0) for x in d)
            lines.append(f'|{g}|{m}|{med:+.4f}|[{lo:+.4f}, {hi:+.4f}]|{better}/{len(d)}|')
    Path(out_md).write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
