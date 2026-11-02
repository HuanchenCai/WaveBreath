"""Round-10 step 2: parameter-layer evaluation of existing OOF predictions and rule
baselines, following the round-10 parameter-evaluation pre-registration (extractor frozen in step 1)."""
import json
from pathlib import Path

import numpy as np

from breath_parameters import FS, f1, inhalations, match, onsets_to_events, record_parameters
from diag_r9_sawtooth_baseline import runs, sawtooth

ROOT = Path(__file__).resolve().parent
MODELS = ('wavlm_large16_tcn', 'loss_ccc100')
KEYS = ('f1_0.4s', 'f1_1.0s', 'onset_abs_err_s', 'interval_abs_err_s', 'count_rel_err', 'depth_spearman')
LOWER_BETTER = {'onset_abs_err_s', 'interval_abs_err_s', 'AB_lag_abs_err_s'}


def load_preds(name):
    out = {}
    for fold in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{fold}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                out[(m['singer'], str(m['number']))] = z['prediction'][off[j]:off[j + 1]]
    return out


def pause_rule(y, voiced):
    """Direct event rule: each unvoiced run >= 2 frames that ends before the record end."""
    on = [s for s, e in runs(~voiced) if e - s >= 2 and e < len(voiced)]
    ev_rule = onsets_to_events(on, len(y))
    res = {}
    for k, c in enumerate('AB'):
        ref = inhalations(y[:, k])
        f04, m04 = f1(ref[:, 0], ev_rule[:, 0], 2)
        f10, _ = f1(ref[:, 0], ev_rule[:, 0], 5)
        iy = np.diff(ref[:, 0]) / FS; ip = np.diff(ev_rule[:, 0]) / FS
        iref = float(np.median(iy)) if len(iy) else float('nan'); iest = float(np.median(ip)) if len(ip) else float('nan')
        res[c] = {'f1_0.4s': f04, 'f1_1.0s': f10,
                  'onset_abs_err_s': float(np.median([abs(ref[i, 0] - ev_rule[j, 0]) / FS for i, j in m04])) if m04 else float('nan'),
                  'interval_abs_err_s': abs(iref - iest), 'count_rel_err': (len(ev_rule) - len(ref)) / max(len(ref), 1),
                  'depth_spearman': float('nan')}
    ea = inhalations(y[:, 0]); eb = inhalations(y[:, 1])
    m = match(ea[:, 0], eb[:, 0], 5)
    ly = float(np.median([eb[j, 0] - ea[i, 0] for i, j in m])) / FS if len(m) >= 3 else float('nan')
    res['AB_lag_abs_err_s'] = abs(ly - 0.)  # rule implies identical A/B timing
    return res


def singer_level(rows, getter):
    singers = sorted({r['singer'] for r in rows})
    vals = {}
    for s in singers:
        v = [getter(r) for r in rows if r['singer'] == s]
        v = [x for x in v if np.isfinite(x)]
        if v:
            vals[s] = float(np.median(v))
    return vals


def boot_ci(diffs, seed=20260930, n=2000):
    d = np.array(list(diffs)); rng = np.random.default_rng(seed)
    b = [np.median(rng.choice(d, len(d))) for _ in range(n)]
    return float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_all, off = z['y'], z['offsets']
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        v_all = z['voiced']
    preds = {m: load_preds(m) for m in MODELS}
    rows = []
    for i, a in enumerate(info):
        y = y_all[off[i]:off[i + 1]]; v = v_all[off[i]:off[i + 1]]
        low = {c: bool(a[c]['polarity_index'] <= 0 or a['rel_amp'][k] < .3) for k, c in enumerate('AB')}
        low05 = {c: bool(a[c]['polarity_index'] <= 0 or a['rel_amp'][k] < .5) for k, c in enumerate('AB')}
        r = {'singer': a['singer'], 'number': str(a['number']), 'task': a['task'], 'low_clarity': low, 'low_clarity_0.5': low05}
        for m in MODELS:
            r[m] = record_parameters(y, preds[m][(a['singer'], str(a['number']))])
        saw = sawtooth(v); r['sawtooth'] = record_parameters(y, np.c_[saw, saw])
        r['pause_rule'] = pause_rule(y, v)
        rows.append(r)
    methods = list(MODELS) + ['sawtooth', 'pause_rule']
    lines = ['# Round 10: parameter-level evaluation (exploratory)', '',
             'Within segment, then singer median, then median across 32 singers. Time in seconds. Higher F1 is better; lower error is better.', '']

    def table(subset, title):
        out = [f'## {title} ({len(subset)} segments)', '',
               '|Method|Belt|Onset F1 ±0.4s|Onset F1 ±1.0s|Onset error|Breath-group duration error|Event-count relative error|Depth Spearman|',
               '|---|---|---|---|---|---|---|---|']
        for m in methods:
            for c in 'AB':
                cells = []
                for k in KEYS:
                    s = singer_level(subset, lambda r: r[m][c][k])
                    cells.append(f'{np.median(list(s.values())):.3f}' if s else '—')
                out.append(f'|{m}|{c}|' + '|'.join(cells) + '|')
        out += ['', '|Method|Dual-belt inhalation lag error (s)|', '|---|---|']
        for m in methods:
            s = singer_level(subset, lambda r: r[m]['AB_lag_abs_err_s'])
            out.append(f'|{m}|{np.median(list(s.values())):.3f}|' if s else f'|{m}|—|')
        return out + ['']

    lines += table(rows, 'All')
    for t in ('song', 'scale', 'reading'):
        lines += table([r for r in rows if r['task'] == t], f'Task={t}')
    for c in 'AB':
        lines += table([r for r in rows if r['low_clarity'][c]], f'Belt {c} low clarity (threshold 0.3)')
        lines += table([r for r in rows if not r['low_clarity'][c]], f'Belt {c} normal clarity (threshold 0.3)')
    lines += ['## Paired comparisons (singer-level median difference, singer-bootstrap 95% CI)', '', '|Comparison|Belt|Metric|Median diff|95% CI|', '|---|---|---|---|---|']
    for a_, b_ in (('wavlm_large16_tcn', 'pause_rule'), ('wavlm_large16_tcn', 'sawtooth'), ('loss_ccc100', 'wavlm_large16_tcn')):
        for c in 'AB':
            for k in ('f1_0.4s', 'onset_abs_err_s', 'interval_abs_err_s'):
                sa = singer_level(rows, lambda r: r[a_][c][k]); sb = singer_level(rows, lambda r: r[b_][c][k])
                common = sorted(set(sa) & set(sb))
                med, lo, hi = boot_ci([sa[s] - sb[s] for s in common])
                lines.append(f'|{a_} − {b_}|{c}|{k}|{med:.3f}|[{lo:.3f}, {hi:.3f}]|')
        sa = singer_level(rows, lambda r: r[a_]['AB_lag_abs_err_s']); sb = singer_level(rows, lambda r: r[b_]['AB_lag_abs_err_s'])
        common = sorted(set(sa) & set(sb)); med, lo, hi = boot_ci([sa[s] - sb[s] for s in common])
        lines.append(f'|{a_} − {b_}|AB|AB_lag_abs_err_s|{med:.3f}|[{lo:.3f}, {hi:.3f}]|')
    ly = singer_level(rows, lambda r: r['wavlm_large16_tcn']['AB_lag_ref_s'])
    lp = singer_level(rows, lambda r: r['wavlm_large16_tcn']['AB_lag_est_s'])
    lines += ['', f'Measured B-A inhalation lag singer median {np.median(list(ly.values())):.2f} s; main model prediction {np.median(list(lp.values())):.2f} s.']
    out = ROOT / 'diagnostics_r10'
    (out / 'parameter_level_evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'parameter_eval_rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
