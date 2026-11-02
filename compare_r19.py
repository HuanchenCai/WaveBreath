"""Round-19 comparison (see the round-19 pre-registration). Writes round19_model_improvement_results.md.
A: whole-record window (r13b_<fam>_wavlm16_joint_win1024) vs batch 1 (r13_<fam>).
B: per-singer calibration on the non-calibration recordings: linear / finetune vs none.
C: LoRA encoder (r19c_convlstm_lora) vs batch-1 single model with the same seed (r13_convlstm_seed0).
All paired at singer level (2000-sample bootstrap); parameters from breath_param_set."""
import json
from pathlib import Path

import numpy as np

from breath_param_set import pair_params
from compare_r11_r12 import singer_med
from diag_r10_parameter_eval import boot_ci
from summarize_r13 import table

ROOT = Path(__file__).resolve().parent
INFO = {(a['singer'], str(a['number'])): a for a in json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))}
with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
    VOICED, OFF = z['voiced'], z['offsets']
KEYS = list(INFO)


def load(name):
    rows = {}
    for f in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{f}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                rows[(m['singer'], str(m['number']))] = (z['target'][off[j]:off[j + 1], :2], z['prediction'][off[j]:off[j + 1], :2])
    return rows


def voiced(key):
    i = KEYS.index(key); return VOICED[OFF[i]:OFF[i + 1]]


def param_rows(name):
    out = {}
    for key, (y, p) in load(name).items():
        sy, sp, pair = pair_params(y, p, voiced(key))
        out[key] = {'A_ccc': None, 'ab_corr_err': abs(sp['dual']['ab_corr'] - sy['dual']['ab_corr']),
                    'paradox_err': abs(sp['dual']['paradox_frac'] - sy['dual']['paradox_frac']),
                    'A_rate_err': abs(sp['A']['rate_per_min'] - sy['A']['rate_per_min']),
                    'B_rate_err': abs(sp['B']['rate_per_min'] - sy['B']['rate_per_min']),
                    'A_f1': pair['A']['f1_0.4s'], 'B_f1': pair['B']['f1_0.4s'],
                    'ab_corr_pred': sp['dual']['ab_corr'], 'ab_corr_meas': sy['dual']['ab_corr']}
        a, b = y, p
        for k, c in enumerate('AB'):
            x, q = a[:, k], b[:, k]
            out[key][f'{c}_ccc'] = float(2 * np.mean((x - x.mean()) * (q - q.mean())) / (x.var() + q.var() + (x.mean() - q.mean()) ** 2 + 1e-8))
    return out


def smed(rows, k):
    by = {}
    for key, r in rows.items():
        v = r[k]
        if v is not None and np.isfinite(v):
            by.setdefault(key[0], []).append(v)
    return {s: float(np.median(v)) for s, v in by.items()}


def compare(new, base, title):
    a, b = param_rows(new), param_rows(base)
    common = set(a) & set(b); a = {k: a[k] for k in common}; b = {k: b[k] for k in common}
    L = [f'### {title}', '', f'`{new}` vs `{base}` ({len(common)} segments)', '', '|Metric|Control|New|Diff [95%]|Singers improved|', '|---|---|---|---|---|']
    for k, lab, higher in (('A_ccc', 'Chest waveform CCC', True), ('B_ccc', 'Abdomen waveform CCC', True), ('A_f1', 'Chest onset F1 ±0.4', True),
                           ('B_f1', 'Abdomen onset F1 ±0.4', True), ('A_rate_err', 'Chest rate error (breaths/min)', False),
                           ('B_rate_err', 'Abdomen rate error (breaths/min)', False), ('ab_corr_err', 'A-B correlation error', False), ('paradox_err', 'Paradox fraction error', False)):
        sa, sb = smed(a, k), smed(b, k); s = sorted(set(sa) & set(sb)); d = [sa[x] - sb[x] for x in s]
        m, lo, hi = boot_ci(d); better = sum((v > 0) if higher else (v < 0) for v in d)
        flag = ' ✓' if (lo > 0 if higher else hi < 0) else (' ✗' if (hi < 0 if higher else lo > 0) else '')
        L.append(f'|{lab}|{np.median(list(sb.values())):.3f}|{np.median(list(sa.values())):.3f}|{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{flag}|{better}/{len(d)}|')
    L.append(f'\nPredicted A-B correlation median: control {np.median([r["ab_corr_pred"] for r in b.values()]):.2f}, new {np.median([r["ab_corr_pred"] for r in a.values()]):.2f} (measured {np.median([r["ab_corr_meas"] for r in a.values()]):.2f})\n')
    return L


def main():
    L = ['# Round 19: model improvement results', '', '✓ = stable improvement (CI excludes 0), ✗ = stable deterioration.', '', '## A. Whole-record context (window 1024 frames)', '']
    for fam in ('convlstm', 'bigru', 'unet'):
        n = f'r13b_{fam}_wavlm16_joint_win1024'
        if (ROOT / 'family_runs' / n / 'fold5_waveforms.npz').exists():
            L += compare(n, f'r13_{fam}', fam)
    L += ['## B. Per-singer calibration (each singer\'s longest segment used for calibration, evaluated on the rest)', '']
    for fam in ('convlstm', 'unet'):
        for c in ('linear', 'finetune'):
            n = f'r19b_{fam}_{c}'
            if (ROOT / 'family_runs' / n / 'fold5_waveforms.npz').exists():
                L += compare(n, f'r19b_{fam}_none', f'{fam} {c}')
    seed0 = ROOT / 'family_runs' / 'r13_convlstm_seed0'
    if not (seed0 / 'fold5_waveforms.npz').exists():
        seed0.mkdir(exist_ok=True)
        for f in range(1, 6):
            with np.load(ROOT / 'family_runs' / 'r13_convlstm' / f'fold{f}_waveforms.npz') as z:
                np.savez_compressed(seed0 / f'fold{f}_waveforms.npz', prediction=z['prediction_seeds'][0], target=z['target'], metadata=z['metadata'], offsets=z['offsets'])
    if (ROOT / 'family_runs' / 'r19c_convlstm_lora_w128' / 'fold5_waveforms.npz').exists() and (ROOT / 'family_runs' / 'r19c_convlstm_frozen_w128' / 'fold5_waveforms.npz').exists():
        L += ['## C. Encoder LoRA fine-tuning (paired with the frozen-encoder control of the same pipeline, 25.6 s window, same epochs and seed)', '']
        L += compare('r19c_convlstm_lora_w128', 'r19c_convlstm_frozen_w128', 'convlstm LoRA vs frozen')
    (ROOT / 'round19_model_improvement_results.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
