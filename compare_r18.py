"""Round 18 comparison.
(1) 5-Hz parameter set (round-16 pipeline) for: r13_convlstm, r13c_convlstm_50hz, round-17 variants,
    r18_convlstm_composite_50hz.
(2) 50-Hz-only parameters for the two 50-Hz models: chest-abdomen correlation, paradox fraction
    (direction over 0.2-s differences), B-A inhalation lag, onset/inspiration-duration errors.
Singer-level paired differences vs r13c_convlstm_50hz with 2000-sample bootstrap CIs."""
import json
from pathlib import Path

import numpy as np

from breath_parameters_fs import record_parameters_fs
from diag_r10_parameter_eval import boot_ci
from summarize_r13 import med, table

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / 'r16_param_validation.json').read_text(encoding='utf-8'))
M5 = ['r13_convlstm', 'r13c_convlstm_50hz', 'r13b_convlstm_wavlm16_joint_lossdiffcorr',
      'r13b_convlstm_wavlm16_joint_lossdiffcorr_w3', 'r13b_convlstm_wavlm16_joint_lossdiffcorr_w10', 'r18_convlstm_composite_50hz']
LAB = {'r13_convlstm': 'Conv-LSTM 5 Hz default loss', 'r13c_convlstm_50hz': 'Conv-LSTM 50 Hz default loss',
       'r13b_convlstm_wavlm16_joint_lossdiffcorr': '5 Hz diff + corr (weight 1)', 'r13b_convlstm_wavlm16_joint_lossdiffcorr_w3': '5 Hz diff + corr (weight 3)',
       'r13b_convlstm_wavlm16_joint_lossdiffcorr_w10': '5 Hz diff + corr (weight 10)', 'r18_convlstm_composite_50hz': '* 50 Hz composite loss'}


def load50(name):
    out = {}
    for f in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{f}_waveforms_50hz.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                out[(m['singer'], str(m['number']))] = (z['target'][off[j]:off[j + 1]], z['prediction'][off[j]:off[j + 1]])
    return out


def dual50(w):
    d = (w[10:] - w[:-10]) / (w.std(0) + 1e-9)
    mv = (np.abs(d[:, 0]) > .5) & (np.abs(d[:, 1]) > .5)
    return float(np.corrcoef(w.T)[0, 1]), float((np.sign(d[mv, 0]) != np.sign(d[mv, 1])).mean()) if mv.any() else np.nan


def rows50(name):
    out = []
    for (s, n), (y, p) in load50(name).items():
        cy, py_ = dual50(y), dual50(p); par = record_parameters_fs(y, p, 50)
        out.append({'singer': s, 'ab_corr_err': abs(py_[0] - cy[0]), 'paradox_err': abs(py_[1] - cy[1]),
                    'ab_corr_meas': cy[0], 'ab_corr_pred': py_[0], 'paradox_meas': cy[1], 'paradox_pred': py_[1],
                    'lag_err': par['AB_lag_abs_err_s'], 'lag_ref': par['AB_lag_ref_s'], 'lag_est': par['AB_lag_est_s'],
                    'A_onset': par['A']['onset_abs_err_s'], 'B_onset': par['B']['onset_abs_err_s'],
                    'A_insp': par['A']['insp_dur_abs_err_s'], 'B_insp': par['B']['insp_dur_abs_err_s'],
                    'A_f1': par['A']['f1_0.4s'], 'B_f1': par['B']['f1_0.4s']})
    return out


def smed(rows, k):
    by = {}
    for r in rows:
        if np.isfinite(r[k]):
            by.setdefault(r['singer'], []).append(r[k])
    return {s: float(np.median(v)) for s, v in by.items()}


def main():
    L = ['# Round 18: 50 Hz composite loss results', '', '## 1. 5 Hz level (round-16 parameter set; 50 Hz models averaged back to 5 Hz)', '',
         '|Model|Waveform CCC chest/abdomen|A-B corr meas/pred|A-B corr CCC|Paradox meas/pred|Paradox CCC|Chest rate CCC|Abdomen rate CCC|Chest expiratory slope CCC|Onset F1 ±0.4 chest/abdomen|',
         '|---|---|---|---|---|---|---|---|---|---|']
    for m in M5:
        if m not in R:
            continue
        s = R[m]; t = table(m); d = s['dual']
        L.append(f'|{LAB[m]}|{med(t, "A_ccc"):.3f}/{med(t, "B_ccc"):.3f}|{d["ab_corr"]["meas_median"]:.2f}/{d["ab_corr"]["pred_median"]:.2f}|{d["ab_corr"]["ccc"]:.2f}|'
                 f'{d["paradox_frac"]["meas_median"]:.2f}/{d["paradox_frac"]["pred_median"]:.2f}|{d["paradox_frac"]["ccc"]:.2f}|'
                 f'{s["signal"]["A_rate_per_min"]["ccc"]:.2f}|{s["signal"]["B_rate_per_min"]["ccc"]:.2f}|{s["signal"]["A_exp_slope_sd_s"]["ccc"]:.2f}|'
                 f'{s["pair"]["A_f1_0.4s"]["median"]:.2f}/{s["pair"]["B_f1_0.4s"]["median"]:.2f}|')
    a, b = rows50('r18_convlstm_composite_50hz'), rows50('r13c_convlstm_50hz')
    L += ['', '## 2. 50 Hz level (only the two 50 Hz models)', '',
          'Singer-level medians; "diff" is composite minus default loss, singer-level median (negative = better for errors; positive = better for F1); brackets are 95% bootstrap CIs.', '',
          '|Metric|Default loss|Composite loss|Diff [95%]|', '|---|---|---|---|']
    for k, lab in (('ab_corr_err', 'A-B correlation abs. error'), ('paradox_err', 'Paradox fraction abs. error'), ('lag_err', 'A-B inhalation lag error (s)'),
                   ('A_onset', 'Chest onset error (s)'), ('B_onset', 'Abdomen onset error (s)'), ('A_insp', 'Chest inspiration-duration error (s)'),
                   ('B_insp', 'Abdomen inspiration-duration error (s)'), ('A_f1', 'Chest onset F1 ±0.4 s'), ('B_f1', 'Abdomen onset F1 ±0.4 s')):
        sa, sb = smed(a, k), smed(b, k); common = sorted(set(sa) & set(sb))
        m_, lo, hi = boot_ci([sa[s] - sb[s] for s in common])
        L.append(f'|{lab}|{np.median(list(sb.values())):.3f}|{np.median(list(sa.values())):.3f}|{m_:+.3f} [{lo:+.3f}, {hi:+.3f}]|')
    for name, rr in (('Measured', None), ('Default loss', b), ('Composite loss', a)):
        pass
    L += ['', f'Measured A-B correlation median {np.median([r["ab_corr_meas"] for r in a]):.2f}; predicted: default {np.median([r["ab_corr_pred"] for r in b]):.2f}, composite {np.median([r["ab_corr_pred"] for r in a]):.2f}.',
          f'Measured paradox fraction median {np.nanmedian([r["paradox_meas"] for r in a]):.2f}; predicted: default {np.nanmedian([r["paradox_pred"] for r in b]):.2f}, composite {np.nanmedian([r["paradox_pred"] for r in a]):.2f}.',
          f'Measured abdomen-minus-chest inhalation lag median {np.nanmedian([r["lag_ref"] for r in a]):+.2f} s; predicted: default {np.nanmedian([r["lag_est"] for r in b]):+.2f}, composite {np.nanmedian([r["lag_est"] for r in a]):+.2f}.']
    d = json.loads((ROOT / 'family_runs' / 'r18_convlstm_composite_50hz' / 'results.json').read_text(encoding='utf-8'))
    L += ['', f'A-B correlation weight w_AB selected in the inner loop of each fold: {[f["w_ab"] for f in d["folds"]]}']
    (ROOT / 'round18_composite_50hz_results.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
