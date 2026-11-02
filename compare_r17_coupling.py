"""Round 17: chest-abdomen coordination parameters, coupling losses vs default loss (same family).
Singer-level paired differences of |pred - meas| (from parameter_set.csv), bootstrap 95% CI."""
import csv
import json
from pathlib import Path
import numpy as np
from diag_r10_parameter_eval import boot_ci

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / 'r16_param_validation.json').read_text(encoding='utf-8'))


def rows(name):
    return list(csv.DictReader(open(ROOT / 'family_runs' / name / 'parameter_set.csv', encoding='utf-8-sig')))


def singer_err(rs, col):
    by = {}
    for r in rs:
        m, p = float(r[f'meas_{col}']), float(r[f'pred_{col}'])
        if np.isfinite(m) and np.isfinite(p):
            by.setdefault(r['singer'], []).append(abs(p - m))
    return {s: float(np.median(v)) for s, v in by.items()}


def main():
    L = ['# Round 17: explicit modelling of chest-abdomen coordination', '',
         'Paired with batch 1 of the same family (default loss). For chest-abdomen parameters the "error difference" is the median singer-level change in |pred - meas| (negative = smaller error, i.e. improvement); brackets are singer-bootstrap 95% CIs.', '',
         '|Model|A-B corr: meas/pred median|CCC|Error diff [95%]|Paradox fraction: meas/pred median|CCC|Error diff [95%]|A-B lag error diff|Waveform CCC A/B|Onset F1 ±0.4 A/B|', '|---|---|---|---|---|---|---|---|---|---|']
    for fam in ('convlstm', 'unet'):
        base = f'r13_{fam}'; rb = rows(base)
        names = [base, f'r13b_{fam}_wavlm16_joint_lossdiff', f'r13b_{fam}_wavlm16_joint_lossdiffcorr']
        names += [f'r13b_{fam}_wavlm16_joint_lossdiffcorr_w{w}' for w in (3, 10) if f'r13b_{fam}_wavlm16_joint_lossdiffcorr_w{w}' in R]
        for name in names:
            s = R[name]; rn = rows(name); cells = []
            for col in ('ab_corr', 'paradox_frac'):
                if name == base:
                    cells.append('—')
                else:
                    a, b = singer_err(rn, col), singer_err(rb, col)
                    m, lo, hi = boot_ci([a[k] - b[k] for k in sorted(set(a) & set(b))])
                    cells.append(f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]')
            if name == base:
                lag = '—'
            else:
                a, b = singer_err(rn, 'ab_lag_s'), singer_err(rb, 'ab_lag_s')
                m, lo, hi = boot_ci([a[k] - b[k] for k in sorted(set(a) & set(b))]); lag = f'{m:+.3f} [{lo:+.3f}, {hi:+.3f}]'
            d = s['dual']; sig = s['signal']
            wave = '—'
            L.append(f'|{name}|{d["ab_corr"]["meas_median"]:.2f}/{d["ab_corr"]["pred_median"]:.2f}|{d["ab_corr"]["ccc"]:.2f}|{cells[0]}|'
                     f'{d["paradox_frac"]["meas_median"]:.2f}/{d["paradox_frac"]["pred_median"]:.2f}|{d["paradox_frac"]["ccc"]:.2f}|{cells[1]}|{lag}|'
                     f'{{wave_{name}}}|{s["pair"]["A_f1_0.4s"]["median"]:.2f}/{s["pair"]["B_f1_0.4s"]["median"]:.2f}|')
    from summarize_r13 import table, med
    text = '\n'.join(L)
    for line in L:
        if '{wave_' in line:
            name = line.split('{wave_')[1].split('}')[0]
            t = table(name); text = text.replace('{wave_' + name + '}', f'{med(t, "A_ccc"):.3f}/{med(t, "B_ccc"):.3f}')
    (ROOT / 'round17_coupling_results.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
