"""Round 26 comparison: each best-of-N rule vs the recommended model (paired at singer level), plus the
large-inhalation metrics of round 21 (within-record depth slope, top-10% depth ratio). Writes
round26_best_of_n_results.md."""
from pathlib import Path

import numpy as np

from compare_r19 import compare, load
from plot_round_compare import depth_pairs

ROOT = Path(__file__).resolve().parent
BASE = 'r15_ens_convlstm_unet'
RULES = [('r26_bon_mean16', 'Mean of 16'), ('r26_bon_seed0', 'Single sample'), ('r26_bon_oracle', 'Oracle pick (upper bound)'),
         ('r26_bon_rule', 'Phonation-rule pick'), ('r26_bon_consensus', 'Pick closest to recommended model')]


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


def summary(name):
    by = {}
    for key, (y, p) in load(name).items():
        row = []
        for k in range(2):
            dm, dp, top = depth_pairs(y, p, k)
            slope = np.polyfit(dm, dp, 1)[0] if len(dm) >= 4 else np.nan
            ratio = np.median(dp[top] / dm[top]) if top.any() else np.nan
            row += [ccc(y[:, k], p[:, k]), slope, ratio]
        by.setdefault(key[0], []).append(row)
    return np.nanmedian([np.nanmedian(v, axis=0) for v in by.values()], axis=0)


def main():
    L = ['# Round 26: best-of-N sampling with reward-based selection - results', '', 'Pre-registration: round26 best-of-N pre-registration. All variants are scored against the same target (r11_tgt_dc), singer medians.', '',
         '## 1. Overview', '', '|Variant|Chest CCC|Abdomen CCC|Chest depth slope|Abdomen depth slope|Chest top-10% ratio|Abdomen top-10% ratio|', '|---|---|---|---|---|---|---|']
    for name, lab in [(BASE, 'Recommended model')] + RULES:
        s = summary(name)
        L.append(f'|{lab}|{s[0]:.3f}|{s[3]:.3f}|{s[1]:.2f}|{s[4]:.2f}|{s[2]:.2f}|{s[5]:.2f}|')
    L += ['', '## 2. Paired comparisons (vs recommended model)', '', '✓ = stable improvement, ✗ = stable deterioration.', '']
    for name, lab in RULES:
        L += compare(name, BASE, lab)
    (ROOT / 'round26_best_of_n_results.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
