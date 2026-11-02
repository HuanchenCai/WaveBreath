"""Single-factor effects: each follow-up run vs the batch-1 result of the SAME family
(they differ only in that factor). Singer-level paired differences, 2000-sample bootstrap."""
from pathlib import Path
import numpy as np
from diag_r10_parameter_eval import boot_ci
from summarize_r13 import table
from compare_r11_r12 import singer_med

ROOT = Path(__file__).resolve().parent
FACTORS = [('wavlm16_separate', 'Per-task separate training'), ('wavlm16_tasklabel', 'Joint training + task label'), ('wavlm_all_joint', 'WavLM all 25 layers weighted'),
           ('mert_all_joint', 'MERT all 25 layers weighted'), ('wavlm16_egg_joint', 'WavLM16 + EGG'), ('egg_only_joint', 'EGG only')]
METRICS = [('ccc', 'CCC'), ('r', 'r'), ('f1_0.4s', 'Onset F1±0.4s')]


def main():
    lines = ['# Rounds 13/14 single-factor effects (paired with batch 1 of the same family)', '',
             'Differences are singer-level median differences, brackets are singer-bootstrap 95% CIs, "improved" is the number of singers (of 32) for whom the metric improved. Only CIs excluding 0 are considered stable.', '',
             '|Family|Factor|Metric|A diff [95%]|A improved|B diff [95%]|B improved|', '|---|---|---|---|---|---|---|']
    for fam in ('convlstm', 'unet'):
        base = table(f'r13_{fam}')
        runs = [(f'r13b_{fam}_{k}', lab) for k, lab in FACTORS] + [(f'r13c_{fam}_50hz', '50 Hz two-scale (averaged back to 5 Hz)')]
        for name, lab in runs:
            t = table(name)
            for k, ml in METRICS:
                cells = []
                for c in 'AB':
                    a, b = singer_med(t, f'{c}_{k}'), singer_med(base, f'{c}_{k}')
                    d = [a[s] - b[s] for s in sorted(set(a) & set(b))]
                    m, lo, hi = boot_ci(d)
                    flag = ' ✓' if lo > 0 else (' ✗' if hi < 0 else '')
                    cells.append(f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}|{sum(x > 0 for x in d)}/{len(d)}')
                lines.append(f'|{fam}|{lab}|{ml}|' + '|'.join(cells) + '|')
    lines += ['', '✓: CI entirely above 0 (stable improvement); ✗: CI entirely below 0 (stable deterioration).']
    (ROOT / 'round13_single_factor_effects.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
