"""Round-10 step 1: measured-only validity of the breath-parameter extractor.
No model predictions are read here. Checks: self-comparison sanity, event rates,
intervals, and agreement of inhalation rises with voice pauses."""
import json
from pathlib import Path

import numpy as np

from breath_parameters import inhalations, record_parameters

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_all, off = z['y'], z['offsets']
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        voiced_all = z['voiced']; assert (z['offsets'] == off).all()
    stats = {c: {'rate_per_min': [], 'interval_s': [], 'rise_unvoiced_frac': [], 'onset_in_or_near_pause': []} for c in 'AB'}
    self_f1 = []
    per_task = {}
    for i, a in enumerate(rows):
        y = y_all[off[i]:off[i + 1]]; v = voiced_all[off[i]:off[i + 1]]
        sp = record_parameters(y, y); self_f1.append(min(sp['A']['f1_0.4s'], sp['B']['f1_0.4s']))
        for k, c in enumerate('AB'):
            ev = inhalations(y[:, k])
            stats[c]['rate_per_min'].append(len(ev) / (len(y) / 5 / 60))
            if len(ev) > 1:
                stats[c]['interval_s'].append(float(np.median(np.diff(ev[:, 0]) / 5)))
            if len(ev):
                rise = np.concatenate([np.arange(int(t), int(p) + 1) for t, p, _ in ev])
                stats[c]['rise_unvoiced_frac'].append(float((~v[rise]).mean()))
                near = [(~v[max(0, int(t) - 2):int(t) + 3]).any() for t, _, _ in ev]
                stats[c]['onset_in_or_near_pause'].append(float(np.mean(near)))
                per_task.setdefault((a['task'], c), []).append(float((~v[rise]).mean()))
    lines = ['# Round 10 step 1: extractor validity on measured belts (no model predictions read)', '',
             f'- Minimum self-comparison F1: {min(self_f1):.3f} (should be 1)',
             f'- Median unvoiced-frame fraction over records: {np.median([1 - a["voiced_fraction"] for a in rows]):.2f} (chance reference)', '',
             '|Belt|Inhalations per min median [IQR]|Median interval s|Median fraction of rise in unvoiced frames|Median fraction of onsets with a pause within ±0.4 s|', '|---|---|---|---|---|']
    for c in 'AB':
        s = stats[c]; q = np.percentile(s['rate_per_min'], [25, 50, 75])
        lines.append(f"|{c}|{q[1]:.1f} [{q[0]:.1f}, {q[2]:.1f}]|{np.median(s['interval_s']):.2f}|"
                     f"{np.median(s['rise_unvoiced_frac']):.2f}|{np.median(s['onset_in_or_near_pause']):.2f}|")
    lines += ['', '|Task|Belt|Median unvoiced fraction of rises|', '|---|---|---|']
    for (t, c), v in sorted(per_task.items()):
        lines.append(f'|{t}|{c}|{np.median(v):.2f}|')
    (ROOT / 'diagnostics_r10').mkdir(exist_ok=True)
    (ROOT / 'diagnostics_r10' / 'extractor_validity.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
