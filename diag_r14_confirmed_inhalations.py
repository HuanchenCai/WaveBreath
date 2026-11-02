"""Round-14 (post hoc, disclosed): a belt inhalation is "sound-confirmed" when an audible
breath sound overlaps its rise interval [trough - 0.2 s, peak + 0.2 s]. The window was set
after seeing that breath-sound onsets cluster 0.2-0.6 s before the belt peak (inside the rise),
not around the trough. Re-evaluates existing model predictions on confirmed vs unconfirmed
belt inhalations (recall of predicted onsets within 0.4 s / 1.0 s). No training."""
import json
from pathlib import Path
import numpy as np
from breath_parameters import inhalations
from compare_r11_r12 import load

ROOT = Path(__file__).resolve().parent
MODELS = ['r11_tgt_dc', 'r13_convlstm', 'r13_bigru', 'r13_tcn']


def main():
    ref = json.loads((ROOT / 'diagnostics_r10' / 'breath_sound_reference.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        y_all, off = z['y'], z['offsets']
    preds = {m: load(m) for m in MODELS if (ROOT / 'family_runs' / m / 'fold5_waveforms.npz').exists()}
    stats = {}
    for i, r in enumerate(ref):
        y = y_all[off[i]:off[i + 1]]; bs = r['breath_sounds_s']; key = (r['singer'], r['number'])
        for k, c in enumerate('AB'):
            ev = inhalations(y[:, k])
            conf = [any(s <= p / 5 + .2 and e >= t / 5 - .2 for s, e in bs) for t, p, _ in ev]
            used = sum(any(s <= p / 5 + .2 and e >= t / 5 - .2 for t, p, _ in ev) for s, e in bs)
            st = stats.setdefault((r['task'], c), {'belt': 0, 'conf': 0, 'sounds': 0, 'sounds_in_rise': 0})
            st['belt'] += len(ev); st['conf'] += sum(conf); st['sounds'] += len(bs); st['sounds_in_rise'] += used
            for m, P in preds.items():
                pe = inhalations(P[key][1][:, k])[:, 0]
                for (t, _, _), ok in zip(ev, conf):
                    d = np.min(np.abs(pe - t)) / 5 if len(pe) else 99
                    s2 = stats.setdefault((m, r['task'], c, ok), [0, 0, 0])
                    s2[0] += 1; s2[1] += d <= .4; s2[2] += d <= 1.0
    lines = ['# Breath-sound-confirmed inhalations and model recall (post hoc)', '',
             'Confirmed = breath sound overlaps the belt rise [trough - 0.2 s, peak + 0.2 s]. This window was set after seeing that breath sounds cluster 0.2-0.6 s before the peak (post-hoc correction).', '',
             '## Correspondence between belt inhalations and breath sounds', '', '|Task|Belt|Belt inhalations|Fraction confirmed by breath sound|Breath sounds|Fraction within a belt rise|', '|---|---|---|---|---|---|']
    for t in ('song', 'scale', 'reading'):
        for c in 'AB':
            s = stats[(t, c)]
            lines.append(f"|{t}|{c}|{s['belt']}|{s['conf'] / s['belt']:.2f}|{s['sounds']}|{s['sounds_in_rise'] / max(s['sounds'], 1):.2f}|")
    lines += ['', '## Model recall of confirmed/unconfirmed inhalations (predicted onset within ±0.4 s / ±1.0 s of measured trough)', '',
              '|Model|Task|Belt|Confirmed recall ±0.4s|Confirmed ±1.0s|Unconfirmed recall ±0.4s|Unconfirmed ±1.0s|', '|---|---|---|---|---|---|---|']
    for m in preds:
        for t in ('song', 'scale', 'reading'):
            for c in 'AB':
                a = stats.get((m, t, c, True), [1, 0, 0]); b = stats.get((m, t, c, False), [1, 0, 0])
                lines.append(f'|{m}|{t}|{c}|{a[1] / a[0]:.2f}|{a[2] / a[0]:.2f}|{b[1] / b[0]:.2f}|{b[2] / b[0]:.2f}|')
    (ROOT / 'diagnostics_r10' / 'confirmed_inhalations_model_recall.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
