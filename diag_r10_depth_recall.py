"""Round-10 post-hoc sensitivity (added after seeing step-2 results): recall of measured
inhalations by depth bin, and precision of predicted events by predicted depth bin.
Depth in units of each waveform's own std."""
import json
from pathlib import Path
import numpy as np
from breath_parameters import inhalations, match
from diag_r10_parameter_eval import load_preds

ROOT = Path(__file__).resolve().parent
BINS = [(.5, 1.), (1., 1.5), (1.5, 2.5), (2.5, 99)]


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_all, off = z['y'], z['offsets']
    pred = load_preds('wavlm_large16_tcn')
    lines = ['# Post-hoc sensitivity: recall and precision by inhalation depth (main model)', '',
             'Depth = peak - trough, in units of each waveform\'s SD. Post-hoc, not pre-registered.', '']
    for tol, name in ((2, '±0.4s'), (5, '±1.0s')):
        lines += [f'## Tolerance {name}', '', '|Task|Belt|Depth bin|Measured events|Recall|Predicted events|Precision|', '|---|---|---|---|---|---|---|']
        for task in ('all', 'song', 'scale', 'reading'):
            for k, c in enumerate('AB'):
                acc = {b: [0, 0, 0, 0] for b in BINS}
                for i, a in enumerate(info):
                    if task != 'all' and a['task'] != task:
                        continue
                    y = y_all[off[i]:off[i + 1], k]; p = pred[(a['singer'], str(a['number']))][:, k]
                    ey, ep = inhalations(y), inhalations(p)
                    m = match(ey[:, 0], ep[:, 0], tol); my = {i_ for i_, _ in m}; mp = {j for _, j in m}
                    for b in BINS:
                        iy = [q for q in range(len(ey)) if b[0] <= ey[q, 2] < b[1]]
                        ip = [q for q in range(len(ep)) if b[0] <= ep[q, 2] < b[1]]
                        acc[b][0] += len(iy); acc[b][1] += sum(q in my for q in iy)
                        acc[b][2] += len(ip); acc[b][3] += sum(q in mp for q in ip)
                for b in BINS:
                    n, h, n2, h2 = acc[b]
                    lines.append(f'|{task}|{c}|{b[0]}–{b[1] if b[1] < 99 else "∞"}|{n}|{h / max(n, 1):.2f}|{n2}|{h2 / max(n2, 1):.2f}|')
        lines.append('')
    (ROOT / 'diagnostics_r10' / 'recall_by_depth_posthoc.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
