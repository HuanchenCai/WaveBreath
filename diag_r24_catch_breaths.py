"""Catch breaths: inhalations that happen without a real pause.
Measured inhalations (frozen extractor) per band are classified by their rise interval
[trough, peak]: 'pause breath' if it overlaps an unvoiced run >= 0.4 s, otherwise 'catch breath'
(inside phrasing or in a gap < 0.4 s). Support: share of each class confirmed by an audible
breath sound (round-14 detector, overlap with the rise +-0.2 s). Detection: predicted inhalation
onset within 0.4 s / 1.0 s of the measured trough. The pause rule cannot detect catch breaths by
definition. Figures: recall bars and example catch breaths. No training."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from breath_parameters import inhalations
from compare_r19 import load
from diag_r9_sawtooth_baseline import runs

ROOT = Path(__file__).resolve().parent
TASKS = {'song': 'Song', 'scale': 'Scale', 'reading': 'Reading'}


def main(models=('r15_ens_convlstm_unet', 'r13_convlstm')):
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        V, off = z['voiced'], z['offsets']
    sounds = {(r['singer'], r['number']): r['breath_sounds_s'] for r in
              json.loads((ROOT / 'diagnostics_r10' / 'breath_sound_reference.json').read_text(encoding='utf-8'))}
    data = {m: load(m) for m in models}
    stats, examples = {}, []
    for i, a in enumerate(info):
        key = (a['singer'], str(a['number'])); v = V[off[i]:off[i + 1]]
        pauses = [(s, e) for s, e in runs(~v) if e - s >= 2]
        y = data[models[0]][key][0]
        for k, c in enumerate('AB'):
            ev = inhalations(y[:, k])
            for t, p, d in ev:
                t, p = int(t), int(p)
                kind = 'pause' if any(s <= p and e >= t for s, e in pauses) else 'catch'
                snd = any(s <= p / 5 + .2 and e >= t / 5 - .2 for s, e in sounds[key])
                rec = {'task': a['task'], 'band': c, 'kind': kind, 'sound': snd, 'depth': d}
                for m in models:
                    pe = inhalations(data[m][key][1][:, k])[:, 0]
                    dist = np.min(np.abs(pe - t)) / 5 if len(pe) else 99
                    rec[m] = (dist <= .4, dist <= 1.0)
                stats.setdefault((a['task'], c), []).append(rec)
                if kind == 'catch' and c == 'A' and d >= 1.5 and len(examples) < 6 and a['task'] == 'song':
                    if not any(ex[0] == key for ex in examples):
                        examples.append((key, t, p, {m: data[m][key][1] for m in models}, y))
    L = ['# Catch breaths (inhalations without a pause within a phrase), no training', '',
         'Classification: inhalation rise overlapping a pause of >=0.4 s = pause breath, otherwise = catch breath. Breath-sound confirmed = breath sound detected within ±0.2 s of the rise. Detected = predicted onset within ±0.4 / ±1.0 s of the measured trough. A pause rule cannot detect catch breaths by definition.', '',
         '|Task|Belt|Inhalations|Catch-breath share|Catch breaths confirmed by breath sound|Pause breaths confirmed|Mean depth catch/pause|' + '|'.join(f'{m} catch detected ±0.4/±1.0|{m} pause detected ±0.4/±1.0' for m in models) + '|',
         '|---|---|---|---|---|---|---|' + '---|---|' * len(models)]
    bars = {}
    for (t, c), rs in sorted(stats.items()):
        cat = [r for r in rs if r['kind'] == 'catch']; pau = [r for r in rs if r['kind'] == 'pause']
        cells = [f'{len(rs)}', f'{len(cat) / len(rs):.0%}', f'{np.mean([r["sound"] for r in cat]):.0%}', f'{np.mean([r["sound"] for r in pau]):.0%}',
                 f'{np.mean([r["depth"] for r in cat]) / np.mean([r["depth"] for r in pau]):.2f}']
        for m in models:
            for grp in (cat, pau):
                cells.append(f'{np.mean([r[m][0] for r in grp]):.2f}/{np.mean([r[m][1] for r in grp]):.2f}')
        L.append(f'|{TASKS[t]}|{c}|' + '|'.join(cells) + '|')
        bars[(t, c)] = [np.mean([r[models[0]][1] for r in cat]), np.mean([r[models[0]][1] for r in pau])]
    out = ROOT / 'round_figures'; out.mkdir(exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1 + len(examples), 1, figsize=(15, 3 + 2.6 * len(examples)), constrained_layout=True)
    labs = [f'{TASKS[t]} - {"chest" if c == "A" else "abdomen"}' for t, c in bars]; x = np.arange(len(labs))
    axs[0].bar(x - .2, [b[0] for b in bars.values()], .4, color='#d1495b', label='Catch breath (no pause)')
    axs[0].bar(x + .2, [b[1] for b in bars.values()], .4, color='#2864aa', label='Pause breath')
    axs[0].set_xticks(x); axs[0].set_xticklabels(labs); axs[0].set_ylabel('Model detection rate (±1 s)'); axs[0].legend()
    axs[0].set_title('Recommended model: detection of the two breath types (a pause rule detects 0 catch breaths by definition)')
    for ax, (key, t, p, preds, y) in zip(axs[1:], examples):
        s0, s1 = max(0, t - 40), min(len(y), p + 40); tt = np.arange(s0, s1) / 5
        ax.plot(tt, y[s0:s1, 0], 'k', lw=1.5, label='Measured chest belt')
        for (m, pr), col in zip(preds.items(), ('#d1495b', '#2f8f5b')):
            ax.plot(tt, pr[s0:s1, 0], color=col, lw=1, label=m)
        ax.axvspan(t / 5, p / 5, color='orange', alpha=.25, label='Catch breath (measured rise)')
        i = [n for n, a in enumerate(info) if (a['singer'], str(a['number'])) == key][0]
        v = V[off[i]:off[i + 1]][s0:s1]
        ax.fill_between(tt, ax.get_ylim()[0], ax.get_ylim()[0] + .3, where=v, color='gray', alpha=.4, step='mid', label='Voiced')
        ax.set_title(f'Catch-breath example (song) {t / 5:.1f}-{p / 5:.1f} s', fontsize=9); ax.legend(fontsize=7, ncol=5, loc='upper right')
    fig.savefig(out / 'catch_breaths.png', dpi=85); plt.close(fig)
    (ROOT / 'catch_breath_analysis.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
