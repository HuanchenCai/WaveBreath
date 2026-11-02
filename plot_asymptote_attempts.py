"""Model-space asymptote: every attempt with complete 197-segment outer predictions, scored with the
same metric (singer-median CCC and r, chest/abdomen), ordered by when it was run (file time of
results), with the running best. Reference lines: original main model and the belt-autoencoder
oracle (belt reconstructed from the true belt, round 5/6; upper bound for a 32x4 latent, not a
voice model). Output: round_figures/asymptote_attempts.png + asymptote_attempts.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def score(d):
    by = {}
    for f in range(1, 6):
        with np.load(d / f'fold{f}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                y = z['target'][off[j]:off[j + 1], :2]; p = z['prediction'][off[j]:off[j + 1], :2]
                v = []
                for k in range(2):
                    a, b = y[:, k], p[:, k]
                    v += [2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8), np.corrcoef(a, b)[0, 1]]
                by.setdefault(m['singer'], []).append(v)
    n = sum(len(v) for v in by.values())
    return n, np.median([np.median(v, axis=0) for v in by.values()], axis=0)


def group(name):
    for pre, g in (('r19b_', 'Personal calibration'), ('r19c_', 'Encoder fine-tuning'), ('r2', 'Round 20+'), ('r1', 'Rounds 11-19'), ('loss_', 'Loss functions'),
                   ('large_latent', 'Generative/latent'), ('large_', 'Whisper family'), ('wavlm', 'WavLM family'), ('reference', 'Whisper family')):
        if name.startswith(pre):
            return g
    return 'Other'


def main():
    pts = []
    for d in sorted((ROOT / 'family_runs').iterdir()):
        if not d.is_dir() or not all((d / f'fold{k}_waveforms.npz').exists() for k in range(1, 6)):
            continue
        if d.name.startswith(('r19b_', 'r13_convlstm_seed0')) or '_band' in d.name:  # personalised subset; seed0 duplicate; single-belt models
            continue
        n, s = score(d)
        if n != 197:
            continue
        t = (d / 'results.json').stat().st_mtime if (d / 'results.json').exists() else (d / 'fold5_waveforms.npz').stat().st_mtime
        pts.append({'name': d.name, 'group': group(d.name), 'time': t, 'A_ccc': s[0], 'A_r': s[1], 'B_ccc': s[2], 'B_r': s[3]})
    pts.sort(key=lambda p: p['time'])
    (ROOT / 'round_figures' / 'asymptote_attempts.json').write_text(json.dumps(pts, ensure_ascii=False, indent=1), encoding='utf-8')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    groups = sorted({p['group'] for p in pts}); cmap = plt.get_cmap('tab10')
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for ax, key, lab in zip(axs.flat, ('A_ccc', 'B_ccc', 'A_r', 'B_r'), ('Chest CCC', 'Abdomen CCC', 'Chest r', 'Abdomen r')):
        x = np.arange(len(pts)); y = np.array([p[key] for p in pts])
        for gi, g in enumerate(groups):
            idx = [i for i, p in enumerate(pts) if p['group'] == g]
            ax.scatter(x[idx], y[idx], s=28, color=cmap(gi % 10), label=g, alpha=.85)
        ax.step(x, np.maximum.accumulate(y), where='post', color='black', lw=1.5, label='Running best')
        base = next((p[key] for p in pts if p['name'] == 'wavlm_large16_tcn'), None)
        if base:
            ax.axhline(base, color='gray', ls=':', lw=1, label='Baseline model (WavLM layer 16 + TCN)')
        ax.set_title(f'{lab}: all {len(pts)} attempts (ordered by completion time)', fontsize=11); ax.set_xlabel('Attempt index (chronological)'); ax.set_ylabel(lab)
        ax.set_ylim(max(0, y.min() - .05), min(1, y.max() + .05))
    axs[0, 0].legend(fontsize=7, ncol=2, loc='lower right')
    fig.suptitle('Model-space asymptote: all attempts on the same split and metric (singer median)', fontsize=13)
    fig.savefig(ROOT / 'round_figures' / 'asymptote_attempts.png', dpi=90); plt.close(fig)
    best = {k: max(pts, key=lambda p: p[k]) for k in ('A_ccc', 'B_ccc', 'A_r', 'B_r')}
    for k, p in best.items():
        print(k, f'{p[k]:.3f}', p['name'])
    print(len(pts), 'attempts')


if __name__ == '__main__':
    main()
