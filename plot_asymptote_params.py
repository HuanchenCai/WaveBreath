"""Model-space asymptote with model size on the x-axis: every attempt (same split, same metric:
singer-median CCC / r) plotted against trainable parameters (log scale; frozen encoders excluded).
Black step line = best score achieved by any model of at most that size (capacity envelope).
Hollow markers = ensembles (sum of member parameters)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
pts = [p for p in json.loads((ROOT / 'round_figures' / 'asymptote_attempts.json').read_text(encoding='utf-8')) if 'params' in p]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
groups = sorted({p['group'] for p in pts}); cmap = plt.get_cmap('tab10')
fig, axs = plt.subplots(2, 2, figsize=(16, 9.5), constrained_layout=True)
for ax, key, lab in zip(axs.flat, ('A_ccc', 'B_ccc', 'A_r', 'B_r'), ('Chest CCC', 'Abdomen CCC', 'Chest r', 'Abdomen r')):
    for gi, g in enumerate(groups):
        sub = [p for p in pts if p['group'] == g]
        ens = [p for p in sub if p['name'].startswith(('r15_ens', 'r20_'))]; single = [p for p in sub if p not in ens]
        ax.scatter([p['params'] for p in single], [p[key] for p in single], s=30, color=cmap(gi % 10), label=g, alpha=.85)
        if ens:
            ax.scatter([p['params'] for p in ens], [p[key] for p in ens], s=60, facecolors='none', edgecolors=cmap(gi % 10), lw=1.5)
    srt = sorted(pts, key=lambda p: p['params']); x = [p['params'] for p in srt]; env = np.maximum.accumulate([p[key] for p in srt])
    ax.step(x, env, where='post', color='black', lw=1.5, label='Best score up to this size')
    base = next(p[key] for p in pts if p['name'] == 'wavlm_large16_tcn'); ax.axhline(base, color='gray', ls=':', lw=1, label='Baseline model (WavLM layer 16 + TCN)')
    ax.set_xscale('log'); ax.set_xlabel('Trainable parameters (log scale; frozen encoder excluded)'); ax.set_ylabel(lab)
    ys = [p[key] for p in pts]; ax.set_ylim(np.percentile(ys, 3) - .03, max(ys) + .02); ax.set_title(lab, fontsize=11)
axs[0, 0].legend(fontsize=7, ncol=2, loc='lower right')
fig.suptitle(f'Model-size asymptote: {len(pts)} attempts, same split and metric (hollow = ensemble)', fontsize=13)
fig.savefig(ROOT / 'round_figures' / 'asymptote_params.png', dpi=90)
print('ok')
