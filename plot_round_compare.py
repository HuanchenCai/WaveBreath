"""Shared per-round comparison figures (user rule: every round ships figures, not only numbers).
Usage: python plot_round_compare.py OUT_STEM "TITLE" model1=label1 model2=label2 ...
Produces OUT_STEM_waveforms.png (fixed S1-S6 recordings, measured vs each model, chest/abdomen),
OUT_STEM_depth.png (matched inhalation depth: predicted vs measured, per model, with slope) and
OUT_STEM_summary.png (bar chart: waveform CCC, depth slope, top-10% depth ratio, A-B correlation).
Recordings missing from a model (e.g. calibration segments) are left blank for that model."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from breath_parameters import inhalations, match
from compare_r19 import load

ROOT = Path(__file__).resolve().parent
COLORS = ['#7f7f7f', '#d1495b', '#2f8f5b', '#e08a1e', '#7b4fb0', '#1f9bb5']
TASK = {'song': 'Song', 'scale': 'Scale', 'reading': 'Reading'}


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


def depth_pairs(y, p, k):
    sd = y[:, k].std() + 1e-9; ey, ep = inhalations(y[:, k]), inhalations(p[:, k]); m = match(ey[:, 0], ep[:, 0], 2)
    dy = (y[ey[:, 1].astype(int), k] - y[ey[:, 0].astype(int), k]) / sd if len(ey) else np.array([])
    dm = np.array([dy[i] for i, _ in m]); dp = np.array([(p[int(ep[j, 1]), k] - p[int(ep[j, 0]), k]) / sd for _, j in m])
    top = dm >= np.quantile(dy, .9) if len(dy) >= 6 else np.zeros(len(dm), bool)
    return dm, dp, top


def main():
    stem, title, specs = sys.argv[1], sys.argv[2], [s.split('=', 1) for s in sys.argv[3:]]
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    sel = json.loads((ROOT / 'r13_gallery' / 'selection_key_singer_names.json').read_text(encoding='utf-8'))
    data = {m: load(m) for m, _ in specs}
    # 1. waveforms
    fig, axs = plt.subplots(len(sel), 2, figsize=(18, 2.6 * len(sel)), constrained_layout=True, squeeze=False)
    for r, s in enumerate(sel):
        key = (s['singer'], s['number']); y = next((d[key][0] for d in data.values() if key in d), None)
        if y is None:
            continue
        t = np.arange(len(y)) / 5
        for c in range(2):
            ax = axs[r, c]; ax.plot(t, y[:, c], color='black', lw=1.4, label='Measured')
            parts = []
            for (m, lab), col in zip(specs, COLORS):
                if key in data[m]:
                    p = data[m][key][1]; ax.plot(t, p[:, c], color=col, lw=1, alpha=.85, label=lab)
                    parts.append(f'{lab} {ccc(y[:, c], p[:, c]):.2f}')
            ax.set_title(f'{s["code"]} ({TASK[s["task"]]}) {"A chest" if c == 0 else "B abdomen"} | CCC: ' + '; '.join(parts), fontsize=8)
            ax.set_xlim(0, t[-1]); ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7, ncol=len(specs) + 1, loc='upper right')
    fig.suptitle(f'{title}: the same 6 recordings (black = measured)', fontsize=12)
    fig.savefig(ROOT / f'{stem}_waveforms.png', dpi=85); plt.close(fig)
    # 2. depth scatter + 3. summary
    fig, axs = plt.subplots(2, len(specs), figsize=(4 * len(specs), 7.5), constrained_layout=True, squeeze=False)
    summ = {}
    for j, (m, lab) in enumerate(specs):
        for c in range(2):
            dms, dps, tops, cc, abp, abm = [], [], [], [], [], []
            for key, (y, p) in data[m].items():
                dm, dp, top = depth_pairs(y, p, c); dms += list(dm); dps += list(dp); tops += list(top)
                cc.append(ccc(y[:, c], p[:, c]))
                if c == 0:
                    abp.append(np.corrcoef(p.T)[0, 1]); abm.append(np.corrcoef(y.T)[0, 1])
            dms, dps, tops = np.array(dms), np.array(dps), np.array(tops, bool)
            ax = axs[c, j]; ax.scatter(dms, dps, s=4, alpha=.25, color=COLORS[j % len(COLORS)])
            ax.scatter(dms[tops], dps[tops], s=6, alpha=.6, color='black', label='Largest 10% inhalations')
            sl, ic = np.polyfit(dms, dps, 1); lim = np.percentile(dms, 99)
            ax.plot([0, lim], [0, lim], 'k--', lw=1); ax.plot([0, lim], [ic, ic + sl * lim], color=COLORS[j % len(COLORS)], lw=2)
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
            ax.set_title(f'{lab} | {"chest" if c == 0 else "abdomen"} | slope {sl:.2f}', fontsize=9); ax.set_xlabel('Measured inhalation depth (SD)', fontsize=8)
            ax.set_ylabel('Predicted inhalation depth', fontsize=8)
            summ.setdefault(lab, {})[f'CCC {"chest" if c == 0 else "abdomen"}'] = np.median(cc)
            summ[lab][f'Depth slope {"chest" if c == 0 else "abdomen"}'] = sl
            summ[lab][f'Top-inhalation ratio {"chest" if c == 0 else "abdomen"}'] = float(np.median(dps[tops] / dms[tops]))
            if c == 0:
                summ[lab]['A-B corr (measured %.2f)' % np.median(abm)] = float(np.median(abp))
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f'{title}: inhalation depth, predicted vs measured (dashed = identity, coloured = fitted slope)', fontsize=12)
    fig.savefig(ROOT / f'{stem}_depth.png', dpi=85); plt.close(fig)
    keys = list(next(iter(summ.values())))
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(keys)), 4.2), constrained_layout=True)
    w = .8 / len(specs)
    for j, (_, lab) in enumerate(specs):
        vals = [summ[lab][k] for k in keys]; xs = np.arange(len(keys)) + j * w
        ax.bar(xs, vals, w, color=COLORS[j % len(COLORS)], label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + .01, f'{v:.2f}', ha='center', fontsize=6, rotation=90)
    ax.set_xticks(np.arange(len(keys)) + w * (len(specs) - 1) / 2); ax.set_xticklabels(keys, fontsize=8)
    ax.axhline(1, color='gray', lw=.8, ls=':'); ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left'); ax.set_title(f'{title}: key metrics (depth slope and top-inhalation ratio ideally 1)', fontsize=11)
    fig.savefig(ROOT / f'{stem}_summary.png', dpi=90); plt.close(fig)
    import base64
    imgs = ''.join(f'<h2>{h}</h2><img src="data:image/png;base64,{base64.b64encode((ROOT / f"{stem}_{s}.png").read_bytes()).decode()}">'
                   for s, h in (('summary', 'Key metrics'), ('waveforms', 'Waveforms of the same 6 recordings'), ('depth', 'Inhalation depth: predicted vs measured')))
    (ROOT / f'{stem}.html').write_text(f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        '<style>body{margin:0;background:#fff;color:#1d1d1f;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:0 auto;padding:16px}img{width:100%;height:auto}</style>'
        f'</head><body><main><h1>{title}</h1>{imgs}</main></body></html>', encoding='utf-8')
    print('written', stem)


if __name__ == '__main__':
    main()
