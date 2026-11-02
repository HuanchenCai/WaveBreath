"""Learning curve and asymptote (see the learning-curve pre-registration).
For each run (fraction f, draw d): mean number of training singers over folds and the
singer-median CCC / r for chest and abdomen. Fit y = a - b * n^(-c) (a = asymptote) with
scipy curve_fit; 2000 bootstrap resamples over singers of the per-singer scores give a CI for a.
Output: round_figures/learning_curve.png, learning_curve.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
RUNS = [(f, d) for f in (0.25, .5, .75) for d in (0, 1)] + [(1.0, 0)]


def model(n, a, b, c):
    return a - b * np.power(n, -c)


def per_singer(name):
    by = {}
    for fold in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{fold}_waveforms.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                y = z['target'][off[j]:off[j + 1]]; p = z['prediction'][off[j]:off[j + 1]]
                v = []
                for k in range(2):
                    a, b = y[:, k], p[:, k]
                    v += [2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8), np.corrcoef(a, b)[0, 1]]
                by.setdefault(m['singer'], []).append(v)
    return {s: np.median(v, axis=0) for s, v in by.items()}


def main():
    pts = []
    for f, d in RUNS:
        name = 'r13b_convlstm_wavlm16_joint' + ('' if f == 1.0 else f'_frac{f:g}_d{d}')
        res = json.loads((ROOT / 'family_runs' / name / 'results.json').read_text(encoding='utf-8'))
        ntrain = np.mean([g['n_train'] for fo in res['folds'] for g in fo['groups'].values()])
        singers_per_rec = None
        pts.append({'name': name, 'frac': f, 'draw': d, 'n_train_recordings': float(ntrain), 'scores': per_singer(name)})
    # convert recordings to singers: about 197/32 recordings per singer
    for p in pts:
        p['n_singers'] = p['n_train_recordings'] / (197 / 32)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.8), constrained_layout=True)
    out = {}
    rng = np.random.default_rng(0)
    for ax, (k, lab) in zip(axs, ((0, 'Chest CCC'), (2, 'Abdomen CCC'), (1, 'Chest r'), (3, 'Abdomen r'))):
        n = np.array([p['n_singers'] for p in pts]); singers = sorted(pts[0]['scores'])
        y = np.array([np.median([p['scores'][s][k] for s in singers]) for p in pts])
        ax.scatter(n, y, s=40, color='#2864aa', zorder=3)
        try:
            popt, _ = curve_fit(model, n, y, p0=[y.max() + .02, .5, .7], bounds=([0, 0, .05], [1.2, 10, 3]), maxfev=20000)
            boots = []
            for _ in range(2000):
                ss = rng.choice(singers, len(singers))
                yb = np.array([np.median([p['scores'][s][k] for s in ss]) for p in pts])
                try:
                    boots.append(curve_fit(model, n, yb, p0=popt, bounds=([0, 0, .05], [1.2, 10, 3]), maxfev=5000)[0][0])
                except RuntimeError:
                    pass
            lo, hi = np.percentile(boots, [2.5, 97.5])
            xs = np.linspace(n.min() * .8, 200, 400)
            ax.plot(xs, model(xs, *popt), color='#d1495b', lw=2, label=f'Fitted asymptote {popt[0]:.3f} (lower CI {lo:.3f}, upper bound not reached)' if hi >= 1.19 else f'Fitted asymptote {popt[0]:.3f} [{lo:.3f}, {hi:.3f}]')
            ax.axhline(popt[0], color='#d1495b', ls='--', lw=1)
            out[lab] = {'asymptote': popt[0], 'ci': [lo, hi], 'b': popt[1], 'c': popt[2], 'points': list(zip(n.tolist(), y.tolist())),
                        'pred_at_64_singers': float(model(64, *popt)), 'pred_at_100_singers': float(model(100, *popt))}
        except RuntimeError:
            out[lab] = {'points': list(zip(n.tolist(), y.tolist())), 'fit': 'failed'}
        ax.set_xscale('log'); ax.set_xlabel('Training singers (log scale)'); ax.set_ylabel(lab); ax.set_title(lab); ax.legend(fontsize=8, loc='lower right')
        ax.axvline(26, color='gray', ls=':', lw=1)
    fig.suptitle('Data-size asymptote: performance vs number of training singers (Conv-LSTM, fixed outer test singers; dotted line = current ~26 training singers)', fontsize=12)
    fig.savefig(ROOT / 'round_figures' / 'learning_curve.png', dpi=90); plt.close(fig)
    (ROOT / 'round_figures' / 'learning_curve.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: {kk: v[kk] for kk in v if kk != 'points'} for k, v in out.items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
