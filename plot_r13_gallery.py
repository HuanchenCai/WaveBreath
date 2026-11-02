"""Round-13 waveform gallery: every configuration on the SAME six recordings.

Selection rule (fixed before plotting, independent of the new models): using the original
main model r11_tgt_dc, per task (song / scale / reading) take the recordings closest to the
25th and 75th percentile of the mean A/B CCC, with six distinct singers. Full recordings are
shown; titles give whole-record r and CCC. Singer names are replaced by codes.
Output: round13_waveform_gallery.html (self-contained, offline) + PNGs in r13_gallery/.
"""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_r11_r12 import load

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'r13_gallery'
CONFIGS = [
    ('r11_tgt_dc', 'Baseline model (WavLM layer 16 + TCN, untuned, single seed)'),
    ('r13_tcn', 'Batch 1 TCN (temporal convolutional network), tuned + 3-seed ensemble'),
    ('r13_bigru', 'Batch 1 BiGRU (bidirectional gated recurrent unit)'),
    ('r13_convlstm', 'Batch 1 Conv-LSTM (convolution + long short-term memory)'),
    ('r13_transformer', 'Batch 1 Transformer (self-attention)'),
    ('r13_unet', 'Batch 1 U-Net (multi-scale encoder-decoder)'),
    ('r13b_convlstm_wavlm16_separate', 'Conv-LSTM: song/speech/scale trained separately'),
    ('r13b_convlstm_wavlm16_tasklabel', 'Conv-LSTM: joint training + task label'),
    ('r13b_convlstm_wavlm_all_joint', 'Conv-LSTM: weighted input from all 25 WavLM layers'),
    ('r13b_convlstm_mert_all_joint', 'Conv-LSTM: all-layer input from the MERT music encoder'),
    ('r13b_convlstm_wavlm16_egg_joint', 'Conv-LSTM: WavLM + EGG (electroglottography) features'),
    ('r13b_convlstm_egg_only_joint', 'Conv-LSTM: EGG features only (no microphone)'),
    ('r13b_convlstm_wavlm16_egg_tasklabel', 'Conv-LSTM: task label + EGG'),
    ('r13c_convlstm_50hz', 'Conv-LSTM: 50 Hz two-scale (plotted at native 50 Hz)'),
    ('r13b_unet_wavlm16_separate', 'U-Net: trained separately'),
    ('r13b_unet_wavlm16_tasklabel', 'U-Net: joint training + task label'),
    ('r13b_unet_wavlm_all_joint', 'U-Net: weighted input from all 25 WavLM layers'),
    ('r13b_unet_mert_all_joint', 'U-Net: MERT all-layer input'),
    ('r13b_unet_wavlm16_egg_joint', 'U-Net: WavLM + EGG'),
    ('r13b_unet_egg_only_joint', 'U-Net: EGG only'),
    ('r13b_unet_wavlm16_egg_tasklabel', 'U-Net: task label + EGG'),
    ('r13c_unet_50hz', 'U-Net: 50 Hz two-scale (plotted at native 50 Hz)'),
    ('r15_ens_convlstm_unet', 'Recommended: Conv-LSTM + U-Net cross-family ensemble'),
]
TASK_CN = {'song': 'Song', 'scale': 'Scale', 'reading': 'Reading'}


def ccc(a, b):
    return 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8)


def load50(name):
    out = {}
    for f in range(1, 6):
        with np.load(ROOT / 'family_runs' / name / f'fold{f}_waveforms_50hz.npz') as z:
            meta = json.loads(str(z['metadata'])); off = z['offsets']
            for j, m in enumerate(meta):
                out[(m['singer'], str(m['number']))] = (z['target'][off[j]:off[j + 1]], z['prediction'][off[j]:off[j + 1]])
    return out


def select(base):
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    task = {(a['singer'], str(a['number'])): a['task'] for a in info}
    score = {k: np.mean([ccc(y[:, c], p[:, c]) for c in range(2)]) for k, (y, p) in base.items()}
    chosen, used = [], set()
    for t in ('song', 'scale', 'reading'):
        keys = sorted([k for k in score if task[k] == t], key=lambda k: score[k])
        for q in (.25, .75):
            target = np.quantile([score[k] for k in keys], q)
            for k in sorted(keys, key=lambda k: abs(score[k] - target)):
                if k[0] not in used:
                    chosen.append((k, t, q)); used.add(k[0]); break
    return chosen


def main():
    OUT.mkdir(exist_ok=True)
    base = load('r11_tgt_dc')
    chosen = select(base)
    codes = {k: f'S{i + 1}' for i, (k, _, _) in enumerate(chosen)}
    (OUT / 'selection_key_singer_names.json').write_text(json.dumps(
        [{'code': codes[k], 'singer': k[0], 'number': k[1], 'task': t, 'baseline_quantile': q} for k, t, q in chosen],
        ensure_ascii=False, indent=1), encoding='utf-8')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    sections, summary = [], []
    for name, label in CONFIGS:
        data = load50(name) if name.startswith('r13c_') else load(name)
        fs = 50 if name.startswith('r13c_') else 5
        fig, axs = plt.subplots(len(chosen), 2, figsize=(16, 2.3 * len(chosen)), constrained_layout=True, squeeze=False)
        vals = []
        for row, (k, t, q) in enumerate(chosen):
            y, p = data[k]; tt = np.arange(len(y)) / fs
            for c in range(2):
                ax = axs[row, c]
                ax.plot(tt, y[:, c], color='#2864aa', lw=1.1, label='Measured')
                ax.plot(tt, p[:, c], color='#d1495b', lw=1.1, alpha=.85, label='Predicted')
                r = np.corrcoef(y[:, c], p[:, c])[0, 1]; cc = ccc(y[:, c], p[:, c]); vals.append((c, r, cc))
                ax.set_title(f'{codes[k]} ({TASK_CN[t]}) - {"A chest" if c == 0 else "B abdomen"} - full record r={r:.2f}  CCC={cc:.2f}', fontsize=9)
                ax.set_xlim(0, tt[-1]); ax.tick_params(labelsize=7)
                if row == len(chosen) - 1:
                    ax.set_xlabel('Time (s)', fontsize=8)
                if c == 0:
                    ax.set_ylabel('Relative amplitude', fontsize=8)
        axs[0, 0].legend(fontsize=8, loc='upper right')
        fig.suptitle(label, fontsize=12)
        png = OUT / f'{name}.png'; fig.savefig(png, dpi=90); plt.close(fig)
        mA = np.median([cc for c, r, cc in vals if c == 0]); mB = np.median([cc for c, r, cc in vals if c == 1])
        summary.append((label, mA, mB))
        b64 = base64.b64encode(png.read_bytes()).decode()
        sections.append(f'<section id="{name}"><h2>{label}</h2><p class="meta">Median CCC on these 6 segments: A {mA:.2f} / B {mB:.2f} (scores on all 197 segments are in the results summary)</p>'
                        f'<img alt="{label}" src="data:image/png;base64,{b64}"></section>')
        print(name, 'done', flush=True)
    toc = ''.join(f'<li><a href="#{n}">{l}</a></li>' for n, l in CONFIGS)
    rows = ''.join(f'<tr><td>{l}</td><td>{a:.2f}</td><td>{b:.2f}</td></tr>' for l, a, b in summary)
    sel = ''.join(f'<li>{codes[k]}: {TASK_CN[t]}, baseline-model score near the {int(q * 100)}th percentile of this task</li>' for k, t, q in chosen)
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waveform gallery by configuration</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#ececf0;--mut:#a0a0a8;--line:#2c2c32}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:16px}} img{{width:100%;height:auto;background:#fff;border-radius:6px}}
section{{border-top:1px solid var(--line);padding-top:12px;margin-top:20px}} .meta{{color:var(--mut)}}
table{{border-collapse:collapse}} td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left}}
</style></head><body><main>
<h1>Round 13: waveform predictions of each configuration on the same recordings</h1>
<p>Blue = measured breathing belt, red = model prediction (outer held-out singers, i.e. singers the model has not seen). A = chest belt, B = abdominal belt. The y-axis is per-segment standardized relative amplitude (not millilitres). r = Pearson correlation; CCC = concordance correlation coefficient (penalizes shape, offset and amplitude mismatch; closer to 1 is better).</p>
<p><b>Selection rule (not chosen by the new models)</b>: based on the baseline model's scores, for each task one segment near the 25th and one near the 75th percentile, 6 segments from 6 different singers; all configurations use the same recordings at full length. Singer names are replaced by codes.</p><ul>{sel}</ul>
<p class="meta">Note: the 6 segments are for visual inspection only and are not representative; overall scores are in the round-13 results summary and single-factor reports. Outer singers were inspected repeatedly in earlier rounds; results are exploratory.</p>
<h2>Median CCC on these 6 segments</h2><table><tr><th>Configuration</th><th>A chest</th><th>B abdomen</th></tr>{rows}</table>
<h2>Contents</h2><ol>{toc}</ol>{''.join(sections)}</main></body></html>"""
    (ROOT / 'round13_waveform_gallery.html').write_text(html, encoding='utf-8')
    print('html written')


if __name__ == '__main__':
    main()
