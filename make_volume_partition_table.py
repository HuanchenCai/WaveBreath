"""Detailed table: chest (A), abdomen (B), sum S = A + B (lung-volume proxy, uncalibrated),
partition D = B - A, and D_perp (D after removing S), measured vs predicted, for the key models.
Singer-level medians with 2000-sample singer bootstrap 95% CI; share of singers with r > 0.8;
paired singer-level difference S vs A. Outputs volume_partition_table.md / .html and a figure."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_r19 import load
from diag_r10_parameter_eval import boot_ci

ROOT = Path(__file__).resolve().parent
MODELS = [('r11_tgt_dc', 'Baseline model (WavLM layer 16 + TCN)'), ('r15_ens_convlstm_unet', 'Recommended model (Conv-LSTM + U-Net ensemble)'),
          ('r13_convlstm', 'Conv-LSTM (round 13 tuned)'), ('r13_unet', 'U-Net (round 13 tuned)'),
          ('r13b_convlstm_wavlm16_joint_lossdiffcorr', 'Conv-LSTM + coupling loss (round 17)'),
          ('r19b_unet_none', 'U-Net uncalibrated (other segments)'), ('r19b_unet_finetune', 'U-Net personal calibration (other segments)'),
          ('large_latent4_diffusion', 'Diffusion generator (mean of 4 samples)')]
SIG = [('A', 'Chest belt'), ('B', 'Abdominal belt'), ('S', 'Chest + abdomen sum'), ('D', 'Abdomen - chest partition'), ('Dp', 'Partition (sum removed)')]
TASK = {'all': 'All', 'song': 'Song', 'scale': 'Scale', 'reading': 'Reading'}


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


def resid(x, z):
    Z = np.c_[z, np.ones(len(z))]
    return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]


def signals(w):
    a, b = w[:, 0], w[:, 1]
    return {'A': a, 'B': b, 'S': a + b, 'D': b - a}


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    task = {(a['singer'], str(a['number'])): a['task'] for a in info}
    res = {}
    for name, lab in MODELS:
        data = load(name)
        for t in TASK:
            by = {}
            for k, (y, p) in data.items():
                if t != 'all' and task[k] != t:
                    continue
                sy, sp = signals(y), signals(p)
                sy['Dp'], sp['Dp'] = resid(sy['D'], sy['S']), resid(sp['D'], sp['S'])
                by.setdefault(k[0], []).append({s: (np.corrcoef(sy[s], sp[s])[0, 1], ccc(sy[s], sp[s])) for s, _ in SIG})
            per_singer = {s: {sig: np.median([[v[sig][0], v[sig][1]] for v in vs], axis=0) for sig, _ in SIG} for s, vs in by.items()}
            out = {}
            for sig, _ in SIG:
                rs = [v[sig][0] for v in per_singer.values()]; cs = [v[sig][1] for v in per_singer.values()]
                out[sig] = {'r': boot_ci(rs), 'ccc': boot_ci(cs), 'r_gt_0.8': float(np.mean(np.array(rs) > .8)), 'n': len(rs)}
            d = [v['S'][0] - v['A'][0] for v in per_singer.values()]
            out['S_minus_A_r'] = boot_ci(d) + (float(np.mean(np.array(d) > 0)),)
            res[(name, t)] = out
    # ---------- markdown ----------
    L = ['# Summary table: chest, abdomen, sum (lung-volume proxy) and partition (2026-10-04, exploratory)', '',
         'Definitions: A = chest belt, B = abdominal belt (each standardized within segment, uncalibrated); **S = A + B (lung-volume proxy)**; D = B - A (partition); D_perp = partition component after removing S.',
         'Each cell is "singer median [singer-bootstrap 95% CI]"; r = Pearson correlation, CCC = concordance correlation coefficient; "r>0.8" = fraction of singers with r above 0.8. Outer singers were inspected repeatedly during exploration, so results are optimistic and need confirmation on new singers.', '',
         '## 1. All recordings (197 segments; the two personal-calibration rows use the 165 non-calibration segments)', '',
         '|Model|Signal|r|CCC|Fraction of singers r>0.8|', '|---|---|---|---|---|']
    for name, lab in MODELS:
        o = res[(name, 'all')]
        for sig, slab in SIG:
            v = o[sig]
            L.append(f'|{lab}|{slab}|{v["r"][0]:.3f} [{v["r"][1]:.3f}, {v["r"][2]:.3f}]|{v["ccc"][0]:.3f} [{v["ccc"][1]:.3f}, {v["ccc"][2]:.3f}]|{v["r_gt_0.8"]:.0%}|')
    L += ['', '## 2. Is the sum better than the chest belt alone? (singer-level paired difference in r, S - A)', '', '|Model|All|Song|Scale|Reading|', '|---|---|---|---|---|']
    for name, lab in MODELS:
        L.append(f'|{lab}|' + '|'.join(f'{res[(name, t)]["S_minus_A_r"][0]:+.3f} [{res[(name, t)]["S_minus_A_r"][1]:+.3f}, {res[(name, t)]["S_minus_A_r"][2]:+.3f}] ({res[(name, t)]["S_minus_A_r"][3]:.0%} of singers positive)' for t in TASK) + '|')
    L += ['', '## 3. By task (r, singer median)', '', '|Model|Task|Chest|Abdomen|Sum|Partition|Partition (sum removed)|', '|---|---|---|---|---|---|---|']
    for name, lab in MODELS:
        for t in ('song', 'scale', 'reading'):
            o = res[(name, t)]
            L.append(f'|{lab}|{TASK[t]}|' + '|'.join(f'{o[s]["r"][0]:.3f}' for s, _ in SIG) + '|')
    rec = res[('r15_ens_convlstm_unet', 'all')]; cal, unc = res[('r19b_unet_finetune', 'all')], res[('r19b_unet_none', 'all')]
    L += ['', '## 4. Key points', '',
          f'1. **The chest + abdomen sum is recovered best**: recommended model S r = {rec["S"]["r"][0]:.3f}, CCC = {rec["S"]["ccc"][0]:.3f}, above the chest belt ({rec["A"]["r"][0]:.3f}) and abdominal belt ({rec["B"]["r"][0]:.3f}) alone; {rec["S"]["r_gt_0.8"]:.0%} of singers have r>0.8.',
          f'2. **The partition is recovered worst**: D r = {rec["D"]["r"][0]:.3f}, CCC = {rec["D"]["ccc"][0]:.3f}; after removing the sum r = {rec["Dp"]["r"][0]:.3f}.',
          f'3. **Personal calibration improves the partition, not the sum**: on the same non-calibration recordings, S r {unc["S"]["r"][0]:.3f} -> {cal["S"]["r"][0]:.3f}, D r {unc["D"]["r"][0]:.3f} -> {cal["D"]["r"][0]:.3f}.',
          '4. Caveat (see diag_r27_partition_null.py): most of the sum/partition gap is arithmetic. With belts standardized within segment, var(D)/var(S) = (1 - rho)/(1 + rho); under an independent-error null the expected r is about 0.92 for S and 0.59 for D, so only the remaining partition deficit reflects the model.',
          '5. Limitations: uncalibrated, so S is only a lung-volume proxy (equal chest/abdomen weights); measured partition variance is only about 20% of the sum, so low partition r may partly reflect measurement noise; calibration and test-retest data are needed.',
          '', 'Figure: `round_figures/volume_partition.png`.']
    md = '\n'.join(L)
    (ROOT / 'volume_partition_table.md').write_text(md, encoding='utf-8')
    # ---------- figure ----------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    cols = ['#2864aa', '#d1495b', '#2f8f5b', '#e08a1e', '#7b4fb0']
    ax = axs[0]; w = .16; x = np.arange(len(MODELS))
    for j, (sig, slab) in enumerate(SIG):
        m = [res[(n, 'all')][sig]['r'] for n, _ in MODELS]
        ax.bar(x + (j - 2) * w, [v[0] for v in m], w, yerr=[[v[0] - v[1] for v in m], [v[2] - v[0] for v in m]], color=cols[j], label=slab, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([l.split(' (')[0] for _, l in MODELS], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('r (singer median, 95% CI)'); ax.set_ylim(0, 1); ax.set_title('Recovery of chest, abdomen, sum and partition by model')
    ax = axs[1]; x = np.arange(3)
    for j, (sig, slab) in enumerate(SIG):
        ax.bar(x + (j - 2) * w, [res[('r15_ens_convlstm_unet', t)][sig]['r'][0] for t in ('song', 'scale', 'reading')], w, color=cols[j], label=slab)
    ax.set_xticks(x); ax.set_xticklabels(['Song', 'Scale', 'Reading']); ax.set_ylim(0, 1); ax.set_title('Recommended model by task')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle('Chest + abdomen sum vs chest-abdomen partition', fontsize=13)
    out = ROOT / 'round_figures' / 'volume_partition.png'; fig.savefig(out, dpi=90, bbox_inches='tight'); plt.close(fig)
    # ---------- html ----------
    rows, html_body, table = md.split('\n'), [], []
    for line in rows + ['']:
        if line.startswith('|'):
            table.append(line); continue
        if table:
            hdr = table[0].strip('|').split('|')
            html_body.append('<div class="tw"><table><tr>' + ''.join(f'<th>{h}</th>' for h in hdr) + '</tr>' +
                             ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r.strip('|').split('|')) + '</tr>' for r in table[2:]) + '</table></div>')
            table = []
        if line.startswith('# '):
            html_body.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('## '):
            html_body.append(f'<h2>{line[3:]}</h2>')
        elif line.strip():
            html_body.append(f'<p>{line}</p>')
    body = '\n'.join(html_body).replace('**', '')
    img = f'<img src="data:image/png;base64,{base64.b64encode(out.read_bytes()).decode()}">'
    (ROOT / 'volume_partition_table.html').write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Volume and partition table</title>'
        '<style>:root{--bg:#fff;--fg:#1d1d1f;--line:#e3e3e8;--head:#f3f3f6}@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--line:#2c2c32;--head:#1f1f24}}'
        'body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1300px;margin:0 auto;padding:16px}'
        'img{width:100%;height:auto;background:#fff}.tw{overflow-x:auto}table{border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}th{background:var(--head)}</style>'
        f'</head><body><main>{img}{body}</main></body></html>', encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
