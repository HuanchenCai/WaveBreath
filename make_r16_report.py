"""Build the round-16 parameter-validation report (Markdown + self-contained HTML with figures)
from r16_param_validation.json and the record-level rows of the recommended/baseline models."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'r16_figures'
REC, BASE = 'r15_ens_convlstm_unet', 'r11_tgt_dc'
LABEL = {
    'wavlm_large16_tcn': 'Early baseline WavLM-L16 + TCN (original loss)', 'wavlm_large8_transformer': 'Early WavLM-L8 + Transformer',
    'wavlm_large16_transformer': 'Early WavLM-L16 + Transformer', 'wavlm_large24_transformer': 'Early WavLM-L24 + Transformer',
    'wavlm4_transformer': 'Early WavLM-Base L4 + Transformer', 'wavlm7_transformer': 'Early WavLM-Base L7 + Transformer',
    'wavlm12_transformer': 'Early WavLM-Base L12 + Transformer', 'wavlm7_tcn': 'Early WavLM-Base L7 + TCN',
    'large_tcn_control': 'Early Whisper-Large + TCN', 'large_transformer': 'Early Whisper-Large + Transformer',
    'large_wave_unet': 'Early Whisper-Large + Wave-U-Net', 'large_swin_unet': 'Early Whisper-Large + windowed-attention U-Net',
    'large_nab_inspired': 'Early Whisper-Large + NAB-inspired', 'large_aligned_latent4': 'Early dual-latent aligned mapping',
    'large_latent4_flow': 'Early latent Flow generator', 'large_latent4_diffusion': 'Early latent Diffusion generator',
    'reference_large_shape': 'Early Whisper-Large + shape loss',
    'loss_r010': 'Loss: +0.1x(1-r)', 'loss_r030': 'Loss: +0.3x(1-r)', 'loss_r100': 'Loss: +1x(1-r)',
    'loss_ccc030': 'Loss: +0.3x(1-CCC)', 'loss_ccc100': 'Loss: +1x(1-CCC)', 'loss_r030_ccc030': 'Loss: +0.3(1-r)+0.3(1-CCC)',
    'r11_tgt_dc': 'Round 11 DC-removed target (= baseline model reference)', 'r11_tgt_dc_singer': 'Round 11 singer-level scale target',
    'r12_tgt_dc_lam0_seedmean': 'Round 12 multitask lambda=0 (mean of 3 seeds)', 'r12_tgt_dc_lam0.3_seedmean': 'Round 12 multitask lambda=0.3',
    'r12_tgt_dc_lam1_seedmean': 'Round 12 multitask lambda=1',
    'r13_tcn': 'Round 13 TCN tuned + ensemble', 'r13_bigru': 'Round 13 BiGRU', 'r13_convlstm': 'Round 13 Conv-LSTM',
    'r13_transformer': 'Round 13 Transformer', 'r13_unet': 'Round 13 U-Net',
    'r13b_convlstm_wavlm16_separate': 'Conv-LSTM per-task training', 'r13b_convlstm_wavlm16_tasklabel': 'Conv-LSTM task label',
    'r13b_convlstm_wavlm_all_joint': 'Conv-LSTM WavLM all layers', 'r13b_convlstm_mert_all_joint': 'Conv-LSTM MERT all layers',
    'r13b_convlstm_wavlm16_egg_joint': 'Conv-LSTM + EGG', 'r13b_convlstm_egg_only_joint': 'Conv-LSTM EGG only',
    'r13b_convlstm_wavlm16_egg_tasklabel': 'Conv-LSTM task label + EGG', 'r13c_convlstm_50hz': 'Conv-LSTM 50 Hz',
    'r13b_unet_wavlm16_separate': 'U-Net per-task training', 'r13b_unet_wavlm16_tasklabel': 'U-Net task label',
    'r13b_unet_wavlm_all_joint': 'U-Net WavLM all layers', 'r13b_unet_mert_all_joint': 'U-Net MERT all layers',
    'r13b_unet_wavlm16_egg_joint': 'U-Net + EGG', 'r13b_unet_egg_only_joint': 'U-Net EGG only',
    'r13b_unet_wavlm16_egg_tasklabel': 'U-Net task label + EGG', 'r13c_unet_50hz': 'U-Net 50 Hz',
    'r15_ens_convlstm_unet': '* Recommended: Conv-LSTM + U-Net cross-family ensemble'}
PN = {'rate_per_min': 'Breathing rate', 'interval_s': 'Breath-group duration', 'insp_dur_s': 'Inspiration duration', 'depth_sd': 'Inhalation depth',
      'exp_slope_sd_s': 'Expiratory slope', 'ab_corr': 'Chest-abdomen correlation', 'ab_lag_s': 'Chest-abdomen lag', 'paradox_frac': 'Paradoxical-motion fraction'}
UNIT = {'rate_per_min': 'breaths/min', 'interval_s': 's', 'insp_dur_s': 's', 'depth_sd': 'SD', 'exp_slope_sd_s': 'SD/s',
        'ab_corr': '', 'ab_lag_s': 's', 'paradox_frac': 'fraction'}
SIG = ['rate_per_min', 'interval_s', 'insp_dur_s', 'depth_sd', 'exp_slope_sd_s']
DUAL = ['ab_corr', 'ab_lag_s', 'paradox_frac']


def grade(c):
    return 'good' if c >= .8 else ('moderate' if c >= .6 else ('weak' if c >= .4 else 'poor'))


def img(path):
    return f'<img alt="{path.stem}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def main():
    FIG.mkdir(exist_ok=True)
    R = json.loads((ROOT / 'r16_param_validation.json').read_text(encoding='utf-8'))
    rows = json.loads((ROOT / 'r16_recommended_rows.json').read_text(encoding='utf-8'))
    base_rows = json.loads((ROOT / 'r16_baseline_rows.json').read_text(encoding='utf-8'))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
    # Figure 1: measured vs predicted scatter for the recommended model
    items = [(c, k) for c in 'AB' for k in SIG] + [(None, k) for k in DUAL]
    fig, axs = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
    colors = {'song': '#2864aa', 'scale': '#d1495b', 'reading': '#3f9a5a'}
    for ax, (c, k) in zip(axs.flat, items):
        for t, col in colors.items():
            sub = [r for r in rows if r['task'] == t]
            m = [r['meas'][c][k] if c else r['meas']['dual'][k] for r in sub]
            p = [r['pred'][c][k] if c else r['pred']['dual'][k] for r in sub]
            ax.scatter(m, p, s=10, alpha=.6, color=col, label={'song': 'Song', 'scale': 'Scale', 'reading': 'Reading'}[t])
        allv = [v for r in rows for v in ((r['meas'][c][k], r['pred'][c][k]) if c else (r['meas']['dual'][k], r['pred']['dual'][k])) if np.isfinite(v)]
        lo, hi = np.percentile(allv, [1, 99]); ax.plot([lo, hi], [lo, hi], color='gray', lw=1, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        s = R[REC]['signal'][f'{c}_{k}'] if c else R[REC]['dual'][k]
        ax.set_title(f'{("Belt " + c + " ") if c else "Dual-belt "}{PN[k]}  CCC={s["ccc"]:.2f}', fontsize=10)
        ax.set_xlabel(f'Measured ({UNIT[k]})' if UNIT[k] else 'Measured', fontsize=8); ax.set_ylabel('Predicted', fontsize=8); ax.tick_params(labelsize=7)
    axs.flat[-2].axis('off'); axs.flat[-1].axis('off')
    axs.flat[0].legend(fontsize=8)
    fig.suptitle('Recommended model: measured vs predicted per-segment parameters (one point per recording; dashed = identity)', fontsize=13)
    f1p = FIG / 'recommended_scatter.png'; fig.savefig(f1p, dpi=90); plt.close(fig)
    # Figure 2: heat map of models x key agreement metrics
    cols = [('A_rate_per_min', 'signal', 'A rate'), ('B_rate_per_min', 'signal', 'B rate'), ('A_interval_s', 'signal', 'A group dur.'),
            ('B_interval_s', 'signal', 'B group dur.'), ('A_insp_dur_s', 'signal', 'A insp. dur.'), ('B_insp_dur_s', 'signal', 'B insp. dur.'),
            ('A_depth_sd', 'signal', 'A depth'), ('B_depth_sd', 'signal', 'B depth'), ('A_exp_slope_sd_s', 'signal', 'A exp. slope'),
            ('B_exp_slope_sd_s', 'signal', 'B exp. slope'), ('ab_corr', 'dual', 'A-B corr'), ('paradox_frac', 'dual', 'Paradox frac.'),
            ('A_f1_0.4s', 'pair', 'A onset F1'), ('B_f1_0.4s', 'pair', 'B onset F1'), ('A_depth_rank_rho', 'pair', 'A depth rank'),
            ('B_depth_rank_rho', 'pair', 'B depth rank')]
    models = sorted(R, key=lambda m: list(LABEL).index(m) if m in LABEL else 999)
    M = np.array([[R[m][g][k]['ccc'] if g != 'pair' else R[m][g][k]['median'] for k, g, _ in cols] for m in models])
    fig, ax = plt.subplots(figsize=(14, 15), constrained_layout=True)
    im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for *_, l in cols], rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL.get(m, m) for m in models], fontsize=8)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if M[i, j] < .55 else 'black')
    fig.colorbar(im, ax=ax, shrink=.5, label='CCC (parameter agreement) or F1/Spearman (per-segment paired metrics)')
    ax.set_title('Parameter recovery for all 50 models (closer to 1 is better)', fontsize=13)
    f2p = FIG / 'all_models_heatmap.png'; fig.savefig(f2p, dpi=90); plt.close(fig)

    # ---------- text ----------
    L = ['# Round 16: breathing-parameter validation - predicted vs measured for all models (2026-10-01)', '',
         'Purpose: with waveforms predicted, this round tests how far the **breathing parameters** extracted from predicted waveforms differ from those extracted from the measured signals. **50 models** in total (the 3 seeds of each round-12 lambda are averaged into one model). Per-segment parameter sets are saved in `family_runs/<model>/parameter_set.csv`, summary numbers in `r16_param_validation.json`. Evaluation only, no training. Outer singers were inspected repeatedly in earlier rounds, so results are exploratory.', '',
         '## Abbreviations and terms', '',
         '- **CCC** (concordance correlation coefficient): penalizes lack of correlation, overall offset and scale mismatch; graded here as >=0.8 good, 0.6-0.8 moderate, 0.4-0.6 weak, <0.4 poor.',
         '- **Bias**: median of predicted minus measured; **MAE** (mean absolute error; here per-segment absolute error, median per singer, then median across singers); brackets are singer-bootstrap 95% CIs.',
         '- **F1**: harmonic mean of precision and recall of inhalation-onset detection; **Spearman**: rank correlation.',
         '- **Belt A** = chest belt, **belt B** = abdominal belt; amplitude parameters are in units of the signal\'s own SD (belts uncalibrated, cannot be converted to millilitres).',
         '- Model names: TCN (temporal convolutional network), BiGRU (bidirectional gated recurrent unit), Conv-LSTM (convolution + long short-term memory), U-Net (multi-scale encoder-decoder), WavLM/Whisper/MERT (pretrained speech/music encoders), EGG (electroglottography).', '',
         '## Parameter set definitions', '',
         '|Parameter|Definition|',
         '|---|---|',
         '|Breathing rate|Inhalations per minute (frozen trough-to-peak extractor)|',
         '|Breath-group duration|Median interval between consecutive inhalation onsets|',
         '|Inspiration duration|Median trough-to-peak duration (5 Hz, 0.2 s resolution)|',
         '|Inhalation depth|Median peak-minus-trough amplitude, in SD units|',
         '|Expiratory slope|Median linear decline rate of the belt within each voiced phrase (phrases from microphone voicing, shared by measured and predicted)|',
         '|Chest-abdomen correlation|Pearson correlation of chest and abdominal belt within a segment|',
         '|Chest-abdomen lag|Median abdominal minus chest onset of matched inhalations|',
         '|Paradoxical-motion fraction|Fraction of frames with clear motion on both belts in which they move in opposite directions|',
         '|Onset F1, onset error|Predicted and measured inhalation onsets paired within ±0.4 s|',
         '|Within-segment depth ranking|Spearman correlation of matched inhalation depths within a segment|',
         '|Physiological regularities R1-R3|Per-singer phrases: R1 inhalation depth ~ next-phrase duration (partial correlation controlling pause duration), R2 expiratory slope ~ SPL, R3 inhalation depth ~ SPL|', '',
         '## 1. Measured parameter reference values (197 segments, medians)', '',
         '|Parameter|A chest belt|B abdominal belt|', '|---|---|---|']
    for k in SIG:
        L.append(f'|{PN[k]} ({UNIT[k]})|{R[BASE]["signal"][f"A_{k}"]["meas_median"]:.2f}|{R[BASE]["signal"][f"B_{k}"]["meas_median"]:.2f}|')
    for k in DUAL:
        L.append(f'|{PN[k]}{" (" + UNIT[k] + ")" if UNIT[k] else ""}|{R[BASE]["dual"][k]["meas_median"]:.2f} (dual belt)||')
    L += ['', 'Measured: about 16-17 inhalations per minute when singing/reading, breath groups about 3 s, inspiration about 0.8-1.0 s; median chest-abdomen correlation 0.68, about 22% of moving frames show opposite chest and abdominal motion, and abdominal onsets lead the chest by about 0.2 s on average (one frame at 5 Hz).', '',
          '## 2. Parameter recovery of the recommended model (Conv-LSTM + U-Net cross-family ensemble) vs the baseline model', '',
          '|Parameter|Belt|Measured median|Predicted median|Bias|MAE [95%]|CCC (recommended)|Grade|CCC (baseline)|', '|---|---|---|---|---|---|---|---|---|']
    for c in 'AB':
        for k in SIG:
            s = R[REC]['signal'][f'{c}_{k}']; b = R[BASE]['signal'][f'{c}_{k}']
            L.append(f'|{PN[k]} ({UNIT[k]})|{c}|{s["meas_median"]:.2f}|{s["pred_median"]:.2f}|{s["bias"]:+.2f}|{s["mae"]:.2f} [{s["mae_ci"][0]:.2f}, {s["mae_ci"][1]:.2f}]|{s["ccc"]:.2f}|{grade(s["ccc"])}|{b["ccc"]:.2f}|')
    for k in DUAL:
        s = R[REC]['dual'][k]; b = R[BASE]['dual'][k]
        L.append(f'|{PN[k]}|dual|{s["meas_median"]:.2f}|{s["pred_median"]:.2f}|{s["bias"]:+.2f}|{s["mae"]:.2f} [{s["mae_ci"][0]:.2f}, {s["mae_ci"][1]:.2f}]|{s["ccc"]:.2f}|{grade(s["ccc"])}|{b["ccc"]:.2f}|')
    L += ['', '|Per-segment paired metric|A (recommended)|B (recommended)|A (baseline)|B (baseline)|', '|---|---|---|---|---|']
    for k in ('f1_0.4s', 'f1_1.0s', 'onset_err_s', 'depth_rank_rho'):
        L.append(f'|{ {"f1_0.4s": "Onset F1 ±0.4 s", "f1_1.0s": "Onset F1 ±1.0 s", "onset_err_s": "Onset time error (s)", "depth_rank_rho": "Within-segment depth ranking"}[k]}|'
                 f'{R[REC]["pair"][f"A_{k}"]["median"]:.2f}|{R[REC]["pair"][f"B_{k}"]["median"]:.2f}|{R[BASE]["pair"][f"A_{k}"]["median"]:.2f}|{R[BASE]["pair"][f"B_{k}"]["median"]:.2f}|')
    L += ['', '|Physiological regularity (per-singer coef.)|Measured median|Predicted median|Across-singer corr meas vs pred|Sign agreement|', '|---|---|---|---|---|']
    for key, name in (('R1_A', 'R1 inhalation depth ~ phrase duration (chest)'), ('R2_A', 'R2 expiratory slope ~ SPL (chest)'), ('R3_A', 'R3 inhalation depth ~ SPL (chest)'),
                      ('R1_B', 'R1 (abdomen)'), ('R2_B', 'R2 (abdomen)'), ('R3_B', 'R3 (abdomen)')):
        s = R[REC]['relations'][key]
        L.append(f'|{name}|{s["meas_median"]:+.2f}|{s["pred_median"]:+.2f}|{s["across_singer_r"]:+.2f}|{s["sign_agree"]:.0%}|')
    L += ['', '### By task (CCC)', '', '|Parameter|Song A/B|Scale A/B|Reading A/B|', '|---|---|---|---|']
    for k in SIG:
        L.append(f'|{PN[k]}|' + '|'.join(f'{R[REC]["by_task"][t][f"A_{k}"]:.2f}/{R[REC]["by_task"][t][f"B_{k}"]:.2f}' for t in ('song', 'scale', 'reading')) + '|')
    L.append('|Onset F1 ±0.4 s (median)|' + '|'.join(f'{R[REC]["by_task"][t]["A_f1_0.4s"]:.2f}/{R[REC]["by_task"][t]["B_f1_0.4s"]:.2f}' for t in ('song', 'scale', 'reading')) + '|')
    # best model per parameter
    L += ['', '## 3. Best model for each parameter (out of 50)', '', '|Parameter|Best model|Value|Recommended model|Baseline model|', '|---|---|---|---|---|']
    for key, g, lab in cols:
        val = lambda m: R[m][g][key]['ccc'] if g != 'pair' else R[m][g][key]['median']
        best = max(R, key=val)
        L.append(f'|{lab}|{LABEL.get(best, best)}|{val(best):.2f}|{val(REC):.2f}|{val(BASE):.2f}|')
    # full matrix
    L += ['', '## 4. Parameter agreement for all models', '', 'Record-level parameters are CCC, per-segment paired metrics are singer medians; R1-R3 are across-singer correlations of measured vs predicted per-singer coefficients (chest).', '',
          '|Model|' + '|'.join(l for *_, l in cols) + '|R1 chest|R2 chest|R3 chest|', '|---|' + '---|' * (len(cols) + 3)]
    for m in models:
        vals = [R[m][g][k]['ccc'] if g != 'pair' else R[m][g][k]['median'] for k, g, _ in cols]
        rel = [R[m]['relations'][f'{x}_A']['across_singer_r'] for x in ('R1', 'R2', 'R3')]
        L.append(f'|{LABEL.get(m, m)}|' + '|'.join(f'{v:.2f}' for v in vals + rel) + '|')
    # findings
    s = R[REC]
    L += ['', '## 5. Main findings', '',
          f'1. **Reliably recovered parameters** (recommended model CCC >= 0.8): chest breathing rate ({s["signal"]["A_rate_per_min"]["ccc"]:.2f}), chest within-phrase expiratory slope ({s["signal"]["A_exp_slope_sd_s"]["ccc"]:.2f}). Abdominal rate is close ({s["signal"]["B_rate_per_min"]["ccc"]:.2f}). The relative size of inhalations within a segment is also recovered well (chest depth ranking {s["pair"]["A_depth_rank_rho"]["median"]:.2f}).',
          f'2. **Moderate**: chest breath-group duration ({s["signal"]["A_interval_s"]["ccc"]:.2f}), abdominal expiratory slope ({s["signal"]["B_exp_slope_sd_s"]["ccc"]:.2f}); most inhalation onsets match within ±1 s (F1 chest {s["pair"]["A_f1_1.0s"]["median"]:.2f} / abdomen {s["pair"]["B_f1_1.0s"]["median"]:.2f}), but only about half to two thirds at ±0.4 s.',
          f'3. **Poorly recovered**: inspiration duration (CCC about 0.3, limited by the 5 Hz resolution; 50 Hz models do not improve it); inhalation depth compared across recordings (CCC {s["signal"]["A_depth_sd"]["ccc"]:.2f}/{s["signal"]["B_depth_sd"]["ccc"]:.2f}, predictions systematically deeper by about +0.4 SD: predicted waveforms are "cleaner" with fewer small fluctuations, which inflates the SD-normalized peak-trough difference); abdominal breath-group duration ({s["signal"]["B_interval_s"]["ccc"]:.2f}).',
          f'4. **Chest-abdomen coordination is largely not recovered**: measured median chest-abdomen correlation {s["dual"]["ab_corr"]["meas_median"]:.2f}, predicted {s["dual"]["ab_corr"]["pred_median"]:.2f}; paradoxical-motion fraction measured {s["dual"]["paradox_frac"]["meas_median"]:.2f}, predicted {s["dual"]["paradox_frac"]["pred_median"]:.2f}; across-recording CCC only {s["dual"]["ab_corr"]["ccc"]:.2f}/{s["dual"]["paradox_frac"]["ccc"]:.2f}. All 50 models behave this way (the "A-B corr" and "Paradox frac." columns of the heatmap are uniformly low). **Models tend to make the abdominal belt follow the chest belt**; individual partitioning and paradoxical motion are not recovered. This agrees with round 14 (abdominal inhalation timing in songs is hard to infer) and is the main current limitation.',
          f'5. **Better waveforms do not imply better parameters**: the cross-family ensemble has the highest waveform CCC but averages away small inhalations, so the predicted breathing rate is too low (chest {s["signal"]["A_rate_per_min"]["bias"]:+.1f}, abdomen {s["signal"]["B_rate_per_min"]["bias"]:+.1f} breaths/min) and chest-rate CCC drops from {R[BASE]["signal"]["A_rate_per_min"]["ccc"]:.2f} (baseline) to {s["signal"]["A_rate_per_min"]["ccc"]:.2f}; the paradoxical fraction is pushed down further. If the endpoint is breathing rate, choose the model by the parameter endpoint, not by waveform CCC.',
          f'6. **Part of the overall agreement comes from differences between tasks**: overall chest-rate CCC {s["signal"]["A_rate_per_min"]["ccc"]:.2f}, but within a task only ' + ', '.join(f'{ {"song": "song", "scale": "scale", "reading": "reading"}[t]} {s["by_task"][t]["A_rate_per_min"]:.2f}' for t in ("song", "scale", "reading")) + '. Reading and singing already differ in breathing rate, so a model gains some agreement simply by telling tasks apart; papers should report within-task agreement.',
          '7. **Physiological regularities**: all models reproduce the direction of R1 "inhalation depth anticipates phrase duration" (100% sign agreement on the chest belt) and most directions of R2/R3; across-singer correlations of individual coefficients are 0.3-0.65.',
          '8. **Generative models (Flow, Diffusion) and EGG-only models** are worse on most parameters; Transformers are generally weaker; changing loss weights has little effect on parameters.', '',
          '## 6. Recommendations for the paper', '',
          '- **Parameters that can be reported as findings**: chest breathing rate, breath-group duration (moderate), within-phrase expiratory slope, relative inhalation depth within a segment, inhalation onsets (±1 s).',
          '- **Must be stated as limitations**: inspiration duration, absolute inhalation depth across recordings, chest-abdomen coordination (correlation, paradoxical motion, lag). Coordination is exactly what the singing-breathing literature cares about most and should be future work: higher temporal resolution, explicit modelling of the two-belt relation, and confirmation data from new singers.',
          '- **Model selection**: by endpoint - use the cross-family ensemble for waveforms; use a single model (e.g. the baseline or a round-13 single family) for breathing rate, avoiding rate underestimation caused by ensemble smoothing.', '',
          '## 7. Limitations', '',
          '- Outer singers have been inspected over many rounds; all rankings are exploratory and the final comparison must be done on new singers.',
          '- The parameter extractor uses fixed rules, partially validated with breath sounds (round 14); there is no manual-annotation gold standard yet.',
          '- At 5 Hz, temporal parameters (inspiration duration, chest-abdomen lag) lack resolution; 50 Hz models are evaluated here after averaging to 5 Hz.',
          '- Belts are uncalibrated, so all amplitude parameters are relative.', '',
          '## Figures', '', '- `r16_figures/recommended_scatter.png`: measured vs predicted per-segment parameters of the recommended model (coloured by task).',
          '- `r16_figures/all_models_heatmap.png`: parameter-agreement heatmap of all 50 models.']
    md = '\n'.join(L)
    (ROOT / 'round16_parameter_validation_report.md').write_text(md, encoding='utf-8')
    # HTML (tables rendered simply from markdown pipes)
    def md2html(text):
        out, table = [], []
        for line in text.split('\n'):
            if line.startswith('|'):
                table.append(line); continue
            if table:
                hdr = [c for c in table[0].strip('|').split('|')]
                body = [r.strip('|').split('|') for r in table[2:]]
                out.append('<div class="tw"><table><tr>' + ''.join(f'<th>{h}</th>' for h in hdr) + '</tr>' +
                           ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in body) + '</table></div>')
                table = []
            if line.startswith('# '):
                out.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                out.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                out.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('- '):
                out.append(f'<p>• {line[2:]}</p>')
            elif line.strip():
                out.append(f'<p>{line}</p>')
        return '\n'.join(out)
    import re
    body = md2html(md)
    body = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', body); body = re.sub(r'`(.+?)`', r'<code>\1</code>', body)
    body = body.replace('<h2>Figures</h2>', '<h2>Figures</h2>' + '<h3>Recommended model: measured vs predicted per-segment parameters</h3>' + img(f1p) + '<h3>Parameter-agreement heatmap of all models</h3>' + img(f2p))
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breathing parameter validation report</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e8;--head:#f3f3f6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141416;--fg:#ececf0;--mut:#a0a0a8;--line:#2c2c32;--head:#1f1f24}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:16px}} img{{width:100%;height:auto;background:#fff;border-radius:6px}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;margin:8px 0}} th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}}
th{{background:var(--head)}} code{{font-size:13px}}
</style></head><body><main>{body}</main></body></html>"""
    (ROOT / 'round16_parameter_validation_report.html').write_text(html, encoding='utf-8')
    print(md[:3000])


if __name__ == '__main__':
    main()
