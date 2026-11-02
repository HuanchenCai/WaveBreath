"""Durable partial/final model-family report and singer-level paired comparisons."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_waveform_families import EXPERIMENTS


def summarize_rows(folds):
    rows = [r for f in folds for r in f['test_segments']]
    return {c: {k: float(np.median([r[c][k] for r in rows])) for k in
                ('r', 'ccc', 'nrmse', 'peak_f1', 'trough_f1', 'std_ratio')} for c in ('A', 'B')}


def singer_values(result, channel, metric):
    groups = defaultdict(list)
    for fold in result['folds']:
        for row in fold['test_segments']:
            groups[row['singer']].append(row[channel][metric])
    return {s: float(np.median(values)) for s, values in groups.items()}


def paired_summary(result, reference):
    out = {}
    for channel in ('A', 'B'):
        out[channel] = {}
        for metric in ('r', 'ccc', 'nrmse', 'trough_f1'):
            current = singer_values(result, channel, metric)
            baseline = singer_values(reference, channel, metric)
            assert set(current) == set(baseline)
            delta = np.array([current[s] - baseline[s] for s in sorted(current)])
            rng = np.random.default_rng(20260930)
            bootstrap = np.median(delta[rng.integers(0, len(delta), size=(3000, len(delta)))], axis=1)
            out[channel][metric] = {'median_delta': float(np.median(delta)),
                'bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
                'improved_singers': int(np.sum(delta < 0 if metric == 'nrmse' else delta > 0)),
                'total_singers': len(delta)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--outputs', type=Path, required=True)
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()
    args.outputs.mkdir(parents=True, exist_ok=True)
    runs = args.root / 'family_runs'
    data, status = {}, []
    for name in ('reference_large_shape', *EXPERIMENTS):
        path = runs / name / 'results.json'
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            data[name] = result
            state = 'done' if result.get('completed') else f'{len(result["folds"])}/5 folds'
        else:
            state = 'pending'
        if (runs / f'{name}_failure.json').exists() and not data.get(name, {}).get('completed'):
            state = 'failed, to fix'
        status.append((name, state))
    layers = [f'wavlm{layer}_transformer' for layer in (4, 7, 12)]
    if all(data.get(name, {}).get('completed') for name in layers):
        selected = []
        for fold in range(1, 6):
            options = [(name, next(f for f in data[name]['folds'] if f['fold'] == fold)) for name in layers]
            name, result = max(options, key=lambda item: item[1]['best_inner_r'])
            selected.append({**result, 'selected_configuration': name})
        data['wavlm_inner_selected_transformer'] = {'completed': True, 'folds': selected, 'summary': summarize_rows(selected)}
        status.append(('wavlm_inner_selected_transformer', 'inner-loop layer selection, summarized'))
    if all(data.get(name, {}).get('completed') for name in EXPERIMENTS):
        selected = []
        for fold in range(1, 6):
            options = [(name, next(f for f in data[name]['folds'] if f['fold'] == fold)) for name in EXPERIMENTS]
            name, result = max(options, key=lambda item: item[1]['best_inner_r'])
            selected.append({**result, 'selected_configuration': name})
        data['inner_selected_family'] = {'completed': True, 'folds': selected, 'summary': summarize_rows(selected)}
        status.append(('inner_selected_family', 'inner-loop model selection, summarized'))
        (runs / 'inner_selected_family.json').write_text(json.dumps(data['inner_selected_family'], ensure_ascii=False, indent=2), encoding='utf-8')
    completed = {n: d for n, d in data.items() if d.get('completed')}
    comparisons = {}
    reference = completed.get('reference_large_shape')
    if reference:
        comparisons = {n: paired_summary(d, reference) for n, d in completed.items() if n != 'reference_large_shape'}
    (runs / 'paired_summary.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    matched = {}
    for candidate, control in (
            ('large_wave_unet', 'large_tcn_control'),
            ('large_swin_unet', 'large_wave_unet'),
            ('large_nab_inspired', 'large_swin_unet'),
            ('large_aligned_latent4', 'large_tcn_control'),
            ('wavlm7_tcn', 'large_tcn_control'),
            ('wavlm_large16_tcn', 'wavlm7_tcn'),
            ('wavlm7_transformer', 'wavlm7_tcn'),
            ('wavlm_large16_transformer', 'wavlm_large16_tcn')):
        if candidate in completed and control in completed:
            matched[f'{candidate}__vs__{control}'] = {
                'candidate': candidate, 'control': control,
                'paired': paired_summary(completed[candidate], completed[control])}
    (runs / 'matched_ablation_summary.json').write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding='utf-8')
    active_path = runs / 'current_progress.json'
    active = json.loads(active_path.read_text(encoding='utf-8')) if active_path.exists() else {}
    if all(name in completed for name in EXPERIMENTS):
        active = {'phase': 'all configurations finished 5 folds', 'completed': len(EXPERIMENTS), 'expected': len(EXPERIMENTS)}
        (runs / 'study_completion.json').write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Full model-family experiments: progress and results', '',
        'All entries are task-adapted implementations for this project and cannot be called full reproductions of the original MD-ViSCo, NABNet, PPGFlowECG or RDDM code.', '',
        '197 segments, 32 singers; fixed five folds with singers held out, inner early stopping by singer; at most 20 epochs, AdamW 3e-4, batch 32. Generative models get 12 extra epochs of joint AE pre-training, average 4 fixed noise seeds with 10 sampling steps, and never pick samples by the ground truth. All A/B are independent measured targets.', '',
        f'Current progress: {json.dumps(active, ensure_ascii=False)}', '',
        '| Configuration | Status | A/B r | A/B CCC | A/B NRMSE | A/B trough F1 |',
        '|---|---|---|---|---|---|']
    for name, state in status:
        result = completed.get(name)
        if result:
            s = result['summary']
            values = [f'{s["A"][key]:.3f} / {s["B"][key]:.3f}' for key in ('r', 'ccc', 'nrmse', 'trough_f1')]
        else:
            values = ['—'] * 4
        lines.append('| ' + ' | '.join((name, state, *values)) + ' |')
    lines.extend(['', '## Per-singer paired comparison vs previous best', '',
                  'Each singer\'s segment metrics are first reduced to a median, then the candidate-minus-reference difference is computed; CIs are 3000 singer-level bootstrap resamples, uncorrected for multiple comparisons, exploratory only.', '',
                  '| Configuration | A delta r median [95% CI] | B delta r median [95% CI] | Singers improved A/B |',
                  '|---|---|---|---|'])
    for name, comparison in comparisons.items():
        values = []
        for channel in ('A', 'B'):
            c = comparison[channel]['r']
            lo, hi = c['bootstrap_95_interval']
            values.append(f'{c["median_delta"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        improved = f'{comparison["A"]["r"]["improved_singers"]}/32, {comparison["B"]["r"]["improved_singers"]}/32'
        lines.append('| ' + ' | '.join((name, *values, improved)) + ' |')
    lines.extend(['', '## Scope of interpretation', '',
        '- WavLM compares Base-Plus layers 4/7/12 and Large layers 8/16/24; Whisper uses Large-v3. Each layer is trained separately and layer choice may only use inner validation.',
        '- wave_unet is a multi-scale 1D encoder-decoder; swin_unet adds shifted-window attention; nab_inspired adds a bidirectional LSTM in the skip fusion and does not fully reproduce UNet++/BiConvLSTM.',
        '- aligned_latent jointly trains voice/belt encoders, a belt decoder, a voice-private latent and reconstruction; flow/diffusion pre-train the same joint AE, then freeze it and fit the conditional latent distribution. No polarity flipping, shifting or sample selection based on test waveforms.',
        '- This is a fixed-budget screen of model families, not proof that every model was fully tuned or reached its capacity. Outer singers were used repeatedly during development; final conclusions still need confirmatory validation after freezing the pipeline.',
        '- Resume entry: run_waveform_families.py; each family_runs configuration directory stores results.json, per-fold weights and full OOF waveforms. Failures are never recorded as completed.'])
    if 'inner_selected_family' in data:
        lines.extend(['', '## Models selected by inner validation in this round', ''])
        lines.extend(f'- Fold {f["fold"]}: {f["selected_configuration"]}, inner singer correlation {f["best_inner_r"]:.4f}' for f in data['inner_selected_family']['folds'])
    report = '\n'.join(lines) + '\n'
    (args.outputs / 'family_study_progress.md').write_text(report, encoding='utf-8')
    (args.root / 'family_study_progress.md').write_text(report, encoding='utf-8')
    if args.plots:
        for name, result in completed.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
            for row, (singer, number) in enumerate((('P29', '3'), ('P24', '12'))):
                fold = next(f for f in result['folds'] if singer in f['test_singers'])
                source = fold.get('selected_configuration', name)
                with np.load(runs / source / f'fold{fold["fold"]}_waveforms.npz', allow_pickle=False) as cache:
                    meta = json.loads(str(cache['metadata']))
                    i = next(i for i, m in enumerate(meta) if m['singer'] == singer and m['number'] == number)
                    begin, end = cache['offsets'][i:i + 2]
                    y, p = cache['target'][begin:end], cache['prediction'][begin:end]
                length = min(len(y), 150) if row == 0 else len(y)
                for c, channel in enumerate(('A', 'B')):
                    ax = axes[row, c]
                    t = np.arange(length) / 5
                    ax.plot(t, y[:length, c], label='Measured', color='#235fa4')
                    ax.plot(t, p[:length, c], label='Predicted', color='#14946c')
                    r = np.corrcoef(y[:, c], p[:, c])[0, 1]
                    ax.set_title(f'{singer} #{number}, belt {channel}, full-recording r={r:.2f}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Relative belt units')
            axes[0, 0].legend()
            fig.suptitle(name)
            fig.savefig(args.outputs / f'{name}_AB_waveforms.png', dpi=150)
            plt.close(fig)
    print(json.dumps({'completed': sum(name in completed for name in EXPERIMENTS), 'expected': len(EXPERIMENTS), 'active': active}, ensure_ascii=False))


if __name__ == '__main__':
    main()
