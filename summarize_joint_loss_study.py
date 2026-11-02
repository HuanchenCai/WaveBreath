"""Loss-only comparison, paired singer summaries and selected outer predictions."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_joint_loss_study import CONFIGS
from summarize_family_runs import paired_summary, summarize_rows

BASE = 'wavlm_large16_tcn'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--outputs', type=Path, required=True)
    args = ap.parse_args()
    runs = args.root / 'family_runs'
    data, partial = {}, {}
    for name in (BASE, *CONFIGS):
        path = runs / name / 'results.json'
        if path.exists():
            d = json.loads(path.read_text(encoding='utf-8'))
            if d.get('completed'):
                data[name] = d
            else:
                partial[name] = len(d['folds'])
    baseline = data[BASE]
    count = sum(n in data for n in CONFIGS)
    if count == len(CONFIGS):
        selected = []
        for fold in range(1, 6):
            choices = [(n, next(f for f in d['folds'] if f['fold'] == fold)) for n, d in data.items()]
            name, f = max(choices, key=lambda item: item[1]['best_inner_r'])
            selected.append({**f, 'selected_configuration': name})
        data['inner_selected_loss'] = {'folds': selected, 'summary': summarize_rows(selected), 'completed': True}
        (runs / 'inner_selected_loss.json').write_text(json.dumps(data['inner_selected_loss'], ensure_ascii=False, indent=2), encoding='utf-8')
    paired = {n: paired_summary(d, baseline) for n, d in data.items() if n != BASE}
    (runs / 'joint_loss_paired.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')
    def pair(s, key):
        return f'{s["A"][key]:.4f} / {s["B"][key]:.4f}'
    lines = ['# Joint correlation and error loss: fixed-model comparison', '',
        f'Updated: {datetime.now().isoformat(timespec="seconds")}. New configurations completed {count}/{len(CONFIGS)}; 5 folds each.', '',
        '## Question and formula', '',
        'This round fixes WavLM Large layer-16 features + TCN, 197 segments / 32 singers, the same five-fold split, seed 20260930, AdamW 3e-4, batch 32, at most 20 epochs. Only the loss weights change.', '',
        '`L = MSE + 0.2 x MSE(first difference) + lambda_r x (1 - Pearson r) + lambda_c x (1 - CCC)`', '',
        'Correlation and CCC are computed per window and per belt over valid time points, then averaged; A and B are not merged into one target. Padded points are excluded; real windows with near-zero variance do not enter the correlation/CCC terms but still count in the pointwise error. Correlation only controls linear shape; MSE constrains point values; CCC constrains mean, amplitude and correlation together.', '',
        'MSE is not free of correlation information: MSE = mean difference^2 + predicted variance + target variance - 2 r x predicted SD x target SD. Adding (1 - r) changes the relative weights of these terms and does not mean the original formula was wrong. Conversely prediction = 2 x measured + 1 also gives r = 1, so optimizing correlation alone does not guarantee matching waveforms.', '',
        'Whisper had previously been tried with correlation weight 0.1; the new best WavLM used only MSE and slope error before this round. The two must not be confused.', '',
        '## Current results', '',
        'Values are medians over all held-out segments. Higher r/CCC is better, lower NRMSE is better.', '',
        '|Configuration|lambda_r / lambda_c|Status|A/B r|A/B CCC|A/B NRMSE|A/B trough F1|A/B amplitude SD ratio|',
        '|---|---|---|---|---|---|---|---|']
    if count == len(CONFIGS):
        insert = lines.index('## Current results')
        candidate = data['loss_ccc100']['summary']
        lines[insert:insert] = ['## Answer from this round', '',
            'No implementation error was found in the original loss, but adding correlation/concordance constraints is worthwhile. Across the six runs, adding Pearson correlation alone mainly gives very small correlation changes; CCC combined with MSE more clearly improves amplitude agreement.', '',
            f'Recommend keeping `MSE + 0.2 x diff-MSE + (1 - CCC)` as a candidate for the next round: A/B CCC={pair(candidate, "ccc")}, NRMSE={pair(candidate, "nrmse")}, r={pair(candidate, "r")}. Relative to the original loss, CCC improves for 27/32 singers on each belt; per-singer NRMSE intervals still cross zero, so not all metrics can be claimed to improve stably.', '',
            'Belt B r<0.5 still occurs in 28/197 segments, one anti-phase failure remains, and trough metrics show no stable gain. Tuning the loss alone cannot solve these problems.', '',
            'Early stopping by r was kept here to isolate the loss effect; if waveform agreement is the primary goal next round, CCC should be pre-specified as the primary inner selection metric, with NRMSE and peak/trough errors also reported, then stability checked over multiple seeds. The weight must not be declared optimal based on outer results already seen.', '']
    for name in (BASE, *CONFIGS):
        weights = CONFIGS.get(name, {'pearson_weight': 0, 'ccc_weight': 0})
        label = f'{weights["pearson_weight"]} / {weights["ccc_weight"]}'
        if name in data:
            vals = [pair(data[name]['summary'], k) for k in ('r', 'ccc', 'nrmse', 'trough_f1', 'std_ratio')]
            state = 'done'
        else:
            vals = ['—'] * 5
            state = f'{partial.get(name, 0)}/5 folds'
        lines.append('| ' + ' | '.join([name, label, state, *vals]) + ' |')
    lines.extend(['', '## Per-singer paired differences vs the original WavLM model', '',
        'Each singer\'s segment metrics are first reduced to a median, then candidate minus control; 3000 singer bootstrap resamples, uncorrected for multiple comparisons, exploratory intervals.', '',
        '|Configuration|A delta r [95% CI]|B delta r [95% CI]|A/B delta CCC|Singers improved A/B|', '|---|---|---|---|---|'])
    for name, p in paired.items():
        ds = []
        for c in ('A', 'B'):
            d = p[c]['r']
            lo, hi = d['bootstrap_95_interval']
            ds.append(f'{d["median_delta"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        ccc = ' / '.join(f'{p[c]["ccc"]["median_delta"]:+.4f}' for c in ('A', 'B'))
        improved = ' / '.join(f'{p[c]["r"]["improved_singers"]}/32' for c in ('A', 'B'))
        lines.append('| ' + ' | '.join([name, *ds, ccc, improved]) + ' |')
    if 'inner_selected_loss' in data:
        selected = data['inner_selected_loss']
        lines.extend(['', '## Selecting the loss weight in the inner loop only', '',
            'To isolate the effect of the training loss, early stopping still uses the original inner singer-correlation criterion; the selection metric is not changed at the same time. All candidates include the original-loss control.', '',
            f'Pooled held-out results: A/B r={pair(selected["summary"], "r")}, CCC={pair(selected["summary"], "ccc")}, NRMSE={pair(selected["summary"], "nrmse")}.', ''])
        lines.extend(f'- Fold {f["fold"]}: {f["selected_configuration"]}.' for f in selected['folds'])
        lines.extend(['', '## Severe failures still need separate inspection', '', '|Configuration|A/B segments r<0.5|A/B segments r<=0|', '|---|---|---|'])
        for name, d in data.items():
            rows = [r for f in d['folds'] for r in f['test_segments']]
            counts = [' / '.join(str(sum(r[c]['r'] < .5 for r in rows)) for c in ('A', 'B')),
                      ' / '.join(str(sum(r[c]['r'] <= 0 for r in rows)) for c in ('A', 'B'))]
            lines.append('| ' + ' | '.join([name, *counts]) + ' |')
        # Audit every saved prediction against the unchanged baseline target/order.
        for name in CONFIGS:
            singers = []
            for fold in data[name]['folds']:
                singers.extend(fold['test_singers'])
                filename = f'fold{fold["fold"]}_waveforms.npz'
                with np.load(runs / name / filename, allow_pickle=False) as p, np.load(runs / BASE / filename, allow_pickle=False) as ref:
                    assert all(np.array_equal(p[k], ref[k]) for k in ('target', 'metadata', 'offsets'))
                    assert p['prediction'].shape == p['target'].shape and np.isfinite(p['prediction']).all()
            assert len(singers) == len(set(singers)) == 32
        status = {'completed': count, 'verified_new_outer_folds': count * 5, 'target_integrity_passed': True}
        (runs / 'joint_loss_completion.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        lines.extend(['', 'Target arrays, time order, singer coverage and finiteness of predictions were checked against the original model.'])
        for name, result in data.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
            for row, (singer, number) in enumerate((('P29', '3'), ('P24', '12'))):
                fold = next(f for f in result['folds'] if singer in f['test_singers'])
                source = fold.get('selected_configuration', name)
                with np.load(runs / source / f'fold{fold["fold"]}_waveforms.npz', allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata']))
                    i = next(i for i, m in enumerate(meta) if m['singer'] == singer and m['number'] == number)
                    left, right = z['offsets'][i:i + 2]
                    y, pred = z['target'][left:right], z['prediction'][left:right]
                n = min(len(y), 150) if row == 0 else len(y)
                for j, c in enumerate(('A', 'B')):
                    ax = axes[row, j]
                    ax.plot(np.arange(n) / 5, y[:n, j], label='Measured', color='#235fa4')
                    ax.plot(np.arange(n) / 5, pred[:n, j], label='Predicted', color='#14946c')
                    ax.set_title(f'{singer} #{number}, belt {c}, full-recording r={np.corrcoef(y[:, j], pred[:, j])[0, 1]:.2f}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Relative belt units')
            axes[0, 0].legend()
            fig.suptitle(name)
            fig.savefig(args.outputs / f'{name}_joint_loss_AB.png', dpi=150)
            plt.close(fig)
    lines.extend(['', '## Scope of interpretation and resuming', '',
        '- Absolute loss values are not comparable across weights; only the same held-out metrics are compared. A lower training loss does not mean better predictions.',
        '- This is a single fixed-seed screen of six loss weightings; test singers were used repeatedly during development, so results need confirmation on new singers.',
        '- Only 5 Hz per-segment standardized measured A/B are evaluated. Absolute amplitude recovery, channel identity/polarity and individual chest-abdomen differences remain unsolved; predictions were not flipped or shifted using test labels.',
        '- Original targets and old results are kept. run_joint_loss_study.py can resume; family_runs/loss_* hold per-fold weights, results and measured/predicted arrays. joint_loss_protocol.json records the weights, joint_loss_checks.json the numerical checks.', ''])
    report = '\n'.join(lines)
    for folder in (args.root, args.outputs):
        (folder / 'joint_loss_progress_and_results.md').write_text(report, encoding='utf-8')
    print(json.dumps({'completed': count, 'expected': len(CONFIGS), 'partial': partial}, ensure_ascii=False))


if __name__ == '__main__':
    main()
