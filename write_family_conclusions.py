"""Build the Chinese study handover from saved, audited experiment outputs."""
import argparse
import json
from pathlib import Path
from run_waveform_families import EXPERIMENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--outputs', type=Path, required=True)
    args = ap.parse_args()
    runs = args.root / 'family_runs'
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    data = {name: read(runs / name / 'results.json') for name in EXPERIMENTS}
    assert all(d.get('completed') and len(d['folds']) == 5 for d in data.values())
    audit = read(runs / 'completion_integrity_and_failures.json')
    assert set(audit) == set(data)
    runtime = read(runs / 'head_runtime.json')
    paired = read(runs / 'paired_summary.json')
    oracle = read(runs / 'latent_oracle_diagnostics.json')
    matched = read(runs / 'matched_ablation_summary.json')
    selection = read(runs / 'inner_selected_family.json')
    ref = read(runs / 'reference_large_shape' / 'results.json')
    rank = sorted(data, key=lambda n: sum(data[n]['summary'][c]['r'] for c in ('A', 'B')), reverse=True)
    best = rank[0]
    def pair(summary, key):
        return f'{summary["A"][key]:.3f} / {summary["B"][key]:.3f}'
    lines = ['# Full model experiments: conclusions, failures and resume entry', '',
        'Date: 2026-09-30. This round completed training and held-out prediction for 16 configurations and 80 outer folds, plus a re-evaluation of the previous round\'s weights as a control. All 197 segments and 32 singers covered; A and B compared separately with the measured waveforms.', '',
        '## Core results', '',
        f'- Ranked by the mean of A/B median outer segment correlations, the best is `{best}`, A/B r={pair(data[best]["summary"], "r")}; this ranking is exploratory and not an unbiased model selection.',
        f'- Selecting the configuration per fold by inner singer validation only, pooled held-out A/B r={pair(selection["summary"], "r")}, CCC={pair(selection["summary"], "ccc")}, NRMSE={pair(selection["summary"], "nrmse")}.',
        f'- Previous-round reference A/B r={pair(ref["summary"], "r")}, CCC={pair(ref["summary"], "ccc")}, NRMSE={pair(ref["summary"], "nrmse")}. High correlation does not mean amplitude and local peaks/troughs match.',
        '- This round covers the literature-suggested multi-scale U-Net, windowed attention, recurrent skip fusion, joint dual latents, conditional Flow and conditional Diffusion, plus two WavLM sizes and different layers. These are task-adapted implementations, not full reproductions of the authors\' code.', '',
        '## All configurations', '',
        'Higher r and CCC are better, lower NRMSE is better; values are medians of segment metrics over 197 segments, A and B not merged.', '',
        '|Configuration|A/B r|A/B CCC|A/B NRMSE|A/B trough F1|A/B segments r<0.5|',
        '|---|---|---|---|---|---|']
    for name in rank:
        s = data[name]['summary']
        bad = ' / '.join(str(audit[name]['channels'][c]['segments_r_below_half']) for c in ('A', 'B'))
        lines.append('| ' + ' | '.join([name, *(pair(s, k) for k in ('r', 'ccc', 'nrmse', 'trough_f1')), bad]) + ' |')
    lines.extend(['', '## Per-singer differences and selection risk', '',
        'Each singer\'s segments are aggregated first and then models compared; this differs from taking the median over all pooled segments. Intervals are 3000 singer bootstrap resamples, uncorrected for multiple comparisons, exploratory only.', '',
        '|Configuration|A delta r [95% CI]|B delta r [95% CI]|Singers improved A/B|', '|---|---|---|---|'])
    for name in (*rank[:3], 'inner_selected_family'):
        values = []
        for c in ('A', 'B'):
            d = paired[name][c]['r']
            lo, hi = d['bootstrap_95_interval']
            values.append(f'{d["median_delta"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        counts = ' / '.join(f'{paired[name][c]["r"]["improved_singers"]}/32' for c in ('A', 'B'))
        lines.append('| ' + ' | '.join([name, *values, counts]) + ' |')
    lines.extend(['', 'Inner-loop selection results:', ''])
    lines.extend(f'- Fold {f["fold"]}: {f["selected_configuration"]}.' for f in selection['folds'])
    lines.extend(['', '## Architecture ablations with matched settings', '',
        'Head comparisons fix the speech encoder and waveform loss; encoder comparisons fix the TCN head and loss. Parameter counts differ and the two WavLM sizes use different layer indices, so differences cannot be attributed entirely to one module or size. Windowed attention vs a U-Net without attention, and recurrent fusion vs windowed-attention U-Net, are the more direct comparisons.', '',
        '|Candidate vs control|A delta r [95% CI]|B delta r [95% CI]|', '|---|---|---|'])
    for item in matched.values():
        values = []
        for c in ('A', 'B'):
            d = item['paired'][c]['r']
            lo, hi = d['bootstrap_95_interval']
            values.append(f'{d["median_delta"]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        lines.append('| ' + ' | '.join([f'{item["candidate"]} vs {item["control"]}', *values]) + ' |')
    lines.extend(['', '## Where does the dual latent get stuck?', '',
        'Voice features -> voice encoder -> shared latent -> mapping -> belt latent -> belt decoder -> predicted measured A/B. During training there is also a belt encoder, a reverse shared-latent mapping and a voice-private latent; prediction never reads the belt ground truth.', '',
        'A 128-frame x 2-channel belt window has 256 numbers, compressed to 32 frames x 4 channels = 128 numbers, which does reduce the total dimension. The voice shared latent is 32 x 16, plus a private latent.', '',
        '|Configuration|Measured-belt self-reconstruction r (diagnostic)|Deterministic voice bridge r|Final model voice prediction r|', '|---|---|---|---|'])
    for name, d in oracle.items():
        lines.append('| ' + ' | '.join([name, pair(d['target_belt_oracle_not_voice_prediction'], 'r'),
            pair(d['deterministic_voice_bridge'], 'r'), pair(data[name]['summary'], 'r')]) + ' |')
    lines.extend(['',
        'Self-reconstruction of the target is clearly higher than voice prediction, so the cross-modal mapping is the larger error source; this is not a strict attainable upper bound and cannot count as voice-prediction performance. Flow/Diffusion did not improve results with the current architecture and budget; this does not imply they are ineffective after full tuning.', '',
        '## Measured speed', '',
        'Single 25.6 s window, batch=1, median of 30 runs after GPU warm-up, prediction head only, excluding Whisper/WavLM encoding time; parameter counts include auxiliary training modules. These comparisons are not end-to-end real-time claims.', '',
        '|Configuration|Head ms|Trainable parameters|', '|---|---|---|'])
    for name in ('large_tcn_control', 'wavlm7_tcn', 'wavlm_large16_tcn', 'large_wave_unet', 'large_swin_unet', 'large_aligned_latent4', 'large_latent4_flow', 'large_latent4_diffusion'):
        d = runtime[name]
        lines.append(f'|{name}|{d["head_only_batch1_median_ms"]:.2f}|{d["training_model_parameters"]:,}|')
    lines.extend(['', '## Fair comparison and open issues', '',
        '- Same fixed five-fold singer split; inner 20% of singers used for early stopping to choose epochs, then refit on outer training singers. Speech feature standardization is fitted on training singers only.',
        '- New models all use AdamW lr 3e-4, weight decay 1e-4, batch 32, at most 20 epochs, waveform loss MSE + 0.2 slope error; a TCN control with the same loss is included. Generative models get 12 extra epochs of AE pre-training, a larger compute budget.',
        '- This round is a fixed-budget grid over architectures and encoder layers, not a full search over width, depth and learning rate for every model; only one training seed.',
        '- All inference uses the same 5 Hz time axis and 128-frame windows. This is offline bidirectional prediction and cannot be claimed as real-time causal prediction; recovering every detail of the original high-rate belt signal is not shown.',
        '- Targets are first decimated to 50 Hz with a 1.5 Hz low-pass, then averaged over 0.2 s to 5 Hz; each belt in each segment has its measured median subtracted and is divided by its SD. Scores therefore concern standardized shape and relative amplitude only; offset, absolute amplitude and physical lung volume from voice alone are not addressed. The anatomical identity and polarity of A/B still need confirmation from acquisition records.',
        '- Four fixed random seeds, ten-step Flow/DDIM sampling, mean of four predictions; generated samples are never chosen using the real test waveforms.',
        '- Test belt polarity was not flipped post hoc, predictions were not shifted and hard cases were not removed. P24 #12 is kept; anti-phase A/B alone does not establish an acquisition error.',
        '- Five-fold targets, order, singer coverage and finiteness of predictions were checked for all configurations. Severe-failure rates are saved in completion_integrity_and_failures.json; average scores must not hide these failures.',
        '- These outer singers have been inspected repeatedly over development; inner selection this round reduces direct winner-picking bias but cannot undo historical development bias. Confirmatory conclusions require external validation on new singers after freezing the protocol.', '',
        '## Priorities and next experiments', '',
        '1. Verify chest/abdomen identity of A/B, wiring polarity, synchronization and abnormal segments; keep the original labels, and any correction must be backed by acquisition evidence.',
        '2. Choose two or three candidates by per-singer paired gain, CCC and severe-failure rate; next round search only channels 32/64, depth 2/3, learning rate 1e-4/3e-4 in the inner loop and confirm stability with multiple seeds.',
        '3. For the latent direction, first optimize the paired shared representation, reverse auxiliary loss weight and temporal resolution; current results do not support simply adding random generation steps.',
        '4. Check label-resampling fidelity and edge peaks/troughs at 5/10/25 Hz; after obtaining a new-singer confirmation set, freeze all preprocessing, hyper-parameter selection and evaluation rules.', '',
        '## Results and resuming', '',
        f'- Project directory: `{args.root}`.',
        '- Entry points: README.md, progress_log.md, this report, family_study_progress.md.',
        '- Each family_runs/<config> directory stores results.json, five-fold weights and five-fold full measured/predicted waveforms; completed folds are skipped when resuming.',
        '- family_runs/large_aligned_latent is an unfinished trial run before the total-dimension fix and is not among the 16 configurations. The valid version has latent4 in its name.',
        '- Raw recordings and belt signals are unchanged; feature caches are kept in the project directory and public pretrained models cached locally can rebuild them.',
        '- In figures blue is measured and green is predicted, A and B shown separately; a typical case and a failure case are always shown together.', '',
        'The methodological basis is in the waveform-to-waveform model literature review; this report only states experiments completed locally and does not borrow scores from papers.', ''])
    text = '\n'.join(lines)
    for directory in (args.root, args.outputs):
        (directory / 'family_study_conclusions.md').write_text(text, encoding='utf-8')
    print(json.dumps({'completed': len(data), 'exploratory_top': best}, ensure_ascii=False))


if __name__ == '__main__':
    main()
