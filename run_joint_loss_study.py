"""Reuse identical model, splits, optimizer and early stopping; vary loss only."""
import argparse
import functools
import json
import time
from pathlib import Path
import torch
import run_waveform_families as runner
from joint_waveform_losses import joint_loss

CONFIGS = {
    'loss_r010': {'pearson_weight': .1, 'ccc_weight': 0.},
    'loss_r030': {'pearson_weight': .3, 'ccc_weight': 0.},
    'loss_r100': {'pearson_weight': 1., 'ccc_weight': 0.},
    'loss_ccc030': {'pearson_weight': 0., 'ccc_weight': .3},
    'loss_ccc100': {'pearson_weight': 0., 'ccc_weight': 1.},
    'loss_r030_ccc030': {'pearson_weight': .3, 'ccc_weight': .3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--experiments', nargs='+', choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--pretrain', type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    protocol = {'configs': CONFIGS, 'derivative_weight': .2, 'mse_weight': 1.,
        'encoder': 'microsoft/wavlm-large layer16', 'head': 'tcn', 'epochs_max': args.epochs,
        'reference': 'wavlm_large16_tcn', 'selection': 'unchanged inner singer median AB Pearson r',
        'seed': 20260930, 'scope': 'exploratory loss ablation; previous outer subjects repeatedly inspected',
        'masking': 'statistics separately over time for each window and belt; flat targets excluded from r/CCC only'}
    destination = args.root / 'joint_loss_protocol.json'
    if destination.exists():
        assert json.loads(destination.read_text(encoding='utf-8')) == protocol, 'Changed protocol needs new run names'
    else:
        destination.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    for name in args.experiments:
        weights = CONFIGS[name]
        runner.EXPERIMENTS[name] = ('wavlm_large_layer16_voice_AB_5hz.npz', 'tcn')
        runner.waveform_loss = functools.partial(joint_loss, **weights)
        runner.run_experiment(args, name, device)
        path = args.root / 'family_runs' / name / 'results.json'
        result = json.loads(path.read_text(encoding='utf-8'))
        result['loss_configuration'] = {**weights, 'mse_weight': 1., 'derivative_weight': .2}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'requested': args.experiments, 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')}), flush=True)


if __name__ == '__main__':
    main()
