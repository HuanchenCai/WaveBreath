"""Held-out target-code oracle and deterministic bridge diagnostics, not deployed inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run_waveform_families import evaluate, load_records
from waveform_family_models import make_model, LatentGenerator


class TargetOracle(nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, y):
        return self.ae.belt_decoder(self.ae.belt(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    records = load_records(args.root / 'whisper_large_v3_voice_AB_5hz.npz')
    target_records = [(m, y, y) for m, _, y in records]
    outputs = {}
    for name in ('large_aligned_latent4', 'large_latent4_flow', 'large_latent4_diffusion'):
        path = args.root / 'family_runs' / name
        result = json.loads((path / 'results.json').read_text(encoding='utf-8'))
        oracle_rows, bridge_rows = [], []
        for fold in result['folds']:
            checkpoint = torch.load(path / fold['checkpoint'], map_location='cpu', weights_only=False)
            model = make_model(checkpoint['family'], checkpoint['input_dim'])
            model.load_state_dict(checkpoint['model'])
            ae = model.ae if isinstance(model, LatentGenerator) else model
            indices = [i for i, (m, _, _) in enumerate(records) if m['singer'] in fold['test_singers']]
            oracle_rows.extend(evaluate(TargetOracle(ae), target_records, indices, np.zeros(2), np.ones(2), 'cpu'))
            bridge_rows.extend(evaluate(ae, records, indices, checkpoint['mean'], checkpoint['std'], 'cpu'))
        outputs[name] = {kind: {c: {metric: float(np.median([row[c][metric] for row in rows]))
                           for metric in ('r', 'ccc', 'nrmse')} for c in ('A', 'B')}
                         for kind, rows in (('target_belt_oracle_not_voice_prediction', oracle_rows),
                                            ('deterministic_voice_bridge', bridge_rows))}
    (args.root / 'family_runs' / 'latent_oracle_diagnostics.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
