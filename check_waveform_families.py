"""Integration check on real cached windows; never counted as model evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from waveform_family_models import make_model, waveform_loss, LatentGenerator
from run_waveform_families import load_records, normalize
from dual_belt_cv import DualWindows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    records = load_records(args.root / 'whisper_large_v3_voice_AB_5hz.npz')
    mean, std = normalize(records, [0, 1])
    ds = DualWindows(records, [0, 1], mean, std)
    batch = [ds[i] for i in range(2)]
    x, y, mask = [torch.stack([row[j] for row in batch]).cuda() for j in range(3)]
    results = []
    for name in ('tcn', 'transformer', 'wave_unet', 'swin_unet', 'nab_inspired', 'aligned_latent', 'flow', 'diffusion'):
        torch.manual_seed(123)
        model = make_model(name, x.shape[-1]).cuda()
        ae = model.ae if isinstance(model, LatentGenerator) else model
        if hasattr(ae, 'belt'):
            assert ae.belt(y).numel() < y.numel(), 'Belt latent must reduce total dimension'
            assert ae.voice(x).numel() < x.numel(), 'Voice latent must reduce total dimension'
        if isinstance(model, LatentGenerator):
            model.freeze_ae()
        model.train()
        loss = model.loss(x, y, mask) if hasattr(model, 'loss') else waveform_loss(model(x), y, mask)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            full = model(x)
            alone = model(x[:1])
        assert full.shape == y.shape and torch.isfinite(full).all()
        difference = float((full[:1] - alone).abs().max())
        assert difference < .005, (name, difference)
        results.append({'family': name, 'loss_finite': True, 'gradient_finite': True,
                        'output_shape': list(full.shape), 'batch_invariance_max_error': difference})
        del model
    # The feature extraction must preserve the exact target labels, offsets and recording order.
    with np.load(args.root / 'whisper_large_v3_voice_AB_5hz.npz') as reference:
        for layer in (4, 7, 12):
            with np.load(args.root / f'wavlm_layer{layer}_voice_AB_5hz.npz') as cache:
                for key in ('y', 'offsets', 'metadata'):
                    assert np.array_equal(reference[key], cache[key]), (layer, key)
    (args.root / 'waveform_family_integration_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
