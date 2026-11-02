"""Confirm large-encoder features retain the same targets and timeline."""
import argparse
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
args = ap.parse_args()
results = []
with np.load(args.root / 'whisper_large_v3_voice_AB_5hz.npz', allow_pickle=False) as ref:
    for layer in (8, 16, 24):
        filename = f'wavlm_large_layer{layer}_voice_AB_5hz.npz'
        with np.load(args.root / filename, allow_pickle=False) as cache:
            for key in ('y', 'offsets', 'metadata'):
                assert np.array_equal(cache[key], ref[key]), (layer, key)
            x = cache['x']
            assert x.shape == (len(ref['y']), 1024) and np.isfinite(x).all()
            assert str(cache['feature_source']) == 'microsoft/wavlm-large'
            results.append({'file': filename, 'shape': list(x.shape), 'target_alignment_verified': True})
(args.root / 'wavlm_large_cache_checks.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results))
