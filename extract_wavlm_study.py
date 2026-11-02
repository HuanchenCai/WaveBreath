"""Frozen WavLM layers, aligned to existing 5-Hz Voice/belt records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor, WavLMModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--model', default='microsoft/wavlm-base-plus')
    ap.add_argument('--layers', nargs='+', type=int, default=[4, 7, 12])
    ap.add_argument('--prefix', default='wavlm')
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = args.model
    layers = tuple(args.layers)
    with (args.root / 'breath_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(args.root / 'whisper_base_voice_AB_5hz.npz', allow_pickle=False) as cache:
        offsets, metadata, targets = cache['offsets'], cache['metadata'], cache['y']
    assert len(rows) == len(offsets) - 1
    metadata_rows = json.loads(str(metadata))
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = WavLMModel.from_pretrained(model_id).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    width = model.config.hidden_size
    progress_path = args.root / f'{args.prefix}_extraction.progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'completed': 0}
    arrays = {}
    for layer in layers:
        path = args.root / f'{args.prefix}_layer{layer}.working.npy'
        arrays[layer] = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                                dtype='float16', shape=(int(offsets[-1]), width))
    print(f'model={model_id} device={device} resume={progress["completed"]}/{len(rows)}', flush=True)
    for i in range(progress['completed'], len(rows)):
        audio, sr = sf.read(rows[i]['audio'], dtype='float32', always_2d=True)
        voice = resample_poly(audio[:, 0], 16000, sr)
        n = int(offsets[i + 1] - offsets[i])
        parts = {layer: [] for layer in layers}
        for left in range(0, len(voice), 480000):
            count = min(150, n - left // 3200)
            if count <= 0:
                break
            chunk = voice[left:left + 480000]
            # Frame centers follow the 400-sample receptive field, 320-sample stride.
            inp = extractor(chunk, sampling_rate=16000, return_tensors='pt').input_values.to(
                device=device, dtype=next(model.parameters()).dtype)
            with torch.inference_mode():
                hidden = model(inp, output_hidden_states=True).hidden_states
            for layer in layers:
                values = hidden[layer][0].float().cpu().numpy()
                centers = (np.arange(len(values)) * 320 + 199.5) / 16000
                pooled = []
                for frame in range(count):
                    selected = (centers >= frame * .2) & (centers < (frame + 1) * .2)
                    if selected.any():
                        pooled.append(values[selected].mean(axis=0))
                    else:
                        pooled.append(values[np.argmin(abs(centers - (frame + .5) * .2))])
                parts[layer].append(np.asarray(pooled, dtype='float16'))
        for layer in layers:
            feature = np.concatenate(parts[layer])
            assert feature.shape == (n, width) and np.isfinite(feature).all()
            arrays[layer][offsets[i]:offsets[i + 1]] = feature
            arrays[layer].flush()
        progress_path.write_text(json.dumps({'completed': i + 1, 'total': len(rows), 'model': model_id,
                                             'layers': layers, 'last_recording': metadata_rows[i]}), encoding='utf-8')
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f'extracted {i + 1}/{len(rows)}', flush=True)
    outputs = []
    for layer in layers:
        path = args.root / f'{args.prefix}_layer{layer}_voice_AB_5hz.npz'
        np.savez_compressed(path, x=np.asarray(arrays[layer]), y=targets, offsets=offsets,
                            metadata=metadata, feature_source=np.array(model_id), layer=np.array(layer))
        outputs.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.root / f'{args.prefix}_feature_manifest.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(json.dumps(outputs), flush=True)


if __name__ == '__main__':
    main()
