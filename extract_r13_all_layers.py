"""Round-13 batch 2: all hidden layers of a frozen SSL encoder pooled to the same 5-Hz grid
as every earlier cache (frame k covers [0.2k, 0.2k+0.2) s; encoder frame centres from its
stride/receptive field). Output: float16 memmap (frames, layers, dim) in a LOCAL cache dir
(fast disk), plus a manifest with hashes in the study folder. Resumable per recording."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
MODELS = {'wavlm_large': {'id': 'microsoft/wavlm-large', 'sr': 16000, 'stride': 320, 'rf': 400, 'remote': False},
          'mert330m': {'id': 'm-a-p/MERT-v1-330M', 'sr': 24000, 'stride': 320, 'rf': 400, 'remote': True}}


def load(spec, device):
    from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor, WavLMModel
    if spec['remote']:
        model = AutoModel.from_pretrained(spec['id'], trust_remote_code=True)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(spec['id'], trust_remote_code=True)
    else:
        model = WavLMModel.from_pretrained(spec['id']); fe = Wav2Vec2FeatureExtractor.from_pretrained(spec['id'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, fe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS), required=True)
    ap.add_argument('--cache', type=Path, default=Path.home() / 'breath_r13_cache')
    ap.add_argument('--rate', type=int, default=5, help='output frame rate in Hz (5 or 50)')
    ap.add_argument('--layers', type=int, nargs='*', help='subset of hidden-state indices (default: all)')
    args = ap.parse_args()
    spec = MODELS[args.model]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        offsets, metadata = z['offsets'], z['metadata']
    meta = json.loads(str(metadata))
    model, fe = load(spec, device)
    keep = args.layers if args.layers else list(range(model.config.num_hidden_layers + 1))
    layers = len(keep); dim = model.config.hidden_size; up = args.rate // 5; hop = 1 / args.rate
    args.cache.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{'all_layers' if not args.layers else 'L' + '-'.join(map(str, keep))}_{args.rate}hz"
    path = args.cache / f'{tag}.f16.npy'
    arr = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+', dtype='float16',
                                    shape=(int(offsets[-1]) * up, layers, dim))
    prog = args.cache / f'{tag}_progress.json'
    start = json.loads(prog.read_text())['completed'] if prog.exists() else 0
    chunk = 30 * spec['sr']; per_chunk = 150
    for i in range(start, len(rows)):
        assert meta[i]['singer'] == rows[i]['singer_en'] and str(meta[i]['number']) == rows[i]['number']
        audio, sr = sf.read(rows[i]['audio'], dtype='float32', always_2d=True)
        voice = resample_poly(audio[:, 0], spec['sr'], sr)
        n = int(offsets[i + 1] - offsets[i]) * up; parts = []
        for c0 in range(0, len(voice), chunk):
            count = min(per_chunk * up, n - c0 // (spec['sr'] // args.rate))
            if count <= 0:
                break
            inp = fe(voice[c0:c0 + chunk], sampling_rate=spec['sr'], return_tensors='pt').input_values.to(device)
            with torch.inference_mode():
                hs = model(inp, output_hidden_states=True).hidden_states
            h = torch.stack([hs[k] for k in keep], dim=1)[0].float().cpu().numpy().transpose(1, 0, 2)  # (frames, layers, dim)
            centres = (np.arange(len(h)) * spec['stride'] + (spec['rf'] - 1) / 2) / spec['sr']
            pooled = []
            for k in range(count):
                sel = (centres >= k * hop) & (centres < (k + 1) * hop)
                pooled.append(h[sel].mean(axis=0) if sel.any() else h[np.argmin(abs(centres - (k + .5) * hop))])
            parts.append(np.asarray(pooled, dtype='float16'))
        feat = np.concatenate(parts)
        assert feat.shape == (n, layers, dim) and np.isfinite(feat).all()
        arr[offsets[i] * up:offsets[i + 1] * up] = feat; arr.flush()
        prog.write_text(json.dumps({'completed': i + 1, 'total': len(rows)}))
        if (i + 1) % 10 == 0:
            print(f'{args.model} {i + 1}/{len(rows)}', flush=True)
    manifest = {'model': spec['id'], 'layers': keep, 'dim': dim, 'frames': int(offsets[-1]) * up, 'path': str(path),
                'grid': f'{args.rate} Hz, frame k = [k/rate, (k+1)/rate) s, mean of encoder frames by centre',
                'recordings': len(rows)}
    (ROOT / f'r13_{tag}_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print('done', manifest, flush=True)


if __name__ == '__main__':
    main()
