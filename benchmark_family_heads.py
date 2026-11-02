"""Compare waveform-head latency on actual held-out windows; encoder cost excluded."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_waveform_families import EXPERIMENTS
from waveform_family_models import make_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for name, (cache_file, family) in EXPERIMENTS.items():
        directory = args.root / 'family_runs' / name
        result_file = directory / 'results.json'
        if not result_file.exists() or not json.loads(result_file.read_text(encoding='utf-8')).get('completed'):
            continue
        state = torch.load(directory / 'fold1.pt', map_location='cpu', weights_only=False)
        with np.load(args.root / cache_file, allow_pickle=False) as cache:
            meta = json.loads(str(cache['metadata']))
            i = next(i for i, m in enumerate(meta) if m['singer'] in state['test_singers'])
            begin, end = cache['offsets'][i:i + 2]
            chunk = cache['x'][begin:min(begin + 128, end)].astype('float32')
        normalized = (chunk - state['mean']) / state['std']
        normalized = np.pad(normalized, ((0, 128 - len(normalized)), (0, 0)))
        x = torch.from_numpy(normalized[None]).to(device)
        model = make_model(family, state['input_dim']).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        samples = []
        with torch.inference_mode():
            for _ in range(5):
                model(x)
            for _ in range(30):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
        results[name] = {'head_only_batch1_median_ms': float(np.median(samples)),
                          'training_model_parameters': sum(p.numel() for p in model.parameters()),
                          'encoder_excluded': True, 'device': device, 'window_s': 25.6}
        del model
    (args.root / 'family_runs' / 'head_runtime.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
