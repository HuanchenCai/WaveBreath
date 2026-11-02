"""Recover frozen previous-best OOF waveforms and CCC for paired family comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_waveform_families import evaluate, load_records
from whisper_large_shape_tcn_cv import FeatureTCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    records = load_records(args.root / 'whisper_large_v3_voice_AB_5hz.npz')
    old = json.loads((args.root / 'whisper_large_shape_tcn_AB_cv.json').read_text(encoding='utf-8'))
    out = args.root / 'family_runs' / 'reference_large_shape'
    out.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in old['folds']:
        state = torch.load(args.root / fold['checkpoint'], map_location='cpu', weights_only=False)
        model = FeatureTCN(records[0][1].shape[1])
        model.load_state_dict(state['model'])
        indices = [i for i, (meta, _, _) in enumerate(records) if meta['singer'] in fold['test_singers']]
        rows = evaluate(model, records, indices, state['mean'], state['std'], 'cpu', out / f'fold{fold["fold"]}_waveforms.npz')
        folds.append({'fold': fold['fold'], 'test_singers': fold['test_singers'], 'test_segments': rows})
    rows = [r for f in folds for r in f['test_segments']]
    summary = {c: {key: float(np.median([r[c][key] for r in rows])) for key in ('r', 'ccc', 'nrmse', 'peak_f1', 'trough_f1', 'std_ratio')} for c in ('A', 'B')}
    (out / 'results.json').write_text(json.dumps({'completed': True, 'experiment': 'reference_large_shape',
        'scope': 'frozen prior checkpoints, reevaluated in batches; no refit', 'summary': summary, 'folds': folds}, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
