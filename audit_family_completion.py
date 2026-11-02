"""Verify complete outer predictions and summarize failure tails without relabeling."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_waveform_families import EXPERIMENTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    runs = args.root / 'family_runs'
    outputs = {}
    for name in EXPERIMENTS:
        result = json.loads((runs / name / 'results.json').read_text(encoding='utf-8'))
        assert result['completed'] and len(result['folds']) == 5, name
        seen_singers, rows = [], []
        for fold in result['folds']:
            number = fold['fold']
            seen_singers.extend(fold['test_singers'])
            rows.extend(fold['test_segments'])
            with np.load(runs / name / f'fold{number}_waveforms.npz', allow_pickle=False) as pred, \
                 np.load(runs / 'reference_large_shape' / f'fold{number}_waveforms.npz', allow_pickle=False) as ref:
                assert np.array_equal(pred['metadata'], ref['metadata']), name
                assert np.array_equal(pred['offsets'], ref['offsets']), name
                assert np.array_equal(pred['target'], ref['target']), name
                assert pred['prediction'].shape == pred['target'].shape, name
                assert np.isfinite(pred['prediction']).all(), name
        assert len(seen_singers) == len(set(seen_singers)) == 32, name
        assert len(rows) == 197, name
        assert len({(r['singer'], r['style'], r['number']) for r in rows}) == 197, name
        channels = {}
        for c in ('A', 'B'):
            groups = defaultdict(list)
            for row in rows:
                groups[row['singer']].append(row[c]['r'])
            ordered = sorted(rows, key=lambda r: r[c]['r'])
            channels[c] = {
                'segments_r_nonpositive': sum(r[c]['r'] <= 0 for r in rows),
                'segments_r_below_half': sum(r[c]['r'] < .5 for r in rows),
                'singer_median_r': float(np.median([np.median(v) for v in groups.values()])),
                'worst_three': [{k: r[k] for k in ('singer', 'style', 'number')} | {'r': r[c]['r']} for r in ordered[:3]],
            }
        outputs[name] = {'verified_folds': 5, 'verified_singers': 32, 'verified_segments': 197,
                         'identical_outer_targets_and_order': True, 'channels': channels}
    destination = runs / 'completion_integrity_and_failures.json'
    destination.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'verified_configurations': len(outputs), 'file': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
