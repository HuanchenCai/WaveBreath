"""Round-11 stage B: same WavLM-L16 + TCN, folds, budget, loss and inner-r selection as
family_runs/wavlm_large16_tcn; ONLY the target definition changes. New config dirs."""
import sys
from pathlib import Path

import numpy as np

import run_waveform_families as R

ROOT = Path(__file__).resolve().parent
BASE = 'wavlm_large_layer16_voice_AB_5hz.npz'
_orig = R.load_records


def load_records(path):
    path = Path(path)
    if '__' not in path.name:
        return _orig(path)
    base, target = path.name.split('__')
    records = _orig(path.parent / base)
    with np.load(path.parent / f'r11_{target}_y.npz') as z:
        y, off = z['y'], z['offsets']
    assert len(off) == len(records) + 1
    out = []
    for i, (m, x, _) in enumerate(records):
        yy = y[off[i]:off[i + 1]]
        assert len(yy) == len(x)
        out.append((m, x, yy))
    return out


R.load_records = load_records
R.EXPERIMENTS.update({'r11_tgt_dc': (f'{BASE}__tgt_dc', 'tcn'),
                      'r11_tgt_dc_singer': (f'{BASE}__tgt_dc_singer', 'tcn')})

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--root', str(ROOT), '--experiments', *(sys.argv[1:] or ['r11_tgt_dc', 'r11_tgt_dc_singer'])]
    R.main()
