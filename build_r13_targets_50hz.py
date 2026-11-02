"""Round-13 batch 3 target: identical to tgt_dc (DC removed before resample_poly to 50 Hz,
4th-order 1.5-Hz Butterworth zero-phase low-pass) but WITHOUT the 10-sample averaging, i.e.
kept at 50 Hz; per-record median/std standardization. Frame 10k..10k+9 corresponds to 5-Hz
frame k. Only the sampling rate differs from the batch-1 target."""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent
FILT = signal.butter(4, 1.5, fs=50, output='sos')


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        y5, off, meta = z['y'], z['offsets'], z['metadata']
    out, checks = [], []
    for i, r in enumerate(rows):
        belt, _ = sf.read(r['belt'], dtype='float32', always_2d=True)
        d = signal.sosfiltfilt(FILT, signal.resample_poly(belt - belt.mean(axis=0), 1, 882, axis=0), axis=0)
        n = int(off[i + 1] - off[i]) * 10
        assert len(d) >= n
        y = d[:n]; y = (y - np.median(y, axis=0)) / np.maximum(np.std(y, axis=0), 1e-6)
        back = y.reshape(-1, 10, 2).mean(axis=1)
        checks.append([float(np.corrcoef(back[:, k], y5[off[i]:off[i + 1], k])[0, 1]) for k in range(2)])
        out.append(y.astype('float32'))
    np.savez_compressed(ROOT / 'r13_tgt_dc_50hz_y.npz', y=np.concatenate(out), offsets=off * 10, metadata=meta)
    c = np.array(checks)
    info = {'min_r_5hz_equivalence': c.min(axis=0).tolist(), 'median': np.median(c, axis=0).tolist()}
    (ROOT / 'r13_tgt_dc_50hz_checks.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print(info)


if __name__ == '__main__':
    main()
