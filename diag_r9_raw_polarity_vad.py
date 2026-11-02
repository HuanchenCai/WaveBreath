"""Round-9 raw-signal audit: target edge artefacts, voice activity at record edges,
and a label-independent polarity index for each belt channel.

Polarity index (per record, per belt): mean 5-Hz slope of the standardized belt in
unvoiced frames minus that in voiced frames. Phonation deflates the lungs and pauses
are where singers inhale, so an "expansion = positive" channel is expected to give a
positive index. The index uses only the voice audio and the measured belt, never the
model prediction, so it is evidence independent of prediction quality. It cannot by
itself separate a wiring/sign inversion from genuine paradoxical motion.

Nothing here changes labels or cached targets.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

FILT = signal.butter(4, 1.5, fs=50, output='sos')


def pipeline_target(belt, n, remove_dc_first=False):
    b = belt - belt.mean(axis=0) if remove_dc_first else belt
    down = signal.resample_poly(b, 1, 882, axis=0)
    down = signal.sosfiltfilt(FILT, down, axis=0)
    y = down[:n * 10].reshape(n, 10, 2).mean(axis=1)
    return y, (y - np.median(y, axis=0)) / np.maximum(np.std(y, axis=0), 1e-6)


def voice_rms_db(audio, sr, n):
    hop = int(round(sr * .2))
    v = audio[:n * hop, 0].astype('float64')
    v = v - v.mean()
    frames = v[:n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--config', default='wavlm_large16_tcn')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    with (args.root / 'breath_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(args.root / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_cache, offsets = z['y'], z['offsets']
        meta = json.loads(str(z['metadata']))
    preds = {}
    run = args.root / 'family_runs' / args.config
    for fold in range(1, 6):
        with np.load(run / f'fold{fold}_waveforms.npz') as z:
            m = json.loads(str(z['metadata'])); off = z['offsets']
            for j, mm in enumerate(m):
                preds[(mm['singer'], str(mm['number']), mm['style'])] = z['prediction'][off[j]:off[j + 1]]
    out = []
    for i, row in enumerate(rows):
        m = meta[i]
        assert m['singer'] == row['singer_en'] and str(m['number']) == row['number']
        n = int(offsets[i + 1] - offsets[i])
        belt, sr = sf.read(row['belt'], dtype='float32', always_2d=True)
        audio, asr = sf.read(row['audio'], dtype='float32', always_2d=True)
        raw_y, y = pipeline_target(belt, n)
        assert np.allclose(y, y_cache[offsets[i]:offsets[i + 1]], atol=1e-4)
        _, y_dc = pipeline_target(belt, n, remove_dc_first=True)
        db = voice_rms_db(audio, asr, n)
        voiced = db > np.percentile(db, 90) - 25
        slope = np.gradient(y, axis=0) * 5  # standardized units per second
        p = preds[(m['singer'], str(m['number']), m['style'])]
        pslope = np.gradient(p, axis=0) * 5
        rec = {**m, 'frames': n, 'voiced_fraction': float(voiced.mean()),
               'voiced_first2s': float(voiced[:10].mean()), 'voiced_last2s': float(voiced[-10:].mean()),
               'first_voiced_s': float(np.argmax(voiced) / 5), 'last_voiced_s_from_end': float(np.argmax(voiced[::-1]) / 5),
               'edge_artifact_maxabs_first2s': float(np.abs(y - y_dc)[:10].max()),
               'edge_artifact_maxabs_last2s': float(np.abs(y - y_dc)[-10:].max()),
               'edge_artifact_maxabs_interior': float(np.abs(y - y_dc)[10:-10].max()),
               'raw_dc': belt.mean(axis=0).tolist(), 'raw_std': belt.std(axis=0).tolist(),
               'raw_clip_fraction': (np.abs(belt) > .99).mean(axis=0).tolist(),
               'belt_file_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(row['belt']))),
               'measured_AB_r': float(np.corrcoef(y.T)[0, 1]),
               'predicted_AB_r': float(np.corrcoef(p.T)[0, 1])}
        for k, c in enumerate('AB'):
            if voiced.all() or not voiced.any():
                idx = pidx = float('nan')
            else:
                idx = float(slope[~voiced, k].mean() - slope[voiced, k].mean())
                pidx = float(pslope[~voiced, k].mean() - pslope[voiced, k].mean())
            rec[c] = {'polarity_index': idx, 'pred_polarity_index': pidx,
                      'r': float(np.corrcoef(y[:, k], p[:, k])[0, 1])}
        # edge vs interior error, split by voicing
        se = ((p - y) ** 2).mean(axis=1)
        rec['mse_voiced'] = float(se[voiced].mean()) if voiced.any() else float('nan')
        rec['mse_unvoiced'] = float(se[~voiced].mean()) if (~voiced).any() else float('nan')
        out.append(rec)
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(rows)}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'raw_polarity_vad.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('done', len(out))


if __name__ == '__main__':
    main()
