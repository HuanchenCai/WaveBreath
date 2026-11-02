"""Round-11 targets (stage B). Same records/offsets as the main WavLM-L16 cache.
tgt_dc        : DC removed before resample_poly; per-record median/std standardization.
tgt_dc_singer : DC removed; per-record median removed; divided by the median raw std of
                the SAME singer's OTHER recordings (leave-one-recording-out singer scale,
                emulating a per-singer calibration at deployment).
Only y is stored; x comes from wavlm_large_layer16_voice_AB_5hz.npz."""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from diag_r9_raw_polarity_vad import pipeline_target

ROOT = Path(__file__).resolve().parent


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_old, offsets, metadata = z['y'], z['offsets'], z['metadata']
    meta = json.loads(str(metadata))
    raws, dc_std = [], []
    for i, r in enumerate(rows):
        assert meta[i]['singer'] == r['singer_en'] and str(meta[i]['number']) == r['number']
        n = int(offsets[i + 1] - offsets[i])
        belt, _ = sf.read(r['belt'], dtype='float32', always_2d=True)
        raw, std_y = pipeline_target(belt, n, remove_dc_first=True)
        raws.append(raw); dc_std.append(std_y)
    singers = np.array([m['singer'] for m in meta])
    raw_std = np.array([x.std(axis=0) for x in raws])
    y_singer, scales = [], []
    for i, raw in enumerate(raws):
        others = np.flatnonzero((singers == singers[i]) & (np.arange(len(raws)) != i))
        scale = np.median(raw_std[others], axis=0)
        scales.append(scale.tolist())
        y_singer.append(((raw - np.median(raw, axis=0)) / scale).astype('float32'))
    y_dc = np.concatenate(dc_std).astype('float32'); y_sg = np.concatenate(y_singer)
    interior = np.concatenate([np.arange(offsets[i] + 10, offsets[i + 1] - 10) for i in range(len(raws))])
    check = {'max_abs_diff_dc_vs_old_interior': float(np.abs(y_dc - y_old)[interior].max()),
             'max_abs_diff_dc_vs_old_all': float(np.abs(y_dc - y_old).max()),
             'singer_scaled_record_std_median': np.median([y.std(axis=0) for y in y_singer], axis=0).tolist(),
             'singer_scaled_record_std_p5_p95': np.percentile([y.std(axis=0) for y in y_singer], [5, 95], axis=0).tolist()}
    for name, y in (('tgt_dc', y_dc), ('tgt_dc_singer', y_sg)):
        assert np.isfinite(y).all() and len(y) == len(y_old)
        np.savez_compressed(ROOT / f'r11_{name}_y.npz', y=y, offsets=offsets, metadata=metadata)
    (ROOT / 'r11_target_checks.json').write_text(json.dumps({**check, 'loro_scales': scales}, indent=1), encoding='utf-8')
    print(json.dumps(check, indent=1))


if __name__ == '__main__':
    main()
