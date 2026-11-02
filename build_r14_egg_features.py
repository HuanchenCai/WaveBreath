"""Round-14: frame-level glottal features from the EGG channel (channel 1), on the same
5-Hz grid as all caches (mean of 20-ms sub-frames inside each 200-ms frame).

Per 40-ms window, hop 20 ms (centre = sub-frame centre):
  level_db     : EGG 50-2000 Hz RMS level (dB, relative to the record's 95th percentile)
  periodicity  : max normalized autocorrelation for F0 in 60-1200 Hz
  log_f0       : log2 F0 from that autocorrelation peak (0 when periodicity < 0.5)
  cq25         : criterion-level contact quotient = fraction of samples above min+25% of the
                 peak-to-peak range (only when periodicity >= 0.5, else 0)
  degg_peak    : 99th percentile of dEGG/dt normalized by the window range (closing speed)
EGG polarity is set per record so that contact is positive: the closing (rising) flank is
steeper than the opening flank, i.e. |99th pct of dEGG| > |1st pct of dEGG| on voiced windows.
Motivation: lung volume changes the glottal voice source (Iwarsson et al. 1998)."""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent
SR = 16000


def window_feats(w):
    w = w - w.mean(); rng = w.max() - w.min() + 1e-9
    ac = np.correlate(w, w, 'full')[len(w) - 1:]; ac = ac / (ac[0] + 1e-12)
    lo, hi = SR // 1200, SR // 60
    lag = lo + int(np.argmax(ac[lo:hi])); per = float(ac[lag])
    d = np.diff(w) / rng
    if per < .5:
        return [per, 0., 0., float(np.percentile(d, 99))]
    cq = float((w > w.min() + .25 * rng).mean())
    return [per, float(np.log2(SR / lag)), cq, float(np.percentile(d, 99))]


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        off, metadata = z['offsets'], z['metadata']
    sos = signal.butter(4, [50, 2000], btype='band', fs=SR, output='sos')
    feats, flips = [], []
    for i, r in enumerate(rows):
        a, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
        e = signal.sosfiltfilt(sos, signal.resample_poly(a[:, 1], SR, sr))
        n5 = int(off[i + 1] - off[i]); n50 = n5 * 10; hop = SR // 50; half = SR // 50
        # polarity from voiced windows
        lv = np.array([np.sqrt(np.mean(e[k * hop:k * hop + hop] ** 2)) for k in range(n50)])
        voiced = lv > np.percentile(lv, 95) * 10 ** (-25 / 20)
        de = np.diff(e)
        sel = np.concatenate([de[k * hop:k * hop + hop] for k in np.flatnonzero(voiced)[:2000]]) if voiced.any() else de
        flip = abs(np.percentile(sel, 1)) > abs(np.percentile(sel, 99))
        if flip:
            e = -e
        flips.append(bool(flip))
        ref = np.percentile(lv, 95) + 1e-12
        f50 = np.zeros((n50, 5), 'float32')
        for k in range(n50):
            c = k * hop + hop // 2; w = e[max(0, c - half):c + half]
            if len(w) < 64:
                continue
            f50[k] = [20 * np.log10(lv[k] / ref + 1e-6), *window_feats(w)]
        feats.append(f50)
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(rows)}', flush=True)
    f50 = np.concatenate(feats); f5 = f50.reshape(-1, 10, 5).mean(axis=1)
    np.savez_compressed(ROOT / 'r14_egg_features.npz', x5=f5, x50=f50, offsets=off, metadata=metadata,
                        names=np.array(['level_db', 'periodicity', 'log_f0', 'cq25', 'degg_peak']))
    info = {'flipped_records': int(sum(flips)), 'n': len(flips),
            'cq25_voiced_median': float(np.median(f50[:, 3][f50[:, 1] >= .5])),
            'periodicity_ge_0.5_frac': float((f50[:, 1] >= .5).mean())}
    (ROOT / 'r14_egg_features_checks.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
    print(info)


if __name__ == '__main__':
    main()
