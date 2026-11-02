"""Round-14 (analysis, no training): belt-independent inhalation reference from audio.

Channel 0 = microphone, channel 1 = EGG (electroglottograph; ~99% of its loud-frame energy is
below 1 kHz, as expected for vocal-fold contact signals).
Per 10-ms frame (25-ms window):
  egg_on   : EGG 60-1000 Hz band level > (95th percentile of the record) - 25 dB
  breathy  : mic 1-4 kHz band level > (5th percentile of the record) + 10 dB and spectral
             flatness of the 1-4 kHz band > 0.2 (noise-like, not harmonic)
  breath frame = breathy and not egg_on
Breath-sound event = run of breath frames lasting 0.1-1.5 s after closing gaps < 50 ms.
Thresholds fixed a priori from the literature (breath sounds are aperiodic, mid-frequency
noise; Nakano et al. 2008) and NOT tuned on belt agreement.
Outputs per record: event onsets/offsets (s), and agreement with belt inhalations from the
frozen extractor (per band) at 5 Hz.
"""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from breath_parameters import inhalations, match

ROOT = Path(__file__).resolve().parent
HOP = .01


def band_db(P, f, lo, hi):
    return 10 * np.log10(P[(f >= lo) & (f < hi)].sum(0) + 1e-12)


def detect(audio, sr):
    nper = int(.025 * sr); hop = int(HOP * sr)
    f, _, Sm = signal.stft(audio[:, 0], sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    _, _, Se = signal.stft(audio[:, 1], sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    Pm, Pe = np.abs(Sm) ** 2, np.abs(Se) ** 2
    egg = band_db(Pe, f, 60, 1000); egg_on = egg > np.percentile(egg, 95) - 25
    mid = band_db(Pm, f, 1000, 4000)
    band = Pm[(f >= 1000) & (f < 4000)] + 1e-12
    flat = np.exp(np.log(band).mean(0)) / band.mean(0)
    breathy = (mid > np.percentile(mid, 5) + 10) & (flat > .2)
    br = breathy & ~egg_on
    # close gaps < 50 ms
    idx = np.flatnonzero(np.diff(np.r_[0, br.astype(int), 0])); runs = list(zip(idx[::2], idx[1::2]))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    ev = [(s * HOP, e * HOP) for s, e in merged if .1 <= (e - s) * HOP <= 1.5]
    centre_offset = nper / 2 / sr
    return [(a + centre_offset, b + centre_offset) for a, b in ev], float(egg_on.mean())


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        y_all, off = z['y'], z['offsets']
    out = []
    for i, r in enumerate(rows):
        a, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
        ev, egg_frac = detect(a, sr)
        y = y_all[off[i]:off[i + 1]]; dur = len(y) / 5
        rec = {'singer': info[i]['singer'], 'number': str(info[i]['number']), 'task': info[i]['task'],
               'breath_sounds_s': ev, 'egg_active_frac': egg_frac, 'breath_sounds_per_min': len(ev) / dur * 60}
        bs_on = np.array([s for s, _ in ev]) * 5  # 5-Hz frames
        for k, c in enumerate('AB'):
            belt = inhalations(y[:, k])
            on = belt[:, 0]
            # a belt inhalation is "sound-confirmed" if a breath sound starts within [-1.0, +0.6] s of its trough
            conf = [any(-5 <= b - t <= 3 for b in bs_on) for t in on]
            m = match(on, bs_on, 3)
            rec[c] = {'belt_events': len(on), 'belt_per_min': len(on) / dur * 60,
                      'frac_belt_confirmed_by_sound': float(np.mean(conf)) if len(on) else float('nan'),
                      'frac_sounds_matched_by_belt_0.6s': len(m) / max(len(bs_on), 1),
                      'confirmed_onsets_frames': [int(t) for t, ok in zip(on, conf) if ok]}
        out.append(rec)
        if (i + 1) % 20 == 0:
            print(f'{i + 1}/{len(rows)}', flush=True)
    (ROOT / 'diagnostics_r10' / 'breath_sound_reference.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['# Breath-sound reference (independent of the belts)', '',
             'Breath sound = EGG silent and noise-like 1-4 kHz microphone energy lasting 0.1-1.5 s; thresholds set a priori from the literature, not tuned against the belts.',
             '"Belt inhalation confirmed by breath sound" = a breath sound starts within 1.0 s before to 0.6 s after the belt inhalation trough.', '',
             '|Task|Segments|Breath sounds/min|A inhalations/min|B inhalations/min|A confirmed fraction|B confirmed fraction|Breath sounds matched by A (±0.6s)|Matched by B|', '|---|---|---|---|---|---|---|---|---|']
    for t in ('song', 'scale', 'reading', 'all'):
        g = [x for x in out if t == 'all' or x['task'] == t]
        q = lambda f: np.nanmedian([f(x) for x in g])
        lines.append(f"|{t}|{len(g)}|{q(lambda x: x['breath_sounds_per_min']):.1f}|{q(lambda x: x['A']['belt_per_min']):.1f}|"
                     f"{q(lambda x: x['B']['belt_per_min']):.1f}|{q(lambda x: x['A']['frac_belt_confirmed_by_sound']):.2f}|"
                     f"{q(lambda x: x['B']['frac_belt_confirmed_by_sound']):.2f}|{q(lambda x: x['A']['frac_sounds_matched_by_belt_0.6s']):.2f}|"
                     f"{q(lambda x: x['B']['frac_sounds_matched_by_belt_0.6s']):.2f}|")
    (ROOT / 'diagnostics_r10' / 'breath_sound_reference.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
