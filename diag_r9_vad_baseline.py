"""Round-9 rule-only baseline (no training, no target fitting).

Belt proxy = leaky integral of (+1 in unvoiced frames, -1 in voiced frames), i.e.
"inhale in pauses, deflate while phonating". Leak fixed a priori at 0.02/frame
(~10 s time constant) and VAD threshold identical to diag_r9_raw_polarity_vad.py.
Pearson r is offset/scale invariant, so no amplitude is estimated. Compared with the
frozen main model per task type. Saves the per-frame voiced masks for later use.
"""
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from diag_r9_raw_polarity_vad import voice_rms_db

ROOT = Path(__file__).resolve().parent


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    audit = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_all, offsets = z['y'], z['offsets']
    masks = []
    for i, (r, a) in enumerate(zip(rows, audit)):
        n = int(offsets[i + 1] - offsets[i]); y = y_all[offsets[i]:offsets[i + 1]]
        audio, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
        db = voice_rms_db(audio, sr, n)
        voiced = db > np.percentile(db, 90) - 25
        masks.append(voiced)
        drive = np.where(voiced, -1., 1.)
        proxy = signal.lfilter([1.], [1., -.98], drive)
        a['vad_baseline_r'] = [float(np.corrcoef(y[:, k], proxy)[0, 1]) for k in range(2)]
    np.savez_compressed(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz', voiced=np.concatenate(masks), offsets=offsets)
    lines = ['## Voiced/unvoiced rule baseline only (no training)', '', '|Task|Segments|Rule A r|Model A r|Rule B r|Model B r|',
             '|---|---|---|---|---|---|']
    for t in ('song', 'scale', 'reading', 'all'):
        g = [a for a in audit if t == 'all' or a['task'] == t]
        lines.append(f"|{t}|{len(g)}|{np.median([a['vad_baseline_r'][0] for a in g]):.3f}|{np.median([a['A']['r'] for a in g]):.3f}|"
                     f"{np.median([a['vad_baseline_r'][1] for a in g]):.3f}|{np.median([a['B']['r'] for a in g]):.3f}|")
    singers = sorted({a['singer'] for a in audit})
    for k, c in enumerate('AB'):
        d = [np.median([a[c]['r'] - a['vad_baseline_r'][k] for a in audit if a['singer'] == s]) for s in singers]
        lines.append(f'\n{c}: per-singer median of model minus rule {np.median(d):.3f}; model better for {sum(x > 0 for x in d)}/{len(d)} singers')
    (ROOT / 'diagnostics_r9' / 'vad_baseline.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
