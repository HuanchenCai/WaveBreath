"""Round-12 exploratory: inhalation onsets read directly from the onset head (lambda>0 runs)
versus onsets extracted from the predicted waveform. Head onsets = local maxima of the
onset probability above a threshold (0.3 and 0.5 both reported), min distance 5 frames.
Reference = frozen extractor on the measured target. Seeds averaged per segment."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from breath_parameters import f1, inhalations

ROOT = Path(__file__).resolve().parent
TASK = {(a['singer'], str(a['number'])): a['task'] for a in json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))}


def main():
    lines = ['# Direct event-head readout vs waveform extraction (exploratory)', '', '|Configuration|Mode|Task|A F1±0.4s|B F1±0.4s|A F1±1.0s|B F1±1.0s|', '|---|---|---|---|---|---|---|']
    for lam in ('0.3', '1'):
        res = {}
        for s in range(3):
            for fold in range(1, 6):
                with np.load(ROOT / 'family_runs' / f'r12_tgt_dc_lam{lam}_s{s}' / f'fold{fold}_waveforms.npz') as z:
                    meta = json.loads(str(z['metadata'])); off = z['offsets']
                    for j, m in enumerate(meta):
                        y = z['target'][off[j]:off[j + 1]]; p = z['prediction'][off[j]:off[j + 1]]
                        key = (m['singer'], str(m['number']))
                        for k, c in enumerate('AB'):
                            ref = inhalations(y[:, k])[:, 0]
                            est = {'waveform': inhalations(p[:, k])[:, 0]}
                            for th in (.3, .5):
                                est[f'head{th}'] = find_peaks(p[:, 4 + k], height=th, distance=5)[0].astype(float)
                            for way, e in est.items():
                                for tol in (2, 5):
                                    res.setdefault((way, key, c, tol), []).append(f1(ref, e, tol)[0])
        for way in ('waveform', 'head0.3', 'head0.5'):
            for task in ('all', 'song', 'scale', 'reading'):
                cells = []
                for tol in (2, 5):
                    for c in 'AB':
                        keys = {k[1] for k in res if k[0] == way and (task == 'all' or TASK[k[1]] == task)}
                        singers = sorted({k[0] for k in keys})
                        cells.append(np.median([np.median([np.mean(res[(way, k, c, tol)]) for k in keys if k[0] == sg]) for sg in singers]))
                lines.append(f'|lam{lam}|{way}|{task}|' + '|'.join(f'{v:.3f}' for v in (cells[0], cells[1], cells[2], cells[3])) + '|')
    (ROOT / 'diagnostics_r10' / 'event_head_direct_readout.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
