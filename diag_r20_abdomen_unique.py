"""Round 20 (analysis, no training): does the voice carry abdomen information beyond the chest?
Per recording, partial correlations (5 Hz, whole record):
  p1 = pcorr(B_meas, B_pred | A_meas)   abdomen-specific signal recovered beyond the measured chest
  p2 = pcorr(A_meas, A_pred | B_meas)   chest-specific signal beyond the measured abdomen
  p3 = pcorr(B_meas, B_pred | A_pred)   does predicted B add anything beyond predicted A?
Controls: same partial correlation with the prediction taken from ANOTHER recording of the same
length class (shuffled), giving the chance level. Singer-level medians."""
import json
import sys
from pathlib import Path
import numpy as np
from compare_r19 import load

ROOT = Path(__file__).resolve().parent


def pcorr(x, y, z):
    Z = np.c_[z, np.ones(len(z))]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    names = sys.argv[1:] or ['r15_ens_convlstm_unet']
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    task = {(a['singer'], str(a['number'])): a['task'] for a in info}
    L = ['# Round 20 diagnostic: does the voice carry abdomen-specific information?', '',
         'Partial correlation: after controlling for one measured belt, the correlation of a predicted belt with the "remainder" of the other measured belt. Control = prediction from another recording of the same singer (trimmed to equal length). Singer medians.', '',
         '|Model|Task|Abdomen-specific pcorr(B meas, B pred | A meas)|Control|Chest-specific pcorr(A meas, A pred | B meas)|Control|Increment of pred B over pred A pcorr(B meas, B pred | A pred)|', '|---|---|---|---|---|---|---|']
    rng = np.random.default_rng(0)
    for name in names:
        data = load(name); keys = list(data)
        for t in ('all', 'song', 'scale', 'reading'):
            by = {}
            for k in keys:
                if t != 'all' and task[k] != t:
                    continue
                y, p = data[k]
                others = [o for o in keys if o[0] == k[0] and o != k]
                o = others[rng.integers(len(others))] if others else k
                po = data[o][1]; n = min(len(y), len(po))
                v = [pcorr(y[:, 1], p[:, 1], y[:, 0]), pcorr(y[:n, 1], po[:n, 1], y[:n, 0]),
                     pcorr(y[:, 0], p[:, 0], y[:, 1]), pcorr(y[:n, 0], po[:n, 0], y[:n, 1]),
                     pcorr(y[:, 1], p[:, 1], p[:, 0])]
                by.setdefault(k[0], []).append(v)
            med = np.median([np.median(v, axis=0) for v in by.values()], axis=0)
            L.append(f'|{name}|{t}|' + '|'.join(f'{x:+.3f}' for x in med) + '|')
    (ROOT / 'round20_abdomen_specific_information.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
