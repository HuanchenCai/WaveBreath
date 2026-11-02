"""Physical hypothesis test (analysis only): acoustics depend on lung volume / subglottal pressure,
not on how the volume is partitioned between rib cage and abdomen. Uncalibrated proxies (each belt
standardized within the record): volume proxy S = A + B, partition proxy D = B - A.
Per recording: CCC and r between measured and predicted S and D; share of measured variance in S and
D; also the 'partition after removing volume' D_perp = residual of D on S (measured) vs same for
prediction. Singer medians. Caveat: without isovolume calibration A+B is only a rough volume proxy."""
import json, sys
from pathlib import Path
import numpy as np
from compare_r19 import load

ROOT = Path(__file__).resolve().parent


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


def resid(x, z):
    Z = np.c_[z, np.ones(len(z))]
    return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]


def main():
    names = sys.argv[1:]
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    task = {(a['singer'], str(a['number'])): a['task'] for a in info}
    L = ['# Physical hypothesis test: does the voice encode the "total" rather than the "chest-abdomen partition"?', '',
         'S = chest + abdomen (lung-volume proxy, uncalibrated), D = abdomen - chest (partition); D_perp = partition component after removing S. Singer medians.', '',
         '|Model|Task|S r|S CCC|D r|D CCC|D_perp r|Measured var(D)/var(S)|', '|---|---|---|---|---|---|---|---|']
    for name in names:
        data = load(name)
        for t in ('all', 'song', 'scale', 'reading'):
            by = {}
            for k, (y, p) in data.items():
                if t != 'all' and task[k] != t:
                    continue
                Sy, Sp = y[:, 0] + y[:, 1], p[:, 0] + p[:, 1]; Dy, Dp = y[:, 1] - y[:, 0], p[:, 1] - p[:, 0]
                Dyp, Dpp = resid(Dy, Sy), resid(Dp, Sp)
                by.setdefault(k[0], []).append([np.corrcoef(Sy, Sp)[0, 1], ccc(Sy, Sp), np.corrcoef(Dy, Dp)[0, 1], ccc(Dy, Dp),
                                                np.corrcoef(Dyp, Dpp)[0, 1], Dy.var() / Sy.var()])
            m = np.median([np.median(v, axis=0) for v in by.values()], axis=0)
            L.append(f'|{name}|{t}|' + '|'.join(f'{x:.3f}' for x in m) + '|')
    (ROOT / 'volume_vs_partition_hypothesis.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
