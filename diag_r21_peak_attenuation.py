"""Round 21 (analysis): are large inhalations under-predicted? Matched inhalations (frozen
extractor, onsets within 0.4 s); depth = peak - trough in the record's MEASURED std units for both
signals (so prediction shrinkage is visible). Per model: within-record regression slope of
predicted on measured depth (1 = no attenuation), ratio pred/meas for the top-10% and bottom-50%
measured depths of each record, and the R1 relation (depth ~ following phrase duration)."""
import json, sys
from pathlib import Path
import numpy as np
from breath_parameters import inhalations, match
from compare_r19 import load

ROOT = Path(__file__).resolve().parent


def main():
    names = sys.argv[1:]
    L = ['# Round 21 diagnostic: are large inhalations under-predicted?', '', 'Depth = peak - trough; predicted and measured both in units of the record\'s measured SD. Slope = within-record regression slope of predicted on measured depth (1 = no compression), singer medians.', '',
         '|Model|Belt|Within-record slope|Top-10% inhalations: pred/meas|Smallest 50%: pred/meas|Fraction of largest inhalations detected (±0.4 s)|', '|---|---|---|---|---|---|']
    for name in names:
        data = load(name)
        for k, c in enumerate('AB'):
            by = {}
            for key, (y, p) in data.items():
                sd = y[:, k].std() + 1e-9
                ey = inhalations(y[:, k]); ep = inhalations(p[:, k])
                if len(ey) < 6:
                    continue
                dy = (y[ey[:, 1].astype(int), k] - y[ey[:, 0].astype(int), k]) / sd
                m = match(ey[:, 0], ep[:, 0], 2)
                if len(m) < 4:
                    continue
                dp = np.array([(p[int(ep[j, 1]), k] - p[int(ep[j, 0]), k]) / sd for _, j in m]); dm = dy[[i for i, _ in m]]
                slope = np.polyfit(dm, dp, 1)[0]
                hi = dm >= np.quantile(dy, .9); lo = dm <= np.quantile(dy, .5)
                top = set(np.flatnonzero(dy >= np.quantile(dy, .9))); hit = {i for i, _ in m}
                by.setdefault(key[0], []).append([slope, np.median(dp[hi] / dm[hi]) if hi.any() else np.nan,
                                                  np.median(dp[lo] / dm[lo]) if lo.any() else np.nan, len(top & hit) / max(len(top), 1)])
            med = np.nanmedian([np.nanmedian(v, axis=0) for v in by.values()], axis=0)
            L.append(f'|{name}|{c}|{med[0]:.2f}|{med[1]:.2f}|{med[2]:.2f}|{med[3]:.2f}|')
    (ROOT / 'round21_peak_attenuation_diagnostic.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
