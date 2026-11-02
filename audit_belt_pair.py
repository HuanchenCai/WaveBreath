"""Read-only label audit: within-recording belt agreement and short-lag relation."""
import json
from pathlib import Path

import numpy as np
from sequence_model_screen import recordings


def correlation(a, b):
    if min(np.std(a), np.std(b)) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lag_relation(a, b, maximum=15):
    options = []
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:lag]
        elif lag > 0:
            aa, bb = a[:-lag], b[lag:]
        else:
            aa, bb = a, b
        options.append((lag, correlation(aa, bb)))
    return max(options, key=lambda item: abs(item[1]))


root = Path(__file__).resolve().parent
rows = []
for meta, _, y in recordings(root / 'whisper_base_voice_AB_5hz.npz'):
    lag, best_r = lag_relation(y[:, 0], y[:, 1])
    rows.append({**meta, 'belt_AB_r_zero_lag': correlation(y[:, 0], y[:, 1]),
                 'belt_AB_best_abs_r_within_3s': best_r, 'best_lag_frames': lag})

target = [row for row in rows if row['singer'] == 'P24']
summary = {
    'recordings': len(rows),
    'AB_r_median': float(np.median([r['belt_AB_r_zero_lag'] for r in rows])),
    'AB_negative_count': sum(r['belt_AB_r_zero_lag'] < 0 for r in rows),
    'AB_strongly_negative_count': sum(r['belt_AB_r_zero_lag'] < -.5 for r in rows),
    'P24': target,
    'most_negative': sorted(rows, key=lambda r: r['belt_AB_r_zero_lag'])[:12],
}
(root / 'belt_pair_label_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'most_negative'}, ensure_ascii=False, indent=2))
