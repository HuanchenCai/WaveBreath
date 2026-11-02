"""Round-9 second rule baseline, designed AFTER the leaky-integral rule failed (post hoc,
reported as such). Phrase = voiced run after closing gaps < 0.4 s (2 frames); within each
phrase the proxy falls linearly +1 -> -1, within each pause it rises -1 -> +1. No training."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def sawtooth(voiced):
    v = voiced.copy()
    for s, e in runs(~v):
        if e - s <= 2 and s > 0 and e < len(v):
            v[s:e] = True
    proxy = np.zeros(len(v))
    for s, e in runs(v):
        proxy[s:e] = np.linspace(1, -1, e - s)
    for s, e in runs(~v):
        proxy[s:e] = np.linspace(-1, 1, e - s)
    return proxy


def main():
    audit = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        voiced, off = z['voiced'], z['offsets']
    with np.load(ROOT / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as z:
        y_all = z['y']
    for i, a in enumerate(audit):
        y = y_all[off[i]:off[i + 1]]; p = sawtooth(voiced[off[i]:off[i + 1]])
        a['sawtooth_r'] = [float(np.corrcoef(y[:, k], p)[0, 1]) for k in range(2)]
    lines = ['## Phrase sawtooth rule baseline (post-hoc design, no training)', '', '|Task|Segments|Sawtooth A r|Model A r|Sawtooth B r|Model B r|', '|---|---|---|---|---|---|']
    for t in ('song', 'scale', 'reading', 'all'):
        g = [a for a in audit if t == 'all' or a['task'] == t]
        lines.append(f"|{t}|{len(g)}|{np.median([a['sawtooth_r'][0] for a in g]):.3f}|{np.median([a['A']['r'] for a in g]):.3f}|"
                     f"{np.median([a['sawtooth_r'][1] for a in g]):.3f}|{np.median([a['B']['r'] for a in g]):.3f}|")
    br = np.array([a['B']['r'] for a in audit]); sb = np.array([a['sawtooth_r'][1] for a in audit])
    lines.append(f'\nAmong the {int((br < .5).sum())} segments with low B correlation (model r<0.5), median sawtooth-rule r {np.median(sb[br < .5]):.3f}; other segments {np.median(sb[br >= .5]):.3f}')
    (ROOT / 'diagnostics_r9' / 'sawtooth_baseline.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
