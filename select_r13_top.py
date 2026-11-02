"""Pre-registered rule (written before batch 1 finished): rank families by the mean over
the 5 outer folds of the winner's mean inner CCC (3 seeds). Outer test scores are NOT used.
The top-2 families go on to batch 1b (training scheme) and batch 2 (input)."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    rank = {}
    for fam in ('tcn', 'bigru', 'convlstm', 'transformer', 'unet'):
        p = ROOT / 'family_runs' / f'r13_{fam}' / 'results.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        if len(d['folds']) < 5:
            print(fam, 'incomplete', len(d['folds'])); continue
        rank[fam] = float(np.mean([np.mean(f['confirm_inner_ccc'][str(f['winner'])]) for f in d['folds']]))
    order = sorted(rank, key=lambda f: -rank[f])
    out = {'inner_score': rank, 'order': order, 'top2': order[:2]}
    (ROOT / 'r13_batch1_selection.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
