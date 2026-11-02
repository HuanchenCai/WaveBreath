import json
from collections import defaultdict
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
models = {name: json.loads((root / path).read_text(encoding='utf-8')) for name, path in {
    'base': 'whisper_feature_tcn_AB_cv.json',
    'large': 'whisper_large_tcn_AB_cv.json',
}.items()}
values = {}
for name, model in models.items():
    by_singer = defaultdict(lambda: {'A': [], 'B': []})
    for fold in model['folds']:
        for row in fold['test_segments']:
            for channel in ('A', 'B'):
                by_singer[row['singer']][channel].append(row[channel]['r'])
    values[name] = {s: {c: median(r[c]) for c in ('A', 'B')} for s, r in by_singer.items()}
paired = {}
for channel in ('A', 'B'):
    changes = [values['large'][s][channel] - values['base'][s][channel]
               for s in sorted(values['base'])]
    paired[channel] = {'median_delta_r': median(changes),
                       'improved_singers': sum(x > 0 for x in changes),
                       'total_singers': len(changes)}
out = {'summary': {n: d['summary'] for n, d in models.items()}, 'paired': paired}
(root / 'whisper_large_vs_base_paired.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(paired, ensure_ascii=False, indent=2))
