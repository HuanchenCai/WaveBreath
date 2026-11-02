import json
from collections import defaultdict
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
files = {
    'direct_tcn': 'whisper_feature_tcn_AB_cv.json',
    'residual_attention': 'residual_attention_tcn_AB_cv.json',
    'shape_loss_tcn': 'shape_loss_tcn_AB_cv.json',
}
models = {key: json.loads((root / file).read_text(encoding='utf-8')) for key, file in files.items()}
grouped = {}
for name, result in models.items():
    by_singer = defaultdict(lambda: {'A': [], 'B': []})
    for fold in result['folds']:
        for row in fold['test_segments']:
            for channel in ('A', 'B'):
                by_singer[row['singer']][channel].append(row[channel]['r'])
    grouped[name] = {s: {c: median(v[c]) for c in ('A', 'B')} for s, v in by_singer.items()}

comparison = {}
for name in ('residual_attention', 'shape_loss_tcn'):
    comparison[name] = {}
    for channel in ('A', 'B'):
        delta = [grouped[name][s][channel] - grouped['direct_tcn'][s][channel]
                 for s in sorted(grouped['direct_tcn'])]
        comparison[name][channel] = {'median_singer_delta_r': median(delta),
                                     'singers_improved': sum(d > 0 for d in delta),
                                     'total_singers': len(delta)}
out = {'summaries': {n: m['summary'] for n, m in models.items()}, 'paired': comparison}
(root / 'redesign_model_paired_comparison.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(comparison, ensure_ascii=False, indent=2))
