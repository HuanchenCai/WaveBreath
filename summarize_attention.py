import json
from collections import defaultdict
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
names = {
    'direct': 'whisper_feature_tcn_AB_cv.json',
    'convolution_latent': 'latent_voice_mapper_dim4_cv.json',
    'global_attention': 'attention_voice_belt_cv.json',
    'local_attention': 'local_attention_voice_belt_cv.json',
}
data = {name: json.loads((root / file).read_text(encoding='utf-8')) for name, file in names.items()}
by_singer = {}
for name, cv in data.items():
    grouped = defaultdict(lambda: {'A': [], 'B': []})
    for fold in cv['folds']:
        for row in fold['test_segments']:
            for channel in ('A', 'B'):
                grouped[row['singer']][channel].append(row[channel]['r'])
    by_singer[name] = {s: {c: median(v[c]) for c in ('A', 'B')} for s, v in grouped.items()}

out = {'summary': {n: d['summary'] for n, d in data.items()}, 'singer_comparison': {}}
for channel in ('A', 'B'):
    singers = sorted(by_singer['direct'])
    differences = {name: [by_singer[name][s][channel] - by_singer['direct'][s][channel]
                          for s in singers] for name in names if name != 'direct'}
    out['singer_comparison'][channel] = {
        name: {'median_delta_r_vs_direct': median(delta),
               'singers_improved': sum(d > 0 for d in delta),
               'singers_total': len(delta)}
        for name, delta in differences.items()
    }
    out['singer_comparison'][channel]['local_attention_by_singer'] = {
        s: round(d, 3) for s, d in zip(singers, differences['local_attention'])
    }

path = root / 'attention_model_paired_comparison.json'
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out['singer_comparison'], ensure_ascii=False, indent=2))
