"""Round-9: join raw audit with task type and fixed-model scores to test whether
low B correlation tracks target clarity (task, raw amplitude, polarity) rather than
the model. Descriptive/post-hoc only; no labels are changed.
"""
import csv
import json
import ntpath
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / '._Data'


def task_of(row):
    folder = DATA / ntpath.basename(ntpath.dirname(ntpath.dirname(row['belt']))) / row['singer_cn']
    title = ''
    for f in folder.glob(f"{row['singer_cn']}({row['number']})*.wav"):
        title = f.stem.split(')', 1)[1]
    eng = re.sub(r'_Voice_EGG\.wav$', '', ntpath.basename(row['audio']))
    text = title + ' ' + eng
    if re.search(r'\u5ff5|\u6717\u8bfb|\u6b4c\u8bcd|Lyrics', text):  # Chinese keywords for "read / reading / lyrics"
        return 'reading', title or eng
    if re.search(r'\u97f3\u9636|Scale|Ascending|Descending', text):  # Chinese keyword for "scale"
        return 'scale', title or eng
    return 'song', title or eng


def main():
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    audit = json.loads((ROOT / 'diagnostics_r9' / 'raw_polarity_vad.json').read_text(encoding='utf-8'))
    assert len(rows) == len(audit)
    for r, a in zip(rows, audit):
        assert r['singer_en'] == a['singer'] and r['number'] == str(a['number'])
        a['task'], a['title'] = task_of(r)
        # amplitude relative to the singer's median raw std for that channel
    singers = sorted({a['singer'] for a in audit})
    for a in audit:
        same = [b for b in audit if b['singer'] == a['singer']]
        a['rel_amp'] = [a['raw_std'][k] / np.median([b['raw_std'][k] for b in same]) for k in range(2)]
    lines = ['# Target clarity diagnostic: task type, raw amplitude, polarity index', '',
             'Post-hoc descriptive analysis; complete outer held-out predictions of the main model wavlm_large16_tcn; no label changes, no retraining.', '']
    lines += ['## By task type', '', '|Task|Segments|Median A r|Median B r|Segments B r<0.5|Median measured A-B corr|Median B polarity index|Median B relative amplitude|', '|---|---|---|---|---|---|---|---|']
    for t in ('song', 'scale', 'reading'):
        g = [a for a in audit if a['task'] == t]
        lines.append(f"|{t}|{len(g)}|{np.median([a['A']['r'] for a in g]):.3f}|{np.median([a['B']['r'] for a in g]):.3f}|"
                     f"{sum(a['B']['r'] < .5 for a in g)}|{np.median([a['measured_AB_r'] for a in g]):.3f}|"
                     f"{np.median([a['B']['polarity_index'] for a in g]):.3f}|{np.median([a['rel_amp'][1] for a in g]):.2f}|")
    pa = np.array([a['A']['polarity_index'] for a in audit]); pb = np.array([a['B']['polarity_index'] for a in audit])
    lines += ['', '## Model-independent polarity index', '',
              'Definition: mean slope of the standardized measured belt in unvoiced frames minus mean slope in voiced frames (units/s). Expected positive if "expansion is positive".',
              f'- A: {(pa > 0).sum()}/{len(pa)} segments positive, median {np.median(pa):.3f}',
              f'- B: {(pb > 0).sum()}/{len(pb)} segments positive, median {np.median(pb):.3f}', '',
              '|Singer|Segment|Task|A polarity|B polarity|B r|Measured A-B r|B relative amplitude|Voiced fraction|', '|---|---|---|---|---|---|---|---|---|']
    flagged = [a for a in audit if a['B']['polarity_index'] <= 0 or a['A']['polarity_index'] <= 0 or a['B']['r'] < .5]
    for a in sorted(flagged, key=lambda a: a['B']['r']):
        lines.append(f"|{a['singer']}|{a['number']}|{a['task']}|{a['A']['polarity_index']:.3f}|{a['B']['polarity_index']:.3f}|"
                     f"{a['B']['r']:.3f}|{a['measured_AB_r']:.3f}|{a['rel_amp'][1]:.2f}|{a['voiced_fraction']:.2f}|")
    # Spearman associations
    from scipy.stats import spearmanr
    br = np.array([a['B']['r'] for a in audit])
    assoc = {'B_r_vs_B_rel_amp': spearmanr(br, [a['rel_amp'][1] for a in audit]),
             'B_r_vs_B_polarity': spearmanr(br, pb),
             'B_r_vs_measured_AB_r': spearmanr(br, [a['measured_AB_r'] for a in audit]),
             'B_r_vs_voiced_fraction': spearmanr(br, [a['voiced_fraction'] for a in audit])}
    lines += ['', '## Spearman correlations (197 segments, singer clustering not corrected, descriptive only)', '']
    lines += [f'- {k}: rho={v.statistic:.3f}, p={v.pvalue:.2g}' for k, v in assoc.items()]
    ea = [a['edge_artifact_maxabs_first2s'] for a in audit]; eb = [a['edge_artifact_maxabs_last2s'] for a in audit]
    ei = [a['edge_artifact_maxabs_interior'] for a in audit]
    lines += ['', '## Target preprocessing edge artefacts', '',
              'Comparison of the current pipeline with a target that removes DC before resampling (standardized units, max abs difference):',
              f'- First 2 s median {np.median(ea):.4f} (max {np.max(ea):.3f}); last 2 s median {np.median(eb):.4f} (max {np.max(eb):.3f}); interior median {np.median(ei):.5f}',
              '', '## Voicing at record edges', '',
              f"- Median voiced fraction first 2 s {np.median([a['voiced_first2s'] for a in audit]):.2f}; last 2 s {np.median([a['voiced_last2s'] for a in audit]):.2f}; whole record {np.median([a['voiced_fraction'] for a in audit]):.2f}",
              f"- Median first voiced time {np.median([a['first_voiced_s'] for a in audit]):.1f} s; median last voiced time from end {np.median([a['last_voiced_s_from_end'] for a in audit]):.1f} s",
              f"- Median per-segment MSE: voiced frames {np.nanmedian([a['mse_voiced'] for a in audit]):.3f}, unvoiced frames {np.nanmedian([a['mse_unvoiced'] for a in audit]):.3f}"]
    out = ROOT / 'diagnostics_r9'
    (out / 'target_clarity_diagnostic.md').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'target_clarity_rows.json').write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
