"""Round-15 (analysis, no training): do predicted belts reproduce known breathing relations?

Phrases (5 Hz): voiced runs (mic VAD from diag_r9) after closing gaps < 0.4 s, lasting >= 1 s,
with a preceding pause. Per phrase and band:
  depth   = belt at phrase onset minus the minimum in the preceding pause (inhalation size)
  dur     = phrase duration (s)
  slope   = least-squares slope of the belt over the phrase (units/s; expiratory drive)
  spl     = mean microphone level over the phrase (dB re the record's median voiced level)
Relations (Spearman, within singer, pooling that singer's phrases):
  R1 depth ~ dur   (inhalation anticipates phrase length: Whalen & Kinsella-Shaw 1997; Fuchs 2013)
  R2 slope ~ spl   (louder phrases, faster lung-volume decline: Stathopoulos & Sapienza 1993)
  R3 depth ~ spl   (louder phrases, deeper inhalation: Stathopoulos & Sapienza 1993)
Computed on measured belts and on predicted belts (same phrases); agreement = correlation of the
per-singer coefficients and sign agreement. Singers with < 15 phrases are skipped.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

from compare_r11_r12 import load
from diag_r9_raw_polarity_vad import voice_rms_db
from diag_r9_sawtooth_baseline import runs

ROOT = Path(__file__).resolve().parent


def phrases(voiced):
    v = voiced.copy()
    for s, e in runs(~v):
        if e - s <= 2 and s > 0 and e < len(v):
            v[s:e] = True
    out, prev_end = [], None
    for s, e in runs(v):
        if prev_end is not None and e - s >= 5 and s - prev_end >= 1:
            out.append((prev_end, s, e))
        prev_end = e
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'r13_convlstm'
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig'))
            if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        v_all, off = z['voiced'], z['offsets']
    P = load(model)
    spl_cache = ROOT / 'diagnostics_r10' / 'mic_db_5hz.npy'
    if spl_cache.exists():
        db_all = np.load(spl_cache)
    else:
        db_all = np.zeros(int(off[-1]), 'float32')
        for i, r in enumerate(rows):
            a, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
            db_all[off[i]:off[i + 1]] = voice_rms_db(a, sr, int(off[i + 1] - off[i]))
        np.save(spl_cache, db_all)
    by = {}
    for i, a in enumerate(info):
        y, p = P[(a['singer'], str(a['number']))]
        v = v_all[off[i]:off[i + 1]]; db = db_all[off[i]:off[i + 1]]
        ref = np.median(db[v]) if v.any() else 0
        for ps, s, e in phrases(v):
            t = np.arange(e - s) / 5
            rec = {'task': a['task'], 'dur': (e - s) / 5, 'pause': (s - ps) / 5, 'spl': float(db[s:e].mean() - ref)}
            for src, w in (('meas', y), ('pred', p)):
                for k, c in enumerate('AB'):
                    seg = w[s:e, k]
                    rec[f'{src}_{c}_depth'] = float(w[s, k] - w[ps:s + 1, k].min())
                    rec[f'{src}_{c}_slope'] = float(np.polyfit(t, seg, 1)[0]) if len(seg) > 2 else np.nan
            by.setdefault(a['singer'], []).append(rec)
    lines = [f'# Can predicted belts reproduce known breathing regularities? (model `{model}`, exploratory)', '',
             'For each singer all phrases are pooled and Spearman correlations computed; measured and predicted use the same phrases. Only singers with at least 15 phrases are included.', '',
             '|Relation|Belt|Measured: singer-median coef.|Measured: fraction of singers positive|Predicted: singer-median coef.|Predicted: fraction positive|Across-singer corr of meas vs pred coef.|Sign agreement|', '|---|---|---|---|---|---|---|---|']
    rel = {'R1 inhalation depth ~ phrase duration': ('depth', 'dur'), 'R2 expiratory slope ~ SPL': ('slope', 'spl'), 'R3 inhalation depth ~ SPL': ('depth', 'spl')}
    out = {}
    for name, (a_, b_) in rel.items():
        for c in 'AB':
            cm, cp = [], []
            for s, recs in by.items():
                if len(recs) < 15:
                    continue
                def coef(src):
                    xa = [r[f'{src}_{c}_{a_}'] if a_ != 'dur' else r['dur'] for r in recs]
                    xb = [r[b_] if b_ in ('dur', 'spl') else r[f'{src}_{c}_{b_}'] for r in recs]
                    return spearmanr(xa, xb).statistic
                cm.append(coef('meas')); cp.append(coef('pred'))
            cm, cp = np.array(cm), np.array(cp)
            ok = np.isfinite(cm) & np.isfinite(cp); cm, cp = cm[ok], cp[ok]
            out[f'{name}|{c}'] = {'meas': cm.tolist(), 'pred': cp.tolist()}
            lines.append(f'|{name}|{c}|{np.median(cm):+.3f}|{(cm > 0).mean():.2f}|{np.median(cp):+.3f}|{(cp > 0).mean():.2f}|'
                         f'{np.corrcoef(cm, cp)[0, 1]:+.3f}|{(np.sign(cm) == np.sign(cp)).mean():.2f}|')
    # R1 controlling for pause duration: partial Spearman (Pearson on ranks after regressing out pause rank)
    from scipy.stats import rankdata
    def partial(xa, xb, z):
        ra, rb, rz = rankdata(xa), rankdata(xb), rankdata(z)
        res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
        return np.corrcoef(res(ra), res(rb))[0, 1]
    lines += ['', '## R1 partial correlation controlling for pause duration', '', '|Belt|Measured: singer-median partial rho / fraction positive|Predicted: singer-median partial rho / fraction positive|Across-singer corr|', '|---|---|---|---|']
    for c in 'AB':
        cm, cp = [], []
        for s, recs in by.items():
            if len(recs) < 15:
                continue
            z = [r['pause'] for r in recs]; d = [r['dur'] for r in recs]
            cm.append(partial([r[f'meas_{c}_depth'] for r in recs], d, z)); cp.append(partial([r[f'pred_{c}_depth'] for r in recs], d, z))
        cm, cp = np.array(cm), np.array(cp)
        lines.append(f'|{c}|{np.median(cm):+.3f} / {(cm > 0).mean():.2f}|{np.median(cp):+.3f} / {(cp > 0).mean():.2f}|{np.corrcoef(cm, cp)[0, 1]:+.3f}|')
    n_ph = sum(len(v) for v in by.values())
    lines += ['', f'Total phrases {n_ph}; {sum(len(v) >= 15 for v in by.values())} singers included.']
    (ROOT / 'diagnostics_r10' / f'physiology_reproduction_{model}.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'diagnostics_r10' / f'physiology_reproduction_{model}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
