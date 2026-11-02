"""Round 22 (analysis): which breath parameters does the model predict BEYOND what simple
voice-derived rules give? Per phrase (voiced run >=1 s after a pause; same phrases as round 15):
  measured  : inhalation depth before the phrase, expiratory slope during it, end level
              (belt value at phrase end relative to the record median), all in measured-std units
  model     : the same quantities from the predicted belt (recommended ensemble)
  rules     : pause duration, upcoming phrase duration, phrase mean SPL (all from the voice alone);
              rule prediction = within-singer linear regression of the measured quantity on these
              three cues, fitted leave-one-recording-out (no measured belt of the test recording used)
Score: within-singer Spearman with the measured quantity (median over 32 singers), and the
partial Spearman of the model with the measured quantity controlling for the rule prediction
(= information beyond the obvious cues)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from compare_r19 import load
from diag_r9_sawtooth_baseline import runs

ROOT = Path(__file__).resolve().parent


def phrases(v):
    v = v.copy()
    for s, e in runs(~v):
        if e - s <= 2 and s > 0 and e < len(v):
            v[s:e] = True
    out, prev = [], None
    for s, e in runs(v):
        if prev is not None and e - s >= 5 and s - prev >= 1:
            out.append((prev, s, e))
        prev = e
    return out


def pspear(a, b, z):
    ra, rb, rz = rankdata(a), rankdata(b), rankdata(z)
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def main(model='r15_ens_convlstm_unet'):
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        V, off = z['voiced'], z['offsets']
    db = np.load(ROOT / 'diagnostics_r10' / 'mic_db_5hz.npy')
    data = load(model)
    rows = []
    for i, a in enumerate(info):
        key = (a['singer'], str(a['number'])); y, p = data[key]; v = V[off[i]:off[i + 1]]; d = db[off[i]:off[i + 1]]
        ref = np.median(d[v]) if v.any() else 0
        for ps, s, e in phrases(v):
            r = {'singer': a['singer'], 'rec': key[1], 'pause': (s - ps) / 5, 'dur': (e - s) / 5, 'spl': float(d[s:e].mean() - ref)}
            for src, w in (('m', y), ('p', p)):
                for k, c in enumerate('AB'):
                    sd = y[:, k].std() + 1e-9; x = w[:, k] / sd; med = np.median(y[:, k]) / sd
                    r[f'{src}{c}_depth'] = float(x[s] - x[ps:s + 1].min())
                    r[f'{src}{c}_slope'] = float(np.polyfit(np.arange(e - s) / 5, x[s:e], 1)[0])
                    r[f'{src}{c}_end'] = float(x[e - 1] - med)
            rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r['singer'], []).append(r)
    L = [f'# Round 22 diagnostic: what does the model predict beyond "obvious voice cues"? (model `{model}`)', '',
         'Rule prediction = linear regression "measured quantity ~ pause duration + next-phrase duration + mean SPL" fitted on the singer\'s other recordings and applied to this one (leave-one-recording-out, voice cues only).',
         'Values are medians over 32 singers of phrase-level Spearman correlations. "Model\'s extra information" = partial Spearman of model prediction with measurement, controlling for the rule prediction.', '',
         '|Parameter|Belt|Rule vs measured|Model vs measured|Model\'s extra information (partial Spearman)|', '|---|---|---|---|---|']
    for q, lab in (('depth', 'Inhalation depth (air taken before phrase)'), ('slope', 'Within-phrase expiratory slope (air use)'), ('end', 'Relative level at phrase end (air left)')):
        for c in 'AB':
            sr, sm, sp = [], [], []
            for s, rs in by.items():
                recs = sorted({r['rec'] for r in rs})
                if len(rs) < 15 or len(recs) < 2:
                    continue
                rule = np.zeros(len(rs))
                for rec in recs:
                    tr = [r for r in rs if r['rec'] != rec]
                    X = np.array([[r['pause'], r['dur'], r['spl'], 1] for r in tr]); yv = np.array([r[f'm{c}_{q}'] for r in tr])
                    w = np.linalg.lstsq(X, yv, rcond=None)[0]
                    for j, r in enumerate(rs):
                        if r['rec'] == rec:
                            rule[j] = np.array([r['pause'], r['dur'], r['spl'], 1]) @ w
                meas = np.array([r[f'm{c}_{q}'] for r in rs]); mod = np.array([r[f'p{c}_{q}'] for r in rs])
                sr.append(spearmanr(rule, meas).statistic); sm.append(spearmanr(mod, meas).statistic); sp.append(pspear(mod, meas, rule))
            L.append(f'|{lab}|{c}|{np.median(sr):+.2f}|{np.median(sm):+.2f}|{np.median(sp):+.2f} (positive {np.mean(np.array(sp) > 0):.0%})|')
    L += ['', f'Phrases: {len(rows)}.']
    (ROOT / 'round22_beyond_voice_cues.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
