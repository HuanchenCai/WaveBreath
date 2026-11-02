"""Round-16 breath parameter set (evaluation only). Every parameter is computed with the SAME
rules on a measured and a predicted 5-Hz belt pair. Amplitude-type parameters are expressed in
units of the signal's own standard deviation (scale-free), because belts are uncalibrated.

Per signal (per band A/B unless stated):
  rate_per_min       inhalations per minute (frozen extractor `inhalations`)
  interval_s         median interval between inhalation onsets (breath-group duration)
  insp_dur_s         median inhalation duration, trough -> peak
  depth_sd           median inhalation depth (peak - trough) / std
  exp_slope_sd_s     median least-squares slope over voiced phrases (expiratory drive), std/s
  [dual] ab_corr     Pearson correlation between A and B (chest-abdomen coupling)
  [dual] ab_lag_s    median B - A onset lag of matched inhalations (within 1 s, >= 3 pairs)
  [dual] paradox_frac fraction of frames where A and B move in opposite directions
                     (both |slope| > 0.1 std per frame)
Pairwise (measured vs predicted, per band):
  f1_0.4s, f1_1.0s   inhalation-onset detection F1
  onset_err_s        median |onset error| of matched onsets (0.4 s)
  depth_rank_rho     Spearman of matched inhalation depths within the record (>= 4 pairs)
Phrases come from the microphone voicing mask (diag_r9) and are identical for both signals.
"""
import numpy as np
from scipy.stats import spearmanr

from breath_parameters import FS, f1, inhalations, match
from diag_r9_sawtooth_baseline import runs

BANDS = 'AB'


def phrases(voiced):
    v = voiced.copy()
    for s, e in runs(~v):
        if e - s <= 2 and s > 0 and e < len(v):
            v[s:e] = True
    return [(s, e) for s, e in runs(v) if e - s >= 5]


def signal_params(w, voiced):
    """w: (n, 2) belt signal at 5 Hz. Returns dict of per-signal parameters."""
    out, ev = {}, []
    dur_min = len(w) / FS / 60
    ph = phrases(voiced)
    for k, c in enumerate(BANDS):
        x = w[:, k]; sd = x.std() + 1e-9; e = inhalations(x); ev.append(e)
        slopes = [np.polyfit(np.arange(b - a) / FS, x[a:b] / sd, 1)[0] for a, b in ph]
        out[c] = {'rate_per_min': len(e) / dur_min,
                  'interval_s': float(np.median(np.diff(e[:, 0]) / FS)) if len(e) > 1 else np.nan,
                  'insp_dur_s': float(np.median((e[:, 1] - e[:, 0]) / FS)) if len(e) else np.nan,
                  'depth_sd': float(np.median(e[:, 2])) if len(e) else np.nan,
                  'exp_slope_sd_s': float(np.median(slopes)) if slopes else np.nan}
    m = match(ev[0][:, 0], ev[1][:, 0], 5)
    da, db = np.diff(w[:, 0]) / (w[:, 0].std() + 1e-9), np.diff(w[:, 1]) / (w[:, 1].std() + 1e-9)
    moving = (np.abs(da) > .1) & (np.abs(db) > .1)
    out['dual'] = {'ab_corr': float(np.corrcoef(w[:, 0], w[:, 1])[0, 1]),
                   'ab_lag_s': float(np.median([ev[1][j, 0] - ev[0][i, 0] for i, j in m]) / FS) if len(m) >= 3 else np.nan,
                   'paradox_frac': float((np.sign(da[moving]) != np.sign(db[moving])).mean()) if moving.any() else np.nan}
    return out, ev


def pair_params(y, p, voiced):
    sy, ey = signal_params(y, voiced)
    sp, ep = signal_params(p, voiced)
    pair = {}
    for k, c in enumerate(BANDS):
        a, b = ey[k], ep[k]
        f04, m04 = f1(a[:, 0], b[:, 0], 2); f10, _ = f1(a[:, 0], b[:, 0], 5)
        pair[c] = {'f1_0.4s': f04, 'f1_1.0s': f10,
                   'onset_err_s': float(np.median([abs(a[i, 0] - b[j, 0]) / FS for i, j in m04])) if m04 else np.nan,
                   'depth_rank_rho': float(spearmanr([a[i, 2] for i, _ in m04], [b[j, 2] for _, j in m04]).statistic) if len(m04) >= 4 else np.nan}
    return sy, sp, pair
