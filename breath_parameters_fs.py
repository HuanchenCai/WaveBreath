"""Same rules as the frozen breath_parameters.py, expressed in seconds so they apply at any
sampling rate: prominence 0.5 x std, peaks >= 1 s apart, rise <= 3 s, rise >= 0.5 x std;
onset tolerance 0.4 s (primary) and 1.0 s; A-B matching within 1 s.
Adds the inspiration duration (trough -> peak) error for matched events."""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from breath_parameters import match


def inhalations_fs(x, fs, prom=.5):
    x = np.asarray(x, dtype='float64'); s = x.std() + 1e-9
    dist = int(round(fs)); max_rise = int(round(3 * fs))
    peaks, _ = find_peaks(x, prominence=prom * s, distance=dist)
    troughs, _ = find_peaks(-x, prominence=prom * s, distance=dist)
    ev = []
    for p in peaks:
        before = troughs[troughs < p]
        if not len(before):
            continue
        t = before[-1]
        if ev and ev[-1][1] >= t:
            continue
        if p - t <= max_rise and x[p] - x[t] >= prom * s:
            ev.append((int(t), int(p), float((x[p] - x[t]) / s)))
    return np.array(ev, dtype='float64').reshape(-1, 3)


def _f1(ref, est, tol):
    m = match(ref, est, tol)
    if not len(ref) and not len(est):
        return 1., m
    p = len(m) / max(len(est), 1); r = len(m) / max(len(ref), 1)
    return (2 * p * r / (p + r) if p + r else 0.), m


def record_parameters_fs(y, p, fs):
    out = {}
    ey = [inhalations_fs(y[:, k], fs) for k in range(2)]; ep = [inhalations_fs(p[:, k], fs) for k in range(2)]
    for k, c in enumerate('AB'):
        ry, rp = ey[k], ep[k]
        f04, m04 = _f1(ry[:, 0], rp[:, 0], .4 * fs); f10, _ = _f1(ry[:, 0], rp[:, 0], 1. * fs)
        iy = np.diff(ry[:, 0]) / fs; ip = np.diff(rp[:, 0]) / fs
        out[c] = {'n_ref': len(ry), 'n_est': len(rp), 'f1_0.4s': f04, 'f1_1.0s': f10,
                  'onset_abs_err_s': float(np.median([abs(ry[i, 0] - rp[j, 0]) / fs for i, j in m04])) if m04 else float('nan'),
                  'interval_abs_err_s': abs((np.median(iy) if len(iy) else np.nan) - (np.median(ip) if len(ip) else np.nan)),
                  'insp_dur_ref_s': float(np.median((ry[:, 1] - ry[:, 0]) / fs)) if len(ry) else float('nan'),
                  'insp_dur_abs_err_s': float(np.median([abs((ry[i, 1] - ry[i, 0]) - (rp[j, 1] - rp[j, 0])) / fs for i, j in m04])) if m04 else float('nan'),
                  'depth_spearman': float(spearmanr([ry[i, 2] for i, _ in m04], [rp[j, 2] for _, j in m04]).statistic) if len(m04) >= 4 else float('nan')}

    def lag(ea, eb):
        m = match(ea[:, 0], eb[:, 0], fs)
        return (float(np.median([eb[j, 0] - ea[i, 0] for i, j in m])) / fs, len(m)) if len(m) >= 3 else (float('nan'), len(m))
    (ly, ny), (lp, np_) = lag(*ey), lag(*ep)
    out.update({'AB_lag_ref_s': ly, 'AB_lag_est_s': lp, 'AB_lag_abs_err_s': abs(ly - lp), 'AB_pairs_ref': ny, 'AB_pairs_est': np_})
    return out
