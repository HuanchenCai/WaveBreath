"""Breath-parameter extractor (round 10), applied identically to measured and predicted
5-Hz belt waveforms. Rules follow the round-10 parameter-evaluation pre-registration."""
import numpy as np
from scipy.signal import find_peaks

FS = 5.0
PROM = .5        # x waveform std
MIN_DIST = 5     # frames (1 s)
MAX_RISE = 15    # frames (3 s)


def inhalations(x, prom=PROM):
    """Return array of (trough_idx, peak_idx, depth) for each inhalation event."""
    x = np.asarray(x, dtype='float64')
    s = x.std() + 1e-9
    peaks, _ = find_peaks(x, prominence=prom * s, distance=MIN_DIST)
    troughs, _ = find_peaks(-x, prominence=prom * s, distance=MIN_DIST)
    events = []
    for p in peaks:
        before = troughs[troughs < p]
        if not len(before):
            continue
        t = before[-1]
        # the trough must be the last extremum before the peak (no intervening peak)
        if len(events) and events[-1][1] >= t:
            continue
        rise = x[p] - x[t]
        if p - t <= MAX_RISE and rise >= prom * s:
            events.append((int(t), int(p), float(rise / s)))
    return np.array(events, dtype='float64').reshape(-1, 3)


def match(a, b, tol):
    """Greedy one-to-one matching of event times (frames); returns list of (i, j)."""
    pairs = sorted(((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x - y) <= tol))
    used_a, used_b, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); out.append((i, j))
    return out


def f1(ref, est, tol):
    m = match(ref, est, tol)
    if not len(ref) and not len(est):
        return 1., m
    p = len(m) / max(len(est), 1); r = len(m) / max(len(ref), 1)
    return (2 * p * r / (p + r) if p + r else 0.), m


def ab_lag(ea, eb, tol=5):
    """Median B-A onset lag (frames) over matched inhalations, or nan if < 3 pairs."""
    m = match(ea[:, 0], eb[:, 0], tol)
    if len(m) < 3:
        return float('nan'), len(m)
    return float(np.median([eb[j, 0] - ea[i, 0] for i, j in m])), len(m)


def record_parameters(y, p):
    """Compare measured y (n,2) and estimate p (n,2). Returns dict of per-record metrics."""
    out = {}
    ev_y = [inhalations(y[:, k]) for k in range(2)]
    ev_p = [inhalations(p[:, k]) for k in range(2)]
    for k, c in enumerate('AB'):
        ry, rp = ev_y[k], ev_p[k]
        f04, m04 = f1(ry[:, 0], rp[:, 0], 2)
        f10, _ = f1(ry[:, 0], rp[:, 0], 5)
        err = [abs(ry[i, 0] - rp[j, 0]) / FS for i, j in m04]
        iy = np.diff(ry[:, 0]) / FS; ip = np.diff(rp[:, 0]) / FS
        depth = float('nan')
        if len(m04) >= 4:
            from scipy.stats import spearmanr
            depth = float(spearmanr([ry[i, 2] for i, _ in m04], [rp[j, 2] for _, j in m04]).statistic)
        out[c] = {'n_ref': len(ry), 'n_est': len(rp), 'f1_0.4s': f04, 'f1_1.0s': f10,
                  'onset_abs_err_s': float(np.median(err)) if err else float('nan'),
                  'interval_ref_s': float(np.median(iy)) if len(iy) else float('nan'),
                  'interval_est_s': float(np.median(ip)) if len(ip) else float('nan'),
                  'count_rel_err': (len(rp) - len(ry)) / max(len(ry), 1),
                  'depth_spearman': depth}
        out[c]['interval_abs_err_s'] = abs(out[c]['interval_ref_s'] - out[c]['interval_est_s'])
    ly, ny = ab_lag(*ev_y); lp, npair = ab_lag(*ev_p)
    out['AB_lag_ref_s'] = ly / FS; out['AB_lag_est_s'] = lp / FS
    out['AB_lag_abs_err_s'] = abs(ly - lp) / FS
    out['AB_pairs_ref'] = ny; out['AB_pairs_est'] = npair
    return out


def onsets_to_events(onsets, n):
    """Wrap bare onset frames (e.g. from a pause rule) into the (t, p, depth) layout."""
    onsets = np.asarray(sorted(onsets), dtype='float64')
    return np.c_[onsets, np.minimum(onsets + 2, n - 1), np.ones(len(onsets))]
