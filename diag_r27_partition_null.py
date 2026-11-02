"""Arithmetic null for the sum-vs-partition claim: keep each belt's own prediction errors but break their
link to the other belt and to the signal (circular shift of the abdomen error). If the observed partition r is
close to the null, low partition recovery is just the smaller variance of B-A, not missing information."""
import numpy as np
from compare_r19 import load
d = load('r15_ens_convlstm_unet')
rng = np.random.default_rng(0)
by = {}
for k,(y,p) in d.items():
    e = p - y
    S, D = y[:,0]+y[:,1], y[:,1]-y[:,0]
    obsS = np.corrcoef(S, p[:,0]+p[:,1])[0,1]; obsD = np.corrcoef(D, p[:,1]-p[:,0])[0,1]
    # null: same per-belt errors, but cross-belt error dependence broken by circular shift of B error
    sh = rng.integers(len(y)//4, 3*len(y)//4)
    eB = np.roll(e[:,1], sh)
    pn = np.c_[y[:,0]+e[:,0], y[:,1]+eB]
    nS = np.corrcoef(S, pn[:,0]+pn[:,1])[0,1]; nD = np.corrcoef(D, pn[:,1]-pn[:,0])[0,1]
    ce = np.corrcoef(e[:,0], e[:,1])[0,1]
    by.setdefault(k[0], []).append([obsS, nS, obsD, nD, ce])
m = np.median([np.median(v,axis=0) for v in by.values()],axis=0)
print('S obs %.3f null %.3f | D obs %.3f null %.3f | err corr A-B %.2f' % tuple(m))
