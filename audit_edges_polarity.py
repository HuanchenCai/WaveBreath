from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parent / 'outputs'
def score(y,p):
    r=float(np.corrcoef(y,p)[0,1])
    c=2*np.mean((y-y.mean())*(p-p.mean()))/(y.var()+p.var()+(y.mean()-p.mean())**2+1e-8)
    return [r,float(c),float(np.sqrt(np.mean((y-p)**2)))]
def records(folder):
    rows=[]
    for f in sorted(folder.glob('fold*_waveforms.npz')):
        with np.load(f) as z:
            meta=json.loads(str(z['metadata']))
            for i,m in enumerate(meta):
                a,b=z['offsets'][i:i+2]; rows.append((m,z['target'][a:b],z['prediction'][a:b]))
    return rows
def main():
    lines=['# Edge errors and anti-phase belt diagnostics','','Only the saved complete outer held-out predictions are analysed; no retraining, no label changes. Edge trimming is an exploratory sensitivity analysis; the official full-length metrics are kept. RMSE is computed in the original standardized units, without re-standardizing after trimming.','','## Edge sensitivity for all configurations','','All 197 segments compared with 0, 1, 2 and 4 s removed from each end; table values are segment-level medians.','','|Model|Seconds removed per end|A/B r|A/B CCC|A/B RMSE|B r<0.5 segments/singers|','|---|---|---|---|---|---|']
    summary={}; base=None
    for folder in sorted((ROOT/'family_runs').iterdir()):
        if not folder.is_dir() or not (folder/'results.json').exists(): continue
        info=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        if not info.get('completed'): continue
        rs=records(folder)
        if len(rs)!=197: continue
        summary[folder.name]={}
        for sec in (0,1,2,4):
            k=sec*5; vals=[]; bad=[]
            for m,y,p in rs:
                yy=y[k:len(y)-k] if k else y; pp=p[k:len(p)-k] if k else p
                if len(yy)<10: continue
                v=[score(yy[:,c],pp[:,c]) for c in range(2)]; vals.append(v)
                if v[1][0]<.5: bad.append(m)
            med=np.median(vals,axis=0)
            summary[folder.name][sec]={'median':med.tolist(),'bad_B':bad,'n':len(vals)}
            cells=[' / '.join(f'{v:.4f}' for v in med[:,j]) for j in range(3)]
            lines.append('|'+ '|'.join([folder.name,str(sec),*cells,f'{len(bad)}/{len(set(m["singer"] for m in bad))}'])+'|')
        if folder.name=='wavlm_large16_tcn':base=rs
    lines+=['','## Current main model: is low correlation confined to P24?','','|Singer|Segments|B r<0.5|B r<0|','|---|---|---|---|']
    for s in sorted(set(m['singer'] for m,y,p in base)):
        rr=[score(y[:,1],p[:,1])[0] for m,y,p in base if m['singer']==s]
        if min(rr)<.5:lines.append(f'|{s}|{len(rr)}|{sum(r<.5 for r in rr)}|{sum(r<0 for r in rr)}|')
    lines+=['','## P24: polarity-inversion diagnostic','','The sign flip is diagnostic only: it is a counterfactual computed after seeing the ground truth and cannot count as model performance. Polarity inversion is not the same as a time shift.','','|Segment|Duration s|Measured A/B corr|Original B r/CCC/RMSE|Flipped B r/CCC/RMSE|','|---|---|---|---|---|']
    xin=[]
    for m,y,p in base:
        if m['singer'].lower()!='xinjiayi':continue
        xin.append((m,y,p))
        fmt=lambda v:' / '.join(f'{x:.4f}' for x in v)
        lines.append(f'|{m["number"]}|{len(y)/5:.1f}|{np.corrcoef(y.T)[0,1]:.4f}|{fmt(score(y[:,1],p[:,1]))}|{fmt(score(y[:,1],-p[:,1]))}|')
    # Choose six distinct other singers by their median B correlation, at spread ranks.
    singers=sorted(set(m['singer'] for m,y,p in base)-{'P24'})
    singers.sort(key=lambda s:np.median([score(y[:,1],p[:,1])[0] for m,y,p in base if m['singer']==s]))
    chosen=[]
    for idx in np.linspace(0,len(singers)-1,6).round().astype(int):
        rs=[x for x in base if x[0]['singer']==singers[idx]]
        rs.sort(key=lambda x:score(x[1][:,1],x[2][:,1])[0]);chosen.append(rs[len(rs)//2])
    def plot(rs,name,flip=False):
        fig,axs=plt.subplots(len(rs),2,figsize=(17,3.2*len(rs)),squeeze=False,constrained_layout=True)
        for row,(m,y,p) in enumerate(rs):
            t=np.arange(len(y))/5
            for c in range(2):
                ax=axs[row,c];ax.plot(t,y[:,c],label='Measured',color='#2864aa');ax.plot(t,p[:,c],label='Predicted',color='#14946c')
                if flip and c==1:ax.plot(t,-p[:,c],label='Negative prediction (diagnostic only)',color='#cb7434',alpha=.7,ls='--')
                ax.axvspan(0,2,color='gray',alpha=.15);ax.axvspan(max(0,t[-1]-2),t[-1],color='gray',alpha=.15)
                ax.set_title(f'{m["singer"]} #{m["number"]} | {"AB"[c]} | full r={score(y[:,c],p[:,c])[0]:.3f}; trim 2s r={score(y[10:-10,c],p[10:-10,c])[0]:.3f}')
                ax.set_xlim(0,t[-1]);ax.set_xlabel('Time (s)');ax.set_ylabel('Relative belt units')
        axs[0,0].legend();axs[0,1].legend();fig.savefig(OUT/name,dpi=140);plt.close(fig)
    plot(chosen,'other_six_singers_full_waveforms_edge_diagnostics.png')
    plot(xin,'P24_all_segments_polarity_diagnostics.png',True)
    lines+=['','## Display and windowing check','','The old figure top row always showed the first 30 s of P29 #3; the bottom row shows the full record of P24 #12 and is not truncated to 40 s. The r in titles is always computed on the full record.','Inference uses 128-frame (25.6 s) windows with a 64-frame (12.8 s) hop; overlapping windows are averaged with equal weights (no Hann weighting). A tail-aligned window is appended at the end of each record; only records shorter than 128 frames are zero-padded.','Window edges differ from record edges: weighted stitching can reduce window-edge effects but cannot restore context missing at the start and end of a record. Measured and predicted signals must not both be tapered to shrink edge errors artificially.','The new figure shows full records; grey only marks 2 s at each end and does not alter the curves. Other singers are ranked by singer-median B correlation of the main model, six ranks are chosen evenly, and for each the record with median B correlation is shown (post-hoc descriptive display).','','## Figure files','','- other_six_singers_full_waveforms_edge_diagnostics.png','- P24_all_segments_polarity_diagnostics.png']
    (OUT/'edge_polarity_diagnostics.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'edge_polarity_stats.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary['wavlm_large16_tcn'],ensure_ascii=False))
    print('\n'.join(lines[lines.index('## P24: polarity-inversion diagnostic'):][:10]))
if __name__=='__main__':main()
