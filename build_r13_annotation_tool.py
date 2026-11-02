"""Build an offline, blind annotation page for validating the breath-event extractor.
Sample (fixed before viewing any extractor output on these records): seed 20261001,
8 songs + 6 readings + 6 scales, at most one record per singer. The page shows only the
measured A/B belts (and an optional voice envelope, off by default); no predictions and
no extractor events. Singer names are replaced by codes; the key is saved separately."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'annotation_r13'


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'r11_tgt_dc_y.npz') as z:
        y_all, off = z['y'], z['offsets']
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        v_all = z['voiced']
    rng = np.random.default_rng(20261001)
    order = rng.permutation(len(info))
    quota = {'song': 8, 'reading': 6, 'scale': 6}
    used, picked = set(), []
    for i in order:
        a = info[i]
        if quota[a['task']] and a['singer'] not in used:
            quota[a['task']] -= 1; used.add(a['singer']); picked.append(int(i))
    assert len(picked) == 20
    items, key = [], []
    for n, i in enumerate(picked, 1):
        y = y_all[off[i]:off[i + 1]]; v = v_all[off[i]:off[i + 1]].astype(int)
        code = f'R{n:02d}'
        items.append({'id': code, 'task': info[i]['task'], 'A': np.round(y[:, 0], 3).tolist(),
                      'B': np.round(y[:, 1], 3).tolist(), 'voiced': v.tolist()})
        key.append({'id': code, 'index': i, 'singer': info[i]['singer'], 'number': str(info[i]['number']), 'task': info[i]['task']})
    OUT.mkdir(exist_ok=True)
    (OUT / 'annotation_key_DO_NOT_SHOW_ANNOTATOR.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')
    html = TEMPLATE.replace('__DATA__', json.dumps(items))
    (OUT / 'inhalation_blind_annotation_tool.html').write_text(html, encoding='utf-8')
    print('records', [k['id'] + ':' + k['task'] for k in key])


TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind inhalation annotation</title>
<style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--grid:#e6e6ea;--a:#2864aa;--b:#b5561c;--v:#9aa3ad;--mk:#d11a4a;--panel:#f5f5f7}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--mut:#9a9aa2;--grid:#2b2b30;--a:#79a8e6;--b:#e59866;--v:#5e6670;--mk:#ff5c86;--panel:#1e1e22}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:var(--panel);padding:8px 12px;border-radius:8px}
button,select{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--grid);background:var(--bg);color:var(--fg);cursor:pointer}
canvas{width:100%;height:220px;display:block;margin-top:8px;border:1px solid var(--grid);border-radius:6px;cursor:crosshair}
.lab{color:var(--mut);font-size:12px;margin-top:10px}
</style></head><body><main>
<h1 style="font-size:20px">Blind inhalation annotation (measured belts only)</h1>
<p style="color:var(--mut)">On each belt, click at <b>the start of every inhalation (the trough before the rise)</b> to add a mark; click an existing mark (within ±0.4 s) to delete it. 30 s per screen. Progress is saved automatically in this browser; when finished click "Export" and return the JSON file. Do not look at any model output.</p>
<div class="bar">
<select id="rec"></select><button id="prev">◀ Previous</button><span id="pos"></span><button id="next">Next ▶</button>
<label><input type="checkbox" id="vo"> Show voiced segments</label><button id="exp">Export JSON</button><span id="cnt" style="color:var(--mut)"></span>
</div>
<div class="lab">Belt A</div><canvas id="cA"></canvas>
<div class="lab">Belt B</div><canvas id="cB"></canvas>
</main><script>
const DATA=__DATA__, FS=5, WIN=30;
let st={};try{st=JSON.parse(localStorage.getItem('r13_marks')||'{}')}catch(e){}
DATA.forEach(d=>{st[d.id]=st[d.id]||{A:[],B:[]}});
let cur=0,page=0;
const sel=document.getElementById('rec');
DATA.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.id+' ('+(d.A.length/FS).toFixed(0)+' s)';sel.appendChild(o)});
function save(){try{localStorage.setItem('r13_marks',JSON.stringify(st))}catch(e){}}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function draw(){
 const d=DATA[cur],n=d.A.length,pages=Math.ceil(n/FS/WIN),t0=page*WIN,t1=Math.min(t0+WIN,n/FS);
 document.getElementById('pos').textContent=`${t0.toFixed(0)}–${t1.toFixed(0)} s (${page+1}/${pages})`;
 let c=0;DATA.forEach(x=>c+=st[x.id].A.length+st[x.id].B.length);document.getElementById('cnt').textContent='Total marks '+c;
 for(const band of ['A','B']){
  const cv=document.getElementById('c'+band),r=cv.getBoundingClientRect(),dpr=devicePixelRatio||1;
  cv.width=r.width*dpr;cv.height=r.height*dpr;const g=cv.getContext('2d');g.scale(dpr,dpr);
  const W=r.width,H=r.height,s=d[band],i0=Math.floor(t0*FS),i1=Math.ceil(t1*FS);
  let lo=Infinity,hi=-Infinity;for(let i=0;i<n;i++){lo=Math.min(lo,s[i]);hi=Math.max(hi,s[i])}
  const X=t=>(t-t0)/WIN*W,Y=v=>H-8-(v-lo)/(hi-lo+1e-9)*(H-16);
  g.fillStyle=css('--bg');g.fillRect(0,0,W,H);
  if(document.getElementById('vo').checked){g.fillStyle=css('--v');g.globalAlpha=.25;
   for(let i=i0;i<i1;i++)if(d.voiced[i])g.fillRect(X(i/FS),0,W/WIN/FS+1,H);g.globalAlpha=1}
  g.strokeStyle=css('--grid');g.fillStyle=css('--mut');g.font='11px system-ui';
  for(let t=Math.ceil(t0);t<=t1;t+=5){g.beginPath();g.moveTo(X(t),0);g.lineTo(X(t),H);g.stroke();g.fillText(t+'s',X(t)+2,12)}
  g.strokeStyle=css(band==='A'?'--a':'--b');g.lineWidth=1.6;g.beginPath();
  for(let i=i0;i<i1;i++){const x=X(i/FS),yv=Y(s[i]);i===i0?g.moveTo(x,yv):g.lineTo(x,yv)}g.stroke();
  g.strokeStyle=css('--mk');g.lineWidth=2;
  for(const t of st[d.id][band])if(t>=t0&&t<=t1){g.beginPath();g.moveTo(X(t),0);g.lineTo(X(t),H);g.stroke()}
 }
}
for(const band of ['A','B'])document.getElementById('c'+band).addEventListener('click',e=>{
 const r=e.target.getBoundingClientRect(),t=page*WIN+(e.clientX-r.left)/r.width*WIN,m=st[DATA[cur].id][band];
 const j=m.findIndex(x=>Math.abs(x-t)<=.4);if(j>=0)m.splice(j,1);else{m.push(Math.round(t*FS)/FS);m.sort((a,b)=>a-b)}
 save();draw()});
sel.onchange=()=>{cur=+sel.value;page=0;draw()};
document.getElementById('prev').onclick=()=>{if(page>0){page--;draw()}};
document.getElementById('next').onclick=()=>{if((page+1)*WIN<DATA[cur].A.length/FS){page++;draw()}};
document.getElementById('vo').onchange=draw;
document.getElementById('exp').onclick=()=>{const b=new Blob([JSON.stringify({version:'r13',fs:FS,marks_seconds:st},null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='r13_inhalation_annotations.json';a.click()};
addEventListener('resize',draw);addEventListener('keydown',e=>{if(e.key==='ArrowRight')document.getElementById('next').click();if(e.key==='ArrowLeft')document.getElementById('prev').click()});
draw();
</script></body></html>"""

if __name__ == '__main__':
    main()
