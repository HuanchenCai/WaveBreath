"""Round 20 E2: single-sample evaluation of the saved latent Flow / Diffusion models (no training).
The original evaluation averaged 4 fixed-seed samples, which removes the stochastic part again.
Here each seed's sample is stitched and scored on its own and compared with the 4-sample mean.
Metrics per recording: waveform CCC A/B, predicted vs measured chest-abdomen correlation and
paradox fraction (breath_param_set). Outputs round20_generative_single_sample.md."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from breath_param_set import signal_params
from dual_belt_cv import DualWindows
from run_waveform_families import load_records
from waveform_family_models import make_model

ROOT = Path(__file__).resolve().parent
SEEDS = (31, 137, 503, 1009)


@torch.no_grad()
def sample(model, x, seed):
    model.ae.eval()
    c = (model.ae.bridge(model.ae.voice(x)) - model.latent_mean) / model.latent_std
    g = torch.Generator(device=x.device).manual_seed(seed)
    z = torch.randn((1, 4, c.shape[-1]), device=x.device, generator=g).expand(len(x), -1, -1).clone()
    if model.kind == 'flow':
        for step in range(10):
            z = z + model.field(z, c, torch.full((len(x),), step / 10, device=x.device)) / 10
    else:
        for step in range(100, 0, -10):
            eps = model.field(z, c, torch.full((len(x),), step / 100, device=x.device))
            a, prev = model.alpha[step], model.alpha[max(0, step - 10)]
            clean = ((z - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
            z = prev.sqrt() * clean + (1 - prev).sqrt() * eps
    return model.ae.belt_decoder(z * model.latent_std + model.latent_mean)


def ccc(a, b):
    return float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))


def main():
    info = json.loads((ROOT / 'diagnostics_r9' / 'target_clarity_rows.json').read_text(encoding='utf-8'))
    with np.load(ROOT / 'diagnostics_r9' / 'voiced_masks_5hz.npz') as z:
        voiced, off = z['voiced'], z['offsets']
    vmask = {(a['singer'], str(a['number'])): voiced[off[i]:off[i + 1]] for i, a in enumerate(info)}
    records = load_records(ROOT / 'whisper_large_v3_voice_AB_5hz.npz')
    L = ['# Round 20 E2: generative models, single sample vs mean of 4 samples (no training)', '',
         'Single sample = each random seed generated and stitched on its own; mean of 4 = original evaluation. Singer medians; A-B correlation CCC is agreement across the 197 segments.', '',
         '|Model|Mode|Waveform CCC chest/abdomen|Predicted A-B corr median (measured)|A-B corr CCC|Paradox fraction median (measured)|Paradox CCC|', '|---|---|---|---|---|---|---|']
    for name in ('large_latent4_flow', 'large_latent4_diffusion'):
        res = {'mean': {}, **{s: {} for s in SEEDS}}
        for fold in range(1, 6):
            ck = torch.load(ROOT / 'family_runs' / name / f'fold{fold}.pt', map_location='cuda', weights_only=False)
            model = make_model(ck['family'], ck['input_dim']).cuda(); model.load_state_dict(ck['model']); model.eval()
            test = [i for i, (m, _, _) in enumerate(records) if m['singer'] in ck['test_singers']]
            ds = DualWindows(records, test, ck['mean'], ck['std'])
            sums = {s: {i: np.zeros_like(records[i][2], dtype='float64') for i in test} for s in SEEDS}
            cnt = {i: np.zeros(len(records[i][2])) for i in test}
            o = 0
            for x, _, m in DataLoader(ds, batch_size=64):
                outs = {s: sample(model, x.cuda(), s).float().cpu().numpy() for s in SEEDS}
                for j in range(len(x)):
                    i, st = ds.locations[o + j]; n = int(m[j].sum())
                    for s in SEEDS:
                        sums[s][i][st:st + n] += outs[s][j, :n]
                    cnt[i][st:st + n] += 1
                o += len(x)
            for i in test:
                key = (records[i][0]['singer'], str(records[i][0]['number']))
                preds = {s: sums[s][i] / cnt[i][:, None] for s in SEEDS}
                preds['mean'] = np.mean([preds[s] for s in SEEDS], axis=0)
                for k, p in preds.items():
                    res[k][key] = p
            del model; torch.cuda.empty_cache()
        meas = {k: records[[j for j, (m, _, _) in enumerate(records) if (m['singer'], str(m['number'])) == k][0]][2] for k in res['mean']}
        for mode, keys in (('Mean of 4 samples', ['mean']), ('Single sample (each of 4 seeds scored, then averaged)', list(SEEDS))):
            rows = []
            for k, y in meas.items():
                sy, _ = signal_params(y, vmask[k])
                vals = []
                for s in keys:
                    p = res[s][k]; sp, _ = signal_params(p, vmask[k])
                    vals.append([ccc(y[:, 0], p[:, 0]), ccc(y[:, 1], p[:, 1]), sp['dual']['ab_corr'], sp['dual']['paradox_frac']])
                v = np.nanmean(vals, axis=0)
                rows.append((k[0], v, sy['dual']['ab_corr'], sy['dual']['paradox_frac']))
            def smed(idx):
                by = {}
                for s, v, *_ in rows:
                    by.setdefault(s, []).append(v[idx])
                return np.median([np.median(x) for x in by.values()])
            am = np.array([r[2] for r in rows]); ap = np.array([r[1][2] for r in rows])
            pm = np.array([r[3] for r in rows]); pp = np.array([r[1][3] for r in rows])
            ok = np.isfinite(pm) & np.isfinite(pp)
            L.append(f'|{name}|{mode}|{smed(0):.3f}/{smed(1):.3f}|{np.median(ap):.2f} ({np.median(am):.2f})|{ccc(am, ap):.2f}'
                     f'{np.nanmedian(pp):.2f} ({np.nanmedian(pm):.2f})|{ccc(pm[ok], pp[ok]):.2f}|')
        print(name, 'done', flush=True)
    (ROOT / 'round20_generative_single_sample.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
