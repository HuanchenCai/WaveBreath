"""Round 26 (see the round-26 best-of-N pre-registration): best-of-N reranking of latent-diffusion samples.
For every 128-frame window of each outer-fold test recording, draw N=16 samples (independent noise per
window), choose one per window with each reward, stitch by overlap averaging and save
family_runs/r26_bon_<rule>/fold<k>_waveforms.npz against the recommended model's target (r11_tgt_dc)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from compare_r19 import load, voiced
from dual_belt_cv import DualWindows
from run_waveform_families import load_records
from waveform_family_models import make_model

ROOT = Path(__file__).resolve().parent
N = 16
RULES = ('mean16', 'seed0', 'oracle', 'rule', 'consensus')


@torch.no_grad()
def sample(model, x, gen):
    c = (model.ae.bridge(model.ae.voice(x)) - model.latent_mean) / model.latent_std
    z = torch.randn((len(x), 4, c.shape[-1]), device=x.device, generator=gen)
    for step in range(100, 0, -10):
        eps = model.field(z, c, torch.full((len(x),), step / 100, device=x.device))
        a, prev = model.alpha[step], model.alpha[max(0, step - 10)]
        clean = ((z - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-5, 5)
        z = prev.sqrt() * clean + (1 - prev).sqrt() * eps
    return model.ae.belt_decoder(z * model.latent_std + model.latent_mean)


def ccc2(y, p):
    """Mean CCC over the two belts; y, p: (n, 2)."""
    out = []
    for k in range(2):
        a, b = y[:, k], p[:, k]
        out.append(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8))
    return float(np.mean(out))


def rule_reward(p, v):
    """Rises of the sum in unvoiced frames, falls in voiced frames."""
    d = np.diff(p[:, 0] + p[:, 1]); u = 1.0 - v[1:len(d) + 1]
    if d.std() < 1e-9 or u.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(d, u)[0, 1])


def main():
    ref = load('r15_ens_convlstm_unet')  # key -> (target r11_tgt_dc, recommended prediction)
    records = load_records(ROOT / 'whisper_large_v3_voice_AB_5hz.npz')
    rng_seed = 20261005
    for fold in range(1, 6):
        ck = torch.load(ROOT / 'family_runs' / 'large_latent4_diffusion' / f'fold{fold}.pt', map_location='cuda', weights_only=False)
        model = make_model(ck['family'], ck['input_dim']).cuda(); model.load_state_dict(ck['model']); model.eval()
        test = [i for i, (m, _, _) in enumerate(records) if m['singer'] in ck['test_singers']]
        ds = DualWindows(records, test, ck['mean'], ck['std'])
        gen = torch.Generator(device='cuda').manual_seed(rng_seed + fold)
        sums = {r: {i: np.zeros((len(records[i][2]), 2)) for i in test} for r in RULES}
        cnt = {i: np.zeros(len(records[i][2])) for i in test}
        o = 0
        for x, _, m in DataLoader(ds, batch_size=32):
            x = x.cuda()
            cand = np.stack([sample(model, x, gen).float().cpu().numpy() for _ in range(N)], 1)  # (b, N, 128, 2)
            for j in range(len(x)):
                i, st = ds.locations[o + j]; n = int(m[j].sum())
                key = (records[i][0]['singer'], str(records[i][0]['number']))
                y, rec = ref[key][0][st:st + n], ref[key][1][st:st + n]; v = voiced(key)[st:st + n].astype(float)
                c = cand[j, :, :n]
                pick = {'mean16': c.mean(0), 'seed0': c[0],
                        'oracle': c[int(np.argmax([ccc2(y, q) for q in c]))],
                        'rule': c[int(np.argmax([rule_reward(q, v) for q in c]))],
                        'consensus': c[int(np.argmax([ccc2(rec, q) for q in c]))]}
                for r in RULES:
                    sums[r][i][st:st + n] += pick[r]
                cnt[i][st:st + n] += 1
            o += len(x)
        for r in RULES:
            meta, preds, tgts = [], [], []
            for i in test:
                key = (records[i][0]['singer'], str(records[i][0]['number']))
                y = ref[key][0]; p = sums[r][i] / cnt[i][:, None]
                assert len(p) == len(y), (key, len(p), len(y))
                meta.append(records[i][0]); preds.append(p); tgts.append(y)
            off = np.cumsum([0] + [len(p) for p in preds])
            d = ROOT / 'family_runs' / f'r26_bon_{r}'; d.mkdir(exist_ok=True)
            np.savez_compressed(d / f'fold{fold}_waveforms.npz', prediction=np.concatenate(preds).astype('float32'),
                                target=np.concatenate(tgts).astype('float32'), metadata=json.dumps(meta, ensure_ascii=False), offsets=off)
        print('fold', fold, 'done', len(ds), 'windows', flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
