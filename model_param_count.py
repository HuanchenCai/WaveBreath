"""Trainable-parameter count of each attempt (frozen encoders excluded; LoRA adds its adapters).
Per-fold configs (round 13+) differ, so the median over the 5 folds is used. Ensembles count
the sum of their member architectures (one seed each)."""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DIM = {'whisper_large_v3_voice_AB_5hz.npz': 1280, 'wavlm_layer4_voice_AB_5hz.npz': 768, 'wavlm_layer7_voice_AB_5hz.npz': 768,
       'wavlm_layer12_voice_AB_5hz.npz': 768}


def n_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def head(fam, inp, cfg):
    from r13_heads import build
    return n_params(build(fam, inp, {**cfg['arch'], 'dropout': cfg['dropout']}))


def folds_cfg(d):
    return [f.get('winner_config') or f.get('config') for f in d.get('folds', []) if (f.get('winner_config') or f.get('config'))]


def count(name):
    rp = ROOT / 'family_runs' / name / 'results.json'
    d = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {}
    if name.startswith('r12_'):
        from run_r12_multitask import MultiTaskTCN
        return n_params(MultiTaskTCN(1024))
    if name == 'reference_large_shape':
        from whisper_feature_tcn_cv import FeatureTCN
        return n_params(FeatureTCN(1280))
    if 'family' in d and 'cache' in d:  # rounds 7-12 family runner
        from waveform_family_models import make_model
        inp = DIM.get(d['cache'].split('__')[0], 1024)
        return n_params(make_model(d['family'], inp))
    if name == 'r15_ens_convlstm_unet':
        return count('r13_convlstm') + count('r13_unet')
    if name == 'r20_convlstm_separate':
        return 2 * count('r13_convlstm')
    if name.startswith(('r13c_', 'r18_')):
        from run_r13_50hz import TwoScale
        fam = 'unet' if 'unet' in name else 'convlstm'
        return int(np.median([n_params(TwoScale(fam, c)) for c in folds_cfg(d)]))
    fam = next(f for f in ('convlstm', 'unet', 'bigru', 'tcn', 'transformer') if f in name)
    inp = 1024
    if 'egg_only' in name:
        inp = 5
    elif 'egg' in name:
        inp = 1029
    if 'tasklabel' in name:
        inp += 3
    extra = 25 if '_all_' in name else 0
    if name.startswith('r19c_') and 'lora' in name:
        extra = 262144
    cfgs = folds_cfg(d)
    if not cfgs:  # fall back to the family's batch-1 winners
        cfgs = folds_cfg(json.loads((ROOT / 'family_runs' / f'r13_{fam}' / 'results.json').read_text(encoding='utf-8')))
    return int(np.median([head(fam, inp, c) for c in cfgs])) + extra


if __name__ == '__main__':
    pts = json.loads((ROOT / 'round_figures' / 'asymptote_attempts.json').read_text(encoding='utf-8'))
    torch.set_grad_enabled(True)
    for p in pts:
        p['params'] = count(p['name'])
        print(f'{p["name"]:45s} {p["params"]:>10,d}')
    (ROOT / 'round_figures' / 'asymptote_attempts.json').write_text(json.dumps(pts, ensure_ascii=False, indent=1), encoding='utf-8')
