"""Round 19-C: fine-tune the WavLM-Large encoder with LoRA (low-rank adaptation) jointly with the
batch-1 Conv-LSTM head. Single-factor comparison with batch 1: same fold, same winner config,
same refit epoch count, same seed (20260930); only the encoder is adapted.

- Encoder: microsoft/wavlm-large truncated to its first 16 transformer layers; the layer-16 output
  (= hidden state 16 of the frozen cache) is captured with a forward hook. LoRA rank 8, alpha 16
  on q_proj and v_proj of layers 9-16 (indices 8-15); B initialised to zero, so at step 0 the
  features equal the frozen cache exactly.
- 50-Hz frames are averaged in groups of 10 to the 5-Hz grid (frame centres fall in bins exactly);
  features are normalised with the outer-train statistics of the frozen cache.
- Optimiser: AdamW, head lr = winner lr, LoRA lr = 1e-4; gradient checkpointing; bf16 autocast.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from r13_heads import build
from run_r13_followup import load_input, normalize
from run_r13_search import seed_all, seg_metrics
from waveform_family_models import waveform_loss

ROOT = Path(__file__).resolve().parent
SR, HOP5 = 16000, 3200


class LoRA(nn.Module):
    """Weight parametrization W -> W + scale * B @ A (works with WavLM's fused attention path,
    which reads q_proj.weight / v_proj.weight directly)."""
    def __init__(self, out_f, in_f, r=8, alpha=16):
        super().__init__()
        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f) / math.sqrt(in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, W):
        return W + (self.B @ self.A).to(W.dtype) * self.scale


class Encoder(nn.Module):
    def __init__(self, lora=True):
        super().__init__()
        from transformers import WavLMModel
        m = WavLMModel.from_pretrained('microsoft/wavlm-large')
        m.encoder.layers = m.encoder.layers[:16]
        for p in m.parameters():
            p.requires_grad_(False)
        from torch.nn.utils import parametrize
        for layer in (m.encoder.layers[8:16] if lora else []):
            for proj in (layer.attention.q_proj, layer.attention.v_proj):
                parametrize.register_parametrization(proj, 'weight', LoRA(proj.out_features, proj.in_features))
        m.config.layerdrop = 0.0
        m.config.apply_spec_augment = False; m.config.mask_time_prob = 0.0; m.config.mask_feature_prob = 0.0  # no random masking
        for mod in m.modules():  # round-19 C amendment: no encoder dropout, so frozen features are deterministic
            if isinstance(mod, nn.Dropout):
                mod.p = 0.0
            if isinstance(getattr(mod, 'dropout', None), float):
                mod.dropout = 0.0
        self.m = m; self.out = None
        m.encoder.layers[15].register_forward_hook(lambda mod, i, o: setattr(self, 'out', o[0] if isinstance(o, tuple) else o))

    def forward(self, wav):
        wav = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)  # Wav2Vec2FeatureExtractor do_normalize
        self.m(wav)
        h = self.out                                   # (B, frames50, 1024)
        n5 = h.shape[1] // 10
        return h[:, :n5 * 10].reshape(h.shape[0], n5, 10, -1).mean(2)


class Full(nn.Module):
    def __init__(self, head, mean, std, lora=True):
        super().__init__()
        self.enc = Encoder(lora); self.head = head
        self.register_buffer('mean', torch.from_numpy(mean)); self.register_buffer('std', torch.from_numpy(std))

    def forward(self, wav):
        return self.head((self.enc(wav).float() - self.mean) / self.std)


def load_audio(rows, n5s):
    out = []
    for r, n5 in zip(rows, n5s):
        a, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
        v = resample_poly(a[:, 0], SR, sr).astype('float32')
        need = n5 * HOP5 + 80
        v = np.pad(v, (0, max(0, need - len(v))))[:need]
        out.append(v)
    return out


def windows(n5, size):
    s = list(range(0, max(n5 - size + 1, 1), size // 2))
    if not s or s[-1] != max(0, n5 - size):
        s.append(max(0, n5 - size))
    return s


def batch(audio, records, items, size):
    wav, ys, ms = [], [], []
    for i, s in items:
        y = records[i][2][s:s + size]; n = len(y)
        w = audio[i][s * HOP5:(s + size) * HOP5 + 80]
        w = np.pad(w, (0, size * HOP5 + 80 - len(w)))
        wav.append(w); ys.append(np.pad(y, ((0, size - n), (0, 0)))); ms.append(np.r_[np.ones(n), np.zeros(size - n)])
    return torch.from_numpy(np.stack(wav)), torch.from_numpy(np.stack(ys).astype('float32')), torch.from_numpy(np.stack(ms).astype('float32'))


@torch.no_grad()
def predict(model, audio, records, idx, size, device):
    model.eval(); out = {}
    for i in idx:
        n5 = len(records[i][2]); s_all = windows(n5, size)
        acc = np.zeros((n5, 2)); cnt = np.zeros(n5)
        for s in s_all:
            w, _, m = batch(audio, records, [(i, s)], size)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(w.to(device)).float().cpu().numpy()[0]
            n = int(m.sum()); acc[s:s + n] += p[:n]; cnt[s:s + n] += 1
        out[i] = (acc / cnt[:, None]).astype('float32')
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--family', default='convlstm'); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--window', type=int, default=0); ap.add_argument('--frozen', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4); device = 'cuda'
    import csv
    rows = [r for r in csv.DictReader(open(ROOT / 'breath_manifest.csv', encoding='utf-8-sig')) if r['suspect'] != '1' and float(r['duration_s']) >= 10]
    records = load_input('wavlm16', None)
    for r, (m, _, _) in zip(rows, records):
        assert r['singer_en'] == m['singer'] and r['number'] == str(m['number'])
    audio = load_audio(rows, [len(y) for _, _, y in records])
    b1 = json.loads((ROOT / 'family_runs' / f'r13_{args.family}' / 'results.json').read_text(encoding='utf-8'))
    winners = {f['fold']: (f['winner_config'], f['refit_epochs']) for f in b1['folds']}
    singers = np.array([m['singer'] for m, _, _ in records]); styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930).split(np.arange(len(records)), styles, singers))
    out = ROOT / 'family_runs' / (f'r19c_{args.family}_' + ('frozen' if args.frozen else 'lora') + (f'_w{args.window}' if args.window else '')); out.mkdir(parents=True, exist_ok=True)
    rp = out / 'results.json'
    res = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {'name': out.name, 'seed': 20260930, 'folds': []}
    done = {f['fold'] for f in res['folds']}
    for fold, (otr, ote) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.monotonic(); cfg, epochs = winners[fold]
        if args.window:
            cfg = {**cfg, 'window': args.window}
        mean, std = normalize(records, otr); seed_all(20260930)
        head = build(args.family, 1024, {**cfg['arch'], 'dropout': cfg['dropout']})
        model = Full(head, mean, std, lora=not args.frozen).to(device)
        model.enc.m.gradient_checkpointing_enable() if hasattr(model.enc.m, 'gradient_checkpointing_enable') else None
        lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('enc') and 'parametrizations' in n]
        assert len(lora) == (0 if args.frozen else 32), len(lora)
        groups = [{'params': model.head.parameters(), 'lr': cfg['lr']}] + ([{'params': lora, 'lr': 1e-4}] if lora else [])
        opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
        size = cfg['window']; items = [(int(i), s) for i in otr for s in windows(len(records[i][2]), size)]
        rng = np.random.default_rng(20260930 + fold)
        cache = None
        if args.frozen:  # encoder is fixed and deterministic: compute each window's features once
            cache = {}
            model.eval()
            with torch.no_grad():
                for k in range(0, len(items), args.batch):
                    w, _, _ = batch(audio, records, items[k:k + args.batch], size)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        f = model.enc(w.to(device)).float().cpu().half()
                    for j, it in enumerate(items[k:k + args.batch]):
                        cache[it] = f[j]
            print(f'fold{fold} cached {len(cache)} windows {time.monotonic() - t0:.0f}s', flush=True)
        for ep in range(epochs):
            model.train(); model.enc.m.train(); perm = rng.permutation(len(items)); losses = []
            for b0 in range(0, len(perm), args.batch):
                w, y, m = batch(audio, records, [items[k] for k in perm[b0:b0 + args.batch]], size)
                w, y, m = w.to(device), y.to(device), m.to(device)
                opt.zero_grad(set_to_none=True)
                if cache is not None:
                    x = torch.stack([cache[items[k]] for k in perm[b0:b0 + args.batch]]).to(device).float()
                    p = model.head((x - model.mean) / model.std)
                else:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        p = model(w)
                loss = waveform_loss(p.float(), y, m); assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1); opt.step()
                losses.append(float(loss))
            print(f'fold{fold} epoch {ep + 1}/{epochs} loss {np.mean(losses):.4f} {time.monotonic() - t0:.0f}s', flush=True)
        preds = predict(model, audio, records, [int(i) for i in ote], size, device)
        order = [int(i) for i in ote]
        seg = [{**records[i][0], **seg_metrics(records[i][2], preds[i])} for i in order]
        np.savez_compressed(out / f'fold{fold}_waveforms.npz', prediction=np.concatenate([preds[i] for i in order]),
                            target=np.concatenate([records[i][2] for i in order]), metadata=np.array(json.dumps([records[i][0] for i in order])),
                            offsets=np.r_[0, np.cumsum([len(records[i][2]) for i in order])])
        res['folds'].append({'fold': fold, 'config': cfg, 'epochs': epochs, 'test_segments': seg, 'seconds': time.monotonic() - t0})
        rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'== fold{fold} A/B ccc {np.median([r["A"]["ccc"] for r in seg]):.4f}/{np.median([r["B"]["ccc"] for r in seg]):.4f}', flush=True)
        del model, opt; torch.cuda.empty_cache()
    res['completed'] = True; rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
