"""Resumable singer-held-out comparison of task-adapted waveform families."""
import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import DataLoader

from dual_belt_cv import DualWindows, singer_median
from evaluate_breath_pilot import metrics
from full_breath_cv import extrema_score
from waveform_family_models import AlignedLatents, LatentGenerator, make_model, waveform_loss


EXPERIMENTS = {
    'large_tcn_control': ('whisper_large_v3_voice_AB_5hz.npz', 'tcn'),
    'large_transformer': ('whisper_large_v3_voice_AB_5hz.npz', 'transformer'),
    'large_wave_unet': ('whisper_large_v3_voice_AB_5hz.npz', 'wave_unet'),
    'large_swin_unet': ('whisper_large_v3_voice_AB_5hz.npz', 'swin_unet'),
    'large_nab_inspired': ('whisper_large_v3_voice_AB_5hz.npz', 'nab_inspired'),
    'large_aligned_latent4': ('whisper_large_v3_voice_AB_5hz.npz', 'aligned_latent'),
    'large_latent4_flow': ('whisper_large_v3_voice_AB_5hz.npz', 'flow'),
    'large_latent4_diffusion': ('whisper_large_v3_voice_AB_5hz.npz', 'diffusion'),
    'wavlm4_transformer': ('wavlm_layer4_voice_AB_5hz.npz', 'transformer'),
    'wavlm7_transformer': ('wavlm_layer7_voice_AB_5hz.npz', 'transformer'),
    'wavlm12_transformer': ('wavlm_layer12_voice_AB_5hz.npz', 'transformer'),
    'wavlm7_tcn': ('wavlm_layer7_voice_AB_5hz.npz', 'tcn'),
    'wavlm_large8_transformer': ('wavlm_large_layer8_voice_AB_5hz.npz', 'transformer'),
    'wavlm_large16_transformer': ('wavlm_large_layer16_voice_AB_5hz.npz', 'transformer'),
    'wavlm_large24_transformer': ('wavlm_large_layer24_voice_AB_5hz.npz', 'transformer'),
    'wavlm_large16_tcn': ('wavlm_large_layer16_voice_AB_5hz.npz', 'tcn'),
}


def seed_all(seed=20260930):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path):
    # Materialize arrays once; repeated npz member access would decompress per recording.
    with np.load(path, allow_pickle=False) as cache:
        x, y, offsets = cache['x'], cache['y'], cache['offsets']
        meta = json.loads(str(cache['metadata']))
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return [(m, x[offsets[i]:offsets[i + 1]], y[offsets[i]:offsets[i + 1]]) for i, m in enumerate(meta)]


def normalize(records, indices):
    values = np.concatenate([records[i][1] for i in indices]).astype('float32')
    return values.mean(axis=0), np.maximum(values.std(axis=0), 1e-5)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = []
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y, mask) if hasattr(model, 'loss') else waveform_loss(model(x), y, mask)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total.append(float(loss.detach()))
    return float(np.mean(total))


def prepare_generator(model, loader, device, epochs, progress):
    if not isinstance(model, LatentGenerator):
        return
    optimizer = torch.optim.AdamW(model.ae.parameters(), lr=3e-4, weight_decay=1e-4)
    for epoch in range(epochs):
        loss = train_epoch(model.ae, loader, optimizer, device)
        progress(f'latent_pretrain {epoch + 1}/{epochs} loss={loss:.4f}')
    model.freeze_ae()
    chunks = []
    with torch.no_grad():
        for _, y, mask in loader:
            latent = model.ae.belt(y.to(device))
            valid = torch.nn.functional.avg_pool1d(mask.to(device)[:, None], 4) > .99
            chunks.append(latent.permute(0, 2, 1)[valid[:, 0]].cpu())
        values = torch.cat(chunks)
        model.latent_mean.copy_(values.mean(dim=0)[None, :, None])
        model.latent_std.copy_(values.std(dim=0).clamp_min(.05)[None, :, None])


@torch.no_grad()
def evaluate(model, records, indices, mean, std, device, save_path=None):
    model.eval()
    dataset = DualWindows(records, indices, mean, std)
    sums = {int(i): np.zeros_like(records[i][2], dtype='float64') for i in indices}
    counts = {int(i): np.zeros(len(records[i][2]), dtype='float64') for i in indices}
    offset = 0
    for x, _, mask in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0):
        pred = model(x.to(device)).float().cpu().numpy()
        if not np.isfinite(pred).all():
            raise RuntimeError('Non-finite inference output')
        for j, p in enumerate(pred):
            i, start = dataset.locations[offset + j]
            n = int(mask[j].sum())
            sums[i][start:start + n] += p[:n]
            counts[i][start:start + n] += 1
        offset += len(pred)
    rows, predictions, targets, metadata, boundaries = [], [], [], [], [0]
    for i in indices:
        meta, _, y = records[i]
        assert np.all(counts[int(i)] > 0)
        p = (sums[int(i)] / counts[int(i)][:, None]).astype('float32')
        row = dict(meta)
        for channel, name in enumerate(('A', 'B')):
            a, b = y[:, channel], p[:, channel]
            ccc = 2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2 + 1e-8)
            row[name] = {**metrics(a, b), 'ccc': float(ccc), 'std_ratio': float(b.std() / (a.std() + 1e-8)),
                         'peak_f1': extrema_score(a, b, +1)['f1'], 'trough_f1': extrema_score(a, b, -1)['f1']}
        rows.append(row)
        predictions.append(p)
        targets.append(y)
        metadata.append(meta)
        boundaries.append(boundaries[-1] + len(y))
    if save_path:
        np.savez_compressed(save_path, prediction=np.concatenate(predictions), target=np.concatenate(targets),
                            metadata=np.array(json.dumps(metadata)), offsets=np.array(boundaries))
    return rows


def run_experiment(args, name, device):
    cache_name, family = EXPERIMENTS[name]
    records = load_records(args.root / cache_name)
    singers = np.array([m['singer'] for m, _, _ in records])
    styles = np.array([m['style'] for m, _, _ in records])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260930)
                 .split(np.arange(len(records)), styles, singers))
    out = args.root / 'family_runs' / name
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'results.json'
    data = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {
        'experiment': name, 'family': family, 'cache': cache_name,
        'scope': 'task-adapted family implementation, not exact author-model reproduction',
        'epochs_max': args.epochs, 'latent_pretrain_epochs': args.pretrain,
        'selection': 'inner singer split early stopping; outer scores exploratory only',
        'generative_prediction': 'mean of four fixed seeds and ten steps; no target-based selection',
        'folds': []}
    if data['epochs_max'] != args.epochs or data['latent_pretrain_epochs'] != args.pretrain:
        raise RuntimeError('Cannot resume with changed budget')
    finished = {f['fold'] for f in data['folds']}
    for fold, (outer_train, outer_test) in enumerate(folds, 1):
        if fold in finished:
            continue
        start_time = time.monotonic()
        def progress(message):
            status = {'experiment': name, 'fold': fold, 'phase': message, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            (args.root / 'family_runs' / 'current_progress.json').write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
        it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260930 + fold)
                      .split(outer_train, groups=singers[outer_train]))
        train, val = outer_train[it], outer_train[iv]
        assert not set(singers[train]) & set(singers[val])
        assert not set(singers[outer_train]) & set(singers[outer_test])
        mean, std = normalize(records, train)
        seed_all()
        model = make_model(family, records[0][1].shape[1]).to(device)
        loader = DataLoader(DualWindows(records, train, mean, std), batch_size=32, shuffle=True, num_workers=0)
        prepare_generator(model, loader, device, args.pretrain, progress)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
        best, selected, bad, history = -1e9, 1, 0, []
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, loader, optimizer, device)
            val_rows = evaluate(model, records, val, mean, std, device)
            score = singer_median(val_rows, 'r')
            history.append({'epoch': epoch, 'loss': loss, 'val_singer_AB_r': score})
            progress(f'inner {epoch}/{args.epochs} r={score:.4f}')
            if score > best + .002:
                best, selected, bad = score, epoch, 0
            else:
                bad += 1
            if bad >= 5 and epoch >= 8:
                break
        del model, optimizer, loader
        torch.cuda.empty_cache()
        mean, std = normalize(records, outer_train)
        seed_all()
        model = make_model(family, records[0][1].shape[1]).to(device)
        loader = DataLoader(DualWindows(records, outer_train, mean, std), batch_size=32, shuffle=True, num_workers=0)
        prepare_generator(model, loader, device, args.pretrain, progress)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
        for epoch in range(selected):
            train_epoch(model, loader, optimizer, device)
            progress(f'outer_refit {epoch + 1}/{selected}')
        checkpoint = out / f'fold{fold}.pt'
        torch.save({'model': model.state_dict(), 'mean': mean, 'std': std, 'family': family,
                    'input_dim': records[0][1].shape[1], 'selected_epoch': selected,
                    'train_singers': sorted(set(singers[outer_train])), 'test_singers': sorted(set(singers[outer_test]))}, checkpoint)
        rows = evaluate(model, records, outer_test, mean, std, device, out / f'fold{fold}_waveforms.npz')
        data['folds'].append({'fold': fold, 'selected_epoch': selected, 'best_inner_r': best,
                              'history': history, 'test_singers': sorted(set(singers[outer_test])),
                              'test_segments': rows, 'checkpoint': checkpoint.name,
                              'elapsed_s': time.monotonic() - start_time})
        result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{name} fold={fold}/5 epoch={selected} r={singer_median(rows, "r"):.4f} seconds={time.monotonic() - start_time:.1f}', flush=True)
        del model, optimizer, loader
        torch.cuda.empty_cache()
    rows = [row for f in data['folds'] for row in f['test_segments']]
    data['summary'] = {c: {key: float(np.median([r[c][key] for r in rows]))
                            for key in ('r', 'ccc', 'nrmse', 'peak_f1', 'trough_f1', 'std_ratio')} for c in ('A', 'B')}
    data['completed'] = True
    result_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'COMPLETE {name}: {json.dumps(data["summary"])}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--experiments', nargs='+', default=list(EXPERIMENTS))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--pretrain', type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (args.root / 'family_runs').mkdir(exist_ok=True)
    failures = []
    for name in args.experiments:
        try:
            run_experiment(args, name, device)
        except Exception:
            detail = traceback.format_exc()
            failures.append({'experiment': name, 'traceback': detail})
            print(detail, flush=True)
            (args.root / 'family_runs' / f'{name}_failure.json').write_text(json.dumps(failures[-1], indent=2), encoding='utf-8')
    (args.root / 'family_runs' / 'batch_completion.json').write_text(json.dumps({
        'requested': args.experiments, 'failures': failures, 'ended': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=2), encoding='utf-8')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
