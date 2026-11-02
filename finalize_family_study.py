"""Write a durable completion entry only after all experiments pass the audit."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
import torch
from run_waveform_families import EXPERIMENTS

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--outputs', type=Path, required=True)
args = ap.parse_args()
runs = args.root / 'family_runs'
audit = json.loads((runs / 'completion_integrity_and_failures.json').read_text(encoding='utf-8'))
assert set(audit) == set(EXPERIMENTS)
selected = json.loads((runs / 'inner_selected_family.json').read_text(encoding='utf-8'))
metadata = {
    'completed_at_local': datetime.now().isoformat(), 'python': platform.python_version(),
    'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'transformers')},
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'cuda': torch.version.cuda, 'seed': 20260930,
    'source_sha256': {name: hashlib.sha256((args.root / name).read_bytes()).hexdigest() for name in (
        'run_waveform_families.py', 'waveform_family_models.py', 'extract_wavlm_study.py',
        'dual_belt_cv.py', 'build_dual_target_cache.py', 'summarize_family_runs.py')},
    'configurations': list(EXPERIMENTS), 'verified_outer_folds': 5 * len(EXPERIMENTS),
}
(runs / 'reproducibility_manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
(runs / 'current_progress.json').write_text(json.dumps({
    'phase': 'All configurations finished 5 folds; integrity audit and timing done', 'completed': len(EXPERIMENTS),
    'expected': len(EXPERIMENTS), 'time': metadata['completed_at_local']}, ensure_ascii=False, indent=2), encoding='utf-8')
s = selected['summary']
entry = f'''## Full model-family screen completed (2026-09-30)

16 formal configurations and 80 outer folds completed, coverage of 197 segments / 32 singers checked; with inner-loop model selection per fold the A/B correlation is {s['A']['r']:.3f}/{s['B']['r']:.3f} and CCC {s['A']['ccc']:.3f}/{s['B']['ccc']:.3f}. Full model table, paired intervals, failure counts, latent diagnostics and speed are in family_study_conclusions.md and family_study_progress.md. All models are task-adapted implementations, not full reproductions of the papers' code; this is still an exploratory validation on standardized 5 Hz relative waveforms.

Resume entry: run_waveform_families.py --root <current directory>. All 16 configurations are complete; a normal re-run skips finished folds. Changing the architecture or training budget requires a new configuration directory; old results must not be overwritten. Each family_runs/<config> holds five-fold weights, per-segment metrics and full measured/predicted arrays; reproducibility_manifest.json records code hashes and environment. Figures are in family_figures. The old large_aligned_latent trial run is marked excluded; the valid compressed version has latent4 in its name.
'''
for name in ('README.md', 'progress_log.md'):
    path = args.root / name
    old = path.read_text(encoding='utf-8-sig')
    marker = '## Full model-family screen completed (2026-09-30)'
    if marker not in old:
        if name == 'README.md':
            first, rest = old.split('\n', 1)
            old = first + '\n\n' + entry + '\n' + rest
        else:
            old += '\n\n' + entry
        path.write_text(old, encoding='utf-8')
    if name == 'README.md':
        old = path.read_text(encoding='utf-8')
        old = old.replace('Best exploratory results so far and failure examples', 'Round-4 historical exploratory results and failure examples')
        old = old.replace('- Frozen Whisper Base representations + dual-output TCN in five folds', '- Historical round 4: frozen Whisper Base representations + dual-output TCN in five folds')
        path.write_text(old, encoding='utf-8')
decision = args.root / 'pending_decisions.md'
old = decision.read_text(encoding='utf-8-sig')
marker = '## Pending decisions after the full model screen'
if marker not in old:
    decision.write_text(old + '\n\n' + marker + '\n\n'
        '16 configurations finished 5 folds; read family_study_conclusions.md first. Need to confirm A/B identity, polarity and synchronization; abnormal segments must not be relabelled based on model results. Next round: prioritise encoder intermediate layers combined with TCN/multi-scale heads; keep the shared latent mapping as a mechanistic control; capacity/learning-rate grid and multiple seeds in the inner loop. Flow and Diffusion gave no benefit this round. Belt targets are normalized per segment by the measured median and SD, so absolute amplitude recovery remains unsolved. A final submission needs a confirmation set of new singers after freezing the protocol.\n', encoding='utf-8')
shutil.copy2(decision, args.outputs / 'pending_decisions_breath_belt_prediction.md')
figdir = args.root / 'family_figures'
figdir.mkdir(exist_ok=True)
for name in (*EXPERIMENTS, 'reference_large_shape', 'wavlm_inner_selected_transformer', 'inner_selected_family'):
    source = args.outputs / f'{name}_AB_waveforms.png'
    assert source.exists(), source
    shutil.copy2(source, figdir / source.name)
print(json.dumps(metadata, ensure_ascii=False, indent=2))
