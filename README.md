# WaveBreath

Predicts chest and abdominal breathing-belt signals from singing and speech audio, using frozen self-supervised speech encoders.

**Data.** 32 singers and 197 recordings (96 songs, 41 scales, 60 lyric readings), about 5.5 h in total. Each recording has synchronized audio, two respiratory belts and EGG:
- A = chest belt;
- B = abdominal belt.

Targets are 5 Hz belt waveforms, low-passed at 1.5 Hz and standardized within each segment.

**Evaluation.** Five-fold cross-validation grouped by singer, so test singers are never seen in training. The inner loop selects models on held-out training singers.

**This repository contains code and one set of model weights only.** No recordings, belt signals, EGG, features or predictions are included. Participant identifiers in the code are pseudonymised (P01–P32).

## Results (singer-median CCC on held-out singers)

| Model | Chest CCC | Abdomen CCC | Weights in repo |
|---|---|---|---|
| Baseline: WavLM-Large layer 16 + TCN | 0.813 | 0.778 | no |
| Multitask TCN (`run_r12_multitask.py`, lambda = 0.3) | 0.821 | 0.797 | **yes** (`checkpoints/`) |
| Recommended: Conv-LSTM + U-Net ensemble | 0.849 | 0.805 | no (not saved) |
| Best chest / best abdomen over 79 attempts (different runs) | 0.857 | 0.822 | no |

CCC is the concordance correlation coefficient. It penalizes poor correlation, offset and scale mismatch together.

**Caveat.** The same outer test singers were evaluated across many development rounds, so all numbers carry selection bias and are exploratory. A confirmatory test on new singers with a frozen pipeline has not been done yet.

Parameter-level results for the recommended model:

| Parameter | Result |
|---|---|
| Chest breathing rate | CCC 0.83 |
| Chest within-phrase expiratory slope | CCC 0.86 |
| Inhalation-onset F1 within ±1 s | 0.81 (chest) / 0.71 (abdomen) |
| Inspiration duration | poor (CCC ≈ 0.34) |
| Absolute inhalation depth across recordings | poor |
| Chest–abdomen coordination | poor (predicted A–B correlation 0.91 vs 0.68 measured) |

**Sum versus partition.** The chest + abdomen sum is recovered better (r ≈ 0.90) than the abdomen − chest partition (r ≈ 0.44). Most of this gap is arithmetic (see `diag_r27_partition_null.py`): with standardized belts, var(D)/var(S) = (1 − ρ)/(1 + ρ). The belts are uncalibrated, and the sum is not a measured lung volume.

## Checkpoint

`checkpoints/r12_multitask_tcn_lam0.3/fold{1..5}.pt` holds the five outer-fold models of the best run whose weights were saved. Each file contains:
- `model`: state dict for `run_r12_multitask.MultiTaskTCN(1024)`;
- `feature_mean`, `feature_std`: input standardization fitted on that fold's training singers;
- metadata (fold, number of training and test singers, selected epoch).

Singer lists were removed.

```python
import torch
from run_r12_multitask import MultiTaskTCN

ck = torch.load("checkpoints/r12_multitask_tcn_lam0.3/fold1.pt", weights_only=True)
model = MultiTaskTCN(1024)
model.load_state_dict(ck["model"])
model.eval()
# x: (batch, frames, 1024) WavLM-Large layer-16 features at 5 Hz
x = (x - ck["feature_mean"]) / ck["feature_std"]
```

Features are WavLM-Large layer-16 hidden states at 5 Hz (feature-extraction scripts: `extract_wavlm_study.py`, `build_voice_feature_cache.py`). Inference uses 128-frame (25.6 s) windows with a 64-frame hop and equal-weight overlap averaging.

## Code map

| Area | Files |
|---|---|
| Data manifest, targets and features | `build_breath_manifest.py`, `build_dual_target_cache.py`, `build_r11_targets.py`, `build_r13_targets_50hz.py`, `build_voice_feature_cache.py`, `extract_wavlm_study.py`, `extract_r13_all_layers.py`, `extract_whisper_*.py`, `build_r14_egg_features.py` |
| Model heads | `r13_heads.py` (TCN, BiGRU, Conv-LSTM, Transformer, U-Net), `waveform_family_models.py`, `run_r13_50hz.py` (two-scale 50 Hz) |
| Training (nested, singer-grouped CV) | `run_r13_search.py`, `run_r13_followup.py`, `run_r12_multitask.py`, `run_waveform_families.py`, `run_r18_composite_50hz.py`, `run_r19_personalize.py`, `run_r19c_lora.py`, `run_r21_gan.py`, `run_r23_gan2.py`, `run_r26_best_of_n.py` |
| Breathing-parameter extraction (frozen rules) | `breath_parameters.py`, `breath_parameters_fs.py`, `breath_param_set.py` |
| Evaluation and diagnostics | `run_r16_param_validation.py`, `compare_*.py`, `diag_r*.py`, `summarize_*.py` |
| Figures | `plot_round_compare.py`, `plot_r13_gallery.py`, `plot_asymptote_*.py`, `plot_learning_curve.py` |

Files prefixed `r<N>` / `diag_r<N>` belong to development round N. Earlier, unprefixed scripts are from rounds 1–8 (Whisper/WavLM screens, latent and generative models).

Scripts expect the private feature and target caches next to them. They will not run without the data.

## Requirements

Python 3.11+, PyTorch, NumPy, SciPy, scikit-learn, matplotlib, soundfile, transformers (WavLM / Whisper / MERT feature extraction).
