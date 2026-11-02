"""Loss invariants checked on real belt windows; these are not model scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from joint_waveform_losses import loss_terms, joint_loss
from waveform_family_models import waveform_loss

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=Path, required=True)
args = ap.parse_args()
with np.load(args.root / 'wavlm_large_layer16_voice_AB_5hz.npz', allow_pickle=False) as cache:
    y = torch.tensor(cache['y'][:256].reshape(2, 128, 2))
mask = torch.ones(2, 128)
mask[1, 80:] = 0
tests = []
pred = 2 * y + .7
torch.testing.assert_close(joint_loss(pred, y, mask), waveform_loss(pred, y, mask))
tests.append('zero_extra_weights_match_previous_loss')
t = loss_terms(y, y, mask, pearson_weight=1, ccc_weight=1)
assert abs(float(t['total'])) < 1e-6
tests.append('identity_has_zero_loss')
t = loss_terms(pred, y, mask)
assert abs(float(t['one_minus_r'])) < 1e-6 and t['one_minus_ccc'] > .05 and t['mse'] > .05
tests.append('pearson_ignores_affine_error_but_mse_and_ccc_detect_it')
t = loss_terms(-y, y, mask)
assert abs(float(t['one_minus_r']) - 2) < 1e-6
tests.append('inverted_polarity_penalized')
padded_pred, padded_y = pred.clone(), y.clone()
padded_pred[1, 80:] = 1e4
padded_y[1, 80:] = -1e4
torch.testing.assert_close(joint_loss(padded_pred, padded_y, mask, pearson_weight=.3, ccc_weight=.3),
                           joint_loss(pred, y, mask, pearson_weight=.3, ccc_weight=.3))
padded_pred.requires_grad_()
joint_loss(padded_pred, padded_y, mask, pearson_weight=.3, ccc_weight=.3).backward()
assert torch.isfinite(padded_pred.grad).all() and (padded_pred.grad[1, 80:] == 0).all()
tests.append('padding_changes_neither_loss_nor_valid_gradients')
for target in (y, torch.ones_like(y)):
    flat = torch.zeros_like(y, requires_grad=True)
    value = joint_loss(flat, target, mask, pearson_weight=1, ccc_weight=1)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(flat.grad).all()
tests.append('constant_predictions_and_targets_have_finite_gradients')
out = {'passed': tests, 'count': len(tests), 'source': 'real target windows with controlled numerical transformations'}
(args.root / 'joint_loss_checks.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out))
