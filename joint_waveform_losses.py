"""Masked per-window, per-belt shape and error objectives."""
import torch


def loss_terms(pred, target, mask, pearson_weight=0., ccc_weight=0., derivative_weight=.2):
    w = mask[:, :, None]
    channels = pred.shape[-1]
    mse = ((pred - target).square() * w).sum() / (channels * mask.sum().clamp_min(1))
    pairs = (mask[:, 1:] * mask[:, :-1])[:, :, None]
    difference = (pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
    derivative = (difference.square() * pairs).sum() / (channels * pairs.sum().clamp_min(1))
    count = w.sum(dim=1).clamp_min(1)
    mp = (pred * w).sum(dim=1) / count
    mt = (target * w).sum(dim=1) / count
    pc, tc = pred - mp[:, None], target - mt[:, None]
    vp = (pc.square() * w).sum(dim=1) / count
    vt = (tc.square() * w).sum(dim=1) / count
    cov = (pc * tc * w).sum(dim=1) / count
    # A flat target has no defined Pearson correlation; MSE still supervises it.
    valid = ((vt > 1e-6) & (count >= 3)).to(pred.dtype)
    r = cov / (vp.clamp_min(1e-6) * vt.clamp_min(1e-6)).sqrt()
    ccc = 2 * cov / (vp + vt + (mp - mt).square()).clamp_min(1e-6)
    corr_loss = ((1 - r.clamp(-1, 1)) * valid).sum() / valid.sum().clamp_min(1)
    ccc_loss = ((1 - ccc.clamp(-1, 1)) * valid).sum() / valid.sum().clamp_min(1)
    total = mse + derivative_weight * derivative + pearson_weight * corr_loss + ccc_weight * ccc_loss
    return {'mse': mse, 'derivative': derivative, 'one_minus_r': corr_loss,
            'one_minus_ccc': ccc_loss, 'total': total}


def joint_loss(pred, target, mask, **weights):
    return loss_terms(pred, target, mask, **weights)['total']
