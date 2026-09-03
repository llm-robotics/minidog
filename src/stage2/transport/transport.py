import random

import torch as th
import torch.nn.functional as F

from stage2.utils import apply_cfg_dropout
from utils.dist_utils import synchronize_gradients


def _expand_t(t, x):
    return t.view(t.size(0), *([1] * (len(x.size()) - 1)))


def spatial_zscore(feat, alpha=1.0, eps=1e-6):
    """iREPA (arxiv 2512.10794) target normalization: per-channel z-score across the token/
    spatial dimension (dim=1), independently per sample. Removes each channel's own scale/
    offset so the REPA loss compares relative spatial pattern rather than absolute magnitude.
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


def attn_align_loss(pred, target, loss_type="kl", eps=1e-8):
    """Per-sample loss between two pooled [B, N] image-token attention-mass vectors.

    Neither vector need sum to 1 over N as-is (each is a per-token sum over a
    different, unrelated key axis on each side) -- for "kl" both are renormalized
    over N first so the comparison is "where does attention concentrate", not raw
    magnitude. Returns a [B] tensor (unreduced), matching the loss_percep masking
    convention (mask, don't .mean(), so callers can gate by timestep first).
    """
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    if loss_type == "cosine":
        return 1 - F.cosine_similarity(pred, target, dim=-1, eps=eps)
    if loss_type == "kl":
        p = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = pred / pred.sum(dim=-1, keepdim=True).clamp_min(eps)
        return F.kl_div(q.clamp_min(eps).log(), p, reduction="none").sum(dim=-1)
    raise ValueError(f"Unknown attn_align_loss_type: {loss_type}")


def get_time_sampler(time_dist_type: str):
    parts = time_dist_type.split("_")
    name = parts[0]
    if name == "logit-normal":
        assert len(parts) == 3, f"Expected 'logit-normal_MU_SIGMA', got '{time_dist_type}'"
        mu, sigma = float(parts[1]), float(parts[2])
        assert sigma > 0, "sigma must be > 0"
        return lambda bs: (th.randn(bs) * sigma + mu).sigmoid()
    else:
        raise NotImplementedError(f"Unknown time distribution: {time_dist_type}")


class Transport:
    def __init__(self, prediction="velocity", time_dist_type="logit-normal_0_1", time_dist_shift=1.0, time_dist_shift_eval=1.0, t_eps=0.05, percep_loss_t_thresh=0.7):
        self.prediction = prediction
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.time_dist_shift_eval = time_dist_shift_eval
        self.t_eps = t_eps
        self.percep_loss_t_thresh = percep_loss_t_thresh
        self.time_sampler = get_time_sampler(time_dist_type)

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t = self.time_sampler(x1.shape[0]).to(x1)
        t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        return t, x0, x1

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################
    def training_losses(self, model, x1, model_kwargs={}, model_kwargs_null={}, z_clean=None, repa_coeff=None, base_model_coeff=1.0, percep_loss=None, cfg_dropout_prob=0.1, ema_model=None, cls_clean=None, reg_coeff=None, attn_target=None, attn_align_coeff=None, attn_align_layer=None, attn_align_t_thresh=0.7, attn_align_loss_type="kl", repa_spatial_norm=False, repa_spatial_norm_alpha=1.0):
        if repa_spatial_norm and z_clean is not None:
            z_clean = spatial_zscore(z_clean, alpha=repa_spatial_norm_alpha)
        model_kwargs, _ = apply_cfg_dropout(model_kwargs, model_kwargs_null, cfg_dropout_prob)

        t, x0, x1 = self.sample(x1)
        xt = (1 - _expand_t(t, x1)) * x1 + _expand_t(t, x1) * x0
        vt = (xt - x1) / _expand_t(t, xt).clamp_min(self.t_eps)

        enable_repa = z_clean is not None and repa_coeff is not None
        enable_reg = cls_clean is not None and reg_coeff is not None
        enable_attn_align = attn_target is not None and attn_align_coeff is not None
        # Append cls noising using the same t and formula
        if enable_reg:
            cls_x0 = th.randn_like(cls_clean)
            cls_t = (1 - _expand_t(t, cls_clean)) * cls_clean + _expand_t(t, cls_clean) * cls_x0
            v_cls = (cls_t - cls_clean) / _expand_t(t, cls_t).clamp_min(self.t_eps)
            model_kwargs = {**model_kwargs, "cls_t": cls_t}

        zt_pred = None
        attn_pred = None
        if enable_repa or enable_attn_align:
            model_output = model(
                xt, t,
                return_intermediate=enable_repa,
                return_attn_layer=attn_align_layer if enable_attn_align else None,
                **model_kwargs,
            )
            if enable_repa and enable_attn_align:
                model_output, zt_pred, attn_pred = model_output
            elif enable_repa:
                model_output, zt_pred = model_output
            else:
                model_output, attn_pred = model_output
        else:
            model_output = model(xt, t, **model_kwargs)

        # Handle multi-output models: REG (full, cls), IG (full, base), or REG+IG (full, base, cls)
        base_output = None
        cls_pred = None
        if isinstance(model_output, tuple):
            if len(model_output) == 3:
                model_output, base_output, cls_pred = model_output
            elif len(model_output) == 2:
                if enable_reg:
                    model_output, cls_pred = model_output
                else:
                    model_output, base_output = model_output

        # Compute loss
        terms = {'loss': self.compute_loss(model_output, vt, xt, t)}
        if base_output is not None:
            loss_base = self.compute_loss(base_output, vt, xt, t)
            terms['loss'] = terms['loss'] + base_model_coeff * loss_base
            terms['loss_base'] = loss_base
        loss_repa = th.tensor(0.0, device=x1.device)
        if enable_repa and zt_pred is not None:
            loss_repa = repa_coeff * F.mse_loss(zt_pred, z_clean)
        terms['loss_repa'] = loss_repa
        loss_reg = th.tensor(0.0, device=x1.device)
        if enable_reg and cls_pred is not None:
            loss_reg = reg_coeff * F.mse_loss(self.convert_model_pred(cls_pred, cls_t, t), v_cls)
        terms['loss_reg'] = loss_reg

        loss_attn_align = th.tensor(0.0, device=x1.device)
        if enable_attn_align and attn_pred is not None:
            # Only align once the latent is clean enough to have coherent spatial
            # content to align against -- same idiom as loss_percep's t-mask below.
            mask = t < attn_align_t_thresh
            loss_attn_align = attn_align_coeff * attn_align_loss(attn_pred, attn_target, attn_align_loss_type) * mask
        terms['loss_attn_align'] = loss_attn_align

        if percep_loss is not None:
            assert self.prediction == "x"
            # Mask based on t < percep_loss_t_thresh
            mask = t < self.percep_loss_t_thresh
            terms['loss_percep'] = percep_loss(model_output, x1) * mask  # [B]

        return terms

    def post_backward(self, model):
        pass

    def convert_model_pred(self, output, xt, t):
        # Unify model output to v-pred
        if self.prediction == "velocity":
            return output
        elif self.prediction == "x":
            t_safe = _expand_t(t, xt).clamp_min(self.t_eps)
            return (xt - output) / t_safe

    def compute_loss(self, output, vt, xt, t):
        output = self.convert_model_pred(output, xt, t)
        return (output - vt) ** 2

    def get_drift(self):
        def body_fn(x, t, h, model, **model_kwargs):
            cls_t = model_kwargs.get('cls_t')
            if cls_t is not None:
                x_pred, cls_pred = model(x, t, **model_kwargs)
                return self.convert_model_pred(x_pred, x, t), self.convert_model_pred(cls_pred, cls_t, t)
            model_output = model(x, t, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            return self.convert_model_pred(model_output, x, t)
        return body_fn
