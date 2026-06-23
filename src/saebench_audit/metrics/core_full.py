"""
Full SAEBench Core metric set — faithful port of sae_bench/evals/core/main.py.

Beyond Loss Recovered + L0 (in core_loss_recovered.py), this computes every scalar Core metric:
  model behavior  : kl_div_score, kl_div_with_sae, kl_div_with_ablation
  model perf      : ce_loss_score (Loss Recovered), ce components
  reconstruction  : explained_variance, explained_variance_legacy, mse, cossim
  shrinkage       : l2_norm_in, l2_norm_out, l2_ratio, relative_reconstruction_bias
  sparsity        : l0, l1
  misc            : frac_alive, freq_over_1_percent, freq_over_10_percent,
                    average_max_encoder_cosine_sim, average_max_decoder_cosine_sim

Every formula mirrors the cited SAEBench function; verified by tests/test_core_oracle.py against the
verbatim code on the real transformer_lens model.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .core_loss_recovered import (get_decoder_layers, not_special_mask, per_token_ce,
                                   ResidPostIntervention)


# --------------------------------------------------------------------------- #
# KL divergence per token (get_recons_loss::kl): sum_v p_orig * (log p_orig - log p_new)
# --------------------------------------------------------------------------- #
def kl_per_token(original_logits, new_logits):
    p = F.softmax(original_logits, dim=-1)
    logp = torch.log(p)
    logq = torch.log(F.softmax(new_logits, dim=-1))
    return (p * (logp - logq)).sum(dim=-1)          # [B, S]


@torch.no_grad()
def recons_logits_and_ce(model, sae, layer, batch_tokens, exclude_special_from_recon, special_ids):
    """Return logits + per-token CE for orig / sae-recon / zero-ablation."""
    layer_module = get_decoder_layers(model)[layer]
    out = {}
    lo = model(batch_tokens).logits
    out["orig_logits"] = lo; out["ce_orig"] = per_token_ce(lo, batch_tokens)

    rmask = (not_special_mask(batch_tokens, special_ids) if exclude_special_from_recon
             else torch.ones_like(batch_tokens, dtype=torch.bool))

    def recon_hook(m, i, o):
        is_t = isinstance(o, tuple); x = o[0] if is_t else o
        xhat = sae.decode(sae.encode(x.to(torch.float32))).to(x.dtype)
        xhat = torch.where(rmask[..., None], xhat, x)
        return (xhat,) + tuple(o[1:]) if is_t else xhat

    h = layer_module.register_forward_hook(recon_hook)
    lr = model(batch_tokens).logits; h.remove()
    out["sae_logits"] = lr; out["ce_sae"] = per_token_ce(lr, batch_tokens)

    def zero_hook(m, i, o):
        is_t = isinstance(o, tuple); x = o[0] if is_t else o
        z = torch.zeros_like(x)
        return (z,) + tuple(o[1:]) if is_t else z

    h = layer_module.register_forward_hook(zero_hook)
    lz = model(batch_tokens).logits; h.remove()
    out["abl_logits"] = lz; out["ce_abl"] = per_token_ce(lz, batch_tokens)
    return out


# --------------------------------------------------------------------------- #
# Max cosine similarity between columns of a (D, F) matrix — verbatim port of
# main.py::calculate_max_cosine_sim.
# --------------------------------------------------------------------------- #
def calculate_max_cosine_sim(encoder_DF, batch_size=256):
    enc = F.normalize(encoder_DF, p=2, dim=0)
    Fn = enc.shape[1]
    out = torch.empty(Fn, dtype=enc.dtype, device=enc.device)
    for start in range(0, Fn, batch_size):
        end = min(start + batch_size, Fn)
        sims = enc[:, start:end].t() @ enc            # [C, F]
        for col in range(start, end):
            sims[col - start, col] = float("-inf")
        out[start:end] = sims.max(dim=1).values
    return out


def sae_W_enc_W_dec(sae):
    """Return (W_enc [D,F], W_dec [F,D]) in sae_lens convention for any of our SAE classes."""
    if hasattr(sae, "encoder"):                       # Linear-based (Standard/TopK/BatchTopK/Gated)
        return sae.encoder.weight.detach().T, sae.decoder.weight.detach().T
    return sae.W_enc.detach(), sae.W_dec.detach()     # JumpReLU / Matryoshka


@torch.no_grad()
def compute_core_full(model, sae, layer, recon_batches, sparsity_batches, special_ids,
                      exclude_special_from_recon=False):
    layer_module = get_decoder_layers(model)[layer]
    dsae = sae.dict_size

    # ---- reconstruction batches: CE + KL ----
    ce_o, ce_s, ce_a, kl_s, kl_a = [], [], [], [], []
    for bt in recon_batches:
        r = recons_logits_and_ce(model, sae, layer, bt, exclude_special_from_recon, special_ids)
        m = not_special_mask(bt, special_ids)                      # [B,S]
        mce = m[:, :-1]                                            # CE is [B,S-1]
        ce_o.append(r["ce_orig"][mce]); ce_s.append(r["ce_sae"][mce]); ce_a.append(r["ce_abl"][mce])
        kl_s.append(kl_per_token(r["orig_logits"], r["sae_logits"])[m])
        kl_a.append(kl_per_token(r["orig_logits"], r["abl_logits"])[m])
    ce_orig = torch.cat(ce_o).mean(); ce_sae = torch.cat(ce_s).mean(); ce_abl = torch.cat(ce_a).mean()
    kl_sae = torch.cat(kl_s).mean(); kl_abl = torch.cat(kl_a).mean()

    # ---- sparsity / variance batches ----
    l0s, l1s, l2in, l2out, l2ratio, relbias = [], [], [], [], [], []
    mses, evlegacy, cossims = [], [], []
    mss, mapd, msrs = [], [], []                       # explained-variance (new) pieces
    total_feat_acts = torch.zeros(dsae, dtype=torch.float64)
    total_tokens = 0
    for bt in sparsity_batches:
        cap = {}
        h = layer_module.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        x = cap["x"].to(torch.float32)
        feats = sae.encode(x)
        xhat = sae.decode(feats)
        mask = not_special_mask(bt, special_ids)
        fmask = mask.reshape(-1)
        xin = x.reshape(-1, x.shape[-1])[fmask]
        fac = feats.reshape(-1, dsae)[fmask]
        xo = xhat.reshape(-1, x.shape[-1])[fmask]

        l2_in = torch.norm(xin, dim=-1); l2_out = torch.norm(xo, dim=-1)
        l2_in_div = l2_in.clone(); l2_in_div[l2_in_div.abs() < 1e-4] = 1
        l2in.append(l2_in); l2out.append(l2_out); l2ratio.append(l2_out / l2_in_div)
        relbias.append((torch.norm(xo, dim=-1).pow(2).mean() / (xin * xo).sum(-1).mean()).unsqueeze(0))

        l0s.append((fac != 0).sum(-1).float()); l1s.append(fac.sum(-1))

        rss = (xin - xo).pow(2).sum(-1)
        mses.append(rss / fmask.sum())
        bvs = (xin - xin.mean(dim=0)).pow(2).sum(-1)
        evlegacy.append(1 - rss / bvs)
        mss.append(xin.pow(2).sum(-1).mean(dim=0))
        mapd.append(xin.pow(2).mean(dim=0))
        msrs.append(rss.mean(dim=0))
        xn = xin / torch.norm(xin, dim=-1, keepdim=True)
        xhn = xo / torch.norm(xo, dim=-1, keepdim=True)
        cossims.append((xn * xhn).sum(-1))

        total_feat_acts += (fac > 0).double().sum(dim=0).cpu()
        total_tokens += int(fmask.sum().item())

    # explained variance (new formula, verbatim aggregation)
    mss_t = torch.stack(mss).mean(dim=0)
    mapd_t = torch.cat(mapd).mean(dim=0)
    total_variance = mss_t - mapd_t ** 2
    residual_variance = torch.stack(msrs).mean(dim=0)
    explained_variance = (1 - residual_variance / total_variance).item()

    # feature density / misc
    density = (total_feat_acts / max(total_tokens, 1))
    frac_alive = (density > 0).float().mean().item()
    freq_over_1pct = (density > 0.01).float().mean().item()
    freq_over_10pct = (density > 0.1).float().mean().item()

    W_enc, W_dec = sae_W_enc_W_dec(sae)
    avg_max_enc = calculate_max_cosine_sim(W_enc).mean().item()
    avg_max_dec = calculate_max_cosine_sim(W_dec.T).mean().item()

    return {
        # model behavior preservation
        "kl_div_score": ((kl_abl - kl_sae) / kl_abl).item(),
        "kl_div_with_sae": kl_sae.item(), "kl_div_with_ablation": kl_abl.item(),
        # model performance preservation
        "ce_loss_score": ((ce_abl - ce_sae) / (ce_abl - ce_orig)).item(),
        "loss_recovered": ((ce_abl - ce_sae) / (ce_abl - ce_orig)).item(),
        "ce_loss_with_sae": ce_sae.item(), "ce_loss_without_sae": ce_orig.item(),
        "ce_loss_with_ablation": ce_abl.item(),
        # reconstruction quality
        "explained_variance": explained_variance,
        "explained_variance_legacy": torch.cat(evlegacy).mean().item(),
        "mse": torch.cat(mses).mean().item(),
        "cossim": torch.cat(cossims).mean().item(),
        # shrinkage
        "l2_norm_in": torch.cat(l2in).mean().item(),
        "l2_norm_out": torch.cat(l2out).mean().item(),
        "l2_ratio": torch.cat(l2ratio).mean().item(),
        "relative_reconstruction_bias": torch.cat(relbias).mean().item(),
        # sparsity
        "l0": torch.cat(l0s).mean().item(),
        "l1": torch.cat(l1s).mean().item(),
        # misc
        "frac_alive": frac_alive,
        "freq_over_1_percent": freq_over_1pct,
        "freq_over_10_percent": freq_over_10pct,
        "average_max_encoder_cosine_sim": avg_max_enc,
        "average_max_decoder_cosine_sim": avg_max_dec,
    }
