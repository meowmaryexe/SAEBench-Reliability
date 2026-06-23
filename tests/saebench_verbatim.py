"""
VERBATIM transcription of the loss-recovered + L0 arithmetic from SAEBench
`sae_bench/evals/core/main.py` (adamkarvonen/SAEBench, fetched 2026-06-22).

Only adaptations (documented): we keep the standard residual-stream path (no attention-head
reshaping), the SAE exposes `.encode` / `.decode`, and `model` is a transformer_lens
HookedTransformer loaded with `from_pretrained_no_processing` (so its `hook_resid_post` equals
the raw HF residual stream). Every line below mirrors the cited SAEBench source function.

This module is used by tests/test_core_oracle.py as the ground-truth oracle that our package
implementation (saebench_audit.metrics.core_loss_recovered.saebench_core_eval) must match.
"""
import torch
from functools import partial


@torch.no_grad()
def get_recons_loss(sae, model, batch_tokens, hook_name, ignore_tokens=frozenset(),
                    exclude_special_tokens_from_reconstruction=False):
    # --- mirrors main.py::get_recons_loss (lines ~639-786) ---
    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", loss_per_token=True)

    if len(ignore_tokens) > 0 and exclude_special_tokens_from_reconstruction:
        mask = torch.logical_not(torch.any(
            torch.stack([batch_tokens == t for t in ignore_tokens], dim=0), dim=0))
    else:
        mask = torch.ones_like(batch_tokens, dtype=torch.bool)

    def standard_replacement_hook(activations, hook):
        reconstructed = sae.decode(sae.encode(activations)).to(activations.dtype)
        reconstructed = torch.where(mask[..., None], reconstructed, activations)
        return reconstructed

    def standard_zero_ablate_hook(activations, hook):
        return torch.zeros_like(activations)

    _, recons_ce_loss = model.run_with_hooks(
        batch_tokens, return_type="both",
        fwd_hooks=[(hook_name, partial(standard_replacement_hook))], loss_per_token=True)
    _, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens, return_type="both",
        fwd_hooks=[(hook_name, standard_zero_ablate_hook)], loss_per_token=True)

    return {"ce_loss_with_sae": recons_ce_loss,
            "ce_loss_without_sae": original_ce_loss,
            "ce_loss_with_ablation": zero_abl_ce_loss}


@torch.no_grad()
def downstream_reduction(per_batch_metrics, batch_tokens_list, ignore_tokens):
    # --- mirrors main.py::get_downstream_reconstruction_metrics reduction (lines ~404-432) ---
    keys = ["ce_loss_with_sae", "ce_loss_without_sae", "ce_loss_with_ablation"]
    acc = {k: [] for k in keys}
    for metrics, batch_tokens in zip(per_batch_metrics, batch_tokens_list):
        for name in keys:
            value = metrics[name]
            if len(ignore_tokens) > 0:
                mask = torch.logical_not(torch.any(
                    torch.stack([batch_tokens == t for t in ignore_tokens], dim=0), dim=0))
                if value.shape[1] != mask.shape[1]:
                    mask = mask[:, :-1]
                value = value[mask]
            acc[name].append(value)
    out = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    out["ce_loss_score"] = ((out["ce_loss_with_ablation"] - out["ce_loss_with_sae"]) /
                            (out["ce_loss_with_ablation"] - out["ce_loss_without_sae"]))
    return out


@torch.no_grad()
def get_recons_and_kl(sae, model, batch_tokens, hook_name, ignore_tokens=frozenset(),
                      exclude_special_tokens_from_reconstruction=False):
    """get_recons_loss extended with the KL block (main.py kl()), returning per-token CE + KL."""
    original_logits, original_ce = model(batch_tokens, return_type="both", loss_per_token=True)
    if len(ignore_tokens) > 0 and exclude_special_tokens_from_reconstruction:
        mask = torch.logical_not(torch.any(
            torch.stack([batch_tokens == t for t in ignore_tokens], dim=0), dim=0))
    else:
        mask = torch.ones_like(batch_tokens, dtype=torch.bool)

    def repl(activations, hook):
        rec = sae.decode(sae.encode(activations)).to(activations.dtype)
        return torch.where(mask[..., None], rec, activations)

    def zero(activations, hook):
        return torch.zeros_like(activations)

    recons_logits, recons_ce = model.run_with_hooks(
        batch_tokens, return_type="both", fwd_hooks=[(hook_name, partial(repl))], loss_per_token=True)
    zero_logits, zero_ce = model.run_with_hooks(
        batch_tokens, return_type="both", fwd_hooks=[(hook_name, zero)], loss_per_token=True)

    def kl(a, b):
        pa = torch.nn.functional.softmax(a, dim=-1)
        return (pa * (torch.log(pa) - torch.log(torch.nn.functional.softmax(b, dim=-1)))).sum(-1)

    return {"ce_loss_with_sae": recons_ce, "ce_loss_without_sae": original_ce,
            "ce_loss_with_ablation": zero_ce,
            "kl_div_with_sae": kl(original_logits, recons_logits),
            "kl_div_with_ablation": kl(original_logits, zero_logits)}


@torch.no_grad()
def reduce_recons_and_kl(per_batch, batch_tokens_list, ignore_tokens):
    keys = ["ce_loss_with_sae", "ce_loss_without_sae", "ce_loss_with_ablation",
            "kl_div_with_sae", "kl_div_with_ablation"]
    acc = {k: [] for k in keys}
    for metrics, bt in zip(per_batch, batch_tokens_list):
        for name in keys:
            value = metrics[name]
            if len(ignore_tokens) > 0:
                mask = torch.logical_not(torch.any(
                    torch.stack([bt == t for t in ignore_tokens], dim=0), dim=0))
                if value.shape[1] != mask.shape[1]:
                    mask = mask[:, :-1]
                value = value[mask]
            acc[name].append(value)
    out = {k: torch.cat(v).mean().item() for k, v in acc.items()}
    out["ce_loss_score"] = ((out["ce_loss_with_ablation"] - out["ce_loss_with_sae"]) /
                            (out["ce_loss_with_ablation"] - out["ce_loss_without_sae"]))
    out["kl_div_score"] = ((out["kl_div_with_ablation"] - out["kl_div_with_sae"]) /
                           out["kl_div_with_ablation"])
    return out


@torch.no_grad()
def sparsity_variance_metrics(sae, model, batch_tokens_list, hook_name, hook_layer, ignore_tokens):
    """Verbatim port of main.py::get_sparsity_and_variance_metrics (scalar outputs)."""
    import einops
    md = {k: [] for k in ["l2_norm_in", "l2_norm_out", "l2_ratio", "relative_reconstruction_bias",
                          "l0", "l1", "explained_variance_legacy", "mse", "cossim"]}
    mss, mapd, msrs = [], [], []
    total_feature_acts = None; total_tokens = 0
    for bt in batch_tokens_list:
        mask = torch.logical_not(torch.any(
            torch.stack([bt == t for t in ignore_tokens], dim=0), dim=0)) if len(ignore_tokens) > 0 \
            else torch.ones_like(bt, dtype=torch.bool)
        fmask = mask.flatten()
        _, cache = model.run_with_cache(bt, prepend_bos=False, names_filter=[hook_name],
                                        stop_at_layer=hook_layer + 1)
        act = cache[hook_name]
        feats = sae.encode(act); out = sae.decode(feats)
        xin = einops.rearrange(act, "b c d -> (b c) d")[fmask]
        fac = einops.rearrange(feats, "b c d -> (b c) d")[fmask]
        xo = einops.rearrange(out, "b c d -> (b c) d")[fmask]
        l2_in = torch.norm(xin, dim=-1); l2_out = torch.norm(xo, dim=-1)
        l2id = l2_in.clone(); l2id[l2id.abs() < 1e-4] = 1
        md["l2_norm_in"].append(l2_in); md["l2_norm_out"].append(l2_out)
        md["l2_ratio"].append(l2_out / l2id)
        md["relative_reconstruction_bias"].append(
            (torch.norm(xo, dim=-1).pow(2).mean() / (xin * xo).sum(-1).mean()).unsqueeze(0))
        md["l0"].append((fac != 0).sum(-1).float()); md["l1"].append(fac.sum(-1))
        rss = (xin - xo).pow(2).sum(-1)
        md["mse"].append(rss / fmask.sum())
        bvs = (xin - xin.mean(dim=0)).pow(2).sum(-1)
        md["explained_variance_legacy"].append(1 - rss / bvs)
        mss.append(xin.pow(2).sum(-1).mean(dim=0)); mapd.append(xin.pow(2).mean(dim=0))
        msrs.append(rss.mean(dim=0))
        xn = xin / torch.norm(xin, dim=-1, keepdim=True)
        xhn = xo / torch.norm(xo, dim=-1, keepdim=True)
        md["cossim"].append((xn * xhn).sum(-1))
        fb = (feats * mask.unsqueeze(-1) > 0).float().sum(dim=1).sum(dim=0)
        total_feature_acts = fb if total_feature_acts is None else total_feature_acts + fb
        total_tokens += int(mask.sum().item())
    out = {k: torch.cat(v).mean().item() for k, v in md.items()}
    mss_t = torch.stack(mss).mean(dim=0); mapd_t = torch.cat(mapd).mean(dim=0)
    out["explained_variance"] = (1 - torch.stack(msrs).mean(dim=0) / (mss_t - mapd_t ** 2)).item()
    density = total_feature_acts / total_tokens
    out["frac_alive"] = (density > 0).float().mean().item()
    out["freq_over_1_percent"] = (density > 0.01).float().mean().item()
    out["freq_over_10_percent"] = (density > 0.1).float().mean().item()
    return out


@torch.no_grad()
def sparsity_l0(sae, model, batch_tokens_list, hook_name, hook_layer, ignore_tokens):
    # --- mirrors main.py::get_sparsity_and_variance_metrics L0 path (lines ~486-562) ---
    import einops
    l0_list = []
    for batch_tokens in batch_tokens_list:
        if len(ignore_tokens) > 0:
            mask = torch.logical_not(torch.any(
                torch.stack([batch_tokens == t for t in ignore_tokens], dim=0), dim=0))
        else:
            mask = torch.ones_like(batch_tokens, dtype=torch.bool)
        flattened_mask = mask.flatten()
        _, cache = model.run_with_cache(
            batch_tokens, prepend_bos=False, names_filter=[hook_name],
            stop_at_layer=hook_layer + 1)
        original_act = cache[hook_name]
        sae_feature_activations = sae.encode(original_act)
        flattened = einops.rearrange(sae_feature_activations, "b ctx d -> (b ctx) d")[flattened_mask]
        l0_list.append((flattened != 0).sum(dim=-1).float())
    return torch.cat(l0_list).mean().item()
