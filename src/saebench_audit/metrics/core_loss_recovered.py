"""
Core / Loss Recovered metric (Karvonen et al., SAEBench, Eq. 4).

  Loss Recovered = (H* - H0) / (H_orig - H0)
    H_orig : CE of the unmodified model
    H*     : CE with resid_post(layer L) replaced by the SAE reconstruction x_hat
    H0     : CE with resid_post(layer L) zero-ablated

Independent reimplementation — imports nothing from SAEBench / dictionary_learning; uses only the
released SAE *weights* (via saebench_audit.sae_models). Supports both tokenization paths:
  - tokenize_mode="packed"        → SAEBench core/main.py (paper Table 4, OpenWebText ctx128)
  - tokenize_mode="per_document"  → dictionary_learning loss_recovered (bundled eval_results.json)

The runner is checkpoint-resumable (one JSONL line per batch) so it survives strict per-process
wall-clock limits: call run_core(...) repeatedly until it returns done=True.
"""
from __future__ import annotations
import os, time
import torch
import torch.nn.functional as F

from ..schema import CoreConfig
from ..io import (get_decoder_layers, prepare_pool, pad_batch, load_ckpt, append_ckpt,
                  load_model_and_tokenizer, load_local_sae)

METRIC_NAME = "core_loss_recovered"


# --------------------------------------------------------------------------- #
# Forward-pass intervention on resid_post(layer L) via a forward hook.
# --------------------------------------------------------------------------- #
class ResidPostIntervention:
    def __init__(self, model, layer, sae, mode):
        self.layer_module = get_decoder_layers(model)[layer]
        self.sae, self.mode, self.handle = sae, mode, None

    def _hook(self, module, inputs, output):
        if self.mode == "orig":
            return output
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        if self.mode == "recon":
            dt = self.sae.encoder.weight.dtype if hasattr(self.sae, "encoder") else x.dtype
            new = self.sae(x.to(dt)).to(x.dtype)
        elif self.mode == "zero":
            new = torch.zeros_like(x)
        else:
            raise ValueError(self.mode)
        return (new,) + tuple(output[1:]) if is_tuple else new

    def __enter__(self):
        if self.mode != "orig":
            self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()


def masked_ce(logits, input_ids, attn):
    """Next-token CE over real (non-pad) targets only. For packed batches attn is all-ones,
    making this identical to a plain flattened cross-entropy."""
    sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    lab = input_ids[:, 1:].reshape(-1)
    v = attn[:, 1:].reshape(-1).bool()
    return F.cross_entropy(sl[v], lab[v])


@torch.no_grad()
def process_batch(model, sae, layer, id_lists, pad_id):
    """Run the 3 forward passes for one batch and return per-batch metrics (incl. L0)."""
    input_ids, attn = pad_batch(id_lists, pad_id)
    layer_module = get_decoder_layers(model)[layer]

    cap = {}
    h = layer_module.register_forward_hook(
        lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
    l_orig = masked_ce(model(input_ids, attention_mask=attn).logits, input_ids, attn).item()
    h.remove()
    with ResidPostIntervention(model, layer, sae, "recon"):
        l_rec = masked_ce(model(input_ids, attention_mask=attn).logits, input_ids, attn).item()
    with ResidPostIntervention(model, layer, sae, "zero"):
        l_zero = masked_ce(model(input_ids, attention_mask=attn).logits, input_ids, attn).item()
    frac = (l_rec - l_zero) / (l_orig - l_zero)

    feats = sae.encode(cap["x"].to(torch.float32))
    valid = attn.bool().reshape(-1)
    f_flat = feats.reshape(-1, sae.dict_size)[valid]
    l0 = (f_flat != 0).float().sum(dim=-1).mean().item()
    return {"loss_original": l_orig, "loss_reconstructed": l_rec, "loss_zero": l_zero,
            "frac_recovered": frac, "l0": l0, "n_seqs": len(id_lists)}


def run_core(cfg: CoreConfig, local_model: str, local_sae_dir: str, arch: str,
             workdir: str, max_seconds: float = 28.0, verbose: bool = True):
    """Resumable Core eval. Returns (done: bool, ckpt_path: str, n_batches: int).

    Writes one JSONL line per completed batch to workdir/raw.jsonl; deterministic and independent
    of how batches are split across invocations (fixed pre-tokenized pool).
    """
    os.makedirs(workdir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    n_seqs = cfg.n_reconstruction_seqs
    n_batches = (n_seqs + cfg.batch_size_prompts - 1) // cfg.batch_size_prompts

    _, tok = _tokenizer_only(local_model)
    pool_path = os.path.join(workdir, f"pool_{cfg.tokenize_mode}_n{n_seqs}_c{cfg.context_size}.pt")
    pool = prepare_pool(cfg, tok, n_seqs, pool_path)
    n_batches = (len(pool) + cfg.batch_size_prompts - 1) // cfg.batch_size_prompts

    ckpt = os.path.join(workdir, "raw.jsonl")
    done = load_ckpt(ckpt)
    todo = [bi for bi in range(n_batches) if bi not in done]
    if verbose:
        print(f"[core] mode={cfg.tokenize_mode} done={len(done)}/{n_batches} todo={len(todo)}", flush=True)
    if not todo:
        return True, ckpt, n_batches

    model, tok = load_model_and_tokenizer(local_model, device=cfg.device)
    sae, _, _ = load_local_sae(local_sae_dir, arch, device=cfg.device)
    pad_id = tok.pad_token_id
    bs = cfg.batch_size_prompts

    t0, processed = time.time(), 0
    for bi in todo:
        if time.time() - t0 > max_seconds and processed > 0:
            break
        rec = process_batch(model, sae, cfg.layer, pool[bi * bs:(bi + 1) * bs], pad_id)
        rec["bi"] = bi
        append_ckpt(ckpt, rec)
        processed += 1
        if verbose:
            cur = load_ckpt(ckpt); rows = list(cur.values())
            rh = sum(r["loss_original"] for r in rows) / len(rows)
            rf = sum(r["frac_recovered"] for r in rows) / len(rows)
            print(f"  bi={bi} Horig={rec['loss_original']:.3f} H0={rec['loss_zero']:.3f} "
                  f"frac={rec['frac_recovered']:.4f} L0={rec['l0']:.1f} | "
                  f"running[{len(rows)}] Horig={rh:.3f} frac={rf:.4f} ({time.time()-t0:.0f}s)", flush=True)

    done = load_ckpt(ckpt)
    return len(done) >= n_batches, ckpt, n_batches


def _tokenizer_only(model_src):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return None, tok


# =========================================================================== #
# EXACT SAEBench core/main.py methodology (sae_bench/evals/core/main.py)
#
# Verified against the source (commit fetched 2026-06-22). Key points, with the
# SAEBench function each mirrors:
#   - SAE forward            : sae.decode(sae.encode(act))            [get_recons_loss]
#   - zero ablation          : torch.zeros_like(resid_post)          [standard_zero_ablate_hook]
#   - reconstruction sites   : ALL positions by default; special tokens kept only if
#                              exclude_special_tokens_from_reconstruction=True  [standard_replacement_hook]
#   - per-token CE           : transformer_lens loss_per_token (log_softmax + gather)
#   - CE reduction           : EXCLUDE ignore_tokens {bos,eos,pad} via mask[:, :-1],
#                              concat over batches, then .mean()      [get_downstream_reconstruction_metrics]
#   - ce_loss_score          : (ce_abl - ce_sae) / (ce_abl - ce_orig)
#   - L0                      : (acts != 0).sum(-1), EXCLUDE special tokens, .mean()  [get_sparsity_and_variance_metrics]
#   - model                  : HookedTransformer.from_pretrained_no_processing → raw activations
#                              identical to the HF residual stream (so this HF impl matches).
# The ignore_tokens set is {pad,eos,bos} in the released run (main.py line ~1101).
# =========================================================================== #

def per_token_ce(logits, tokens):
    """transformer_lens loss_per_token equivalent → shape [B, S-1]."""
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    ll = logp.gather(-1, tokens[:, 1:, None])[..., 0]
    return -ll


def not_special_mask(tokens, special_ids):
    """Boolean mask, True where the token is NOT a special token (bos/eos/pad)."""
    m = torch.ones_like(tokens, dtype=torch.bool)
    for tid in special_ids:
        if tid is not None:
            m &= tokens != tid
    return m


@torch.no_grad()
def saebench_recons_per_token(model, sae, layer, batch_tokens, special_ids,
                              exclude_special_from_recon=False):
    """Replicates get_recons_loss: returns per-token CE for orig / sae / zero-ablation [B,S-1]."""
    layer_module = get_decoder_layers(model)[layer]
    ce_orig = per_token_ce(model(batch_tokens).logits, batch_tokens)

    rmask = (not_special_mask(batch_tokens, special_ids) if exclude_special_from_recon
             else torch.ones_like(batch_tokens, dtype=torch.bool))

    def recon_hook(m, i, o):
        is_t = isinstance(o, tuple); x = o[0] if is_t else o
        xhat = sae.decode(sae.encode(x.to(torch.float32))).to(x.dtype)
        xhat = torch.where(rmask[..., None], xhat, x)
        return (xhat,) + tuple(o[1:]) if is_t else xhat

    h = layer_module.register_forward_hook(recon_hook)
    ce_sae = per_token_ce(model(batch_tokens).logits, batch_tokens); h.remove()

    def zero_hook(m, i, o):
        is_t = isinstance(o, tuple); x = o[0] if is_t else o
        z = torch.zeros_like(x)
        return (z,) + tuple(o[1:]) if is_t else z

    h = layer_module.register_forward_hook(zero_hook)
    ce_abl = per_token_ce(model(batch_tokens).logits, batch_tokens); h.remove()
    return ce_orig, ce_sae, ce_abl


@torch.no_grad()
def saebench_core_eval(model, sae, layer, token_batches, special_ids,
                       exclude_special_from_recon=False):
    """Exact core/main.py reconstruction + sparsity metrics over a list of token batches.

    token_batches: list of LongTensor [B, S] (e.g. packed, BOS-prefixed contexts).
    special_ids:   iterable of {bos,eos,pad} token ids to exclude from CE mean and L0.
    """
    layer_module = get_decoder_layers(model)[layer]
    co, cs, ca, l0s = [], [], [], []
    for batch_tokens in token_batches:
        ceo, ces, cea = saebench_recons_per_token(
            model, sae, layer, batch_tokens, special_ids, exclude_special_from_recon)
        m = not_special_mask(batch_tokens, special_ids)[:, :-1]   # trim to per-token-loss length
        co.append(ceo[m]); cs.append(ces[m]); ca.append(cea[m])
        # L0 over non-special tokens
        cap = {}
        h = layer_module.register_forward_hook(
            lambda mod, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(batch_tokens); h.remove()
        feats = sae.encode(cap["x"].to(torch.float32)).reshape(-1, sae.dict_size)
        fm = not_special_mask(batch_tokens, special_ids).reshape(-1)
        l0s.append((feats[fm] != 0).sum(dim=-1).float())
    ce_without = torch.cat(co).mean().item()
    ce_with = torch.cat(cs).mean().item()
    ce_abl = torch.cat(ca).mean().item()
    score = (ce_abl - ce_with) / (ce_abl - ce_without)
    return {"loss_recovered": score, "ce_loss_score": score,
            "ce_loss_without_sae": ce_without, "ce_loss_with_sae": ce_with,
            "ce_loss_with_ablation": ce_abl, "l0": torch.cat(l0s).mean().item()}
