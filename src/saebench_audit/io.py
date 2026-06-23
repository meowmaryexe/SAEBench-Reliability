"""
I/O and data utilities shared across metrics: model/SAE loading, dataset tokenization into
evaluation windows, and JSONL checkpoint read/write for resumable runs.
"""
from __future__ import annotations
import json, os
import torch

from .schema import CoreConfig
from .sae_models import load_sae


# ---------------------------------------------------------------------------
# Model-type-aware access to the transformer block whose OUTPUT is resid_post.
# ---------------------------------------------------------------------------
def get_decoder_layers(model):
    if hasattr(model, "gpt_neox"):                                   # Pythia / GPT-NeoX
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):   # Gemma / Llama / Qwen
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2
        return model.transformer.h
    raise ValueError(f"Unknown model architecture for {type(model)}")


def load_model_and_tokenizer(model_src, dtype=torch.float32, device="cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_src, dtype=dtype).eval().to(device)
    return model, tok


def load_local_sae(sae_dir, arch, device="cpu", dtype=torch.float32):
    """Load ae.pt from a local SAE folder; also return bundled config/eval_results if present."""
    ae_path = os.path.join(sae_dir, "ae.pt")
    sae = load_sae(ae_path, arch, device=device, dtype=dtype)
    bundled_config = _maybe_json(os.path.join(sae_dir, "config.json"))
    bundled_eval = _maybe_json(os.path.join(sae_dir, "eval_results.json"))
    return sae, bundled_config, bundled_eval


def _maybe_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Tokenization into evaluation windows (packed or per-document).
# ---------------------------------------------------------------------------
def iter_token_windows(cfg: CoreConfig, tokenizer, n_seqs: int):
    """Yield up to n_seqs token-id lists. Packed → fixed length; per_document → variable length."""
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, split=cfg.dataset_split, streaming=True)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    produced = 0
    eff_len = cfg.context_size - (1 if cfg.prepend_bos else 0)

    if cfg.tokenize_mode == "per_document":
        for ex in ds:
            text = ex[cfg.text_field]
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:eff_len]
            if len(ids) < 2:
                continue
            yield ([bos] + ids) if cfg.prepend_bos else ids
            produced += 1
            if produced >= n_seqs:
                return
        return

    # "packed": concatenate EOS-separated documents into fixed-length windows.
    buf = []
    for ex in ds:
        text = ex[cfg.text_field]
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buf.append(tokenizer.eos_token_id)   # doc separator
        while len(buf) >= eff_len:
            window, buf = buf[:eff_len], buf[eff_len:]
            yield ([bos] + window) if cfg.prepend_bos else window
            produced += 1
            if produced >= n_seqs:
                return


def prepare_pool(cfg: CoreConfig, tokenizer, n_seqs: int, path: str):
    """Deterministically build and cache a fixed pool of token-id windows (list of lists)."""
    if os.path.exists(path):
        return torch.load(path)
    pool = list(iter_token_windows(cfg, tokenizer, n_seqs))
    torch.save(pool, path)
    return pool


def pad_batch(id_lists, pad_id):
    """Right-pad a batch of token-id lists; return (input_ids, attention_mask)."""
    L = max(len(x) for x in id_lists)
    input_ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(id_lists), L), dtype=torch.long)
    for i, x in enumerate(id_lists):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        attn[i, :len(x)] = 1
    return input_ids, attn


# ---------------------------------------------------------------------------
# Resumable JSONL checkpoints.
# ---------------------------------------------------------------------------
def load_ckpt(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["bi"]] = r
    return done


def append_ckpt(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2)
