"""
Reproduce Core / Loss Recovered for a whole SAE SUITE (all architectures x sparsities at one width)
efficiently on CPU.

Key optimization: H_orig (model CE) and H0 (zero-ablation CE) are SAE-INDEPENDENT, so we compute
them ONCE for a shared per-document document pool (caching the layer activations for L0 reuse), then
evaluate each SAE with a single forward pass per batch:
    Loss Recovered = (H* - H0) / (H_orig - H0)

Config matches the bundled eval_results.json path (per-document, ctx 1024, the Pile), so each SAE's
Loss Recovered + L0 are directly comparable to its released values.

Resumable: one line per SAE appended to <workdir>/suite.jsonl; re-invoke until ALL_SAES_DONE.
Downloads each ae.pt on demand and deletes it after eval to bound disk use.
"""
import argparse, json, os, subprocess, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.schema import CoreConfig
from saebench_audit.io import (prepare_pool, pad_batch, load_model_and_tokenizer, load_sae,
                               load_ckpt, append_ckpt, _maybe_json, get_decoder_layers)
from saebench_audit.metrics.core_loss_recovered import ResidPostIntervention, masked_ce

HF_BASE = "https://huggingface.co/{repo}/resolve/main"
# (folder name in the HF repo, loader arch)
ARCHS = {
    "Standard":   ("Standard_pythia-160m-deduped__0108", "standard"),
    "TopK":       ("TopK_pythia-160m-deduped__0108", "topk"),
    "BatchTopK":  ("BatchTopK_pythia-160m-deduped__0108", "batchtopk"),
    "JumpRelu":   ("JumpRelu_pythia-160m-deduped__0108", "jumprelu"),
    "GatedSAE":   ("GatedSAE_pythia-160m-deduped__0108", "gated"),
    "Matryoshka": ("MatryoshkaBatchTopK_pythia-160m-deduped__0108", "matryoshka"),
    "PAnneal":    ("PAnneal_pythia-160m-deduped__0108", "standard"),
}


def curl(url, dest, timeout=40):
    subprocess.run(["curl", "-L", "-C", "-", "-s", "-o", dest, url], timeout=timeout)


def ensure_sae_files(repo, folder, trainer, dst):
    os.makedirs(dst, exist_ok=True)
    base = HF_BASE.format(repo=repo) + f"/{folder}/resid_post_layer_8/trainer_{trainer}"
    for fn in ("config.json", "eval_results.json"):
        if not os.path.exists(os.path.join(dst, fn)):
            curl(f"{base}/{fn}", os.path.join(dst, fn), timeout=20)
    ae = os.path.join(dst, "ae.pt")
    for _ in range(4):
        if os.path.exists(ae) and os.path.getsize(ae) > 24_000_000:
            break
        curl(f"{base}/ae.pt", ae, timeout=40)
    return ae


@torch.no_grad()
def build_baseline(model, tok, cfg, pool, workdir):
    """Compute & cache per-batch H_orig, H0 and the captured layer activations (for L0)."""
    meta_path = os.path.join(workdir, "baseline.json")
    acts_path = os.path.join(workdir, "baseline_acts.pt")
    if os.path.exists(meta_path) and os.path.exists(acts_path):
        return json.load(open(meta_path)), torch.load(acts_path)
    layer_module = get_decoder_layers(model)[cfg.layer]
    bs, pad_id = cfg.batch_size_prompts, tok.pad_token_id
    horig, h0, acts = [], [], []
    n_batches = (len(pool) + bs - 1) // bs
    for bi in range(n_batches):
        ids, attn = pad_batch(pool[bi * bs:(bi + 1) * bs], pad_id)
        cap = {}
        h = layer_module.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        lo = masked_ce(model(ids, attention_mask=attn).logits, ids, attn).item()
        h.remove()
        with ResidPostIntervention(model, cfg.layer, None, "zero"):
            lz = masked_ce(model(ids, attention_mask=attn).logits, ids, attn).item()
        horig.append(lo); h0.append(lz)
        acts.append({"x": cap["x"], "attn": attn, "ids": ids})
        print(f"  [baseline] batch {bi+1}/{n_batches} Horig={lo:.3f} H0={lz:.3f}", flush=True)
    meta = {"horig": horig, "h0": h0, "n_batches": n_batches}
    json.dump(meta, open(meta_path, "w"))
    torch.save(acts, acts_path)
    return meta, acts


@torch.no_grad()
def eval_sae(model, sae, layer, baseline_meta, baseline_acts):
    fracs, l0s = [], []
    for bi, a in enumerate(baseline_acts):
        ids, attn, x = a["ids"], a["attn"], a["x"]
        with ResidPostIntervention(model, layer, sae, "recon"):
            hstar = masked_ce(model(ids, attention_mask=attn).logits, ids, attn).item()
        ho, hz = baseline_meta["horig"][bi], baseline_meta["h0"][bi]
        fracs.append((hstar - hz) / (ho - hz))
        feats = sae.encode(x.to(torch.float32))
        valid = attn.bool().reshape(-1)
        l0s.append((feats.reshape(-1, sae.dict_size)[valid] != 0).float().sum(-1).mean().item())
    return sum(fracs) / len(fracs), sum(l0s) / len(l0s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108")
    ap.add_argument("--local_model", required=True)
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--trainers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n_docs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--sae_tmp", default="/sessions/zealous-gifted-volta/mnt/outputs/_suite_sae_tmp")
    ap.add_argument("--max_seconds", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    torch.manual_seed(0)

    cfg = CoreConfig(layer=args.layer, dataset="monology/pile-uncopyrighted", context_size=1024,
                     prepend_bos=False, tokenize_mode="per_document",
                     batch_size_prompts=args.batch, n_reconstruction_seqs=args.n_docs)

    # job list
    jobs = [(a, t) for a in args.archs for t in args.trainers]
    results_path = os.path.join(args.workdir, "suite.jsonl")
    done = {(r["arch"], r["trainer"]) for r in
            (json.loads(l) for l in open(results_path)) } if os.path.exists(results_path) else set()
    todo = [(a, t) for (a, t) in jobs if (a, t) not in done]
    print(f"[suite] {len(done)}/{len(jobs)} done; {len(todo)} todo", flush=True)
    if not todo:
        print("ALL_SAES_DONE", flush=True); return

    model, tok = load_model_and_tokenizer(args.local_model, device="cpu")
    pool = prepare_pool(cfg, tok, args.n_docs, os.path.join(args.workdir, f"pool_n{args.n_docs}.pt"))
    baseline_meta, baseline_acts = build_baseline(model, tok, cfg, pool, args.workdir)

    t0 = 0.0
    import time as _t; t0 = _t.time(); processed = 0
    for (arch, trainer) in todo:
        if _t.time() - t0 > args.max_seconds and processed > 0:
            break
        folder, loader_arch = ARCHS[arch]
        dst = os.path.join(args.sae_tmp, f"{arch}_t{trainer}")
        ae = ensure_sae_files(args.repo, folder, trainer, dst)
        if not (os.path.exists(ae) and os.path.getsize(ae) > 24_000_000):
            print(f"  [skip] {arch} t{trainer}: ae.pt not fully downloaded", flush=True); continue
        sae = load_sae(ae, loader_arch, device="cpu")
        frac, l0 = eval_sae(model, sae, args.layer, baseline_meta, baseline_acts)
        bundled = _maybe_json(os.path.join(dst, "eval_results.json")) or {}
        rec = {"arch": arch, "trainer": trainer, "loader_arch": loader_arch,
               "loss_recovered": frac, "l0": l0,
               "bundle_frac": bundled.get("frac_recovered"), "bundle_l0": bundled.get("l0")}
        append_ckpt(results_path, rec)
        processed += 1
        df = (frac - rec["bundle_frac"]) if rec["bundle_frac"] else float("nan")
        print(f"  {arch} t{trainer}: LR={frac:.4f} (bundle {rec['bundle_frac']:.4f}, Δ{df:+.4f}) "
              f"L0={l0:.1f} (bundle {rec['bundle_l0']:.1f})", flush=True)
        try:
            os.remove(ae)   # free disk
        except OSError:
            pass

    n_done = len({(r["arch"], r["trainer"]) for r in (json.loads(l) for l in open(results_path))})
    print("ALL_SAES_DONE" if n_done >= len(jobs) else f"PROGRESS {n_done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
