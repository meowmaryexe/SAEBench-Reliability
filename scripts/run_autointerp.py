"""
Run the AutoInterp eval on one SAE with the gpt-4o-mini judge (faithful to SAEBench autointerp).

Stages (all resumable across short calls):
  0. residual cache (SAE-INDEPENDENT, shared across SAEs): tokenize the Pile -> cache resid_post
     activations per batch (bf16). This is the expensive model-forward step; reused for every SAE.
  A. per-SAE example build (cheap): SAE.encode(cached resid) -> sparsity -> select non-dead latents ->
     per-latent example sets (top/iw/random). Cached.
  B. judge loop (resumable): gpt-4o-mini writes an explanation then predicts which sequences activate;
     detection accuracy is scored. One line per latent in <sae_workdir>/scores.jsonl.

CPU note: n_tokens / n_latents are reduced from the paper's 2M / 1000 for CPU feasibility; the
methodology is identical and scales on GPU. Judge = the paper's exact gpt-4o-mini.
"""
import argparse, json, os, pickle, sys, time
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import autointerp as ai
from saebench_audit.metrics.core_loss_recovered import get_decoder_layers
from saebench_audit.sae_models import load_sae


@torch.no_grad()
def build_resid_cache(args, cfg, tok, resid_dir, max_seconds):
    """Resumable: cache resid_post[batch] (bf16) + tokens. Returns True when complete."""
    os.makedirs(resid_dir, exist_ok=True)
    tokens_path = os.path.join(resid_dir, "tokens.pt")
    if os.path.exists(tokens_path):
        tokens = torch.load(tokens_path)
    else:
        print("[0] tokenizing dataset ...", flush=True)
        tokens = ai.load_and_tokenize_dataset(cfg.dataset_name, cfg.llm_context_size, args.n_tokens, tok)
        torch.save(tokens, tokens_path)
    bs = cfg.llm_batch_size
    n_batches = (tokens.shape[0] + bs - 1) // bs
    todo = [b for b in range(n_batches) if not os.path.exists(os.path.join(resid_dir, f"resid_{b}.pt"))]
    if not todo:
        return True, tokens, n_batches
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.local_model, dtype=torch.float32).eval()
    lm = get_decoder_layers(model)[cfg.layer]
    t0 = time.time(); done = 0
    for b in todo:
        if time.time() - t0 > max_seconds and done > 0:
            break
        bt = tokens[b * bs:(b + 1) * bs]
        cap = {}
        h = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        torch.save(cap["x"].to(torch.bfloat16), os.path.join(resid_dir, f"resid_{b}.pt"))
        done += 1
    n_cached = sum(os.path.exists(os.path.join(resid_dir, f"resid_{b}.pt")) for b in range(n_batches))
    print(f"[0] resid cache {n_cached}/{n_batches} batches", flush=True)
    return n_cached >= n_batches, tokens, n_batches


@torch.no_grad()
def build_examples(args, cfg, tok, resid_dir, tokens, n_batches, sae, ex_path):
    if os.path.exists(ex_path):
        return pickle.load(open(ex_path, "rb"))
    import random
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    # SAE.encode over cached resid -> full acts; sparsity; select latents
    running = torch.zeros(sae.dict_size, dtype=torch.float64); total = 0
    per_batch_full = []
    for b in range(n_batches):
        resid = torch.load(os.path.join(resid_dir, f"resid_{b}.pt")).to(torch.float32)
        bt = tokens[b * cfg.llm_batch_size:(b + 1) * cfg.llm_batch_size]
        f = sae.encode(resid)
        km = ai.keep_mask(bt, tok)
        f = f * km[:, :, None]
        running += (f > 0).reshape(-1, sae.dict_size).double().sum(0); total += int(km.sum().item())
        per_batch_full.append(f.to(torch.bfloat16))
    sparsity = (running / max(total, 1)).float()
    latents = ai.select_latents(cfg, sparsity)
    print(f"[A] {len(latents)} non-dead latents (of {sae.dict_size}); building examples", flush=True)
    acts_sel = torch.cat([f[:, :, latents] for f in per_batch_full], dim=0).float()
    del per_batch_full
    gen, score = ai.gather_data(cfg, acts_sel, tokens, latents, tok)
    payload = {"latents": sorted(gen.keys()), "gen": gen, "score": score}
    pickle.dump(payload, open(ex_path, "wb"))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_model", required=True)
    ap.add_argument("--sae_dir", required=True)
    ap.add_argument("--arch", default="standard")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n_tokens", type=int, default=96000)
    ap.add_argument("--n_latents", type=int, default=40)
    ap.add_argument("--resid_dir", required=True, help="shared resid cache (SAE-independent)")
    ap.add_argument("--workdir", required=True, help="per-SAE workdir")
    ap.add_argument("--keyfile", default="/sessions/zealous-gifted-volta/mnt/outputs/.openai_key")
    ap.add_argument("--max_seconds", type=float, default=33.0)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.local_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = ai.AutoInterpConfig(model_name=args.local_model, layer=args.layer,
                              total_tokens=args.n_tokens, n_latents=args.n_latents)

    done, tokens, n_batches = build_resid_cache(args, cfg, tok, args.resid_dir, args.max_seconds)
    if not done:
        print("RESID_CACHE_INCOMPLETE (re-invoke)", flush=True); return

    sae = load_sae(os.path.join(args.sae_dir, "ae.pt"), args.arch, device="cpu")
    payload = build_examples(args, cfg, tok, args.resid_dir, tokens, n_batches, sae,
                             os.path.join(args.workdir, "examples.pkl"))
    gen, score, latents = payload["gen"], payload["score"], payload["latents"]

    scores_path = os.path.join(args.workdir, "scores.jsonl")
    done_l = {json.loads(l)["latent"] for l in open(scores_path)} if os.path.exists(scores_path) else set()
    todo = [L for L in latents if L not in done_l]
    print(f"[B] judge {len(done_l)}/{len(latents)} done; {len(todo)} todo", flush=True)
    if not todo:
        agg(scores_path, args.workdir); print("ALL_LATENTS_DONE", flush=True); return

    import random
    random.seed(cfg.random_seed)
    judge = ai.openai_judge(open(args.keyfile).read().strip())
    t0 = time.time(); n = 0
    f = open(scores_path, "a")
    for L in todo:
        if time.time() - t0 > args.max_seconds and n > 0:
            break
        r = ai.run_single_latent(cfg, judge, gen[L], score[L]) or {"score": None, "explanation": None, "predictions": None, "correct_seqs": None}
        f.write(json.dumps({"latent": L, **r}) + "\n"); f.flush(); n += 1
        print(f"  latent {L}: score={None if r['score'] is None else round(r['score'],3)} \"{(r['explanation'] or '')[:50]}\"", flush=True)
    f.close()
    nd = len({json.loads(l)["latent"] for l in open(scores_path)})
    if nd >= len(latents):
        agg(scores_path, args.workdir); print("ALL_LATENTS_DONE", flush=True)
    else:
        print(f"PROGRESS {nd}/{len(latents)}", flush=True)


def agg(scores_path, workdir):
    rows = [json.loads(l) for l in open(scores_path)]
    s = [r["score"] for r in rows if r["score"] is not None]
    t = torch.tensor(s)
    out = {"autointerp_score": t.mean().item(), "autointerp_std_dev": t.std().item() if len(s) > 1 else 0.0,
           "n_latents_scored": len(s), "per_latent": rows}
    json.dump(out, open(os.path.join(workdir, "result.json"), "w"), indent=2)
    print(f"[agg] autointerp_score={out['autointerp_score']:.4f} std={out['autointerp_std_dev']:.4f} (n={len(s)})", flush=True)


if __name__ == "__main__":
    main(); sys.stdout.flush(); os._exit(0)
