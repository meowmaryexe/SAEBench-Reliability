"""
GPU suite-runner for AutoInterp — the turnkey analog of run_core_gpu.py.

Reads configs/gpu/autointerp_gpu.yaml + configs/registry.yaml, evaluates a whole SAE suite
(7 architectures x 6 sparsities) at paper scale (2M tokens / 1000 latents) with the gpt-4o-mini judge,
and writes per-SAE autointerp_score compared to the published values.

  python scripts/run_autointerp_gpu.py --config configs/gpu/autointerp_gpu.yaml \
      --suite pythia-160m_4k --device cuda --judge_workers 10 \
      --out results/processed/autointerp/pythia-160m_4k_autointerp.json

Efficiency: the residual activations are SAE-INDEPENDENT, so the model forward over the token set is done
ONCE per (model, n_tokens) and cached (bf16); every SAE reuses it (encode is cheap). The LLM judge calls
are run concurrently across latents (ThreadPoolExecutor), like SAEBench. Resumable at the resid-cache and
per-SAE level. NOTE: intended for an A100-class GPU; not exercised on the CPU sandbox. The compute
primitives (gather_data, scoring, prompts) are covered by tests/ on CPU.
"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import autointerp as ai
from saebench_audit.metrics.core_loss_recovered import get_decoder_layers
from saebench_audit.sae_models import load_sae


def resolve_folder(reg, suite_cfg, arch):
    conv = reg["folder_conventions"][suite_cfg["convention"]][arch]
    return conv.format(modeltag=suite_cfg.get("modeltag", ""), date=suite_cfg["date"], wpow=suite_cfg["wpow"])


def curl(url, dst, timeout=600):
    subprocess.run(["curl", "-L", "-C", "-", "-s", "-o", dst, url], timeout=timeout)


def fetch_ae(repo, subpath, dst):
    os.makedirs(dst, exist_ok=True)
    ae = os.path.join(dst, "ae.pt")
    if not (os.path.exists(ae) and os.path.getsize(ae) > 1_000_000):
        curl(f"https://huggingface.co/{repo}/resolve/main/{subpath}/ae.pt", ae)
    return ae


def published_score(suite_repo_name, arch_folder, layer, trainer):
    """Look up the published autointerp_score from a local mirror of sae_bench_results_0125 if present."""
    base = os.environ.get("AUTOINTERP_REF_DIR", "")
    if not base:
        return None
    pat = os.path.join(base, f"*_{arch_folder.split('_')[0]}_*trainer_{trainer}_eval_results.json")
    fs = glob.glob(pat)
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return d.get("eval_result_metrics", {}).get("autointerp", {}).get("autointerp_score")


@torch.no_grad()
def build_resid_cache(args, cfg, tok, model, layer, resid_dir, device):
    os.makedirs(resid_dir, exist_ok=True)
    tokens_path = os.path.join(resid_dir, "tokens.pt")
    if os.path.exists(tokens_path):
        tokens = torch.load(tokens_path)
    else:
        print("[resid] tokenizing ...", flush=True)
        tokens = ai.load_and_tokenize_dataset(cfg.dataset_name, cfg.llm_context_size, cfg.total_tokens, tok)
        torch.save(tokens, tokens_path)
    bs = cfg.llm_batch_size
    n_batches = (tokens.shape[0] + bs - 1) // bs
    lm = get_decoder_layers(model)[layer]
    for b in range(n_batches):
        p = os.path.join(resid_dir, f"resid_{b}.pt")
        if os.path.exists(p):
            continue
        bt = tokens[b * bs:(b + 1) * bs].to(device)
        cap = {}
        h = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        torch.save(cap["x"].to(torch.bfloat16).cpu(), p)
    return tokens, n_batches


@torch.no_grad()
def build_examples(cfg, tok, resid_dir, tokens, n_batches, sae, device):
    running = torch.zeros(sae.dict_size, dtype=torch.float64); total = 0
    per_batch = []
    for b in range(n_batches):
        resid = torch.load(os.path.join(resid_dir, f"resid_{b}.pt")).to(device).float()
        bt = tokens[b * cfg.llm_batch_size:(b + 1) * cfg.llm_batch_size].to(device)
        f = sae.encode(resid)
        km = ai.keep_mask(bt, tok)
        f = f * km[:, :, None]
        running += (f > 0).reshape(-1, sae.dict_size).double().sum(0).cpu(); total += int(km.sum().item())
        per_batch.append(f.to(torch.bfloat16).cpu())
    sparsity = (running / max(total, 1)).float()
    latents = ai.select_latents(cfg, sparsity)
    acts_sel = torch.cat([f[:, :, latents] for f in per_batch], dim=0).float()   # on CPU for gather
    del per_batch
    gen, score = ai.gather_data(cfg, acts_sel, tokens.cpu(), latents, tok)
    return sorted(gen.keys()), gen, score


def judge_suite_sae(cfg, judge, gen, score, latents, workers):
    """Run gen+scoring judge calls concurrently across latents (like SAEBench's ThreadPoolExecutor)."""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(ai.run_single_latent, cfg, judge, gen[L], score[L]): L for L in latents}
        for fut in as_completed(futs):
            L = futs[fut]
            r = fut.result()
            if r is not None:
                results[L] = {"latent": L, **r}
    scores = [r["score"] for r in results.values() if r["score"] is not None]
    t = torch.tensor(scores)
    return {"autointerp_score": t.mean().item() if scores else float("nan"),
            "autointerp_std_dev": t.std().item() if len(scores) > 1 else 0.0,
            "n_latents_scored": len(scores), "per_latent": list(results.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/gpu/autointerp_gpu.yaml"))
    ap.add_argument("--registry", default=os.path.join(ROOT, "configs/registry.yaml"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--trainers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--judge_workers", type=int, default=10)
    ap.add_argument("--keyfile", default=os.path.join(ROOT, "openai_api_key.txt"))
    ap.add_argument("--resid_dir", default=None)
    ap.add_argument("--sae_tmp", default=os.path.join(ROOT, "_ai_sae_tmp"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gcfg = yaml.safe_load(open(args.config))
    reg = yaml.safe_load(open(args.registry))
    suite = reg["sae_suites"][args.suite]
    model_name = reg["models"][suite["model"]]["hf_repo"]
    layer = suite["layer"]
    archs = args.archs or reg["architectures"]
    ev = gcfg["eval"]
    cfg = ai.AutoInterpConfig(
        model_name=model_name, layer=layer, dataset_name=ev["dataset_name"],
        llm_context_size=ev["llm_context_size"], total_tokens=ev["total_tokens"],
        n_latents=ev["n_latents"], dead_latent_threshold=ev["dead_latent_threshold"],
        random_seed=ev["random_seed"], buffer=ev["buffer"], act_threshold_frac=ev["act_threshold_frac"],
        n_top_ex_for_generation=ev["n_top_ex_for_generation"],
        n_iw_sampled_ex_for_generation=ev["n_iw_sampled_ex_for_generation"],
        n_top_ex_for_scoring=ev["n_top_ex_for_scoring"],
        n_iw_sampled_ex_for_scoring=ev["n_iw_sampled_ex_for_scoring"],
        n_random_ex_for_scoring=ev["n_random_ex_for_scoring"],
        max_tokens_in_explanation=ev["max_tokens_in_explanation"],
        use_demos_in_explanation=ev["use_demos_in_explanation"],
        device=args.device)
    device = args.device
    dtype = getattr(torch, gcfg["runtime"]["dtype"])
    resid_dir = args.resid_dir or os.path.join(args.sae_tmp, f"resid_{args.suite}_{cfg.total_tokens}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[gpu] {model_name} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).eval().to(device)

    import random
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    tokens, n_batches = build_resid_cache(args, cfg, tok, model, layer, resid_dir, device)
    print(f"[gpu] resid cache ready ({n_batches} batches); judge workers={args.judge_workers}", flush=True)

    judge = ai.openai_judge(open(args.keyfile).read().strip())
    results = []
    for arch in archs:
        folder = resolve_folder(reg, suite, arch)
        loader_arch = reg["loader_arch"][arch]
        for tr in args.trainers:
            random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
            subpath = f"{folder}/resid_post_layer_{layer}/trainer_{tr}"
            dst = os.path.join(args.sae_tmp, f"{args.suite}_{arch}_t{tr}")
            ae = fetch_ae(suite["hf_repo"], subpath, dst)
            sae = load_sae(ae, loader_arch, device=device, dtype=dtype)
            latents, gen, score = build_examples(cfg, tok, resid_dir, tokens, n_batches, sae, device)
            t0 = time.time()
            r = judge_suite_sae(cfg, judge, gen, score, latents, args.judge_workers)
            pub = published_score(suite["hf_repo"], folder, layer, tr)
            results.append({"suite": args.suite, "arch": arch, "trainer": tr,
                            "autointerp_score": r["autointerp_score"], "autointerp_std_dev": r["autointerp_std_dev"],
                            "n_latents_scored": r["n_latents_scored"], "published": pub,
                            "seconds": round(time.time() - t0, 1)})
            d = "" if pub is None else f" (published {pub:.4f}, Δ{r['autointerp_score']-pub:+.4f})"
            print(f"  {arch} t{tr}: autointerp_score={r['autointerp_score']:.4f}{d}  "
                  f"n={r['n_latents_scored']} ({results[-1]['seconds']}s)", flush=True)
            if gcfg["runtime"].get("download_then_delete_ae", True):
                try:
                    os.remove(ae)
                except OSError:
                    pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"metric": "autointerp", "judge": "gpt-4o-mini", "suite": args.suite,
               "model": model_name, "layer": layer, "eval": ev, "per_sae": results}, open(args.out, "w"), indent=2)
    print(f"[gpu] wrote {args.out} ({len(results)} SAEs)", flush=True)


if __name__ == "__main__":
    main()
