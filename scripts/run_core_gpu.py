"""
GPU runner for the canonical SAEBench Core / Loss Recovered eval (methodology = saebench_core,
PROVEN identical to sae_bench/evals/core/main.py — see docs/logs/08 and tests/test_core_oracle.py).

Reads configs/gpu/core_gpu.yaml + configs/registry.yaml, evaluates a whole SAE suite (7 architectures
x 6 sparsities) at paper scale, and writes per-SAE results compared to the bundled eval_results.json.

  python scripts/run_core_gpu.py --config configs/gpu/core_gpu.yaml --suite gemma-2-2b_65k \
      --out results/processed/core_loss_recovered/gemma-2-2b_65k_saebench_core.json

NOTE: intended for an A100-class GPU; not exercised on the CPU sandbox. The compute primitives it calls
(saebench_core_eval, the SAE loaders) are covered by tests/ on CPU. Downloads each ae.pt on demand and
deletes it after eval (config runtime.download_then_delete_ae).
"""
import argparse, json, os, subprocess, sys, time
import torch
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import core_loss_recovered as core
from saebench_audit.sae_models import load_sae
from saebench_audit.io import _maybe_json


def resolve_folder(reg, suite_cfg, arch):
    conv = reg["folder_conventions"][suite_cfg["convention"]][arch]
    return conv.format(modeltag=suite_cfg.get("modeltag", ""),
                       date=suite_cfg["date"], wpow=suite_cfg["wpow"])


def curl(url, dest, timeout=600):
    subprocess.run(["curl", "-L", "-C", "-", "-s", "-o", dest, url], timeout=timeout)


def fetch_sae(repo, subpath, dst):
    os.makedirs(dst, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main/{subpath}"
    for fn in ("config.json", "eval_results.json", "ae.pt"):
        p = os.path.join(dst, fn)
        if not os.path.exists(p):
            curl(f"{base}/{fn}", p)
    return os.path.join(dst, "ae.pt")


def build_packed_batches(tok, dataset, n_seqs, ctx, batch_size, device):
    """transformer_lens ActivationsStore-style packed, BOS-prefixed contexts."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split="train", streaming=True)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    buf, windows = [], []
    for ex in ds:
        if not ex.get("text"):
            continue
        buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"]); buf.append(tok.eos_token_id)
        while len(buf) >= ctx - 1:
            windows.append([bos] + buf[:ctx - 1]); buf = buf[ctx - 1:]
            if len(windows) >= n_seqs:
                break
        if len(windows) >= n_seqs:
            break
    t = torch.tensor(windows[:n_seqs], dtype=torch.long, device=device)
    return [t[i:i + batch_size] for i in range(0, len(t), batch_size)]


@torch.no_grad()
def l0_over_batches(model, sae, layer, batches, special_ids):
    """Canonical L0 over the (larger) sparsity sample set — matches saebench_core_eval's L0 path."""
    layer_module = core.get_decoder_layers(model)[layer]
    l0s = []
    for bt in batches:
        cap = {}
        h = layer_module.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        feats = sae.encode(cap["x"].to(torch.float32)).reshape(-1, sae.dict_size)
        fm = core.not_special_mask(bt, special_ids).reshape(-1)
        l0s.append((feats[fm] != 0).sum(-1).float())
    return torch.cat(l0s).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/gpu/core_gpu.yaml"))
    ap.add_argument("--registry", default=os.path.join(ROOT, "configs/registry.yaml"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--trainers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--sae_tmp", default=os.path.join(ROOT, "_sae_tmp"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    reg = yaml.safe_load(open(args.registry))
    suite = reg["sae_suites"][args.suite]
    model_name = reg["models"][suite["model"]]["hf_repo"]
    layer = suite["layer"]
    archs = args.archs or reg["architectures"]
    ev, rt = cfg["eval"], cfg["runtime"]
    device = rt["device"]
    dtype = getattr(torch, rt["dtype"])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[gpu] loading {model_name} on {device} ({rt['dtype']}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).eval().to(device)
    special_ids = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}

    print("[gpu] building token pools (OpenWebText, packed, BOS) ...", flush=True)
    recon_batches = build_packed_batches(tok, ev["dataset"], ev["n_reconstruction_seqs"],
                                         ev["context_size"], ev["batch_size_prompts"], device)
    sparsity_batches = build_packed_batches(tok, ev["dataset"], ev["n_sparsity_seqs"],
                                            ev["context_size"], ev["batch_size_prompts"], device)

    results = []
    for arch in archs:
        folder = resolve_folder(reg, suite, arch)
        loader_arch = reg["loader_arch"][arch]
        for tr in args.trainers:
            subpath = f"{folder}/resid_post_layer_{layer}/trainer_{tr}"
            dst = os.path.join(args.sae_tmp, f"{args.suite}_{arch}_t{tr}")
            ae = fetch_sae(suite["hf_repo"], subpath, dst)
            sae = load_sae(ae, loader_arch, device=device, dtype=dtype)
            t0 = time.time()
            rec = core.saebench_core_eval(model, sae, layer, recon_batches, special_ids,
                                          exclude_special_from_recon=ev["exclude_special_tokens_from_reconstruction"])
            l0_full = l0_over_batches(model, sae, layer, sparsity_batches, special_ids)
            bundled = _maybe_json(os.path.join(dst, "eval_results.json")) or {}
            results.append({"suite": args.suite, "arch": arch, "trainer": tr,
                            "loss_recovered": rec["loss_recovered"], "l0": l0_full,
                            "ce_loss_without_sae": rec["ce_loss_without_sae"],
                            "ce_loss_with_sae": rec["ce_loss_with_sae"],
                            "ce_loss_with_ablation": rec["ce_loss_with_ablation"],
                            "bundle_frac": bundled.get("frac_recovered"),
                            "bundle_l0": bundled.get("l0"),
                            "seconds": round(time.time() - t0, 1)})
            print(f"  {arch} t{tr}: LR={rec['loss_recovered']:.4f} L0={l0_full:.1f} "
                  f"({results[-1]['seconds']}s)", flush=True)
            if rt.get("download_then_delete_ae"):
                try:
                    os.remove(ae)
                except OSError:
                    pass

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"methodology": "saebench_core", "suite": args.suite, "model": model_name,
               "layer": layer, "eval": ev, "per_sae": results}, open(args.out, "w"), indent=2)
    print(f"[gpu] wrote {args.out}  ({len(results)} SAEs)", flush=True)


if __name__ == "__main__":
    main()
