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

Checkpointed per SAE: each completed SAE is appended to <out>.progress.jsonl and --out is rewritten,
so re-running the same command resumes instead of redoing the suite. Before any download or GPU work
it HEADs every remaining SAE URL and aborts on a bad path.
"""
import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
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


MIN_AE_BYTES = 1_000_000        # a released ae.pt is >= tens of MB; anything smaller is an error body


def curl(url, dest, timeout=600):
    """Download url -> dest. Returns True on success, False on any HTTP/transport error.

    -f      : fail on HTTP >= 400 rather than writing the error page into dest (a 404 body
              used to land in ae.pt and later die unpickling HTML).
    no -C - : a partial or poisoned file must be re-fetched from scratch, never resumed.
    """
    r = subprocess.run(["curl", "-f", "-L", "-s", "-o", dest, url], timeout=timeout)
    if r.returncode != 0:
        if os.path.exists(dest):
            os.remove(dest)         # never leave a partial/error body behind for the next run
        return False
    return True


def fetch_sae(repo, subpath, dst):
    os.makedirs(dst, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main/{subpath}"
    ae = os.path.join(dst, "ae.pt")
    # An undersized ae.pt is a leftover error page / truncated download from an earlier run.
    # The old `if not os.path.exists(p)` guard made such a file permanently sticky.
    if os.path.exists(ae) and os.path.getsize(ae) < MIN_AE_BYTES:
        os.remove(ae)
    for fn in ("config.json", "ae.pt"):
        p = os.path.join(dst, fn)
        if not os.path.exists(p) and not curl(f"{base}/{fn}", p):
            raise RuntimeError(f"download failed: {base}/{fn}")
    # 42 of the 252 released SAEs ship no eval_results.json — best-effort only
    # (the caller already treats a missing bundle as {}).
    p = os.path.join(dst, "eval_results.json")
    if not os.path.exists(p):
        curl(f"{base}/eval_results.json", p)
    if os.path.getsize(ae) < MIN_AE_BYTES:
        raise RuntimeError(f"ae.pt is only {os.path.getsize(ae)} bytes: {base}/ae.pt")
    return ae


def trainers_for(reg, suite_name, arch, cli_trainers):
    """Trainer indices for one (suite, arch). CLI wins; else registry override; else 0-5.

    The released suites are not uniform: gemma-2-2b_16k batch_top_k is published as
    trainer_6..11 while every other (suite, arch) uses trainer_0..5.
    """
    if cli_trainers:
        return list(cli_trainers)
    ov = (reg.get("trainer_overrides") or {}).get(suite_name) or {}
    return list(ov.get(arch, reg.get("default_trainers", [0, 1, 2, 3, 4, 5])))


def _head_status(url, timeout=30):
    """True = present, False = definitively absent (HTTP 4xx), None = indeterminate."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as e:
        return False if 400 <= e.code < 500 else None
    except Exception:
        return None


def preflight(repo, work, workers=8):
    """HEAD every SAE URL before any GPU work, so a bad path costs seconds not hours.

    work: list of (arch, trainer, subpath). Returns (missing, indeterminate).
    """
    urls = [(a, t, f"https://huggingface.co/{repo}/resolve/main/{sp}/ae.pt") for a, t, sp in work]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        st = list(ex.map(lambda u: _head_status(u[2]), urls))
    missing = [u for u, s in zip(urls, st) if s is False]
    unknown = [u for u, s in zip(urls, st) if s is None]
    return missing, unknown


def ckpt_fingerprint(suite_name, model_name, layer, ev, rt):
    """Identifies the run config; resuming across a changed config would mix incomparable rows."""
    return {"suite": suite_name, "model": model_name, "layer": layer,
            "eval": ev, "dtype": rt["dtype"]}


def load_ckpt_rows(path):
    """(header, rows) from the JSONL sidecar. Tolerates a truncated trailing line."""
    header, rows = None, []
    if not os.path.exists(path):
        return header, rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[ckpt] ignoring truncated trailing line in {path}", flush=True)
                continue
            if "_header" in rec:
                header = rec["_header"]
            else:
                rows.append(rec)
    return header, rows


def append_ckpt_row(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_out(path, payload):
    """Atomic rewrite, so being killed mid-write cannot truncate the result file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


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


@torch.no_grad()
def build_sparsity_cache(model, layer, batches, special_ids, path):
    """Cache resid_post(layer) at non-special positions for the sparsity pool, ONCE per suite.

    The model forward that produces these activations does not depend on the SAE, yet
    l0_over_batches re-runs it for every SAE — 2000 of the ~2800 forward passes per SAE.
    Stored float32, i.e. exactly the tensor `sae.encode(x.to(torch.float32))` consumed
    before, and in the same position order, so L0 is unchanged.
    """
    layer_module = core.get_decoder_layers(model)[layer]
    n = sum(int(core.not_special_mask(bt, special_ids).sum().item()) for bt in batches)
    d = int(model.config.hidden_size)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, d))
    off = 0
    for bt in batches:
        cap = {}
        h = layer_module.register_forward_hook(
            lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        fm = core.not_special_mask(bt, special_ids).reshape(-1)
        x = cap["x"].reshape(-1, d).to(torch.float32)[fm]
        k = int(x.shape[0])
        mm[off:off + k] = x.cpu().numpy()
        off += k
    mm.flush(); del mm
    return n, d


@torch.no_grad()
def l0_from_cache(sae, path, device, chunk_rows=4096):
    """L0 from the cached activations. torch.cat over the same positions in the same order
    as l0_over_batches, so the mean is the same value."""
    mm = np.load(path, mmap_mode="r")
    l0s = []
    for i in range(0, mm.shape[0], chunk_rows):
        x = torch.from_numpy(np.ascontiguousarray(mm[i:i + chunk_rows])).to(device)
        feats = sae.encode(x)
        l0s.append((feats != 0).sum(-1).float().cpu())
    return torch.cat(l0s).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/gpu/core_gpu.yaml"))
    ap.add_argument("--registry", default=os.path.join(ROOT, "configs/registry.yaml"))
    ap.add_argument("--suite", required=True)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--trainers", nargs="*", type=int, default=None,
                    help="override trainer indices for every arch; default = per-suite from the registry")
    ap.add_argument("--sae_tmp", default=os.path.join(ROOT, "_sae_tmp"))
    ap.add_argument("--cache_dir", default=None,
                    help="dir for the SAE-independent activation cache (use fast local NVMe). "
                         "Omit to disable caching and use the original per-SAE path.")
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

    # ---- work list, resume, preflight: all before any download or GPU work ----
    work = []                                   # (arch, trainer, subpath)
    for arch in archs:
        folder = resolve_folder(reg, suite, arch)
        for tr in trainers_for(reg, args.suite, arch, args.trainers):
            work.append((arch, tr, f"{folder}/resid_post_layer_{layer}/trainer_{tr}"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ckpt_path = os.path.splitext(args.out)[0] + ".progress.jsonl"
    fp = ckpt_fingerprint(args.suite, model_name, layer, ev, rt)
    header, done_rows = load_ckpt_rows(ckpt_path)
    if header is not None and header != fp:
        sys.exit(f"[ckpt] {ckpt_path} was written under a different config; resuming would mix "
                 f"incomparable rows.\n       Delete it to start this suite over.")
    done = {(r["arch"], r["trainer"]) for r in done_rows}
    todo = [w for w in work if (w[0], w[1]) not in done]
    print(f"[gpu] suite {args.suite}: {len(work)} SAEs total, {len(done)} already done, "
          f"{len(todo)} to do", flush=True)

    def finalize(rows):
        order = {a: i for i, a in enumerate(archs)}
        rows = sorted(rows, key=lambda r: (order.get(r["arch"], 99), r["trainer"]))
        write_out(args.out, {"methodology": "saebench_core", "suite": args.suite,
                             "model": model_name, "layer": layer, "eval": ev, "per_sae": rows})
        return rows

    if not todo:
        rows = finalize(done_rows)
        print(f"[gpu] wrote {args.out}  ({len(rows)} SAEs, all resumed from checkpoint)", flush=True)
        return

    miss, unknown = preflight(suite["hf_repo"], todo)
    if unknown:
        print(f"[preflight] {len(unknown)} URL(s) indeterminate (network?) — continuing", flush=True)
    if miss:
        print(f"[preflight] FAIL — {len(miss)} SAE path(s) not found on HuggingFace:", flush=True)
        for a, t, u in miss:
            print(f"    {a} trainer_{t}  {u}", flush=True)
        sys.exit(f"[preflight] aborting before any GPU work — fix trainer_overrides in {args.registry}")
    print(f"[preflight] OK — all {len(todo)} remaining SAE paths present", flush=True)

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

    # ---- SAE-independent work, computed once per suite instead of once per SAE ----
    baseline, cache_path = None, None
    if args.cache_dir:
        t0 = time.time()
        print("[cache] computing SAE-independent baseline CE (recon pool) ...", flush=True)
        baseline = core.saebench_core_baseline(model, layer, recon_batches, special_ids)
        cache_path = os.path.join(args.cache_dir, f"sparsity_{args.suite}_{ev['n_sparsity_seqs']}.npy")
        if os.path.exists(cache_path):
            print(f"[cache] reusing activation cache {cache_path}", flush=True)
        else:
            print("[cache] building sparsity-pool activation cache ...", flush=True)
            n, d = build_sparsity_cache(model, layer, sparsity_batches, special_ids, cache_path)
            print(f"[cache] wrote {n} x {d} float32 "
                  f"({n * d * 4 / 1e9:.1f} GB) -> {cache_path}", flush=True)
        print(f"[cache] ready in {time.time() - t0:.0f}s", flush=True)

    if header is None:
        append_ckpt_row(ckpt_path, {"_header": fp})

    results = list(done_rows)
    for arch, tr, subpath in todo:
        loader_arch = reg["loader_arch"][arch]
        dst = os.path.join(args.sae_tmp, f"{args.suite}_{arch}_t{tr}")
        ae = fetch_sae(suite["hf_repo"], subpath, dst)
        sae = load_sae(ae, loader_arch, device=device, dtype=dtype)
        t0 = time.time()
        rec = core.saebench_core_eval(model, sae, layer, recon_batches, special_ids,
                                      exclude_special_from_recon=ev["exclude_special_tokens_from_reconstruction"],
                                      baseline=baseline, compute_l0=(baseline is None))
        l0_full = (l0_from_cache(sae, cache_path, device) if cache_path
                   else l0_over_batches(model, sae, layer, sparsity_batches, special_ids))
        bundled = _maybe_json(os.path.join(dst, "eval_results.json")) or {}
        row = {"suite": args.suite, "arch": arch, "trainer": tr,
               "loss_recovered": rec["loss_recovered"], "l0": l0_full,
               "ce_loss_without_sae": rec["ce_loss_without_sae"],
               "ce_loss_with_sae": rec["ce_loss_with_sae"],
               "ce_loss_with_ablation": rec["ce_loss_with_ablation"],
               "bundle_frac": bundled.get("frac_recovered"),
               "bundle_l0": bundled.get("l0"),
               "seconds": round(time.time() - t0, 1)}
        append_ckpt_row(ckpt_path, row)     # durable before anything downstream can fail
        results.append(row)
        finalize(results)                   # keep --out current after every SAE
        print(f"  {arch} t{tr}: LR={rec['loss_recovered']:.4f} L0={l0_full:.1f} "
              f"({row['seconds']}s)", flush=True)
        if rt.get("download_then_delete_ae"):
            try:
                os.remove(ae)
            except OSError:
                pass

    rows = finalize(results)
    print(f"[gpu] wrote {args.out}  ({len(rows)} SAEs)", flush=True)


if __name__ == "__main__":
    main()
