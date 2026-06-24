"""
Run the FULL Core metric set (saebench_core methodology) on the 4k Pythia-160M suite, CPU, and
compare every metric to the published values (adamkarvonen/sae_bench_results_0125).

Efficiency: the orig/zero-ablation CE (recon baseline) and the layer activations (sparsity set) are
SAE-INDEPENDENT, computed once and cached; per SAE only the reconstruction forward runs, plus cheap
encode/decode on the cached activations. batch_size_prompts=16 (matches the published mse normalization).
Resumable: one line per SAE in <workdir>/full.jsonl.
"""
import argparse, glob, json, os, subprocess, sys, time
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics.core_full import (per_token_ce, not_special_mask, calculate_max_cosine_sim,
                                              sae_W_enc_W_dec, get_decoder_layers)
from saebench_audit.metrics.core_loss_recovered import ResidPostIntervention
from saebench_audit.sae_models import load_sae

REPO = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"
FOLDER = {"Standard": ("Standard", "standard"), "TopK": ("TopK", "topk"),
          "BatchTopK": ("BatchTopK", "batchtopk"), "JumpRelu": ("JumpRelu", "jumprelu"),
          "GatedSAE": ("GatedSAE", "gated"), "Matryoshka": ("MatryoshkaBatchTopK", "matryoshka"),
          "PAnneal": ("PAnneal", "standard")}
REF_DIR = "/sessions/zealous-gifted-volta/mnt/outputs/neuronpedia_core_ref"


def curl(url, dst, t=40):
    subprocess.run(["curl", "-L", "-C", "-", "-s", "-o", dst, url], timeout=t)


def packed_batches(tok, n_seqs, ctx, bs, device, cache):
    if os.path.exists(cache):
        t = torch.load(cache)
    else:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        buf, w = [], []
        for ex in ds:
            if not ex.get("text"):
                continue
            buf.extend(tok(ex["text"], add_special_tokens=False)["input_ids"]); buf.append(tok.eos_token_id)
            while len(buf) >= ctx - 1:
                w.append([bos] + buf[:ctx - 1]); buf = buf[ctx - 1:]
                if len(w) >= n_seqs:
                    break
            if len(w) >= n_seqs:
                break
        t = torch.tensor(w[:n_seqs], dtype=torch.long); torch.save(t, cache)
    return [t[i:i + bs].to(device) for i in range(0, len(t), bs)]


@torch.no_grad()
def build_baseline(model, layer, recon_batches, sparsity_batches, special, path):
    if os.path.exists(path):
        return torch.load(path)
    lm = get_decoder_layers(model)[layer]
    recon = []
    for bt in recon_batches:
        ce_o = per_token_ce(model(bt).logits, bt)
        with ResidPostIntervention(model, layer, None, "zero"):
            ce_a = per_token_ce(model(bt).logits, bt)
        recon.append({"tokens": bt, "ce_orig": ce_o, "ce_abl": ce_a})
    acts = []
    for bt in sparsity_batches:
        cap = {}
        h = lm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        acts.append({"tokens": bt, "x": cap["x"]})
    base = {"recon": recon, "acts": acts}
    torch.save(base, path)
    return base


@torch.no_grad()
def eval_sae(model, sae, layer, base, special, exclude_special=True):
    # reconstruction CE (per-SAE forward) + Loss Recovered.
    # exclude_special=True (the PUBLISHED 0125 setting): keep the ORIGINAL activation at BOS/EOS/PAD
    # positions and reconstruct only the rest (matches core/main.py standard_replacement_hook + mask).
    lm = get_decoder_layers(model)[layer]
    co, cs, ca = [], [], []
    for rb in base["recon"]:
        bt = rb["tokens"]
        rmask = not_special_mask(bt, special) if exclude_special else torch.ones_like(bt, dtype=torch.bool)

        def recon_hook(m, i, o):
            is_t = isinstance(o, tuple); x = o[0] if is_t else o
            xh = sae.decode(sae.encode(x.to(torch.float32))).to(x.dtype)
            xh = torch.where(rmask[..., None], xh, x)
            return (xh,) + tuple(o[1:]) if is_t else xh

        h = lm.register_forward_hook(recon_hook)
        ce_s = per_token_ce(model(bt).logits, bt); h.remove()
        m = not_special_mask(bt, special)[:, :-1]
        co.append(rb["ce_orig"][m]); cs.append(ce_s[m]); ca.append(rb["ce_abl"][m])
    ce_o = torch.cat(co).mean(); ce_s = torch.cat(cs).mean(); ce_a = torch.cat(ca).mean()
    lr = ((ce_a - ce_s) / (ce_a - ce_o)).item()

    # sparsity / variance / density from cached activations
    dsae = sae.dict_size
    l0s, l1s, l2i, l2o, l2r, relb, mses, evl, cos = [], [], [], [], [], [], [], [], []
    mss, mapd, msrs = [], [], []
    tot_acts = torch.zeros(dsae, dtype=torch.float64); tot_tok = 0
    for ab in base["acts"]:
        x = ab["x"].to(torch.float32); bt = ab["tokens"]
        feats = sae.encode(x); xhat = sae.decode(feats)
        fm = not_special_mask(bt, special).reshape(-1)
        xin = x.reshape(-1, x.shape[-1])[fm]; fac = feats.reshape(-1, dsae)[fm]; xo = xhat.reshape(-1, x.shape[-1])[fm]
        li = torch.norm(xin, dim=-1); lo = torch.norm(xo, dim=-1)
        lid = li.clone(); lid[lid.abs() < 1e-4] = 1
        l2i.append(li); l2o.append(lo); l2r.append(lo / lid)
        relb.append((torch.norm(xo, dim=-1).pow(2).mean() / (xin * xo).sum(-1).mean()).unsqueeze(0))
        l0s.append((fac != 0).sum(-1).float()); l1s.append(fac.sum(-1))
        rss = (xin - xo).pow(2).sum(-1); mses.append(rss / fm.sum())
        evl.append(1 - rss / (xin - xin.mean(0)).pow(2).sum(-1))
        mss.append(xin.pow(2).sum(-1).mean(0)); mapd.append(xin.pow(2).mean(0)); msrs.append(rss.mean(0))
        xn = xin / torch.norm(xin, dim=-1, keepdim=True); xhn = xo / torch.norm(xo, dim=-1, keepdim=True)
        cos.append((xn * xhn).sum(-1))
        tot_acts += (fac > 0).double().sum(0).cpu(); tot_tok += int(fm.sum().item())
    density = tot_acts / max(tot_tok, 1)
    We, Wd = sae_W_enc_W_dec(sae)
    mss_t = torch.stack(mss).mean(0); mapd_t = torch.cat(mapd).mean(0)
    return {
        "loss_recovered": lr, "ce_loss_score": lr,
        "explained_variance_new": (1 - torch.stack(msrs).mean(0) / (mss_t - mapd_t ** 2)).item(),
        "explained_variance_legacy": torch.cat(evl).mean().item(),
        "mse": torch.cat(mses).mean().item(), "cossim": torch.cat(cos).mean().item(),
        "l2_norm_in": torch.cat(l2i).mean().item(), "l2_norm_out": torch.cat(l2o).mean().item(),
        "l2_ratio": torch.cat(l2r).mean().item(),
        "relative_reconstruction_bias": torch.cat(relb).mean().item(),
        "l0": torch.cat(l0s).mean().item(), "l1": torch.cat(l1s).mean().item(),
        "frac_alive": (density > 0).float().mean().item(),
        "freq_over_1_percent": (density > 0.01).float().mean().item(),
        "freq_over_10_percent": (density > 0.1).float().mean().item(),
        "average_max_encoder_cosine_sim": calculate_max_cosine_sim(We).mean().item(),
        "average_max_decoder_cosine_sim": calculate_max_cosine_sim(Wd.T).mean().item(),
    }


def published_ref(arch, trainer):
    pat = f"{REF_DIR}/*_{arch}_pythia-160m-deduped__0108_resid_post_layer_8_trainer_{trainer}_eval_results.json"
    fs = glob.glob(pat)
    if not fs:
        return None
    d = json.load(open(fs[0]))["eval_result_metrics"]
    return {"loss_recovered": d["model_performance_preservation"]["ce_loss_score"],
            "explained_variance": d["reconstruction_quality"]["explained_variance"],
            "mse": d["reconstruction_quality"]["mse"], "cossim": d["reconstruction_quality"]["cossim"],
            "l2_ratio": d["shrinkage"]["l2_ratio"],
            "relative_reconstruction_bias": d["shrinkage"]["relative_reconstruction_bias"],
            "l2_norm_in": d["shrinkage"]["l2_norm_in"], "l2_norm_out": d["shrinkage"]["l2_norm_out"],
            "l0": d["sparsity"]["l0"], "l1": d["sparsity"]["l1"],
            "frac_alive": d["misc_metrics"]["frac_alive"],
            "freq_over_1_percent": d["misc_metrics"]["freq_over_1_percent"],
            "freq_over_10_percent": d["misc_metrics"]["freq_over_10_percent"],
            "average_max_encoder_cosine_sim": d["misc_metrics"]["average_max_encoder_cosine_sim"],
            "average_max_decoder_cosine_sim": d["misc_metrics"]["average_max_decoder_cosine_sim"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_model", required=True)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n_recon", type=int, default=64)
    ap.add_argument("--n_sparsity", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--sae_tmp", default="/sessions/zealous-gifted-volta/mnt/outputs/_full_sae_tmp")
    ap.add_argument("--archs", nargs="*", default=list(FOLDER))
    ap.add_argument("--trainers", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--max_seconds", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True); torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.local_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    special = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}

    jobs = [(a, t) for a in args.archs for t in args.trainers]
    res_path = os.path.join(args.workdir, "full.jsonl")
    done = {(r["arch"], r["trainer"]) for r in (json.loads(l) for l in open(res_path))} if os.path.exists(res_path) else set()
    todo = [j for j in jobs if j not in done]
    print(f"[full] {len(done)}/{len(jobs)} done; {len(todo)} todo", flush=True)
    if not todo:
        print("ALL_SAES_DONE", flush=True); return

    model = AutoModelForCausalLM.from_pretrained(args.local_model, dtype=torch.float32).eval()
    rb = packed_batches(tok, args.n_recon, 128, args.batch, "cpu", os.path.join(args.workdir, "recon.pt"))
    sb = packed_batches(tok, args.n_sparsity, 128, args.batch, "cpu", os.path.join(args.workdir, "spars.pt"))
    base = build_baseline(model, args.layer, rb, sb, special, os.path.join(args.workdir, "baseline.pt"))

    t0 = time.time(); n = 0
    for (arch, tr) in todo:
        if time.time() - t0 > args.max_seconds and n > 0:
            break
        folder_pascal, loader = FOLDER[arch]
        sub = f"{folder_pascal}_pythia-160m-deduped__0108/resid_post_layer_{args.layer}/trainer_{tr}"
        dst = os.path.join(args.sae_tmp, f"{arch}_t{tr}"); os.makedirs(dst, exist_ok=True)
        ae = os.path.join(dst, "ae.pt")
        for _ in range(4):
            if os.path.exists(ae) and os.path.getsize(ae) > 24_000_000:
                break
            curl(f"https://huggingface.co/{REPO}/resolve/main/{sub}/ae.pt", ae)
        sae = load_sae(ae, loader, device="cpu")
        mine = eval_sae(model, sae, args.layer, base, special)
        ref = published_ref(arch, tr)
        rec = {"arch": arch, "trainer": tr, "mine": mine, "published": ref}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        dlr = (mine["loss_recovered"] - ref["loss_recovered"]) if ref else float("nan")
        print(f"  {arch} t{tr}: LR={mine['loss_recovered']:.4f} (pub {ref['loss_recovered']:.4f} Δ{dlr:+.4f}) "
              f"L0={mine['l0']:.1f}/{ref['l0']:.1f} l1={mine['l1']:.1f}/{ref['l1']:.1f}", flush=True)
        try:
            os.remove(ae)
        except OSError:
            pass
    nd = len({(r["arch"], r["trainer"]) for r in (json.loads(l) for l in open(res_path))})
    print("ALL_SAES_DONE" if nd >= len(jobs) else f"PROGRESS {nd}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main(); sys.stdout.flush(); os._exit(0)
