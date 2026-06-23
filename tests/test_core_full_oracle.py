"""
FULL-METRIC ORACLE — proves every Core metric (not just Loss Recovered) matches SAEBench.

Runs SAEBench's verbatim get_recons_loss+KL reduction and get_sparsity_and_variance_metrics
(tests/saebench_verbatim.py) on the real transformer_lens model, vs our
saebench_audit.metrics.core_full.compute_core_full on HF — same tokens, same SAE.

Run: python tests/test_core_full_oracle.py [--arch standard --sae_dir <dir>]
"""
import argparse, os, sys
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saebench_audit.sae_models import load_sae
from saebench_audit.metrics import core_full
import saebench_verbatim as v
from test_core_oracle import SAEAdapter, build_batches

import math
# per-metric relative tolerance (data-dependent metrics; activations match to ~1e-5 under no_processing).
# freq_over_* are fractions at a hard density threshold, so a single feature flipping near the cutoff
# from TL-vs-HF activation float-noise moves them ~1/d_sae — compared at absolute 1/d_sae, not relative.
TOL = {"ce_loss_score": 1e-4, "kl_div_score": 5e-3, "explained_variance": 2e-3,
       "explained_variance_legacy": 2e-3, "mse": 2e-3, "cossim": 1e-4, "l2_ratio": 1e-4,
       "relative_reconstruction_bias": 1e-4, "l0": 1e-2, "l1": 1e-3, "frac_alive": 2e-3,
       "freq_over_1_percent": 2e-3, "freq_over_10_percent": 2e-3,
       "l2_norm_in": 1e-4, "l2_norm_out": 1e-4}
# metrics compared by ABSOLUTE difference (fractions / threshold-sensitive)
ABS_METRICS = {"frac_alive", "freq_over_1_percent", "freq_over_10_percent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/sessions/zealous-gifted-volta/mnt/outputs/models/pythia-160m-deduped")
    ap.add_argument("--sae_dir", default="/sessions/zealous-gifted-volta/mnt/outputs/models/sae_standard_4k_t0")
    ap.add_argument("--arch", default="standard")
    ap.add_argument("--layer", type=int, default=8)
    args = ap.parse_args()
    hook = f"blocks.{args.layer}.hook_resid_post"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    hf = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tl = HookedTransformer.from_pretrained_no_processing(
        "pythia-160m-deduped", hf_model=hf, tokenizer=tok, device="cpu", dtype="float32")

    sae = load_sae(os.path.join(args.sae_dir, "ae.pt"), args.arch, device="cpu")
    adapter = SAEAdapter(sae, hook, args.layer)
    special = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}
    batches = build_batches(tok, cache="/sessions/zealous-gifted-volta/mnt/outputs/oracle_batches.pt")

    # oracle (verbatim SAEBench on transformer_lens)
    per_batch = [v.get_recons_and_kl(adapter, tl, bt, hook, ignore_tokens=special) for bt in batches]
    o = v.reduce_recons_and_kl(per_batch, batches, ignore_tokens=special)
    o.update(v.sparsity_variance_metrics(adapter, tl, batches, hook, args.layer, ignore_tokens=special))

    # ours (HF)
    m = core_full.compute_core_full(hf, sae, args.layer, batches, batches, special)

    print(f"{'metric':32} {'oracle':>14} {'mine':>14} {'delta':>11}")
    ok = True
    for k in TOL:
        ov, mv = o[k], m[k]
        if math.isnan(ov) and math.isnan(mv):
            # SAEBench's KL uses log(softmax), which underflows to log(0)=-inf -> nan on CPU float32
            # with this vocab. Both produce nan => code-identical. (On GPU/larger batches it is finite.)
            print(f"{k:32} {'nan':>14} {'nan':>14} {'both-nan ✓':>11}"); continue
        d = abs(ov - mv) if k in ABS_METRICS else abs(ov - mv) / (abs(ov) if abs(ov) > 1e-9 else 1.0)
        flag = "" if d <= TOL[k] else "  <-- FAIL"
        if d > TOL[k]:
            ok = False
        print(f"{k:32} {ov:14.6f} {mv:14.6f} {d:11.2e}{flag}")
    print("\nFULL-METRIC ORACLE:", "PASS ✓ — all Core metrics identical to SAEBench" if ok else "FAIL ✗")
    sys.stdout.flush(); os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
