"""A/B: cached vs uncached Core path must produce the SAME numbers.

This is the direct proof that the caching is arithmetic-neutral — stronger than the oracle
for this specific change, because it compares the new fast path against the exact code the
oracle already validated, on the same tokens and the same SAEs.
"""
import os, sys, importlib.util
import torch

W = "/private/tmp/claude-501/-Users-alor-SAEBench-Reliability/febf8d82-fa49-45b0-978b-bad22bfdcec2/scratchpad/ari-worktree"
sys.path.insert(0, os.path.join(W, "src"))
spec = importlib.util.spec_from_file_location("rcg", os.path.join(W, "scripts/run_core_gpu.py"))
rcg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rcg)
from saebench_audit.metrics import core_loss_recovered as core
from saebench_audit.sae_models import load_sae

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", dtype=torch.float32).eval()
special = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}
LAYER = 8

recon = rcg.build_packed_batches(tok, "Skylion007/openwebtext", 32, 128, 8, "cpu")
spars = rcg.build_packed_batches(tok, "Skylion007/openwebtext", 48, 128, 8, "cpu")
print(f"recon: {len(recon)} batches, sparsity: {len(spars)} batches\n")

SAES = [("standard", f"{W}/_fixtures/sae_standard_4k_t0/ae.pt"),
        ("topk",     f"{W}/_fixtures/arch_saes/dir_TopK/ae.pt"),
        ("jumprelu", f"{W}/_fixtures/arch_saes/dir_JumpRelu/ae.pt"),
        ("gated",    f"{W}/_fixtures/arch_saes/dir_GatedSAE/ae.pt"),
        ("matryoshka", f"{W}/_fixtures/arch_saes/dir_Matryoshka/ae.pt")]

# SAE-independent work, computed once
baseline = core.saebench_core_baseline(model, LAYER, recon, special)
cache = "/private/tmp/claude-501/-Users-alor-SAEBench-Reliability/febf8d82-fa49-45b0-978b-bad22bfdcec2/scratchpad/_abcache.npy"
n, d = rcg.build_sparsity_cache(model, LAYER, spars, special, cache)
print(f"baseline + cache built ({n} positions x {d})\n")

allok = True
for arch, path in SAES:
    if not os.path.exists(path):
        print(f"  skip {arch}"); continue
    sae = load_sae(path, arch, device="cpu")
    old = core.saebench_core_eval(model, sae, LAYER, recon, special,
                                  exclude_special_from_recon=True)
    old_l0 = rcg.l0_over_batches(model, sae, LAYER, spars, special)
    new = core.saebench_core_eval(model, sae, LAYER, recon, special,
                                  exclude_special_from_recon=True,
                                  baseline=baseline, compute_l0=False)
    new_l0 = rcg.l0_from_cache(sae, cache, "cpu")

    keys = ["loss_recovered", "ce_loss_without_sae", "ce_loss_with_sae", "ce_loss_with_ablation"]
    same = all(old[k] == new[k] for k in keys)
    l0same = (old_l0 == new_l0)
    dl0 = abs(old_l0 - new_l0)
    print(f"  {arch:11s} LR {old['loss_recovered']:.12f} -> {new['loss_recovered']:.12f} "
          f"{'IDENTICAL' if same else 'DIFFERS'} | L0 {old_l0:.9f} -> {new_l0:.9f} "
          f"{'IDENTICAL' if l0same else f'd={dl0:.2e}'}")
    if not same or dl0 > 1e-9:
        allok = False

print("\nA/B RESULT:", "PASS — cached path is numerically identical" if allok else "FAIL")
os.remove(cache)
sys.exit(0 if allok else 1)
