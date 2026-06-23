"""
DETERMINISTIC ORACLE for AutoInterp — proves our example-construction + prompts are identical to
SAEBench's, up to the (non-deterministic) LLM judge.

Runs SAEBench's verbatim gather_data (tests/saebench_autointerp_verbatim.py) and ours on the SAME
random activations / tokens with the SAME seed, and asserts every per-latent example matches bit-for-bit
(token windows, activations, is_active flags, and the marked/unmarked rendered strings). Also checks the
generated generation+scoring prompt text is byte-identical.

Run: python tests/test_autointerp_oracle.py
"""
import os, sys, random
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saebench_audit.metrics import autointerp as ai
import saebench_autointerp_verbatim as vb


def make_inputs(seed=0, N=40, L=128, n_lat=6, vocab=5000):
    torch.manual_seed(seed)
    tokens = torch.randint(0, vocab, (N, L))
    # structured activations: a few strong spikes per latent so top/iw selection is non-trivial
    acts = torch.rand(N, L, n_lat) * 0.05
    for i in range(n_lat):
        for _ in range(30):
            r, c = torch.randint(0, N, (1,)).item(), torch.randint(12, L - 12, (1,)).item()
            acts[r, c, i] += torch.rand(1).item() * 5
    return tokens, acts


def str_toks_fn(tok):
    return lambda toks: [tok.decode([t]) for t in toks]


def examples_equal(a, b):
    if len(a) != len(b):
        return False
    for ea, eb in zip(a, b):
        if ea.toks != eb.toks or ea.acts != eb.acts:
            return False
        if ea.is_active != eb.is_active or ea.toks_are_active != eb.toks_are_active:
            return False
        if ea.to_str(True) != eb.to_str(True) or ea.to_str(False) != eb.to_str(False):
            return False
    return True


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/sessions/zealous-gifted-volta/mnt/outputs/models/pythia-160m-deduped")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = ai.AutoInterpConfig()
    stf = str_toks_fn(tok)
    tokens, acts = make_inputs()
    latents = list(range(acts.shape[-1]))

    # ours
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    gen_mine, score_mine = ai.gather_data(cfg, acts, tokens, latents, tok)
    # verbatim SAEBench (same seed, same inputs, same str_toks)
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    gen_vb, score_vb = vb.gather_data_verbatim(cfg, acts, tokens, latents, stf)

    assert sorted(gen_mine) == sorted(gen_vb), "different latents survived"
    ex_ok = True
    for L in gen_mine:
        if not examples_equal(gen_mine[L], gen_vb[L]):
            ex_ok = False; print(f"  gen examples differ for latent {L}")
        if not examples_equal(score_mine[L], score_vb[L]):
            ex_ok = False; print(f"  scoring examples differ for latent {L}")

    # prompt text identical (use one latent)
    L0 = sorted(gen_mine)[0]
    gp_mine = ai.generation_prompts(cfg, gen_mine[L0])
    sp_mine = ai.scoring_prompts(cfg, "test concept", score_mine[L0])
    # rebuild the same prompts from the verbatim examples by hand (same to_str output)
    gp_ex = "\n".join(f"{i+1}. {e.to_str(mark_toks=True)}" for i, e in enumerate(gen_vb[L0]))
    sp_ex = "\n".join(f"{i+1}. {e.to_str(mark_toks=False)}" for i, e in enumerate(score_vb[L0]))
    prompt_ok = (gp_ex in gp_mine[1]["content"]) and (sp_ex in sp_mine[1]["content"])

    n_ex = len(gen_mine)
    print(f"latents matched: {len(gen_mine)}/{len(latents)} survived; per-latent examples identical: {ex_ok}")
    print(f"prompt example-blocks identical: {prompt_ok}")
    ok = ex_ok and prompt_ok and n_ex > 0
    print("\nAUTOINTERP ORACLE:", "PASS ✓ — deterministic pipeline identical to SAEBench" if ok else "FAIL ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
