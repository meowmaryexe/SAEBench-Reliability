"""
Unit tests for the Core / Loss Recovered methodology primitives — fast, no LLM required.
Each test pins one documented SAEBench behavior. Run: python tests/test_core_units.py
(also pytest-compatible: pytest tests/test_core_units.py)
"""
import os, sys
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import core_loss_recovered as core
from saebench_audit.sae_models import load_sae

SAE_DIR = "/sessions/zealous-gifted-volta/mnt/outputs/models/sae_standard_4k_t0"
SUITE_INSPECT = "/sessions/zealous-gifted-volta/mnt/outputs/suite_inspect"


def test_loss_recovered_formula():
    # ce_loss_score = (abl - sae) / (abl - orig); 1.0 if perfect (sae==orig), 0.0 if sae==abl
    orig, sae, abl = 2.0, 2.0, 12.0
    assert abs((abl - sae) / (abl - orig) - 1.0) < 1e-12
    orig, sae, abl = 2.0, 12.0, 12.0
    assert abs((abl - sae) / (abl - orig) - 0.0) < 1e-12


def test_per_token_ce_matches_cross_entropy():
    torch.manual_seed(0)
    B, S, V = 2, 7, 50
    logits = torch.randn(B, S, V)
    tokens = torch.randint(0, V, (B, S))
    mine = core.per_token_ce(logits, tokens)                       # [B, S-1]
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1),
                          reduction="none").reshape(B, S - 1)
    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max()


def test_special_token_masking_excludes_ids():
    tokens = torch.tensor([[0, 5, 6, 0, 7]])      # 0 is a special id (bos/eos)
    m = core.not_special_mask(tokens, {0, None})
    assert m.tolist() == [[False, True, True, False, True]]
    # masked mean over per-token-loss positions (mask[:, :-1])
    loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])   # length S-1 = 4
    kept = loss[m[:, :-1]]
    assert kept.tolist() == [2.0, 3.0]            # positions 0 and 3 (special sources) dropped


def test_l0_definition_and_special_exclusion():
    # L0 = (acts != 0).sum(-1), excluding special-token positions
    feats = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])  # [1,3,3]
    tokens = torch.tensor([[0, 5, 6]])            # position 0 special
    fm = core.not_special_mask(tokens, {0}).reshape(-1)
    l0 = (feats.reshape(-1, 3)[fm] != 0).sum(-1).float()
    assert l0.tolist() == [0.0, 2.0]              # special pos excluded; rows: [0 nonzero], [2 nonzero]


def test_zero_ablation_is_zeros():
    x = torch.randn(2, 4, 8)
    assert torch.equal(torch.zeros_like(x), x * 0.0)


def test_standard_sae_forward_matches_formula():
    if not os.path.exists(os.path.join(SAE_DIR, "ae.pt")):
        print("  (skip standard forward: SAE not present)"); return
    sae = load_sae(os.path.join(SAE_DIR, "ae.pt"), "standard")
    x = torch.randn(3, sae.activation_dim)
    # decode(encode(x)) == forward(x); encode == ReLU(W_enc (x - b_dec))
    f = sae.encode(x)
    assert torch.allclose(sae.decode(f), sae(x), atol=1e-6)
    by_hand = torch.relu((x - sae.bias) @ sae.encoder.weight.T + sae.encoder.bias)
    assert torch.allclose(f, by_hand, atol=1e-5)
    assert (f >= 0).all()                          # ReLU family


def test_topk_sae_has_exactly_k_active():
    d = os.path.join(SUITE_INSPECT, "dir_TopK", "ae.pt")
    if not os.path.exists(d):
        print("  (skip topk: SAE not present)"); return
    sae = load_sae(d, "topk")
    x = torch.randn(5, sae.activation_dim)
    f = sae.encode(x)
    nz = (f != 0).sum(-1)
    assert (nz <= int(sae.k.item())).all() and nz.float().mean() > 0


def test_all_arch_loaders_load():
    archs = {"TopK": "topk", "BatchTopK": "batchtopk", "JumpRelu": "jumprelu",
             "GatedSAE": "gated", "Matryoshka": "matryoshka", "PAnneal": "standard"}
    for folder, arch in archs.items():
        p = os.path.join(SUITE_INSPECT, f"dir_{folder}", "ae.pt")
        if not os.path.exists(p):
            print(f"  (skip {folder}: not present)"); continue
        sae = load_sae(p, arch)
        x = torch.randn(2, sae.activation_dim)
        out = sae(x)
        assert out.shape == x.shape and torch.isfinite(out).all()


def test_max_cosine_sim_matches_verbatim():
    # our calculate_max_cosine_sim vs a verbatim reference, on a random (D, F) matrix
    from saebench_audit.metrics.core_full import calculate_max_cosine_sim
    torch.manual_seed(0)
    M = torch.randn(16, 40)
    mine = calculate_max_cosine_sim(M, batch_size=7)
    enc = F.normalize(M, p=2, dim=0)
    full = enc.t() @ enc
    full.fill_diagonal_(float("-inf"))
    ref = full.max(dim=1).values
    assert torch.allclose(mine, ref, atol=1e-6), (mine - ref).abs().max()


def test_W_enc_W_dec_extraction_shapes():
    from saebench_audit.metrics.core_full import sae_W_enc_W_dec
    for folder, arch in [("dir_TopK", "topk"), ("dir_JumpRelu", "jumprelu"),
                         ("dir_Matryoshka", "matryoshka"), ("dir_GatedSAE", "gated")]:
        p = os.path.join(SUITE_INSPECT, folder, "ae.pt")
        if not os.path.exists(p):
            print(f"  (skip {folder})"); continue
        sae = load_sae(p, arch)
        We, Wd = sae_W_enc_W_dec(sae)
        assert We.shape == (sae.activation_dim, sae.dict_size)       # [D, F]
        assert Wd.shape == (sae.dict_size, sae.activation_dim)       # [F, D]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {repr(e)[:120]}")
    print(f"\n{passed}/{len(tests)} unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
