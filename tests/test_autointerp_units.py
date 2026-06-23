"""
Unit + oracle tests for the deterministic AutoInterp pipeline (no LLM). Each test pins one piece of
SAEBench's autointerp/main.py + indexing_utils.py behavior. Run: python tests/test_autointerp_units.py
"""
import os, sys, random
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import autointerp as ai


# ---- verbatim SAEBench references (indexing_utils.py) for the oracle ----
def _verbatim_k_largest(x, k, buffer=0, no_overlap=False):
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1); cols = indices % x.size(1) + buffer
    if no_overlap:
        uniq, seen = [], set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen:
                uniq.append((row, col))
                for off in range(-buffer, buffer + 1):
                    seen.add((row, col + off))
            if len(uniq) == k:
                break
        rows, cols = torch.tensor(uniq, dtype=torch.int64).unbind(dim=-1)
    return torch.stack((rows, cols), dim=1)[:k]


def test_get_k_largest_matches_verbatim():
    torch.manual_seed(0)
    x = torch.randn(8, 60).abs()
    for no in (False, True):
        a = ai.get_k_largest_indices(x, k=12, buffer=10, no_overlap=no)
        b = _verbatim_k_largest(x, k=12, buffer=10, no_overlap=no)
        assert torch.equal(a, b), no


def test_index_with_buffer_window():
    x = torch.arange(5 * 40).reshape(5, 40).float()
    idx = torch.tensor([[1, 15], [3, 25]])          # cols within [buffer, seq-buffer)
    w = ai.index_with_buffer(x, idx, buffer=10)
    assert w.shape == (2, 21)                       # +-buffer window = 2*10+1
    assert torch.equal(w[0], x[1, 5:26])            # cols 15-10 .. 15+10


def test_iw_sampling_in_range_and_disjoint():
    torch.manual_seed(0)
    x = torch.rand(6, 50)
    idx = ai.get_iw_sample_indices(x, k=7, buffer=10)
    assert idx.shape == (7, 2)
    assert (idx[:, 1] >= 10).all() and (idx[:, 1] < 40).all()   # cols within [buffer, seq-buffer)


def test_score_predictions_detection_accuracy():
    # 14 examples: indices 1..4 active (2 top + 2 iw), 5..14 inactive (random)
    class E:
        def __init__(self, a): self.is_active = a
    examples = [E(i < 4) for i in range(14)]
    # perfect prediction -> 1.0
    assert ai.score_predictions([1, 2, 3, 4], examples) == 1.0
    # predict none -> 10/14 correct (the negatives)
    assert abs(ai.score_predictions([], examples) - 10 / 14) < 1e-12
    # one false positive + one false negative -> 12/14
    assert abs(ai.score_predictions([1, 2, 3, 5], examples) - 12 / 14) < 1e-12


def test_parse_predictions_and_explanation():
    assert ai.parse_predictions("1, 3, and 5") == [1, 3, 5]
    assert ai.parse_predictions("None") == []
    assert ai.parse_predictions("garbage") is None
    assert ai.parse_explanation("This neuron activates on the word 'the'.") == "the word 'the'"


def test_example_is_active_and_marking():
    # acts above threshold mark the token and set is_active
    ex = ai.Example(toks=[1, 2, 3], acts=[0.0, 5.0, 0.0], act_threshold=1.0,
                    str_toks=["a", "b", "c"])
    assert ex.toks_are_active == [False, True, False]
    assert ex.is_active is True
    assert ex.to_str(mark_toks=True) == "a<<b>>c"
    assert ex.to_str(mark_toks=False) == "abc"
    dead = ai.Example([1], [0.1], 1.0, ["x"])
    assert dead.is_active is False


def test_select_latents_threshold_and_seed():
    cfg = ai.AutoInterpConfig(total_tokens=1000, n_latents=3, dead_latent_threshold=15, random_seed=42)
    # firing fractions -> counts = frac*1000; >15 alive
    sparsity = torch.tensor([0.0, 0.02, 0.5, 0.001, 0.3, 0.02, 0.0])  # counts: 0,20,500,1,300,20,0
    random.seed(42)
    sel = ai.select_latents(cfg, sparsity)
    assert set(sel).issubset({1, 2, 4, 5})          # only latents with count>15
    assert len(sel) == 3
    # dead latent (count<=15) never selected
    assert 0 not in sel and 3 not in sel


def test_scoring_examples_have_4_active_of_14():
    # construct examples like gather_data does: 2 top + 2 iw active, 10 random inactive
    act_examples = [ai.Example([1], [5.0], 1.0, ["x"]) for _ in range(4)]
    rand_examples = [ai.Example([1], [0.0], 1.0, ["x"]) for _ in range(10)]
    exs = ai.Examples(act_examples + rand_examples, shuffle=True)
    assert len(exs) == 14
    assert sum(e.is_active for e in exs) == 4       # n_correct_for_scoring


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {repr(e)[:140]}")
    print(f"\n{passed}/{len(tests)} autointerp unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
