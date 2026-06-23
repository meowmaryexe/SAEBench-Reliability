"""
Verbatim prompt assertion — our AutoInterp prompts must be byte-identical to SAEBench's.

The golden strings in tests/fixtures_autointerp_prompts.json were extracted directly from SAEBench
`autointerp/main.py` (get_generation_prompts), so this catches any transcription drift in our copy.
Run: python tests/test_autointerp_prompts.py
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import autointerp as ai

GOLD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "fixtures_autointerp_prompts.json")))


class _E:
    def to_str(self, mark_toks=False): return "x"


def test_generation_system_prompt_verbatim():
    cfg = ai.AutoInterpConfig(use_demos_in_explanation=True)
    sysmsg = ai.generation_prompts(cfg, [_E()])[0]["content"]
    expected = GOLD["generation_system_base"] + GOLD["generation_demos"]
    assert sysmsg == expected, "generation system prompt differs from SAEBench source"


def test_generation_user_prompt_format():
    cfg = ai.AutoInterpConfig()
    user = ai.generation_prompts(cfg, [_E(), _E()])[1]["content"]
    assert user == "The activating documents are given below:\n\n1. x\n2. x"


def test_scoring_system_prompt_invariants():
    cfg = ai.AutoInterpConfig()
    sysmsg = ai.scoring_prompts(cfg, "concept", [_E()] * 14)[0]["content"]
    for phrase in [
        f"be shown {cfg.n_ex_for_scoring} example sequences in random order",
        "comma-separated list of the examples where you think the neuron should activate at least once",
        'on ANY of the words or substrings in the document',
        'If you think there are no examples where the neuron will activate, you should just respond with "None"',
        'You should include nothing else in your response other than comma-separated numbers or the word "None"',
    ]:
        assert phrase in sysmsg, f"scoring prompt missing: {phrase!r}"


def test_scoring_user_prompt_format():
    cfg = ai.AutoInterpConfig()
    user = ai.scoring_prompts(cfg, "the word the", [_E(), _E()])[1]["content"]
    assert user.startswith("Here is the explanation: this neuron fires on the word the.\n\nHere are the examples:")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} prompt tests passed")
    sys.exit(0 if passed == len(tests) else 1)
