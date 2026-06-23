"""
AutoInterp (Automated Interpretability) metric — faithful port of SAEBench
`sae_bench/evals/autointerp/main.py` + `sae_bench_utils/{indexing_utils,activation_collection}.py`.

Pipeline (paper §3.3 / Appendix Table 5):
  1. tokenize the Pile (ctx 128), collect per-latent SAE activations (BOS/EOS/PAD masked).
  2. select n_latents non-dead latents (firing count > dead_latent_threshold), seeded.
  3. per latent build example windows (±buffer): top-k activating, importance-weighted, random.
  4. GENERATION: an LLM (gpt-4o-mini) writes a short explanation from the top examples (<<token>> marked).
  5. SCORING (detection, Paulo et al. 2024): the LLM, given the explanation + a shuffled mix of
     2 top + 2 importance-weighted + 10 random sequences, predicts which activate. Score = detection
     accuracy over all 14 sequences (positives AND negatives). autointerp_score = mean over latents.

The judge is pluggable (`judge_fn(messages, max_tokens) -> str`) so the deterministic pipeline can be
oracle-tested with a mock judge; the real run uses `openai_judge` (gpt-4o-mini).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
import torch
import einops

from .core_loss_recovered import get_decoder_layers


# --------------------------------------------------------------------------- #
# Config (AutoInterpEvalConfig defaults; n_latents/total_tokens reducible for CPU)
# --------------------------------------------------------------------------- #
@dataclass
class AutoInterpConfig:
    model_name: str = "EleutherAI/pythia-160m-deduped"
    layer: int = 8
    dataset_name: str = "monology/pile-uncopyrighted"
    llm_context_size: int = 128
    total_tokens: int = 2_000_000
    n_latents: int = 1000
    dead_latent_threshold: float = 15
    random_seed: int = 42
    buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.01
    max_tokens_in_explanation: int = 30
    use_demos_in_explanation: bool = True
    n_top_ex_for_generation: int = 10
    n_iw_sampled_ex_for_generation: int = 5
    n_top_ex_for_scoring: int = 2
    n_random_ex_for_scoring: int = 10
    n_iw_sampled_ex_for_scoring: int = 2
    llm_batch_size: int = 16
    device: str = "cpu"

    @property
    def n_top_ex(self): return self.n_top_ex_for_generation + self.n_top_ex_for_scoring
    @property
    def n_iw_sampled_ex(self): return self.n_iw_sampled_ex_for_generation + self.n_iw_sampled_ex_for_scoring
    @property
    def n_ex_for_scoring(self): return self.n_top_ex_for_scoring + self.n_random_ex_for_scoring + self.n_iw_sampled_ex_for_scoring
    @property
    def n_correct_for_scoring(self): return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring
    @property
    def max_tokens_in_prediction(self): return 2 * self.n_ex_for_scoring + 5


# --------------------------------------------------------------------------- #
# Indexing utilities (verbatim from sae_bench_utils/indexing_utils.py)
# --------------------------------------------------------------------------- #
def get_k_largest_indices(x, k, buffer=0, no_overlap=False):
    x = x[:, buffer:-buffer]
    indices = x.flatten().argsort(-1, descending=True)
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer
    if no_overlap:
        unique, seen = [], set()
        for row, col in zip(rows.tolist(), cols.tolist()):
            if (row, col) not in seen:
                unique.append((row, col))
                for off in range(-buffer, buffer + 1):
                    seen.add((row, col + off))
            if len(unique) == k:
                break
        rows, cols = torch.tensor(unique, dtype=torch.int64, device=x.device).unbind(dim=-1)
    return torch.stack((rows, cols), dim=1)[:k]


def get_iw_sample_indices(x, k, buffer=0, use_squared_values=True):
    x = x[:, buffer:-buffer]
    if use_squared_values:
        x = x.pow(2)
    probs = x.flatten() / x.sum()
    idx = torch.multinomial(probs, k, replacement=False)
    rows = idx // x.size(1)
    cols = idx % x.size(1) + buffer
    return torch.stack((rows, cols), dim=1)[:k]


def index_with_buffer(x, indices, buffer=0):
    rows, cols = indices.unbind(dim=-1)
    rows = einops.repeat(rows, "k -> k b", b=buffer * 2 + 1)
    cols = einops.repeat(cols, "k -> k b", b=buffer * 2 + 1) + torch.arange(-buffer, buffer + 1, device=cols.device)
    return x[rows, cols]


# --------------------------------------------------------------------------- #
# Example / Examples (verbatim formatting)
# --------------------------------------------------------------------------- #
class Example:
    def __init__(self, toks, acts, act_threshold, str_toks):
        self.toks = toks
        self.acts = acts
        self.act_threshold = act_threshold
        self.str_toks = str_toks
        self.toks_are_active = [a > act_threshold for a in acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks=False):
        return ("".join(f"<<{t}>>" if (mark_toks and a) else t
                        for t, a in zip(self.str_toks, self.toks_are_active))
                .replace("�", "").replace("\n", "↵"))


class Examples:
    def __init__(self, examples, shuffle=False):
        if shuffle:
            random.shuffle(examples)
        else:
            examples = sorted(examples, key=lambda x: max(x.acts), reverse=True)
        self.examples = examples

    def __len__(self): return len(self.examples)
    def __iter__(self): return iter(self.examples)
    def __getitem__(self, i): return self.examples[i]


# --------------------------------------------------------------------------- #
# Dataset tokenization (verbatim from dataset_utils.tokenize_and_concat_dataset)
# --------------------------------------------------------------------------- #
def load_and_tokenize_dataset(dataset_name, ctx_len, num_tokens, tok, add_bos=True):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train", streaming=True)
    docs, total = [], 0
    for row in ds:
        t = row["text"]
        if len(t) > 100:
            docs.append(t); total += len(t)
            if total > num_tokens * 5:
                break
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    full = tok.eos_token.join(docs)
    nch = 20; cl = (len(full) - 1) // nch + 1
    chunks = [full[i * cl:(i + 1) * cl] for i in range(nch)]
    toks = tok(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()
    toks = toks[toks != tok.pad_token_id]
    toks = toks[: num_tokens + ctx_len + 1]
    nb = len(toks) // ctx_len
    toks = toks[: nb * ctx_len].reshape(nb, ctx_len)
    if add_bos:
        toks[:, 0] = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return toks


# --------------------------------------------------------------------------- #
# Activation collection (HF model; resid_post layer; BOS/EOS/PAD masked)
# --------------------------------------------------------------------------- #
def keep_mask(tokens, tok):
    special = torch.zeros_like(tokens, dtype=torch.bool)
    for tid in (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id):
        if tid is not None:
            special |= (tokens == tid)
    return ~special


@torch.no_grad()
def collect_activations(tokens, model, sae, tok, layer, batch_size, selected_latents=None,
                        sparsity_only=False):
    """Per-token SAE activations [N, L, F] (or sparsity [F]), special tokens zeroed."""
    lm = get_decoder_layers(model)[layer]
    out_acts = []
    running = torch.zeros(sae.dict_size, dtype=torch.float32)
    total = 0
    for i in range(0, tokens.shape[0], batch_size):
        bt = tokens[i:i + batch_size]
        cap = {}
        h = lm.register_forward_hook(lambda m, i_, o: cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
        model(bt); h.remove()
        f = sae.encode(cap["x"].to(torch.float32))
        km = keep_mask(bt, tok).to(f.device)
        if sparsity_only:
            fire = (f > 0).float() * km[:, :, None]
            running += einops.reduce(fire, "b l f -> f", "sum")
            total += int(km.sum().item())
        else:
            if selected_latents is not None:
                f = f[:, :, selected_latents]
            f = f * km[:, :, None]
            out_acts.append(f)
    if sparsity_only:
        return running / max(total, 1)
    return torch.cat(out_acts, dim=0)


# --------------------------------------------------------------------------- #
# Per-latent example construction (gather_data, verbatim logic)
# --------------------------------------------------------------------------- #
def _str_toks(tok, toks):
    return [tok.decode([t]) for t in toks]


def gather_data(cfg, acts, tokens, latents, tok):
    """acts: [N, L, len(latents)] for the selected latents. Returns gen/scoring Examples per latent."""
    dataset_size, seq_len = tokens.shape
    gen, score = {}, {}
    for i, latent in enumerate(latents):
        a = acts[..., i].float()
        rand_indices = torch.stack([
            torch.randint(0, dataset_size, (cfg.n_random_ex_for_scoring,)),
            torch.randint(cfg.buffer, seq_len - cfg.buffer, (cfg.n_random_ex_for_scoring,))], dim=-1)
        rand_toks = index_with_buffer(tokens, rand_indices, buffer=cfg.buffer)

        top_idx = get_k_largest_indices(a, k=cfg.n_top_ex, buffer=cfg.buffer, no_overlap=cfg.no_overlap)
        top_toks = index_with_buffer(tokens, top_idx, buffer=cfg.buffer)
        top_vals = index_with_buffer(a, top_idx, buffer=cfg.buffer)
        act_threshold = cfg.act_threshold_frac * top_vals.max().item()

        threshold = top_vals[:, cfg.buffer].min().item()
        a_thresh = torch.where(a >= threshold, 0.0, a)
        if a_thresh[:, cfg.buffer:-cfg.buffer].max() < 1e-6:
            continue
        iw_idx = get_iw_sample_indices(a_thresh, k=cfg.n_iw_sampled_ex, buffer=cfg.buffer)
        iw_toks = index_with_buffer(tokens, iw_idx, buffer=cfg.buffer)
        iw_vals = index_with_buffer(a, iw_idx, buffer=cfg.buffer)

        top_perm = torch.randperm(cfg.n_top_ex)
        top_gen, top_sc = top_perm[:cfg.n_top_ex_for_generation], top_perm[cfg.n_top_ex_for_generation:]
        iw_perm = torch.randperm(cfg.n_iw_sampled_ex)
        iw_gen, iw_sc = iw_perm[:cfg.n_iw_sampled_ex_for_generation], iw_perm[cfg.n_iw_sampled_ex_for_generation:]

        def mk(all_toks, all_acts=None):
            if all_acts is None:
                all_acts = torch.zeros_like(all_toks).float()
            return [Example(t, ac, act_threshold, _str_toks(tok, t))
                    for t, ac in zip(all_toks.tolist(), all_acts.tolist())]

        gen[latent] = Examples(mk(top_toks[top_gen], top_vals[top_gen]) + mk(iw_toks[iw_gen], iw_vals[iw_gen]))
        score[latent] = Examples(
            mk(top_toks[top_sc], top_vals[top_sc]) + mk(iw_toks[iw_sc], iw_vals[iw_sc]) + mk(rand_toks),
            shuffle=True)
    return gen, score


# --------------------------------------------------------------------------- #
# Prompts (verbatim) + parsing + scoring
# --------------------------------------------------------------------------- #
def generation_prompts(cfg, gen_examples):
    ex = "\n".join(f"{i+1}. {e.to_str(mark_toks=True)}" for i, e in enumerate(gen_examples))
    sys = ("We're studying neurons in a neural network. Each neuron activates on some particular "
           "word/words/substring/concept in a short document. The activating words in each document are "
           "indicated with << ... >>. We will give you a list of documents on which the neuron activates, "
           "in order from most strongly activating to least strongly activating. Look at the parts of the "
           "document the neuron activates for and summarize in a single sentence what the neuron is "
           "activating on. Try not to be overly specific in your explanation. Note that some neurons will "
           "activate only on specific words or substrings, but others will activate on most/all words in a "
           "sentence provided that sentence contains some particular concept. Your explanation should cover "
           "most or all activating words (for example, don't give an explanation which is specific to a "
           "single word if all words in a sentence cause the neuron to activate). Pay attention to things "
           "like the capitalization and punctuation of the activating words or concepts, if that seems "
           "relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. "
           "Omit punctuation and formatting. You should avoid giving long lists of words.")
    if cfg.use_demos_in_explanation:
        sys += (" Some examples: \"This neuron activates on the word 'knows' in rhetorical questions\", and "
                "\"This neuron activates on verbs related to decision-making and preferences\", and \"This "
                "neuron activates on the substring 'Ent' at the start of words\", and \"This neuron activates "
                "on text about government economic policy\".")
    else:
        sys += 'Your response should be in the form "This neuron activates on...".'
    return [{"role": "system", "content": sys},
            {"role": "user", "content": f"The activating documents are given below:\n\n{ex}"}]


def scoring_prompts(cfg, explanation, scoring_examples):
    ex = "\n".join(f"{i+1}. {e.to_str(mark_toks=False)}" for i, e in enumerate(scoring_examples))
    example_response = sorted(random.sample(range(1, 1 + cfg.n_ex_for_scoring), k=cfg.n_correct_for_scoring))
    ers = ", ".join(str(i) for i in example_response)
    sys = (f"We're studying neurons in a neural network. Each neuron activates on some particular "
           f"word/words/substring/concept in a short document. You will be given a short explanation of what "
           f"this neuron activates for, and then be shown {cfg.n_ex_for_scoring} example sequences in random "
           f"order. You will have to return a comma-separated list of the examples where you think the neuron "
           f"should activate at least once, on ANY of the words or substrings in the document. For example, "
           f"your response might look like \"{ers}\". Try not to be overly specific in your interpretation of "
           f"the explanation. If you think there are no examples where the neuron will activate, you should "
           f"just respond with \"None\". You should include nothing else in your response other than "
           f"comma-separated numbers or the word \"None\" - this is important.")
    usr = f"Here is the explanation: this neuron fires on {explanation}.\n\nHere are the examples:\n\n{ex}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]


def parse_explanation(raw):
    return raw.split("activates on")[-1].rstrip(".").strip()


def parse_predictions(raw):
    parts = raw.strip().rstrip(".").replace("and", ",").replace("None", "").split(",")
    lst = [p.strip() for p in parts if p.strip() != ""]
    if lst == []:
        return []
    if not all(p.isdigit() for p in lst):
        return None
    return [int(p) for p in lst]


def score_predictions(predictions, scoring_examples):
    classifications = [i in predictions for i in range(1, len(scoring_examples) + 1)]
    correct = [e.is_active for e in scoring_examples]
    return sum(c == cc for c, cc in zip(classifications, correct)) / len(classifications)


# --------------------------------------------------------------------------- #
# OpenAI judge (gpt-4o-mini)
# --------------------------------------------------------------------------- #
def openai_judge(api_key, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    def judge_fn(messages, max_tokens):
        r = client.chat.completions.create(model=model, messages=messages, n=1,
                                           max_tokens=max_tokens, stream=False)
        return r.choices[0].message.content.strip()
    return judge_fn


# --------------------------------------------------------------------------- #
# Latent selection + top-level run
# --------------------------------------------------------------------------- #
def select_latents(cfg, sparsity):
    counts = sparsity * cfg.total_tokens
    alive = torch.nonzero(counts > cfg.dead_latent_threshold).squeeze(1).tolist()
    if len(alive) <= cfg.n_latents:
        return alive
    return random.sample(alive, k=cfg.n_latents)


def run_single_latent(cfg, judge_fn, gen_ex, score_ex):
    expl_raw = judge_fn(generation_prompts(cfg, gen_ex), cfg.max_tokens_in_explanation)
    explanation = parse_explanation(expl_raw)
    preds_raw = judge_fn(scoring_prompts(cfg, explanation, score_ex), cfg.max_tokens_in_prediction)
    preds = parse_predictions(preds_raw)
    if preds is None:
        return None
    return {"explanation": explanation, "predictions": preds,
            "correct_seqs": [i for i, e in enumerate(score_ex, start=1) if e.is_active],
            "score": score_predictions(preds, score_ex)}


def run_autointerp(cfg, model, tok, sae, judge_fn, tokens, sparsity=None, verbose=True):
    random.seed(cfg.random_seed); torch.manual_seed(cfg.random_seed)
    if sparsity is None:
        sparsity = collect_activations(tokens, model, sae, tok, cfg.layer, cfg.llm_batch_size, sparsity_only=True)
    latents = select_latents(cfg, sparsity)
    if verbose:
        print(f"[autointerp] {len(latents)} non-dead latents selected", flush=True)
    acts = collect_activations(tokens, model, sae, tok, cfg.layer, cfg.llm_batch_size, selected_latents=latents)
    gen, score = gather_data(cfg, acts, tokens, latents, tok)
    results = {}
    for latent in sorted(gen.keys()):
        r = run_single_latent(cfg, judge_fn, gen[latent], score[latent])
        if r is not None:
            results[latent] = {"latent": latent, **r}
            if verbose:
                print(f"  latent {latent}: score={r['score']:.3f}  expl=\"{r['explanation'][:60]}\"", flush=True)
    scores = torch.tensor([r["score"] for r in results.values()])
    return {"autointerp_score": scores.mean().item(),
            "autointerp_std_dev": scores.std().item() if len(scores) > 1 else 0.0,
            "n_latents_scored": len(scores), "per_latent": results}
