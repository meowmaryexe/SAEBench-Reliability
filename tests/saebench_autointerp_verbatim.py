"""
VERBATIM transcription of the deterministic data-gathering in SAEBench
`sae_bench/evals/autointerp/main.py` (AutoInterp.gather_data + Example), fetched 2026-06-22.

Only adaptation: `str_toks` are passed in (we verified HF tok.decode([t]) == transformer_lens
to_str_tokens to 100%, so this is faithful), and the indexing helpers are imported from our package
(they are themselves verbatim copies, separately unit-tested vs the source).

Used by tests/test_autointerp_oracle.py as the ground truth that our
saebench_audit.metrics.autointerp.gather_data must match bit-for-bit.
"""
import torch
from saebench_audit.metrics.autointerp import (get_k_largest_indices, get_iw_sample_indices,
                                               index_with_buffer)


class ExampleV:
    def __init__(self, toks, acts, act_threshold, str_toks):
        self.toks = toks
        self.str_toks = str_toks
        self.acts = acts
        self.act_threshold = act_threshold
        self.toks_are_active = [act > act_threshold for act in self.acts]
        self.is_active = any(self.toks_are_active)

    def to_str(self, mark_toks=False):
        return ("".join(f"<<{tok}>>" if (mark_toks and a) else tok
                        for tok, a in zip(self.str_toks, self.toks_are_active))
                .replace("�", "").replace("\n", "↵"))


class ExamplesV:
    def __init__(self, examples, shuffle=False):
        import random
        if shuffle:
            random.shuffle(examples)
        else:
            examples = sorted(examples, key=lambda x: max(x.acts), reverse=True)
        self.examples = examples

    def __len__(self): return len(self.examples)
    def __iter__(self): return iter(self.examples)
    def __getitem__(self, i): return self.examples[i]


def gather_data_verbatim(cfg, acts, tokens, latents, str_toks_fn):
    """Mirror of AutoInterp.gather_data (main.py lines 394-513)."""
    dataset_size, seq_len = tokens.shape
    generation_examples, scoring_examples = {}, {}
    for i, latent in enumerate(latents):
        rand_indices = torch.stack([
            torch.randint(0, dataset_size, (cfg.n_random_ex_for_scoring,)),
            torch.randint(cfg.buffer, seq_len - cfg.buffer, (cfg.n_random_ex_for_scoring,))], dim=-1)
        rand_toks = index_with_buffer(tokens, rand_indices, buffer=cfg.buffer)

        top_indices = get_k_largest_indices(acts[..., i], k=cfg.n_top_ex, buffer=cfg.buffer, no_overlap=cfg.no_overlap)
        top_toks = index_with_buffer(tokens, top_indices, buffer=cfg.buffer)
        top_values = index_with_buffer(acts[..., i], top_indices, buffer=cfg.buffer)
        act_threshold = cfg.act_threshold_frac * top_values.max().item()

        threshold = top_values[:, cfg.buffer].min().item()
        acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
        if acts_thresholded[:, cfg.buffer:-cfg.buffer].max() < 1e-6:
            continue
        iw_indices = get_iw_sample_indices(acts_thresholded, k=cfg.n_iw_sampled_ex, buffer=cfg.buffer)
        iw_toks = index_with_buffer(tokens, iw_indices, buffer=cfg.buffer)
        iw_values = index_with_buffer(acts[..., i], iw_indices, buffer=cfg.buffer)

        rand_top_ex_split_indices = torch.randperm(cfg.n_top_ex)
        top_gen_indices = rand_top_ex_split_indices[:cfg.n_top_ex_for_generation]
        top_scoring_indices = rand_top_ex_split_indices[cfg.n_top_ex_for_generation:]
        rand_iw_split_indices = torch.randperm(cfg.n_iw_sampled_ex)
        iw_gen_indices = rand_iw_split_indices[:cfg.n_iw_sampled_ex_for_generation]
        iw_scoring_indices = rand_iw_split_indices[cfg.n_iw_sampled_ex_for_generation:]

        def create_examples(all_toks, all_acts=None):
            if all_acts is None:
                all_acts = torch.zeros_like(all_toks).float()
            return [ExampleV(t, a, act_threshold, str_toks_fn(t))
                    for (t, a) in zip(all_toks.tolist(), all_acts.tolist())]

        generation_examples[latent] = ExamplesV(
            create_examples(top_toks[top_gen_indices], top_values[top_gen_indices])
            + create_examples(iw_toks[iw_gen_indices], iw_values[iw_gen_indices]))
        scoring_examples[latent] = ExamplesV(
            create_examples(top_toks[top_scoring_indices], top_values[top_scoring_indices])
            + create_examples(iw_toks[iw_scoring_indices], iw_values[iw_scoring_indices])
            + create_examples(rand_toks), shuffle=True)
    return generation_examples, scoring_examples
