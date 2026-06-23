"""
Config + result schemas shared across metrics.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class CoreConfig:
    """Configuration for the Core / Loss Recovered evaluation.

    tokenize_mode:
      - "packed": concatenate EOS-separated documents into fixed ctx windows — matches SAEBench
        core/main.py (transformer_lens ActivationsStore) → paper Table 4 / OWT ctx128 numbers.
      - "per_document": one document per sequence, truncated to ctx — matches dictionary_learning
        loss_recovered() → the bundled eval_results.json. Absolute CE differs between modes;
        frac_recovered (Loss Recovered) is ~invariant. See docs/metric_notes.md.
    """
    model_name: str = "EleutherAI/pythia-160m-deduped"
    layer: int = 8
    dataset: str = "Skylion007/openwebtext"
    dataset_split: str = "train"
    text_field: str = "text"
    context_size: int = 128
    batch_size_prompts: int = 16
    n_reconstruction_seqs: int = 3200      # paper Table 4 (loss recovered)
    n_sparsity_seqs: int = 32000           # paper Table 4 (sparsity)
    prepend_bos: bool = True
    tokenize_mode: str = "packed"
    dtype: str = "float32"
    device: str = "cpu"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Keys produced per batch by the Core metric (one JSON object per line in results/raw/).
CORE_PER_BATCH_KEYS = [
    "bi", "loss_original", "loss_reconstructed", "loss_zero", "frac_recovered", "l0", "n_seqs",
]
# Aggregated keys written to results/processed/.
CORE_AGG_KEYS = ["loss_original", "loss_reconstructed", "loss_zero", "frac_recovered", "l0"]
