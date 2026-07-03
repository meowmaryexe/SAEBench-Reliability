"""
VERBATIM transcription of the per-word `absorption_fraction` arithmetic from upstream SAEBench, for the
oracle in test_absorption_fraction_oracle.py. Copied from `feature_absorption_calculator.py` in each
version (operating on the same tensors the real code uses) — do NOT "clean up" or refactor; the whole
point is byte-for-byte fidelity to upstream so our pure-Python ports in
src/saebench_audit/metrics/absorption_fraction.py can be checked against it.

- v0.3.2  = sae-bench 0.3.2 (@141aff72, PR #48) — the version behind the published `0125` results.
- v0.6.0  = sae-bench 0.6.0 (PR #62) — current code.
"""
import numpy as np
import torch


def fraction_v032_verbatim(sae_act_probe_proj: torch.Tensor, act_probe_proj: float,
                           main_feature_ids: list[int]) -> float:
    # --- verbatim from 0.3.2 feature_absorption_calculator.calculate_absorption ---
    main_feats_probe_proj = torch.sum(sae_act_probe_proj[main_feature_ids]).cpu().item()
    all_feats_probe_proj = torch.sum(sae_act_probe_proj).cpu().item()
    if main_feats_probe_proj >= act_probe_proj or all_feats_probe_proj <= 0:
        absorption_fraction = 0.0
    else:
        absorption_fraction = (all_feats_probe_proj - main_feats_probe_proj) / all_feats_probe_proj
        absorption_fraction = np.clip(absorption_fraction, 0.0, 1.0)
    return float(absorption_fraction)


def fraction_v060_verbatim(sae_act_probe_proj: torch.Tensor, cos_sims: torch.Tensor,
                           act_probe_proj: float, main_feature_ids: list[int],
                           absorption_fraction_probe_cos_sim_threshold: float = 0.1,
                           probe_projection_proportion_threshold: float = 0.4,
                           absorption_fraction_max_absorbing_latents: int = 3) -> float:
    # --- verbatim from 0.6.0 feature_absorption_calculator.calculate_absorption (lines ~170-219) ---
    main_feats_probe_proj = torch.sum(sae_act_probe_proj[main_feature_ids]).cpu().item()
    potential_absorbers_mask = torch.ones(sae_act_probe_proj.size(0), dtype=torch.bool)
    potential_absorbers_mask[main_feature_ids] = False
    potential_absorbers_mask &= cos_sims >= absorption_fraction_probe_cos_sim_threshold
    potential_absorbers_mask &= sae_act_probe_proj > 0
    potential_absorbers_probe_proj = sae_act_probe_proj[potential_absorbers_mask]
    top_potential_absorbers_probe_proj = potential_absorbers_probe_proj.topk(
        k=min(absorption_fraction_max_absorbing_latents, potential_absorbers_probe_proj.numel())
    ).values
    top_potential_absorbers_total_probe_proj = torch.sum(top_potential_absorbers_probe_proj).cpu().item()
    top_potential_absorbers_probe_proj_proportion = (
        top_potential_absorbers_total_probe_proj / act_probe_proj
    )
    if (main_feats_probe_proj >= act_probe_proj
            or top_potential_absorbers_probe_proj_proportion
            < probe_projection_proportion_threshold):
        absorption_fraction = 0.0
    elif main_feats_probe_proj <= 0.0:
        absorption_fraction = 1.0
    else:
        unaccounted_probe_proj = act_probe_proj - main_feats_probe_proj
        absorption_probe_proj = min(top_potential_absorbers_total_probe_proj, unaccounted_probe_proj)
        absorption_fraction = absorption_probe_proj / (absorption_probe_proj + main_feats_probe_proj)
        absorption_fraction = np.clip(absorption_fraction, 0.0, 1.0)
    return float(absorption_fraction)
