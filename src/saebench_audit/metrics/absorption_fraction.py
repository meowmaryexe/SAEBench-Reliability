"""
The two versions of SAEBench's per-word `absorption_fraction` definition, as standalone pure functions.

Why this exists: the published SAEBench absorption_fraction (results repo `sae_bench_results_0125`) was
computed by sae-bench 0.3.2 (@141aff72, PR #48). A later release (0.6.0, PR #62) *redefined* the metric —
adding a cosine gate, a top-k cap, and a proportion floor — which drops the score ~10× on the same SAE.
The `full_absorption` metric is unchanged across versions. See docs/findings/absorption_version_drift.md.

These are faithful ports of the per-word math in `feature_absorption_calculator.py` in each version, kept
dependency-free (pure Python) so the mechanism is testable/auditable without a model, SAE, or GPU. They
are the basis of tests/test_absorption_version_drift.py and are reusable for the Stage-2 audit (toggling
the definition on cached projections).

Terminology (per word / SAE input):
- `act_probe_proj`      : projection of the residual-stream activation onto the unit probe direction.
- `main_feats_probe_proj`: summed probe-projection of the SAE's "main"/split latents for the concept.
- per-latent `probe_proj` = latent_activation * cos_sim(latent decoder, probe); `cos_sim` is that cosine.
"""
from __future__ import annotations


def absorption_fraction_v032(
    act_probe_proj: float,
    main_feats_probe_proj: float,
    all_feats_probe_proj: float,
) -> float:
    """sae-bench 0.3.2 / PR #48 (the version behind the published `0125` results).

    The *loose* definition: the fraction of the total SAE-reconstruction probe-projection that is NOT
    accounted for by the main latents — every non-main latent counts, with no gating.
    """
    if main_feats_probe_proj >= act_probe_proj or all_feats_probe_proj <= 0:
        return 0.0
    frac = (all_feats_probe_proj - main_feats_probe_proj) / all_feats_probe_proj
    return min(max(frac, 0.0), 1.0)


def absorption_fraction_v060(
    act_probe_proj: float,
    main_feats_probe_proj: float,
    candidate_probe_projs: list[float],
    candidate_cos_sims: list[float],
    cos_gate: float = 0.1,
    proportion_floor: float = 0.4,
    max_absorbers: int = 3,
) -> float:
    """sae-bench 0.6.0 / PR #62 (current code).

    The *gated* definition: only non-main latents that are cosine-aligned with the probe (cos >= cos_gate)
    and have positive projection are "potential absorbers"; take the top `max_absorbers` by projection;
    if their share of `act_probe_proj` is below `proportion_floor` the word contributes 0; otherwise the
    absorbed projection is `min(top_total, act - main)` over `(that + main)`.

    `candidate_*` are the per-latent probe projections and cosine sims for NON-main latents (pass all of
    them; the cos/positivity filters are applied here).
    """
    absorbers = [
        p for p, c in zip(candidate_probe_projs, candidate_cos_sims)
        if c >= cos_gate and p > 0
    ]
    top_total = sum(sorted(absorbers, reverse=True)[:max_absorbers])

    if act_probe_proj == 0:
        return 0.0
    if main_feats_probe_proj >= act_probe_proj or (top_total / act_probe_proj) < proportion_floor:
        return 0.0
    if main_feats_probe_proj <= 0.0:
        return 1.0
    unaccounted = act_probe_proj - main_feats_probe_proj
    absorption_proj = min(top_total, unaccounted)
    frac = absorption_proj / (absorption_proj + main_feats_probe_proj)
    return min(max(frac, 0.0), 1.0)
