"""
saebench_audit — independent reimplementations of SAEBench evaluation metrics for the
SAEBench reproducibility + reliability study (Karvonen et al., 2025; Chanin et al., 2026).

Implemented metrics (src/saebench_audit/metrics/):
  - core_loss_recovered   ✅ (Core / Loss Recovered — reproduce-only)
  - scr, tpp, sparse_probing   (scaffolded)

Shared modules: schema (configs), io (model/SAE/data/checkpoint I/O), statistics (aggregation),
sae_models (independent SAE forwards), plotting (SVG figures).

Nothing imports SAEBench / dictionary_learning; only released SAE *weights* are consumed.
"""
__version__ = "0.1.0"
