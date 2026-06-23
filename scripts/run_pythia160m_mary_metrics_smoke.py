"""
Smoke runner for Mary-owned SAEBench metrics on Pythia-160M dictionary-learning SAEs.

Goal:
- Use the actual SAEBench Pythia-160M custom SAE loading path.
- Run only Mary-owned metrics: TPP, SCR, Sparse Probing.
- Start with one SAE location before scaling.
"""

# TODO:
# 1. Import custom SAE loading helpers from SAEBench.
# 2. Enumerate Pythia-160M SAE locations from HuggingFace.
# 3. Filter to one SAE for smoke testing.
# 4. Run TPP, SCR, and Sparse Probing.
# 5. Save outputs under results/raw/reproduction_smoke/.