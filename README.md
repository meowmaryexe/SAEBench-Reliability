## SAEBench Reliability 

Reproducibility and reliability study of Sparse Autoencoder (SAE) evaluation benchmarks.

## Overview

This project aims to reproduce key results from:

* Karvonen et al., SAEBench (2025)
* Chanin et al., Are Sparse Autoencoder Benchmarks Reliable? (2026)

Our current focus is:

1. Reproducing core SAEBench evaluation results on released SAEs.
2. Building a clean and fully reproducible evaluation pipeline.
3. Auditing the reliability and stability of selected SAE benchmark metrics.
4. Investigating the robustness and generalizability of benchmark conclusions across models and evaluation settings.

The project is being conducted as a reproducibility study targeting submission to TMLR and consideration for the NeurIPS 2026 Machine Learning Reproducibility Challenge (MLRC).

## Repository Structure
- `configs/` - Experiment configurations
- `docs/` - Project notes and preregistration
- `figures/` - Generated figures and visualizations
- `results/` - Raw and processed experiment outputs
- `scripts/` - Entry-point scripts
- `src/` - Core source code

# References 
Karvonen et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability.

Chanin et al. (2026). Are Sparse Autoencoder Benchmarks Reliable?
