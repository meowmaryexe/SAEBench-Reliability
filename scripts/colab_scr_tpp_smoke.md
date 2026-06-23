# Colab SCR / TPP Smoke Test

This document records the known-good CUDA smoke test path for SCR and TPP.

## Setup

```bash
git clone https://github.com/adamkarvonen/SAEBench.git
cd SAEBench
pip install -e .
pip install pytest
```

Restart the Colab runtime after installation if Colab warns about imported packages.

## Verify CUDA

```python
import torch
import sae_lens

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("sae_lens OK")
```

## TPP Acceptance Test

```bash
python -m pytest -s tests/acceptance/test_scr_and_tpp.py::test_tpp_end_to_end_different_seed
```

Expected result:

```text
1 passed
```

Observed result on 2026-06-22:

```text
Global mean difference: 0.008533358573913566
Global max difference: 0.012800037860870361
1 passed in 199.37s
```

## SCR Acceptance Test

```bash
python -m pytest -s tests/acceptance/test_scr_and_tpp.py::test_scr_end_to_end_different_seed
```

Expected result:

```text
1 passed
```

Observed result on 2026-06-22:

```text
scr_score dir 1: 0.8357665734093227
scr_score dir 2: 0.5411764623384583
Global mean difference: 0.07869108679148029
Global max difference: 0.07869108679148029
1 passed in 189.76s
```

## Notes

- These are reduced acceptance tests, not full SAEBench reproduction runs.
- They validate that SCR/TPP execute correctly on CUDA.
- Full local Mac MPS execution failed due memory limits.
- Next step is to create our own minimal run script using the same reduced config.