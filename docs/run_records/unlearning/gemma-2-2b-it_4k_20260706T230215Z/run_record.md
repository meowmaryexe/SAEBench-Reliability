# unlearning run record — gemma-2-2b-it_4k

_generated 2026-07-07T01:40:02+00:00_

## v0.6.0 (current) — 42/42 SAEs ok
- sae-bench `0.6.0` · sae_lens `6.45.3` · transformer_lens `2.17.0` · transformers `4.57.6` · torch `2.12.1`
- host `ip-172-31-20-225` · device `cuda` · gpu `NVIDIA A10G, 23028 MiB` · git `00587e4565` · finished `2026-07-07T01:39:50+00:00`

## Per-architecture

| arch | published score | v0.6.0 score |
|---|---|---|
| batchtopk | 0.0413 | 0.0417 |
| gatedsae | 0.0260 | 0.0280 |
| jumprelu | 0.0697 | 0.0738 |
| matryoshkabatchtopk | 0.0260 | 0.0259 |
| panneal | 0.0378 | 0.0389 |
| standard | 0.0475 | 0.0480 |
| topk | 0.0507 | 0.0751 |

## Report
```

=== unlearning_score ===
              arch  published       ours        Δ
         batchtopk     0.0413     0.0417  +0.0005
          gatedsae     0.0260     0.0280  +0.0021
          jumprelu     0.0697     0.0738  +0.0041
matryoshkabatchtopk     0.0260     0.0259  -0.0001
           panneal     0.0378     0.0389  +0.0011
          standard     0.0475     0.0480  +0.0004
              topk     0.0507     0.0751  +0.0244
  Spearman ρ  published↔ours = +0.929

[verdict] unlearning: max |Δ vs published| = 0.0244 (within drift band 0.05) → reproduces published
[verdict] architecture ranking: published↔ours ρ=+0.929 → ranking holds (caveat: scores are near-zero/degenerate on Gemma-2-2B, so ranking is noise-sensitive)
```
