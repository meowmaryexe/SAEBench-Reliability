# Absorption run record — pythia-160m_4k

_generated 2026-07-05T05:09:53+00:00_

## v0.3.2 (published-match) — 42/42 SAEs ok
- sae-bench `0.3.2` · sae_lens `5.3.1` · transformer_lens `2.17.0` · transformers `4.57.6` · torch `2.12.1`
- host `ip-172-31-41-199` · device `cpu` · gpu `None` · git `f4e2a86fc9`(dirty) · finished `2026-07-05T05:09:24+00:00`

## v0.6.0 (current) — 42/42 SAEs ok
- sae-bench `0.6.0` · sae_lens `6.45.2` · transformer_lens `2.17.0` · transformers `4.57.6` · torch `2.12.1`
- host `ip-172-31-41-199` · device `cpu` · gpu `None` · git `f4e2a86fc9`(dirty) · finished `2026-07-05T05:09:49+00:00`

## Per-architecture (fraction | full)

| arch | published frac | 0.3.2 | 0.6.0 | published full | 0.3.2 | 0.6.0 |
|---|---|---|---|---|---|---|
| batchtopk | 0.2839 | 0.2851 | 0.1529 | 0.0543 | 0.0583 | 0.0544 |
| gatedsae | 0.1898 | 0.1947 | 0.0621 | 0.0542 | 0.0557 | 0.0553 |
| jumprelu | 0.2224 | 0.2281 | 0.1082 | 0.0428 | 0.0431 | 0.0401 |
| matryoshkabatchtopk | 0.2904 | 0.2978 | 0.1354 | 0.0559 | 0.0561 | 0.0554 |
| panneal | 0.1121 | 0.1173 | 0.0308 | 0.0290 | 0.0313 | 0.0302 |
| standard | 0.1670 | 0.1694 | 0.0539 | 0.0281 | 0.0312 | 0.0288 |
| topk | 0.2614 | 0.2654 | 0.1270 | 0.0556 | 0.0612 | 0.0641 |

## Report
```

=== absorption_fraction ===
              arch  published     v0.3.2        Δ     v0.6.0        Δ
         batchtopk     0.2839     0.2851  +0.0013     0.1529  -0.1310
          gatedsae     0.1898     0.1947  +0.0048     0.0621  -0.1277
          jumprelu     0.2224     0.2281  +0.0056     0.1082  -0.1142
matryoshkabatchtopk     0.2904     0.2978  +0.0074     0.1354  -0.1550
           panneal     0.1121     0.1173  +0.0051     0.0308  -0.0813
          standard     0.1670     0.1694  +0.0024     0.0539  -0.1131
              topk     0.2614     0.2654  +0.0040     0.1270  -0.1344
  Spearman ρ  published↔0.3.2=+1.000  published↔0.6.0=+0.964  0.3.2↔0.6.0=+0.964

=== full_absorption ===
              arch  published     v0.3.2        Δ     v0.6.0        Δ
         batchtopk     0.0543     0.0583  +0.0040     0.0544  +0.0001
          gatedsae     0.0542     0.0557  +0.0015     0.0553  +0.0011
          jumprelu     0.0428     0.0431  +0.0002     0.0401  -0.0028
matryoshkabatchtopk     0.0559     0.0561  +0.0002     0.0554  -0.0005
           panneal     0.0290     0.0313  +0.0023     0.0302  +0.0012
          standard     0.0281     0.0312  +0.0031     0.0288  +0.0007
              topk     0.0556     0.0612  +0.0056     0.0641  +0.0085
  Spearman ρ  published↔0.3.2=+0.893  published↔0.6.0=+0.929  0.3.2↔0.6.0=+0.893

[verdict] fraction: 0.3.2 max |Δ vs published| = 0.0074 (within drift band 0.05) → reproduces published
[verdict] ranking survives redefinition? published↔0.3.2 ρ=+1.000, published↔0.6.0 ρ=+0.964 → ranking holds
```
