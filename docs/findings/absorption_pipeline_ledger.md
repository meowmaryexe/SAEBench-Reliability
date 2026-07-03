# Absorption pipeline — code-vs-paper ledger

**Owner:** Alor · **Purpose:** cover the "wrap" blind spot. We run upstream's absorption code rather than
reimplement it, so the stages that *feed* the score — GT probes, k-sparse probing, prompt/vocab
construction, and the eval guards — are otherwise unaudited. This is a zero-compute source read of those
stages against the paper's description (Karvonen et al. Appendix D + the eval README). It does **not**
re-derive the pipeline; it documents every undocumented / deviating choice so (a) we know what to copy to
reproduce, and (b) any "hidden choice doing the heavy lifting" is on the record. The absorption *scoring*
itself is covered separately and oracle-verified — see `absorption_version_drift.md`.

Sources read: sae-bench **0.6.0** (`saebench-absorption-env`, current code) with **0.3.2**
(`saebench-0125-env`, `@141aff72`, the version that produced the published `0125` results) for version
deltas. Paths are under `sae_bench/evals/absorption/`. Where a "paper says" cell can't be checked against
the actual Appendix D text (not in hand), it is marked **VERIFY**.

**Buckets:** silent-default (a value the code fixes but the paper doesn't state) · deviation (departs from
cited prior work) · optimistic-selection. **Impact:** cosmetic · plausible (could move a score) ·
outcome-determining (a headline could depend on it).

## TL;DR

- The scoring is not the only version-sensitive part. **Two independent code deltas separate the published
  (0.3.2) numbers from current code (0.6.0):** (1) the `absorption_fraction` formula rework (PR #62 — see
  `absorption_version_drift.md`), and (2) **new aggregation guards** (`min_GT_probe_f1=0.6` per-letter
  filter + `min_feats_for_eval=20` abort) that **did not exist in 0.3.2**, which averaged all 26 letters
  unconditionally. To reproduce the published absorption you must match both.
- Everything the pipeline does is **un-seeded** (`random_seed=42` is declared but never applied), so the
  probe split, probe weights, feature selection, and prompts all vary run-to-run.
- The GT "probe" is a **torch BCE** multi-probe, not the sklearn logistic regression a reader might assume.

## Ledger

### 1. Ground-truth probe — `probing.py`
| item | code does | paper/README says | bucket | impact | version |
|---|---|---|---|---|---|
| probe type | torch **BCE** 26-way one-vs-rest `LinearProbe` (`nn.Linear`), `BCEWithLogitsLoss` with per-class `pos_weight=num_neg/num_pos` (`probing.py:29-47,101`) | "logistic-regression probe" (naive reading) | deviation | plausible — defines the probe direction everything is projected onto | stable |
| optimizer / schedule | Adam, `lr=1e-2` exp-decay to `end_lr`, `weight_decay=1e-4`, `num_epochs=50`, `batch_size=64` (`common.py:279-283`, `probing.py:163-201,395`) | VERIFY (Appendix D) | silent-default | plausible | stable |
| train/test split | **80/20** via `shuffled_vocab = random.sample(vocab, len(vocab))` then split index (`probing.py:276-281`), **no seed** | 80/20 (per code-audit) | silent-default + un-seeded | plausible — which tokens train vs eval; drifts run-to-run | stable |
| probe direction | `LinearProbe.weights = fc.weight` (26×d_model) used for the probe projection + `cos_sim(probe, W_dec)` (`probing.py:41-43`; `k_sparse_probing.py:198-216`) | probe direction | — | — | stable |

### 2. K-sparse probing (main/"split" latent selection) — `k_sparse_probing.py`
| item | code does | paper says | bucket | impact | version |
|---|---|---|---|---|---|
| candidate selection | top-k on a trained **L1 multi-probe** (`l1_decay=0.01`, 50 ep), then sklearn `LogisticRegression(max_iter=500, class_weight="balanced")` on those k SAE columns (`k_sparse_probing.py:57-92,159-179`) | k-sparse probing | silent-default | plausible | stable (0.6.0 adds batching/precalc, numerically equivalent) |
| feature-split → main latents | walk `k=1..max_k_value(=10)`; keep growing while F1 gain `≥ f1_jump_threshold(=0.03)`; **break at first non-jump**; `split_feats` = set at the last kept k; `num_split_features = len-1` (`k_sparse_probing.py:328,422-445`) | "feature splitting" via an F1 jump | silent-default | outcome-determining — `split_feats` are the "main" latents = the absorption denominator | stable |
| determinism | selection arithmetic deterministic given the L1 probe, but that probe trains with unseeded `DataLoader(shuffle=True)` (`probing.py:93`) | — | un-seeded | plausible — main-latent set drifts run-to-run | stable |

### 3. Prompt + vocabulary — `prompting.py`, `vocab.py`
| item | code does | paper says | bucket | impact | version |
|---|---|---|---|---|---|
| ICL prompt | `create_icl_prompt`, `max_icl_examples=10`, `shuffle_examples=True` via `random.sample` **(no seed)**, contamination guard resamples until `word ∉ examples` (≤1000 tries) (`prompting.py:72-142`) | ICL spelling prompt | silent-default + un-seeded | plausible | stable (0.6.0 adds a Mistral leading-space option) |
| read position | `prompt_token_pos = -6` for `"{word} has the first letter:"` (`eval_config.py:38-42`) | position of the read | silent-default | plausible — **tokenizer-specific**; VERIFY for Pythia | stable |
| vocab filter | `get_alpha_tokens`: keep tokens that are **all a–z/A–Z** after one optional leading space; non-empty; **no length limit, no case normalization** (`vocab.py:36-50`) | alphabetic tokens | silent-default | plausible | stable (0.6.0 adds a Mistral token→string fix) |

### 4. Guards + aggregation — `main.py`  ⚠ version-critical
| item | code does | paper says | bucket | impact | version |
|---|---|---|---|---|---|
| per-letter drop | count a letter only if its GT `f1_probe > 0.6` (`main.py:142-147`) | VERIFY | silent-default | **outcome-determining** — changes which letters enter the mean | **0.6.0-only** |
| abort | if `<20` letters have `f1_probe>0.6`, **`break`** → no output for that SAE (`main.py:112-118`) | VERIFY | silent-default | **outcome-determining** — can zero out an SAE | **0.6.0-only** |
| headline | `statistics.mean/stdev` over surviving letters; `f1_probe` = GT logit thresholded at 0.0 on the **test** split (`main.py:168-179`; `k_sparse_probing.py:373-374`) | mean over letters | — | — | mean/stdev stable; **survivor set differs by version** |

**0.3.2 (published) had neither guard** — no `min_GT_probe_f1`/`min_feats_for_eval` config fields, no abort
block, and the aggregation loop appended **all 26 letters unconditionally** (0.3.2 `main.py:122-136`). So
the published mean is over 26 letters; a current-code mean is over the ≤26 that clear F1>0.6 (and may not
run at all). This is a second reason 0.6.0 ≠ published, independent of the fraction formula.

### 5. Seed
`random_seed=42` is a config field, threaded through the CLI and saved to the output JSON, but **never
applied**: no `random.seed` / `torch.manual_seed` / `np.random.seed` / `set_seed` anywhere in the
absorption dir (both versions). Unseeded sources of run-to-run variance: the 80/20 split
(`probing.py:276`), every probe's `DataLoader(shuffle=True)` (`probing.py:93`), and ICL sampling
(`prompting.py` `random.sample`). Bucket: silent-default; impact: plausible (this is the drift we measured,
fraction 0.0163 ± 0.0005 over N=3).

## What you must copy to reproduce the published (0125) absorption
1. **Version**: sae-bench 0.3.2 (`@141aff72`) — this pins both the fraction formula *and* the
   no-guards aggregation (all 26 letters, no abort). Under 0.6.0 you get neither.
2. GT probe = torch BCE multi-probe (not sklearn LR); 80/20 split; k-sparse feature-split with
   `f1_jump_threshold=0.03`, `max_k_value=10`; `prompt_token_pos=-6`; alpha-only vocab.
3. Expect run-to-run drift (no seed); compare within a drift band and, above all, on rankings.

## Follow-ups (not done here)
- Expose the guards (`min_GT_probe_f1`, `min_feats_for_eval`) as config in our wrapper, mirroring the
  four thresholds, so the Stage-2 audit can toggle guard-on vs guard-off and measure the effect on the
  Pythia mean.
- Replace the `VERIFY` cells with exact Appendix D quotes once the paper text is in hand.
- The residual blind spot after this ledger: we've *read* the pipeline but not independently
  re-implemented it, so a bug shared between code and paper description would still pass. Full
  reimplementation remains deferred (absorption is "reproduce-only, low audit yield" per the project plan).
