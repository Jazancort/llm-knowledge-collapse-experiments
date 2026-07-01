# Negative Results — Gemma 3 C5 Exposure Intervention

## Date: 2026-07-01

---

## Hypothesis

If effective training pressure is a general organizing principle, then reducing
synthetic exposure by ~5% should also restore stability at the Gemma 3 boundary
configuration (r=16), analogous to the Qwen r=256 result (+9pp).

## Protocol

- Model: google/gemma-3-1b-it (4-bit NF4 QLoRA)
- Rank: 16 (boundary case for Gemma 3, degrades to ~68.8% by Gen5)
- Intervention: random downsample 5% of synthetic training examples each gen
- Generations: 5
- Seeds: 15 (matching baseline)
- Dataset: TriviaQA rc.nocontext, shuffle seed=15, first 2000 train / next 200 eval
- Script: `scripts/gemma3_c5_correct.py` (identical to sprint2_gemma.py + downsample)

## Expected Result

Retention improvement from ~70% to ~80%+ at Gen5, analogous to Qwen (+9pp).

## Observed Result

| Gen | C5 (5% downsample) | Baseline (no intervention) |
|-----|---------------------|---------------------------|
| 1   | 38/46 (82.6%)       | 38/47 (80.9%)             |
| 2   | 39/46 (84.8%)       | 38/47 (80.9%)             |
| 3   | 36/46 (78.3%)       | 36/47 (76.6%)             |
| 4   | 34/46 (73.9%)       | 34/47 (72.3%)             |
| 5   | 33/46 (71.7%)       | 33/47 (70.2%)             |

Delta at Gen5: **+1.5pp** (within noise, K0 differs by 1 item).

## Conclusion

Under the evaluated Gemma 3 r=16 configuration, a ~5% reduction in synthetic
exposure did not produce a measurable improvement in recursive factual retention.
The intervention that restored near-homeostatic behavior on Qwen r=256 does not
transfer to this Gemma 3 boundary configuration at the same intervention magnitude.

## Interpretation (ordered by plausibility)

1. **Configuration not near an exposure-sensitive boundary.** The Gemma 3 r=16
   operating point may not lie close enough to a threshold where marginal exposure
   reduction can shift the regime.

2. **Intervention magnitude insufficient.** 5% may be too small for this backbone;
   a larger reduction (10-20%) might show an effect.

3. **Exposure sensitivity is backbone-dependent.** Not all pressure axes transfer
   identically across architectures under identical intervention magnitudes.

## What this does NOT demonstrate

- Does NOT prove absence of exposure sensitivity in Gemma 3 (only one magnitude tested)
- Does NOT invalidate the Qwen exposure result (that remains internally valid)
- Does NOT refute the ETP framework (framework predicts backbone-dependent thresholds)

## Impact on Paper

- §4.5 closing adjusted to specify "evaluated Qwen boundary configuration"
- Discussion §5.4 now includes paragraph reporting this negative result
- Reinforces that exposure intervention is configuration-specific, not universal
- Increases credibility (honest reporting of failed replication)

## Earlier Spurious Result (for the record)

An initial version of the experiment (`gemma3_c5_intervention.py`) showed +24.7pp
improvement. This was later shown to be entirely explained by protocol differences
(different prompt, different dataset split, different eval settings) rather than the
downsample intervention. A control experiment (`gemma3_baseline_control.py`) using
the same modified protocol without downsample produced identical results (93.5%),
confirming the artefactual nature of that observation.

## Files

- `scripts/gemma3_c5_correct.py` — correct experiment (sprint2_gemma + downsample only)
- `scripts/gemma3_c5_intervention.py` — initial (FLAWED) version (different protocol)
- `scripts/gemma3_baseline_control.py` — control proving initial result was artefactual
- `outputs/gemma3_c5correct_rank16_seed15/results.json` — correct results
- `outputs/gemma3_c5_intervention/` — artefactual results (do not use)
- `outputs/gemma3_baseline_control/` — control results

## Future Work

- Test larger exposure reductions (10%, 20%, 30%) as dose-response on Gemma 3
- Test C3 (length filtering) instead of C5 (random downsample) on Gemma 3
- Test exposure intervention at a different Gemma 3 rank (e.g., r=8 if boundary)
