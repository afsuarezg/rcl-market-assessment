# Analyses produced by `analyze_results.py`

This document describes each post-estimation analysis run by `analyze_results.py`. Each analysis writes a text report to `results/<dataset>/analysis/NN_<name>.txt`, where `NN` is the analysis number used below.

## Inputs

For each dataset (`nevo`, `blp`) the script reads (preferring the nested `csv/` subdir, falling back to the flat layout):

- `<dataset>/csv/<prefix>multistart_all.csv` — every multistart row (one row per spec × start), with `spec`, `seed`, `start`, `objective`, `price_coef`, `best`, and `init_*` / `est_*` columns.
- `<dataset>/csv/<prefix>elasticities_detail.csv` — one row per (`spec`, `seed`, `product_j`, `product_k`), with `elasticity` and an `own_price` flag (`'True'` when `j == k`).

The BLP files carry a `blp_` prefix; Nevo files do not. Inputs are auto-located via `_find()` and also checked under `/oak/stanford/groups/polinsky/blp_nevo/results`.

## Shared building blocks

- **Validity:** `_valid(r)` returns `True` when `price_coef < 0`. Demand without a negative price coefficient is economically nonsensical, so most aggregates either restrict to valid rows or count invalid rows separately.
- **Best per spec:** `_best_per_spec(rows)` returns one row per spec — the lowest-objective row whose `price_coef < 0`. Falls back to the lowest-objective row overall if every start for a spec is invalid.
- **Best-seed map:** `_best_seed_map(all_rows)` maps `spec → seed` using `_best_per_spec`, then drives every elasticity analysis that operates "on the best start."

Every section runs for both datasets except the Nevo-specific ones — sections 24, 25, 37, 38.

---

## 01 — Objective ranking across specifications

`objective_ranking(rows, label)` (analyze_results.py:178). For each spec, picks the best valid start and prints them in ascending order of `objective`. Columns: `Rank`, `GMM obj`, `price_coef`, `Specification`.

Use it to identify the spec that fits the moments best, and to sanity-check that the implied `price_coef` is negative and reasonable. Section numbers are shared with `rcl_synthetic_data`'s `plot_specs.py`/`GRAPHS.md`, so they are not contiguous here: Nevo-only and market-assessment-only analyses sit at 24, 25, 37, 38, 39.

## 02 — Multi-start convergence stability

`multistart_stability(rows, label)` (analyze_results.py:204). For each spec, computes the spread `max(obj) − min(obj)` across all its random starts, sorted so the most-unstable spec is first. `N starts` counts all starts (including invalid ones).

High spread is a flag that BFGS is finding multiple local optima for that spec.

## 03 — Convergence audit

`convergence_audit(rows, label)` (analyze_results.py:219). Prints every (spec, seed, start) row in load order with `objective`, `price_coef`, `Valid?`, the `best` flag, and an `OK` / `** INVALID (alpha > 0) **` marker. Ends with totals for valid vs invalid starts.

Use it to see precisely which starts misbehaved and which seeds reproduce a given local optimum.

## 04 — Global minimum identification

`global_minimum(rows, label)` (analyze_results.py:238). Restricts to valid starts, then prints every (spec, seed) sorted by `objective` ascending, with `d_from_best = obj − min(obj)`.

Use it to see which spec wins overall and how clustered the top of the leaderboard is.

## 05 — Two-basin analysis

`two_basin_analysis(rows, label)` (analyze_results.py:259). Among valid starts, partitions rows into:
- **Basin A**: `obj − global_min ≤ 5`
- **Basin B**: `obj − global_min > 5`

For each basin reports the seed list, `objective` range, `price_coef` range, and mean ± spread for every `est_*[…]` parameter column.

Use it to characterize the two (or more) optima BFGS lands in — typically a "good" basin near the global minimum and a "bad" basin where parameters are pushed to the boundary. The 5.0 threshold is hardcoded.

## 07 — Price coefficient sensitivity

`price_coef_sensitivity(rows, label)` (analyze_results.py:188). Across the per-spec best valid starts, prints the range and spread of `price_coef`, then a table sorted by `price_coef` with `−1/α` shown as a rough implied-markup approximation (the actual BLP markup is more complex; this is a heuristic only).

Use it to see how much α moves across model choices — a large spread means model selection materially changes elasticity/markup conclusions.

## 19 — Price coefficient across simulations

`price_coef_across_sims(rows, label)` (analyze_results.py:1136). For the spec with the lowest mean objective, prints every seed's `objective`, `price_coef`, and a `yes`/`no` validity flag, plus summary statistics. Counts valid starts and shows the price-coefficient distribution.

A diagnostic that complements analysis 19's mean: if α has a large range or is bimodal across seeds (a common pattern when there are two basins), the "best" α is much less informative.
## 20 — Own-price elasticity summary by spec

`elasticity_own_summary(elas_rows, all_rows, label)` (analyze_results.py:328). For each spec's best valid seed (per `_best_seed_map`), collects the own-price elasticities across all products and prints mean / median / std / min / max, ranked by GMM objective.

Use it to compare the own-price elasticity distribution across specs at their best fits.

## 21 — Elasticity stability across multi-start seeds

`elasticity_multistart_stability(elas_rows, all_rows, label)` (analyze_results.py:356). For every spec that has more than one seed with elasticity data, prints per-product own-price elasticity mean / spread / min / max across seeds.

Use it to verify that elasticities are reproducible across local optima — a large spread for the same product across seeds within one spec means the elasticity estimate is unstable to optimizer randomness.

## 22 — Top substitutes per product

`elasticity_top_substitutes(elas_rows, all_rows, label, top_k=5)` (analyze_results.py:396). Picks the rank-1 spec and its best seed, then for each product `j` lists the top-5 substitutes `k` by cross-price elasticity `e_jk`. For markets with >50 products (BLP), only the 10 most price-elastic products are shown.

Use it to read off who-competes-with-whom under the preferred model.

## 23 — Cross-price elasticity asymmetry

`elasticity_asymmetry(elas_rows, all_rows, label)` (analyze_results.py:444). For the best spec/seed, computes `|e_jk − e_kj|` over all unordered product pairs, reports mean / median / max / percent of pairs above 0.1, and lists the 10 most-asymmetric pairs.

Symmetric cross-elasticities are *not* implied by random-coefficients logit, so a large mean asymmetry is informative — pairs with very asymmetric substitution are worth understanding (often because one product is much larger).

## 24 — Cross-spec Spearman correlation (Nevo only)

`nevo_elasticity_cross_spec_correlation(elas_rows, all_rows)` (analyze_results.py:798). For products common to every spec, computes Spearman rank correlation of own-price elasticities between every pair of specs and prints both the full matrix and a sorted pairwise list.

Use it to see whether different specs rank products consistently (high correlation) or reshuffle them (low correlation). Robustness to spec choice on *rankings* is often easier to argue than robustness on levels.

## 25 — Within-firm vs between-firm substitution (Nevo only)

`nevo_elasticity_firm_substitution(elas_rows, all_rows)` (analyze_results.py:861). Parses firm IDs from product IDs (`F1B04 → F1`) and, for each spec at its best seed, reports mean within-firm cross-elasticity, mean between-firm cross-elasticity, and their ratio. Also prints a firm-by-firm mean cross-elasticity matrix for the best spec.

A within/between ratio > 1 means own-firm products substitute more strongly with each other than with rival firms' products — relevant for portfolio-pricing arguments and merger analysis. Only works for Nevo-style product IDs of the form `F\d+B\d+`.

## 26 — Own-price elasticity cross-spec stability

`elasticity_own_cross_spec_stability(elas_rows, all_rows, label)` (analyze_results.py:502). For products present in every spec's best seed, computes the coefficient of variation `CV = std / |mean|` of own-price elasticity across specs, plus std, mean, min, max, and the full per-spec values. Sorted by CV descending. Limits to 20 most-elastic products when there are >50 products total. Ends with aggregate mean and max CV.

Use it to see which products' elasticities are robust to model choice and which are not.

## 27 — Cross-price elasticity cross-spec stability

`elasticity_cross_cross_spec_stability(elas_rows, all_rows, label, top_k=10)` (analyze_results.py:577). Picks the top-10 substitute pairs from the rank-1 spec, then for each pair computes the CV of `e_jk` across every spec's best seed.

The cross-spec analogue of analysis 26, but for cross-elasticities on the pairs that matter most in the preferred model.

## 28 — Pairwise spec agreement (MAD matrix)

`elasticity_spec_pairwise_mad(elas_rows, all_rows, label)` (analyze_results.py:668). Builds an S×S matrix of mean absolute deviation of own-price elasticities between every pair of specs, computed over products common to that pair. Prints both the matrix and a sorted list of the most divergent spec pairs.

Use it as a complement to analysis 24 (Spearman) — MAD picks up level differences that rank correlation misses.

## 30 — Elasticity pair across all simulations (single spec)

`elasticity_pair_across_sims(elas_rows, all_rows, label)` (analyze_results.py:968). For the spec with the lowest *mean* objective:
1. Selects the cross-elasticity pair `(j, k)` with the highest mean `e_jk` across seeds.
2. For every seed of that spec, prints the four elasticities `e_jj`, `e_kk`, `e_jk`, `e_kj`.
3. Reports per-series mean / std / min / max.

Use it to see how much a single product pair's elasticity panel moves across local optima within the preferred spec.

## 31 — Elasticity pair across specifications (best simulation per spec)

`elasticity_pair_best_sim_across_specs(elas_rows, all_rows, label)` (analyze_results.py:1046). Mirror of analysis 30, but the cross-section is over specifications instead of seeds:
1. From each spec's best valid seed, finds the product pair with the highest cross-coverage; ties broken by mean `e_jk`.
2. For every spec, prints `e_jj`, `e_kk`, `e_jk`, `e_kj` for that pair using only that spec's best seed.
3. Reports per-series mean / std / min / max across specs.

Use it to see how the same product pair's elasticities shift as the model is re-specified.

## 36 — Objective aggregation across specifications

`objective_spec_comparison(rows, label)` (analyze_results.py:928). Unlike analysis 01 (best valid start only), this aggregates *all* starts per spec: N starts, N valid, mean / median / std / min / max / range / CV% of `objective`. Sorted by mean objective ascending. Ends with cross-spec summary statistics.

Use it to identify specs that converge tightly (low CV) versus specs that fluctuate — a spec with the lowest *best* objective but very high CV may just be lucky on one seed.

## 37 — Demographic expansion effect (Nevo only)

`nevo_demographic_expansion(rows)` (analyze_results.py:739). For each fixed X2 (`sugar`, `mushy`), walks through a hardcoded sequence of demographic specifications from sparse to rich, reporting `GMM obj`, `price_coef`, and `Δobj` vs the previous demographic set.

Use it to read the marginal improvement (or harm) from adding `income_squared`, `age`, `child`. Specs not present in the data are silently skipped.

## 38 — X2 characteristic comparison (Nevo only)

`nevo_x2_comparison(rows)` (analyze_results.py:777). Holds demographics fixed at `['income']` and compares `x2 ∈ {sugar, mushy, sugar+mushy}`. Reports `GMM obj` and `price_coef` per X2 choice. Useful for isolating the contribution of each product characteristic to fit.

## 39 — Starting-value sensitivity

`starting_value_sensitivity(rows, label)` (analyze_results.py:303). For every `init_*[…]` parameter column, computes the mean among valid starts and the mean among invalid starts, plus their difference.

Use it to detect whether certain ranges of initial values systematically lead to invalid (α > 0) solutions — i.e., whether starts are "doomed" by their priors. This is a diagnostic for how to draw better initial values in future runs.

---

## How to run

```bash
python analyze_results.py    # writes text reports to results/<dataset>/analysis/
```

BLP and Nevo are wrapped in independent `try/except FileNotFoundError` blocks — missing inputs for one dataset don't block the other.

## Companion: `plot_results.py`

Most of these analyses have a graphical counterpart in `plot_results.py`, written to `results/<dataset>/graphs/NN_<name>.png`. Section 39 (starting-value sensitivity) is the only text-only analysis with no figure; the Nevo-only section 38 (X2 comparison) is plotted only when matching specs exist.
