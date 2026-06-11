# Graphs produced by `plot_results.py`

This document describes each figure written by `plot_results.py`. Files land in `results/<dataset>/graphs/NN_<name>.png`. Numbering follows the analysis numbering in [`ANALYSIS.md`](ANALYSIS.md); see that file for the underlying tables.

## Inputs and shared conventions

Same auto-located inputs as `analyze_results.py`: `<prefix>multistart_all.csv` and `<prefix>elasticities_detail.csv` under `results/<dataset>/csv/` (with `blp_` prefix for BLP). Each figure carries a `· Analysis NN ·` footer pointing back to the matching text report.

Color conventions:

- **`COL_VALID`** — start has `price_coef < 0` (economically valid).
- **`COL_INVALID`** — start has `price_coef ≥ 0`.
- **`COL_BASIN_A`** / **`COL_BASIN_B`** — distance from global minimum ≤ 5 / > 5 (basin classification).
- Diverging heatmaps (`RdBu_r`, `coolwarm_r`, `RdYlGn`) center at 0; sequential heatmaps (`YlGnBu`, `YlOrRd`, `Oranges`, `Blues`) start at 0.
- `*` marker = best start; `o` = other random starts.

---

## 01 — `01_objective_ranking.png`

`plot_objective_ranking` (plot_results.py:149). Horizontal bar chart, one bar per spec showing the GMM objective of its best valid start. Bars sorted ascending (best at top). Color encodes validity (`COL_VALID` vs `COL_INVALID`); the price coefficient `α=...` is annotated to the right of each bar.

Look at the top bar for the preferred model. Look at how flat the top of the leaderboard is — close objectives across several specs mean the choice is less critical.

## 02 — `02_multistart_stability.png`

`plot_multistart_stability` (plot_results.py:199). Scatter with one column per spec, x-axis = spec, y-axis = GMM objective. Each random start is a marker (`*` for the best, `o` for others), colored by validity. Specs sorted left-to-right by minimum objective.

Use it to spot which specs converge tightly (one cluster) vs which scatter across multiple basins (vertical spread).

## 03 — `03_convergence_scatter.png`

`plot_convergence_scatter` (plot_results.py:232). Scatter of every start as one point: x-axis = `price_coef` (α), y-axis = GMM objective. Color = validity, marker = best (`*`) vs other (`o`). Vertical dashed line at α = 0.

This is the most diagnostic plot for basin structure. Vertical bands of points at the same α with very different objectives = repeated convergence to the same parameters from different random starts. Horizontal bands = different α reaching the same objective (parameter-identification problem). Combined with `03_convergence_audit.txt` you can read off which seeds populate each basin.

## 04 — `04_global_minimum.png`

`plot_global_minimum` (plot_results.py:256). Horizontal bar chart of `Δ = objective − global_min` per valid start. Y-axis labels include both `spec` and `seed`. Best start is highlighted; everything else shown as distance above the global minimum.

Use it to gauge how isolated the global minimum is. A single bar at 0 and everything else much higher means the global is hard to find; a cluster of bars near 0 means several starts converged close together.

## 05 — `05_basin_scatter.png`

`plot_basin_scatter` (plot_results.py:280). Scatter of valid starts: x-axis = GMM objective, y-axis = `price_coef`. Points colored by basin (A if `Δ ≤ 5`, B if `Δ > 5`). Horizontal dashed line at α = 0.

Complement to figure 03: same data sliced differently. Use it when you suspect the high- and low-objective basins have systematically different α values — vertical separation between the two colored clouds is the signal.

## 07 — `07_price_coef.png`

`plot_price_coef` (plot_results.py:175). Horizontal bar chart of `price_coef` per spec (best valid start), sorted by `α` ascending. Bars colored by validity. `markup≈-1/α` annotated to the right of each bar. Vertical line at 0 marks the validity boundary.

Use it to see the spread of α across specs. Tightly clustered bars = robust price sensitivity; long spread = α is sensitive to spec choice and so are any downstream markup arguments.

## 19 — `19_price_coef_across_sims.png`

`plot_price_coef_across_sims` (plot_results.py:1167). Scatter of `price_coef` per seed for the lowest-mean-objective spec. X-axis = seed (ordered by seed value), y-axis = α. Points colored by validity. Dashed line at mean α; dotted line at α = 0.

Use it to read off the distribution of α within the preferred spec — bimodal scatter is the classic two-basin signature, which is the visual counterpart of `03_convergence_audit.txt`.
## 20 — `20_own_elas_boxplot.png`

`plot_own_elas_boxplot` (plot_results.py:308). Box plot of own-price elasticities, one box per spec, using the best seed of each spec. Boxes show IQR, median line, whiskers, and outliers; an overlay strip plot jitters individual products to show within-spec dispersion.

Use it to compare elasticity distributions across model choices. A spec with a much wider box (or a different median) is making different elasticity claims than its neighbors.

## 21 — `21_multistart_elas_dotplot.png` (one per multi-seed spec)

`plot_multistart_elas_dotplot` (plot_results.py:347). One figure per spec that has multiple seeds with elasticity data. X-axis = seed, y-axis = own-price elasticity, one line per product across seeds. Lines colored by a hash of the product ID. Filename gets a `_specN` suffix when more than one multi-seed spec exists.

A nearly-flat fan of lines = stable own-price elasticities across local optima. Lines that cross or fan out = elasticity for that product depends on which basin BFGS landed in.

## 22 — `22_cross_elas_heatmap.png`

`plot_cross_elas_heatmap` (plot_results.py:387). Square heatmap of the elasticity matrix `ε_jk` for the rank-1 spec. Diverging colormap (`RdBu_r`) centered at 0: diagonal (own-price) red/negative, off-diagonal (cross-price) blue/positive. Up to 24 products shown by default; for larger markets the 24 most price-elastic products are kept. Annotations included when `n ≤ 30`.

The full picture of who-substitutes-with-whom under the preferred model. Reads as a "substitution map" — bright off-diagonal cells flag tightly competing products.

## 23 — `23_asymmetry_scatter.png`

`plot_asymmetry_scatter` (plot_results.py:446). Scatter of unordered product pairs from the rank-1 spec: x-axis = `e_jk`, y-axis = `e_kj`. Point color encodes `|e_jk − e_kj|` (sequential `YlOrRd`). Dashed 45° line = perfect symmetry.

Points away from the diagonal indicate asymmetric substitution: products whose share absorbs more from rivals than vice versa. Hot-colored, far-from-diagonal points are the most asymmetric — typically big products soaking up substitution.

## 24 — `24_spearman_corr_heatmap.png` (Nevo only)

`plot_nevo_spearman_heatmap` (plot_results.py:803). Square heatmap of Spearman rank correlations of own-price elasticities between every pair of specs, computed over products common to all specs. Diverging colormap (`RdYlGn`) from −1 to +1, centered at 0. Always annotated.

A heatmap close to all-green (ρ near 1) = rankings of price-sensitive products are stable across specs. Patches of yellow or red are where two specs disagree on which products are most elastic.

## 25 — `25_firm_substitution_heatmap.png` (Nevo only)

`plot_nevo_firm_substitution_heatmap` (plot_results.py:866). Square heatmap of mean cross-price elasticities aggregated to firm × firm level for the rank-1 spec. Firm IDs parsed from product IDs as the substring before `B` (`F1B04 → F1`). Sequential `YlOrRd` from 0 to `max(|ε|)`. Always annotated.

The diagonal shows within-firm substitution (mean cross-elasticity among one firm's own products). A bright diagonal vs. dimmer off-diagonal cells = consumers substitute within a firm's portfolio more than across firms — directly relevant to merger arguments.

## 26 — `26_own_stability_heatmap.png`

`plot_own_stability_heatmap` (plot_results.py:497). Two-panel figure: left panel is products × specs of own-price elasticity (diverging `coolwarm_r`, centered at 0); right sidebar is per-product CV across specs (sequential `Oranges`). Products limited to 20 most-elastic when there are >50 in total; rows sorted by CV descending (least-stable first). Annotated when ≤ 25 products and ≤ 12 specs.

Look at the top rows to see which products are most sensitive to model choice. Cells with very different colors across columns = elasticity flips materially as the spec changes.

## 27 — `27_cross_stability_heatmap.png`

`plot_cross_stability_heatmap` (plot_results.py:576). Heatmap of cross-price elasticity for the top-10 substitute pairs (selected from the rank-1 spec) across all specs (best seed each). Sequential `YlGnBu` from 0 to max. Row labels `j→k`; rows sorted by CV descending. Annotated when ≤ 20 rows and ≤ 12 columns.

The cross-elasticity analogue of figure 26, restricted to the pairs that matter most in the preferred model.

## 28 — `28_pairwise_mad_heatmap.png`

`plot_spec_pairwise_mad` (plot_results.py:657). Square heatmap of mean absolute deviation of own-price elasticities between every pair of specs. Sequential `Blues`; diagonal masked. Annotated when ≤ 15 specs.

Dark cells = specs that disagree on absolute elasticities (not just rankings). Complement to figure 24 (Spearman), which only sees rank agreement.

## 30 — `30_elasticity_pair_across_sims.png`

`plot_elasticity_pair_across_sims` (plot_results.py:980). 2 × 2 panel of scatter plots. Cross-section is *seeds* (x-axis = seed); spec held fixed = the lowest-mean-objective spec. Panels: `e_jj` for product j (top-left), `e_kk` for product k (top-right), `e_jk` (bottom-left), `e_kj` (bottom-right), for the pair with the highest mean cross-elasticity in that spec. Dashed horizontal line in each panel = mean across seeds.

Use it to see whether elasticities for a specific high-substitution pair are stable across local optima, or whether they jump when BFGS lands in a different basin.

## 31 — `31_elasticity_pair_best_sim_across_specs.png`

`plot_elasticity_pair_best_sim_across_specs` (plot_results.py:1068). 2 × 2 panel like figure 30, but the cross-section is *specifications* — one point per spec, each from that spec's best seed. X-axis = spec (sorted by objective rank); same four panels (`e_jj`, `e_kk`, `e_jk`, `e_kj`). Dashed horizontal line = mean across specs. Pair selected for maximum spec coverage, ties broken by mean cross-elasticity.

Companion to figure 30. Together they answer "do we get the same elasticities when we change (a) the seed within one spec, vs (b) the spec at its best seed?"

## 36 — `36_objective_spec_comparison.png`

`plot_objective_spec_comparison` (plot_results.py:1136). Horizontal bar chart, one bar per spec = mean GMM objective across that spec's valid starts (rows are pre-filtered by `_filter_valid`, plot_results.py:1486/:1449, so only `price_coef < 0 AND objective ≥ 0` starts reach the plot). Error bars = ±1 std; dashed `--` segments with `|` end-caps mark the min/max range; `n=<count>` (valid starts per spec) annotated to the right. Bars use a single uniform `COL_VALID` color — validity is *not* encoded, since every plotted start is already valid. Y-axis inverted so the lowest-mean (best) spec sits on top; specs sorted by mean ascending.

Unlike figure 01 (best valid start only), this is the *distribution* of objectives across valid starts. A spec with a low mean and a tight error bar = both well-fitting and reliably reachable; low mean with a wide error bar = the headline number depends on a lucky seed. (The companion text report — analysis 36 in [`ANALYSIS.md`](ANALYSIS.md) — tabulates the same valid-start distribution, with `N` / `N valid` columns showing how many starts were dropped per spec.)

## 37 — `37_demographic_expansion.png` (Nevo only)

`plot_nevo_demographic_expansion` (plot_results.py:715). Two-panel figure (`X2=sugar` left, `X2=mushy` right). Each panel has a dual y-axis: solid line = GMM objective (left axis), dashed line = `price_coef` (right axis). X-axis steps through demographic sets from sparse to rich in a hardcoded order.

Use it to read whether richer demographics monotonically improve fit, and how price-sensitivity moves as you add `income_squared`, `age`, `child`.

## 38 — `38_x2_comparison.png` (Nevo only)

`plot_nevo_x2_comparison` (plot_results.py:765). Side-by-side bar chart with dual y-axis: GMM objective (left axis bars) and `price_coef` (right axis bars), holding `demos=['income']`. Three X2 choices: `sugar`, `mushy`, `sugar+mushy`. Skipped if no matching specs exist (the script prints a `[NEVO] No matching specs` notice — happened in the most recent BLP-only run).

## 39

No graph produced — analysis 39 (starting-value sensitivity) is text-only.

---

## How to run

```bash
python plot_results.py   # writes PNGs to results/<dataset>/graphs/
```

BLP and Nevo are wrapped in independent `try/except` blocks, so a missing dataset doesn't block the other.

## Run-time notes

- Loading `<dataset>/csv/elasticities_detail.csv` is the slow step — BLP's is ~880 MB. Expect a delay before the first PNG appears for BLP.
- Heatmaps stop annotating cell values past hardcoded thresholds (e.g., 30 products in figure 22, 25 products in figure 26, 15 specs in figure 28) to keep labels legible.
- Plots that need ≥ 2 specs (figures 26, 27, 28) skip themselves with a `[label] <2 specs` notice when only one spec has data.
