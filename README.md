# RCL Market Assessment

Random Coefficients Logit demand estimation using [PyBLP](https://pyblp.readthedocs.io/) for the Nevo (2000a) cereal and BLP (1995) automobile datasets.

## Overview

This project implements the Berry, Levinsohn & Pakes (1995) demand estimation framework to estimate heterogeneous consumer preferences, compute own- and cross-price elasticities, markups, and simulate mergers. It covers two canonical datasets:

- **Nevo (2000a)**: Fake cereal market with consumer demographics (income, age, child)
- **BLP (1995)**: Real automobile market with consumer income data

All input data is bundled inside the `pyblp` package — no external data files are required.

## Requirements

- Python 3.12+
- Key dependency: `pyblp==1.1.2` (see `requirements.txt` for full list)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Running the Pipeline

### Step 1 — Nevo (2000a) cereal estimation

```bash
python nevo_blp.py
```

Prompts you to select:
1. X2 characteristic combinations (nonlinear demand chars: `sugar`, `mushy`, or both)
2. Demographic variable combinations (`income`, `income_squared`, `age`, `child`, or subsets)
3. Number of random starting points per specification

Writes results to `results/nevo/`.

### Step 2 — BLP (1995) automobile estimation

```bash
python blp_blp.py
```

Same interactive prompts. X2 characteristics are `hpwt`, `air`, `mpd`, `space` (subsets). Demographics use inverse income `I(1/income)`. Includes a supply side with merger simulation.

Writes results to `results/blp/`.

### Step 3 — Post-hoc analysis

```bash
python analyze_results.py
```

Reads the multistart and elasticity CSVs from both result directories and writes 15 analysis reports to `results/nevo/analysis/` and 13 to `results/blp/analysis/`.

## Output Files

Both estimation scripts produce four files per dataset:

| File | Contents |
|---|---|
| `multistart_all.csv` | One row per random start per specification |
| `multistart_best.csv` | Best start (lowest GMM objective) per specification |
| `post_estimation_summary.csv` | Elasticities, diversion ratios, markups, HHI, merger effects |
| `elasticities_detail.csv` | Full own- and cross-price elasticity matrices |

`analyze_results.py` adds text reports:

| File | Dataset | Contents |
|---|---|---|
| `analysis/01_objective_ranking.txt` | Both | GMM objective ranking across specs with price coefficient |
| `analysis/02_multistart_stability.txt` | Both | GMM objective spread across random starts per spec |
| `analysis/03_convergence_audit.txt` | Both | Full start listing flagging invalid solutions (price_coef > 0) |
| `analysis/04_global_minimum.txt` | Both | Global minimum identification among valid starts |
| `analysis/05_two_basin_analysis.txt` | Both | Two-basin structure of the objective surface |
| `analysis/07_price_coef_sensitivity.txt` | Both | Price coefficient range and implied Lerner markups |
| `analysis/20_elasticity_own_summary.txt` | Both | Own-price elasticity distribution per spec ranked by objective |
| `analysis/21_elasticity_multistart_stability.txt` | Both | Product-level own-price elasticity spread across seeds |
| `analysis/22_elasticity_top_substitutes.txt` | Both | Top-5 substitute products per product (best spec) |
| `analysis/23_elasticity_asymmetry.txt` | Both | Distribution of cross-price asymmetry \|e_jk − e_kj\| |
| `analysis/24_elasticity_cross_spec_correlation.txt` | Nevo | Spearman rank correlation of own-price elasticities across specs |
| `analysis/25_elasticity_firm_substitution.txt` | Nevo | Within-firm vs between-firm substitution patterns |
| `analysis/37_demographic_expansion.txt` | Nevo | Effect of adding demographics one at a time (fixed X2) |
| `analysis/38_x2_comparison.txt` | Nevo | X2 characteristic comparison (fixed demographics = income) |
| `analysis/39_starting_value_sensitivity.txt` | Both | Initial parameter differences between valid and invalid starts |

The `results/` directory is git-ignored. CSVs must be regenerated locally.

## Analysis Descriptions

Section numbers are shared with `rcl_synthetic_data` (`plot_specs.py` / `GRAPHS.md`) so that the same analysis carries the same number in both repos. They are therefore not contiguous here: the Nevo-only and market-assessment-only analyses sit at 24, 25, 37, 38, 39.

### Specification & Convergence

These analyses characterize how well each specification fits the data and whether the multi-start optimizer reliably finds the global minimum. All are produced for both the Nevo and BLP datasets.

**01 — Objective ranking.** Ranks all specifications by their best-seed GMM objective value (lower = better fit) and reports the corresponding price coefficient. Use this table to identify the preferred specification and to see how much the objective degrades as you move down the ranking.

**02 — Multi-start convergence stability.** For each specification, shows the spread (max − min) of GMM objectives across all random starts, sorted from most to least variable. A spread near zero means the optimizer reliably converges to the same solution from different starting points; a large spread indicates a rough or multi-modal objective surface.

**03 — Convergence audit.** Lists every random start with its seed, start index, GMM objective, price coefficient, validity flag, and best-start marker. Starts where `price_coef > 0` are flagged as economically invalid (a positive price coefficient implies demand increases with price). Use this table to assess how many starts produce economically sensible solutions.

**04 — Global minimum.** Among all economically valid starts (price_coef < 0), identifies the solution with the lowest GMM objective and ranks the remaining valid starts by their distance from it. A cluster of starts near the same objective value provides confidence that the true global minimum has been found.

**05 — Two-basin analysis.** Classifies valid starts into Basin A (within 5 GMM units of the global minimum) and Basin B (farther away). Reports the range of objectives and price coefficients in each basin, along with mean estimated parameters. Two distinct basins with different parameter values indicate genuine multi-modality in the likelihood surface rather than numerical noise.

**07 — Price coefficient sensitivity.** Reports the range and spread of estimated price coefficients (α) across all specifications, together with the implied Lerner markup approximation −1/α. A wide range signals that the price elasticity estimate is sensitive to specification choice and warrants careful inspection of the preferred spec.

**39 — Starting-value sensitivity.** Compares the mean initial values of sigma (Σ) and pi (Π) parameters between valid and invalid starts. Systematic differences highlight which regions of the parameter space tend to lead the optimizer toward economically invalid solutions, informing better initialization strategies.

### Elasticity Levels & Distribution

These analyses describe the own- and cross-price elasticity estimates produced by the preferred specification and assess their robustness. All four analyses are produced for both datasets.

**20 — Own-price elasticity summary.** For each specification's best seed, computes the mean, median, standard deviation, minimum, and maximum of own-price elasticities across all products, displayed in objective-rank order. This shows both the typical level of price sensitivity in the market and how sensitive the elasticity estimates are to the choice of specification.

**21 — Elasticity multi-start stability.** For specifications estimated with more than one random start, reports the spread (max − min) of each product's own-price elasticity across seeds. A near-zero spread confirms that elasticity estimates are robust to starting values even when the GMM objective surface has multiple modes; a large spread signals that different starts produce economically meaningfully different demand estimates.

**22 — Top substitutes.** Using the best-fitting specification and its best seed, lists the top-5 products with the highest cross-price elasticity e_jk for each product j — that is, the products whose price increase would cause the largest demand increase for j. For datasets with many products (BLP), output is limited to the 10 most price-elastic products to keep the table readable.

**23 — Cross-price asymmetry.** For every product pair (j, k) in the best specification, computes |e_jk − e_kj| and reports distribution statistics (mean, median, maximum) and the share of pairs exceeding a 0.1 threshold. Large asymmetries arise naturally when products have very different market shares — a small product's demand responds strongly to the price of a large competitor, but not vice versa. The top-10 most asymmetric pairs are listed explicitly.

### Elasticity Structure (Nevo only)

These two analyses examine how the substitution structure varies across specifications and across firm boundaries. They are Nevo-specific because Nevo has multiple comparable specifications over the same 24 products and because product IDs encode firm identity directly.

**24 — Cross-spec elasticity correlation** *(Nevo only).* Computes the Spearman rank correlation between every pair of specifications' own-price elasticity vectors over the 24 products. A high correlation (close to 1) means that the ranking of products by price sensitivity is stable across specification choices even when the levels differ — a reassuring robustness check. A low correlation signals that the identity of the most and least elastic products depends heavily on which specification is used.

**25 — Firm substitution patterns** *(Nevo only).* Separates cross-price elasticities into within-firm pairs (both products made by the same manufacturer) and between-firm pairs, for each specification ranked by objective. Reports the mean elasticity in each group and their ratio. A ratio substantially above 1 indicates that consumers treat a firm's own products as closer substitutes for each other than for rival products — a key input to merger simulation and market-power analysis. The best specification's results are also presented as a firm × firm mean cross-elasticity matrix.

### Nevo Fit Diagnostics (Nevo only)

These two analyses probe how the Nevo fit responds to the demographic set and to the choice of nonlinear characteristic. Both are Nevo-specific.

**37 — Demographic expansion** *(Nevo only).* Holds the X2 nonlinear characteristic fixed and adds demographic variables one at a time in order of complexity, showing the change in GMM objective at each step. A negative delta indicates that the added demographic variable improves fit; a positive delta suggests it may be redundant or poorly identified.

**38 — X2 characteristic comparison** *(Nevo only).* Holds demographics fixed at income-only and compares `sugar`, `mushy`, and `['sugar', 'mushy']` as the nonlinear characteristic. Reveals how the choice of product characteristic for random-coefficient heterogeneity affects fit and the estimated price elasticity.

## Remote Server Usage

`analyze_results.py` automatically falls back to the Stanford Oak path
`/oak/stanford/groups/polinsky/blp_nevo/results/` when local result files are absent.
No changes are needed when running on Oak.

For other remote servers, either copy the `results/` directory alongside the script or
run the estimation scripts on the server first to generate the CSVs there.

## Project Structure

```
rcl_market_assessment/
├── nevo_blp.py              # Nevo (2000a) estimation
├── blp_blp.py               # BLP (1995) estimation
├── analyze_results.py       # Post-hoc analysis of multistart results
├── requirements.txt         # Direct dependencies
├── requirements-lock.txt    # Full pinned dependency lock
├── ipynb/                   # Educational reference notebooks (not part of pipeline)
└── results/                 # Generated outputs (git-ignored)
    ├── nevo/
    └── blp/
```

## Key Technical Notes

**Parameter structure**
- **Sigma (Σ)**: Diagonal Cholesky root of the random-coefficient covariance matrix. Sparsity pattern is preserved from initialization.
- **Pi (Π)**: `K2 × D` matrix of demographic interaction coefficients (e.g., price sensitivity with respect to income).
- When demographics are present in BLP, `sigma[1,1]` is fixed to zero and price heterogeneity is captured entirely through `pi[1,0]`.

**Multi-start**
- Start 0 uses published parameter values from the respective paper (or a seeded draw for non-baseline specs).
- Subsequent starts use fresh random draws with deterministic seeds for reproducibility.
- The best start (lowest GMM objective) is selected as the final estimate.

**Supply side**
- Included in BLP only (log-linear marginal cost specification).
- Merger simulation sets firm 2 acquiring firm 1 (standard BLP convention).
- Disabled for Nevo (`include_supply=False`).

**Instruments and identification**
- Automatic order-condition guards prevent solving underidentified specifications by trimming Pi entries as needed.
