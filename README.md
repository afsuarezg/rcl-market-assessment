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

Reads the multistart CSVs from both result directories and writes 9 analysis reports to `results/nevo/analysis/` and `results/blp/analysis/`.

## Output Files

Both estimation scripts produce four files per dataset:

| File | Contents |
|---|---|
| `multistart_all.csv` | One row per random start per specification |
| `multistart_best.csv` | Best start (lowest GMM objective) per specification |
| `post_estimation_summary.csv` | Elasticities, diversion ratios, markups, HHI, merger effects |
| `elasticities_detail.csv` | Full own- and cross-price elasticity matrices |

`analyze_results.py` adds text reports:

| File | Contents |
|---|---|
| `analysis/01_objective_ranking.txt` | GMM objective ranking across Nevo specs |
| `analysis/02_demographic_expansion.txt` | Effect of adding demographics (fixed X2) |
| `analysis/03_x2_comparison.txt` | X2 characteristic comparison (fixed demographics) |
| `analysis/04_price_coef_sensitivity.txt` | Price coefficient range and implied markups |
| `analysis/05_multistart_stability.txt` | Convergence spread across random starts |
| `analysis/06_convergence_audit.txt` | Valid vs invalid BLP starts (price_coef < 0 check) |
| `analysis/07_global_minimum.txt` | Global minimum identification for BLP |
| `analysis/08_two_basin_analysis.txt` | Two-basin structure of the BLP objective |
| `analysis/09_starting_value_sensitivity.txt` | Starting value differences between valid/invalid starts |

The `results/` directory is git-ignored. CSVs must be regenerated locally.

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
