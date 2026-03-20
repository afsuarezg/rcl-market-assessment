# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements **Random Coefficients Logit (BLP) demand estimation** for two standard econometric datasets using the [PyBLP](https://pyblp.readthedocs.io/) package:

- **Nevo (2000a)**: Fake cereal market data with consumer demographics
- **BLP (1995)**: Real automobile market data with consumer income data

The goal is to estimate heterogeneous consumer preferences, compute demand elasticities, markups, and simulate mergers.

## Environment Setup

Dependencies are pinned in `requirements-lock.txt`. To create a virtual environment:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

The key dependency is `pyblp==1.1.2`. All input data is bundled inside the `pyblp` package itself (no external data files needed).

## Running the Scripts

```bash
python nevo_blp.py    # Nevo (2000a) cereal estimation → results/nevo/*.csv
python blp_blp.py     # BLP (1995) automobile estimation → results/blp/*.csv
```

Each script runs multi-start BLP estimation across combinations of X2 characteristics and demographic variables, then exports three CSVs per dataset:
- `multistart_all.csv` — one row per start per specification
- `multistart_best.csv` — best solution per specification
- `post_estimation_summary.csv` — elasticities, diversion ratios, markups, HHI, merger simulation

The tutorial notebooks (`nevo_tutorial_original.ipynb`) are educational references from the pyblp documentation.

## Architecture

Both `nevo_blp.py` and `blp_blp.py` share the same structure:

| Function | Purpose |
|---|---|
| `load_data()` | Loads product and agent data from pyblp built-in datasets |
| `build_problem()` | Constructs `pyblp.Problem` with configurable X2 chars and optional supply side |
| `build_initial_params()` | Generates Sigma (Cholesky root) and Pi (demographic interactions) starting values |
| `solve_spec()` | Solves one specification via BFGS (`gtol=1e-5`, `method='1s'`) with clustered SEs |
| `run_specification()` | End-to-end runner for one spec |
| `run_multistart()` | Runs multiple random starts and selects the best (lowest GMM objective) |
| `compare_results()` / `compare_multistart_results()` | Aggregate results into DataFrames |
| `summarise_post_estimation()` | Computes elasticities, markups, HHI, and merger effects |

**Parameter structure:**
- **Sigma (Σ)**: Diagonal Cholesky root of the random coefficient covariance matrix. Sparsity pattern is preserved from initialization.
- **Pi (Π)**: `K2 × D` matrix of demographic interactions (e.g., price elasticity w.r.t. income).
- **Price coefficient extraction**: When demographics are present, use `pi[1,0]`; otherwise use `sigma[1,1]`.

**Supply side** (BLP only): Log-linear marginal cost specification. Merger simulation sets firm 2 acquiring firm 1. Supply side is disabled for Nevo (`include_supply=False`).

**Main block**: Both scripts define X2 characteristic and demographic variable combinations, loop through them calling `run_multistart()`, then write CSV outputs.

## Key Technical Notes

- The `results/` directory is git-ignored (`.csv` files excluded). Results must be regenerated locally.
- `N_STARTS` in each script controls the number of random restarts per specification (currently 1 for BLP, 2 for Nevo).
- Integration uses product rule (Gauss-Hermite) when agent data with demographics is available, Monte Carlo otherwise.
- Instrument order-condition guards prevent solving under-identified specifications.
