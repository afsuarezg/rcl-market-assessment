#!/usr/bin/env python
# coding: utf-8

"""
blp_blp.py
----------
Reusable BLP estimation functions for the Berry, Levinsohn & Pakes (1995)
automobile dataset.  Mirrors the architecture of nevo_blp.py.
Run as a script to compare estimates across different characteristic specifications.
"""

import pyblp
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, Optional

from solve_diagnostics import extract_solve_diagnostics

pyblp.options.digits = 2
pyblp.options.verbose = False


class StartResult(NamedTuple):
    """Bundles a solved result with the initial parameters used to start it."""
    result:     pyblp.ProblemResults
    sigma_init: np.ndarray
    pi_init:    Optional[np.ndarray]
    seed:       int


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BLP (1995) product and agent data."""
    product_data = pd.read_csv(pyblp.data.BLP_PRODUCTS_LOCATION)
    agent_data   = pd.read_csv(pyblp.data.BLP_AGENTS_LOCATION)
    return product_data, agent_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Problem builder
# ─────────────────────────────────────────────────────────────────────────────

def build_problem(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    include_supply: bool = True,
) -> pyblp.Problem:
    """
    Build a pyblp.Problem for the BLP (1995) automobile data.

    Parameters
    ----------
    product_data, agent_data
        Raw DataFrames from load_data().
    x2_vars
        Product characteristics for X2 beyond the constant and prices.
        E.g. ['hpwt', 'air', 'mpd', 'space'].
    demo_vars
        Demographic variables for the agent formulation.
        If None, no agent formulation is included.
    include_supply
        Whether to include the supply-side (X3) formulation and
        estimate marginal costs via the log-linear specification.
    """
    X1_formulation = pyblp.Formulation('1 + hpwt + air + mpd + space')
    X2_formulation = pyblp.Formulation('1 + prices + ' + ' + '.join(x2_vars))

    if include_supply:
        X3_formulation = pyblp.Formulation(
            '1 + log(hpwt) + air + log(mpg) + log(space) + trend'
        )
        product_formulations = (X1_formulation, X2_formulation, X3_formulation)
    else:
        product_formulations = (X1_formulation, X2_formulation)

    kwargs = {'costs_type': 'log'} if include_supply else {}

    if demo_vars is None:
        return pyblp.Problem(product_formulations, product_data, **kwargs)

    agent_formulation = pyblp.Formulation('0 + ' + ' + '.join(demo_vars))
    return pyblp.Problem(
        product_formulations, product_data,
        agent_formulation, agent_data,
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Initial parameter builder
# ─────────────────────────────────────────────────────────────────────────────

_BLP_X2    = ['hpwt', 'air', 'mpd', 'space']
_BLP_DEMOS = ['I(1 / income)']
_BLP_SIGMA   = np.diag([3.612, 0, 4.628, 1.818, 1.050, 2.056])
_BLP_PI      = np.array([[0], [-43.501], [0], [0], [0], [0]], dtype=float)


def _random_pi_init(
    K2: int,
    n_demo: int,
    n_instruments: Optional[int],
    rng: 'np.random.Generator',
) -> np.ndarray:
    """
    Draw a pi_init matrix with randomly chosen sparsity.

    K2 nonzero sigma entries are assumed fixed; the remaining budget
    (n_instruments - K2) determines how many pi entries can be nonzero.
    Positions are chosen uniformly at random; values from standard_normal.
    If n_instruments is None all entries are activated.
    """
    n_total = K2 * n_demo
    if n_instruments is not None:
        max_nonzero = n_instruments - K2
        if max_nonzero <= 0:
            raise ValueError(
                f"Specification requires at least {K2} instruments for sigma "
                f"but only {n_instruments} excluded instruments are available."
            )
        n_active = min(max_nonzero, n_total)
    else:
        n_active = n_total
    indices = rng.choice(n_total, size=n_active, replace=False)
    pi = np.zeros((K2, n_demo))
    pi.flat[indices] = rng.standard_normal(n_active)
    return pi


def build_initial_params(
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    n_instruments: Optional[int] = None,
    seed: int = 0,
    force_random: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (sigma_init, pi_init) appropriate for the given specification.

    When x2_vars == ['hpwt', 'air', 'mpd', 'space'] and
    demo_vars == ['I(1 / income)'], the BLP (1995) published starting values
    are returned.  Otherwise sigma_init is drawn from Uniform(0, 1) and
    pi_init (when demo_vars is not None) from standard_normal.

    K2 = 2 + len(x2_vars)  (constant + prices + x2_vars)

    Parameters
    ----------
    n_instruments
        Number of excluded demand instruments.  If provided, an order-condition
        check trims pi_init rows to avoid underidentification.
    seed
        RNG seed for reproducibility of random starting values.
    force_random
        Skip the BLP-baseline detection and always draw random values.
    """
    K2 = 2 + len(x2_vars)
    rng = np.random.default_rng(seed)

    # For the exact baseline specification use published sigma; randomise pi with seed.
    if not force_random and x2_vars == _BLP_X2 and demo_vars == _BLP_DEMOS:
        return _BLP_SIGMA.copy(), _random_pi_init(K2, len(demo_vars), n_instruments, rng)

    sigma_init = np.diag(rng.uniform(0, 1, K2))
    if demo_vars is not None:
        # Price heterogeneity is captured via pi (price × demographics);
        # sigma[1,1] must be 0 to match the agent data's integration columns.
        sigma_init[1, 1] = 0.0

    if demo_vars is None:
        return sigma_init, None

    return sigma_init, _random_pi_init(K2, len(demo_vars), n_instruments, rng)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_spec(
    problem: pyblp.Problem,
    sigma_init: np.ndarray,
    pi_init: Optional[np.ndarray] = None,
    gtol: float = 1e-5,
    method: str = '1s',
    include_supply: bool = True,
    clustered: bool = True,
    initial_update: bool = True,
) -> pyblp.ProblemResults:
    """Solve a pyblp.Problem with BFGS and BLP-appropriate options."""
    optimization = pyblp.Optimization('bfgs', {'gtol': gtol})
    kwargs: dict = dict(optimization=optimization, method=method)

    if clustered:
        kwargs['W_type']  = 'clustered'
        kwargs['se_type'] = 'clustered'

    if initial_update:
        kwargs['initial_update'] = True

    if include_supply:
        kwargs['costs_bounds'] = (0.001, None)

    if pi_init is not None:
        return problem.solve(sigma_init, pi_init, **kwargs)
    return problem.solve(sigma_init, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 5. End-to-end convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_specification(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    include_supply: bool = True,
    gtol: float = 1e-5,
    method: str = '1s',
    seed: int = 0,
) -> pyblp.ProblemResults:
    """Build the problem, construct initial parameters, and solve — all in one call."""
    problem = build_problem(
        product_data, agent_data, x2_vars, demo_vars,
        include_supply=include_supply,
    )
    n_instr = len([c for c in product_data.columns if c.startswith('demand_instruments')])
    sigma_init, pi_init = build_initial_params(
        x2_vars, demo_vars, n_instruments=n_instr, seed=seed,
    )
    return solve_spec(
        problem, sigma_init, pi_init,
        gtol=gtol, method=method, include_supply=include_supply,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-start runner
# ─────────────────────────────────────────────────────────────────────────────

def run_multistart(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    n_starts: int = 5,
    base_seed: int = 0,
    include_supply: bool = True,
    gtol: float = 1e-5,
    method: str = '1s',
) -> tuple[list[StartResult], list[tuple[int, Exception]]]:
    """
    Solve a specification n_starts times from different random starting points.

    Start 0 uses BLP (1995) published values for the full spec, or a seeded
    draw otherwise.  Starts 1..n_starts-1 always draw fresh random values
    (force_random=True) with seed = base_seed + start_index.

    A start whose solve raises is collected instead of killing the batch.

    Returns (results, failures): StartResult objects sorted ascending by
    objective (best = index 0), and (seed, exception) pairs for starts that
    raised, so callers can record them as diagnostics.
    """
    problem = build_problem(
        product_data, agent_data, x2_vars, demo_vars,
        include_supply=include_supply,
    )
    n_instr = len([c for c in product_data.columns if c.startswith('demand_instruments')])

    results = []
    failures: list[tuple[int, Exception]] = []
    for i in range(n_starts):
        seed = base_seed + i
        force_random = (i > 0)
        sigma_init, pi_init = build_initial_params(
            x2_vars, demo_vars,
            n_instruments=n_instr,
            seed=seed,
            force_random=force_random,
        )
        try:
            res = solve_spec(
                problem, sigma_init, pi_init,
                gtol=gtol, method=method, include_supply=include_supply,
            )
        except Exception as exc:  # noqa: BLE001 — one bad start must not kill the batch
            print(f"  Dropping seed {seed}: stage-1 solve raised "
                  f"({type(exc).__name__}: {exc}).")
            failures.append((seed, exc))
            continue
        results.append(StartResult(result=res, sigma_init=sigma_init, pi_init=pi_init, seed=seed))

    return sorted(results, key=lambda sr: float(sr.result.objective)), failures


# ─────────────────────────────────────────────────────────────────────────────
# 7. Optimal instruments
# ─────────────────────────────────────────────────────────────────────────────

def is_usable_start(sr: StartResult) -> tuple[bool, str]:
    """Pre-screen a first-stage start before applying optimal instruments.

    Returns (ok, reason). A start is rejected if it failed to converge, has a
    non-finite objective or parameters, or implies non-finite marginal costs.
    The log-linear marginal-cost spec recovers costs as exp(fitted log-cost),
    which overflows to inf for extreme supply-side estimates; those inf costs
    later crash to_problem()'s collinearity check, so we screen them out here.
    """
    res = sr.result
    if not res.converged:
        return False, "did not converge"
    if not np.isfinite(float(res.objective)):
        return False, "non-finite objective"
    for name, arr in (('sigma', res.sigma), ('pi', res.pi),
                      ('beta', res.beta), ('gamma', res.gamma)):
        if arr is not None and not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            return False, f"non-finite {name}"
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            costs = res.compute_costs()
        if not np.all(np.isfinite(costs)):
            return False, "non-finite marginal costs"
    except Exception as e:  # noqa: BLE001 — any failure means the start is unusable
        return False, f"cost computation failed ({type(e).__name__}: {e})"
    return True, "ok"


def apply_optimal_instruments(
    sr: StartResult,
    gtol: float = 1e-5,
    method: str = '1s',
    include_supply: bool = True,
    clustered: bool = True,
) -> StartResult:
    """Re-solve from sr using approximate optimal instruments."""
    instrument_results = sr.result.compute_optimal_instruments(method='approximate')
    updated_problem = instrument_results.to_problem()
    updated_result = solve_spec(updated_problem, sr.result.sigma, sr.result.pi,
                                gtol=gtol, method=method,
                                include_supply=include_supply, clustered=clustered,
                                initial_update=False)
    return StartResult(result=updated_result, sigma_init=sr.result.sigma,
                       pi_init=sr.result.pi, seed=sr.seed)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(results_dict: dict[str, pyblp.ProblemResults]) -> pd.DataFrame:
    """
    Summarise a collection of solved specifications into a tidy DataFrame.

    Columns: price_coef, objective, sigma_0 … sigma_{K2-1},
             and (when demographics are present) pi_{i}_{j} for each entry.

    price_coef is res.pi[1, 0] (price × 1/income interaction) when demographics
    are present, or res.sigma[1, 1] (price random-coef std dev) otherwise.
    Prices are in X2 only (not X1), so res.beta contains [constant, hpwt, …].
    """
    rows = []
    for label, res in results_dict.items():
        row = {
            'label':      label,
            'price_coef': (float(res.pi[1, 0]) if res.pi is not None
                           else float(res.sigma[1, 1])),
            'objective':  float(res.objective),
        }
        for i, v in enumerate(np.diag(res.sigma)):
            row[f'sigma_{i}'] = float(v)
        if res.pi is not None:
            for i in range(res.pi.shape[0]):
                for j in range(res.pi.shape[1]):
                    row[f'pi_{i}_{j}'] = float(res.pi[i, j])
        rows.append(row)
    return pd.DataFrame(rows).set_index('label')


def compare_multistart_results(
    multistart_dict: dict[str, list[StartResult]],
) -> pd.DataFrame:
    """
    Summarise multi-start results into a tidy DataFrame.

    Columns: spec, start, price_coef, objective, best,
             init_sigma_0 … init_sigma_{K2-1},
             (when demographics are present) init_pi_{i}_{j},
             est_sigma_0 … est_sigma_{K2-1},
             and (when demographics are present) est_pi_{i}_{j}.

    price_coef is res.pi[1, 0] (price × 1/income interaction) when demographics
    are present, or res.sigma[1, 1] (price random-coef std dev) otherwise.
    Prices are in X2 only (not X1), so res.beta contains [constant, hpwt, …].
    """
    rows = []
    for label, sr_list in multistart_dict.items():
        for i, sr in enumerate(sr_list):
            row = {
                'spec':       label,
                'start':      i,
                'seed':       sr.seed,
                'price_coef': (float(sr.result.pi[1, 0]) if sr.result.pi is not None
                               else float(sr.result.sigma[1, 1])),
                'objective':  float(sr.result.objective),
                'best':       (i == 0),
            }
            sigma_labels = sr.result.sigma_labels
            pi_labels    = sr.result.pi_labels
            for lbl, v in zip(sigma_labels, np.diag(sr.sigma_init)):
                row[f'init_sigma[{lbl}]'] = float(v)
            if sr.pi_init is not None:
                for r, rlbl in enumerate(sigma_labels):
                    for c, clbl in enumerate(pi_labels):
                        row[f'init_pi[{rlbl},{clbl}]'] = float(sr.pi_init[r, c])
            for lbl, v in zip(sigma_labels, np.diag(sr.result.sigma)):
                row[f'est_sigma[{lbl}]'] = float(v)
            if sr.result.pi is not None:
                for r, rlbl in enumerate(sigma_labels):
                    for c, clbl in enumerate(pi_labels):
                        row[f'est_pi[{rlbl},{clbl}]'] = float(sr.result.pi[r, c])
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Post-estimation summary
# ─────────────────────────────────────────────────────────────────────────────

def summarise_post_estimation(
    multistart_results: dict[str, list[StartResult]],
    product_data: pd.DataFrame,
    include_supply: bool = True,
) -> pd.DataFrame:
    """
    Compute demand- and supply-side post-estimation statistics for each
    specification, using the best start (index 0).

    Demand columns (always present):
        mean_own_elas     — mean own-price elasticity
        mean_outside_div  — mean diversion to outside good

    Supply columns (present when results include a supply side and
    include_supply=True):
        mean_markup       — mean markup  (p − mc) / p
        mean_hhi          — mean HHI across markets
        mean_delta_markup — mean markup change from merging firm 2 → 1
        mean_delta_hhi    — mean HHI change from merging firm 2 → 1
        mean_delta_cs     — mean consumer-surplus change from merging firm 2 → 1

    Supply columns are NaN when the result has no marginal-cost estimates or
    when include_supply=False.

    Parameters
    ----------
    multistart_results:
        Mapping of spec label → list of StartResult (best start at index 0).
    product_data:
        Product-level DataFrame used to build merger firm_ids.
    include_supply:
        If False, skip all supply-side calculations even when the result
        includes marginal-cost estimates. Default True.
    """
    rows = []
    for label, starts in multistart_results.items():
        res = starts[0].result

        mean_own_elas    = float(
            np.asarray(res.extract_diagonal_means(res.compute_elasticities())).mean()
        )
        mean_outside_div = float(
            np.asarray(res.extract_diagonal_means(res.compute_diversion_ratios())).mean()
        )

        row: dict = {
            'label':           label,
            'mean_own_elas':   mean_own_elas,
            'mean_outside_div': mean_outside_div,
            'mean_markup':     np.nan,
            'mean_hhi':        np.nan,
            'mean_delta_markup': np.nan,
            'mean_delta_hhi':  np.nan,
            'mean_delta_cs':   np.nan,
        }

        # Supply-side statistics (only when marginal costs are available and requested)
        if include_supply:
            try:
                costs   = res.compute_costs()
                markups = res.compute_markups(costs=costs)
                hhi     = res.compute_hhi()
                cs      = res.compute_consumer_surpluses()

                row['mean_markup'] = float(np.asarray(markups).mean())
                row['mean_hhi']    = float(np.asarray(hhi).mean())

                # Merger simulation: firm 2 acquires firm 1 (BLP tutorial convention)
                merger_ids    = product_data['firm_ids'].replace(2, 1)
                changed_prices = res.compute_prices(
                    firm_ids=merger_ids, costs=costs
                )
                changed_shares = res.compute_shares(changed_prices)
                changed_markups = res.compute_markups(changed_prices, costs)
                changed_hhi     = res.compute_hhi(
                    firm_ids=merger_ids, shares=changed_shares
                )
                changed_cs      = res.compute_consumer_surpluses(changed_prices)

                row['mean_delta_markup'] = float(
                    np.asarray(changed_markups - markups).mean()
                )
                row['mean_delta_hhi'] = float(
                    np.asarray(changed_hhi - hhi).mean()
                )
                row['mean_delta_cs'] = float(
                    np.asarray(changed_cs - cs).mean()
                )
            except (AttributeError, pyblp.exceptions.MultipleErrors):
                pass

        rows.append(row)

    return pd.DataFrame(rows).set_index('label')


# ─────────────────────────────────────────────────────────────────────────────
# 11. Elasticity export
# ─────────────────────────────────────────────────────────────────────────────

def export_elasticities(
    multistart_results: dict[str, list[StartResult]],
    product_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Export own- and cross-price elasticities aggregated (averaged) over all
    markets, one row per (spec, seed, product_j, product_k).

    Columns: spec, seed, product_j, product_k, elasticity, own_price

    elasticity is the mean of ε_jkt across all markets where both j and k
    appear. seed identifies the best start's random seed, allowing multiple
    runs of the same spec to be distinguished when rows are appended.
    """
    rows = []
    # Sort to match PyBLP's internal product ordering (by market_ids ascending)
    product_data = product_data.sort_values('market_ids').reset_index(drop=True)
    markets = np.sort(product_data['market_ids'].unique())
    id_col = 'product_ids' if 'product_ids' in product_data.columns else 'car_ids'
    for label, starts in multistart_results.items():
        for start in starts:
            res  = start.result
            seed = start.seed
            # compute_elasticities() returns a flat (N,) array; each element is a
            # 1-D array of length J_t — the j-th row of the J_t × J_t matrix.
            elasticities = res.compute_elasticities()

            # Accumulate elasticity values per (product_j, product_k) pair across markets
            pair_vals: dict[tuple, list[float]] = {}
            flat_idx = 0
            for market_id in markets:
                mask = product_data['market_ids'] == market_id
                product_ids = product_data.loc[mask, id_col].values
                J_t = len(product_ids)
                # Stack J_t rows → J_t × J_t matrix
                E_t = np.stack(list(elasticities[flat_idx:flat_idx + J_t]))
                for j, prod_j in enumerate(product_ids):
                    for k, prod_k in enumerate(product_ids):
                        pair = (prod_j, prod_k)
                        pair_vals.setdefault(pair, []).append(float(E_t[j, k]))
                flat_idx += J_t

            for (prod_j, prod_k), vals in pair_vals.items():
                rows.append({
                    'spec':       label,
                    'seed':       seed,
                    'product_j':  prod_j,
                    'product_k':  prod_k,
                    'elasticity': float(np.mean(vals)),
                    'own_price':  prod_j == prod_k,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Script entry point — grid over specifications
# ─────────────────────────────────────────────────────────────────────────────

def _blp_out_dir(results_root: Path) -> Path:
    """Resolve where blp CSVs live: the canonical csv/ subdir, unless only the
    legacy flat dir already holds accumulated data (e.g. older cluster runs).

    This keeps top-up runs appending to the existing history wherever it sits,
    instead of silently starting a fresh file in a different directory.
    """
    blp = results_root / 'blp'
    if (blp / 'blp_multistart_all.csv').exists() \
            and not (blp / 'csv' / 'blp_multistart_all.csv').exists():
        return blp
    return blp / 'csv'


def _all_nonempty_subsets(items: list[str]) -> list[list[str]]:
    return [list(s) for r in range(1, len(items) + 1) for s in combinations(items, r)]


_BLP_X2_VARS   = ['hpwt', 'air', 'mpd', 'space']
_BLP_DEMO_VARS = ['I(1 / income)']
_X2_OPTIONS   = _all_nonempty_subsets(_BLP_X2_VARS)
_DEMO_OPTIONS = _all_nonempty_subsets(_BLP_DEMO_VARS)


def _prompt_combos(options: list[list[str]], label: str) -> list[list[str]]:
    """Print numbered options and return the user-selected subset."""
    print(f"\nAvailable {label} combinations:")
    for i, opt in enumerate(options):
        print(f"  [{i}] {opt}")
    raw = input(f"Select {label} indices (comma-separated, or 'all'): ").strip()
    if raw.lower() == 'all':
        return list(options)
    indices = [int(x.strip()) for x in raw.split(',')]
    return [options[i] for i in indices]


def _dedupe_multistart(df: pd.DataFrame, *, by: Optional[list], reflag_best: bool) -> pd.DataFrame:
    """Collapse appended multistart rows to one per key, keeping the lowest objective.

    The multistart_all / _best CSVs are appended across top-up runs, so the same
    (spec, seed) can recur — an identical re-run, or a divergent start that reused
    the seed. One seed is one simulation, so each (spec, seed) must fold to a
    single row: the best (lowest-objective) start. `by` is the list of key columns
    (None keys on the index, used by the spec-indexed _best file). With
    reflag_best, the `best` flag is rebuilt so exactly one row per spec — its
    global-minimum objective — is True (a plain concat leaves one True per run).
    """
    if 'objective' not in df.columns:
        return df
    df = df.copy()
    if by is not None:                      # column-keyed: concat left duplicate index labels
        df = df.reset_index(drop=True)
    df['__obj'] = pd.to_numeric(df['objective'], errors='coerce')
    df = df.sort_values(['spec', '__obj'] if 'spec' in df.columns else ['__obj'])
    if by is None:                          # index-keyed (spec) for the _best file
        df = df[~df.index.duplicated(keep='first')]
    else:
        df = df.drop_duplicates(subset=by, keep='first')
    if reflag_best and 'best' in df.columns and 'spec' in df.columns:
        df['best'] = False
        df.loc[df.groupby('spec')['__obj'].idxmin(), 'best'] = True
    return df.drop(columns='__obj')


def _append_csv(df: pd.DataFrame, path: Path, *, index: bool = True,
                dedupe: Optional[str] = None) -> None:
    """Write df to path, appending below existing rows if the file exists.

    `dedupe='all'` folds duplicate (spec, seed) rows after the concat (keeping the
    lowest objective and rebuilding `best`); `dedupe='best'` folds the spec-indexed
    best file to one row per spec. Used only for the multistart CSVs — the
    elasticity / post-estimation files legitimately keep many rows per (spec, seed).
    """
    if path.exists():
        existing = pd.read_csv(path, index_col=0 if index else None)
        df = pd.concat([existing, df])
    if dedupe == 'all':
        df = _dedupe_multistart(df, by=['spec', 'seed'], reflag_best=True)
    elif dedupe == 'best':
        df = _dedupe_multistart(df, by=None, reflag_best=False)
    df.to_csv(path, index=index)


def _record_diag(diag_rows, res_or_exc, *, stage, seed, label,
                 note="", always=False) -> None:
    """Append a diagnostics row for a failed / non-converged solve.

    Converged results are skipped unless ``always=True`` (used for pre-filter
    rejections and caught exceptions, which are failure events regardless of the
    underlying solve's convergence flag).
    """
    diag = extract_solve_diagnostics(res_or_exc, stage=stage, seed=seed, note=note)
    if always or diag["outcome"] != "converged":
        diag_rows.append({"spec": label, **diag})


def _flush_diag(diag_rows: list[dict], path: Path, label: str) -> None:
    """Append accumulated diagnostics rows to path and clear the buffer.

    Called after each stage (not just at the end of a spec) so a crash or
    walltime kill cannot lose already-collected diagnostics.
    """
    if diag_rows:
        _append_csv(pd.DataFrame(diag_rows), path, index=False)
        print(f"Saved: {path.name}  [{label}] "
              f"({len(diag_rows)} failed/non-converged row(s))")
        diag_rows.clear()


def main(
    x2_combos:    Optional[list[list[str]]] = None,
    demo_combos:  Optional[list[list[str]]] = None,
    n_starts:     Optional[int]             = None,
    target_seeds: int                       = 40,
) -> None:
    if x2_combos is None:
        x2_combos = _prompt_combos(_X2_OPTIONS, 'x2')
    if demo_combos is None:
        demo_combos = _prompt_combos(_DEMO_OPTIONS, 'demo')

    _OAK_ROOT    = Path('/oak/stanford/groups/polinsky/blp_nevo')
    _LOCAL_ROOT  = Path(__file__).parent / 'results'
    _RESULTS_ROOT = _OAK_ROOT / 'results' if _OAK_ROOT.exists() else _LOCAL_ROOT
    # Write where the accumulated data already lives so top-up runs append to it
    # (canonical csv/ subdir read by count_seeds_per_spec.py / analyze_results.py /
    # plot_results.py, with a fallback to the legacy flat dir).
    OUT_DIR = _blp_out_dir(_RESULTS_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RAW_ALL_CSV  = OUT_DIR / 'blp_multistart_raw_all.csv'
    OPT_ALL_CSV  = OUT_DIR / 'blp_multistart_all.csv'
    OPT_BEST_CSV = OUT_DIR / 'blp_multistart_best.csv'
    POST_CSV     = OUT_DIR / 'blp_post_estimation_summary.csv'
    ELAST_CSV    = OUT_DIR / 'blp_elasticities_detail.csv'
    DIAG_CSV     = OUT_DIR / 'blp_solve_diagnostics.csv'

    product_data, agent_data = load_data()

    def _seeds_by_spec(path: Path) -> dict[str, set[int]]:
        out: dict[str, set[int]] = {}
        if path.exists():
            df = pd.read_csv(path, usecols=['spec', 'seed'])
            df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
            df = df[df['seed'].notna()]
            for spec_label, grp in df.groupby('spec'):
                out[spec_label] = set(grp['seed'].astype(int))
        return out

    # Count against converged seeds (OPT_ALL_CSV / blp_multistart_all.csv): a
    # start that solves in stage 1 but fails the optimal-instruments re-solve
    # never reaches blp_multistart_all.csv, so counting raw stage-1 seeds would
    # mark a spec "done" while it is still short of the target there.
    converged_seeds_per_spec = _seeds_by_spec(OPT_ALL_CSV)
    # Attempted seeds (RAW_ALL_CSV) and failed seeds (DIAG_CSV, for starts that
    # raised before reaching the raw CSV) are unioned in only to pick a
    # collision-free base_seed, never to count progress toward the target.
    attempted_seeds_per_spec = _seeds_by_spec(RAW_ALL_CSV)
    failed_seeds_per_spec    = _seeds_by_spec(DIAG_CSV)

    for x2 in x2_combos:
        for demos in demo_combos:
            label = f"x2={x2} | demos={demos}"

            existing_seeds = converged_seeds_per_spec.get(label, set())
            n_existing     = len(existing_seeds)
            used_seeds     = (existing_seeds
                              | attempted_seeds_per_spec.get(label, set())
                              | failed_seeds_per_spec.get(label, set()))
            base_seed      = max(used_seeds) + 1 if used_seeds else 0

            if n_starts is None:
                n_to_run = max(0, target_seeds - n_existing)
                if n_to_run == 0:
                    print(f"\nSkipping {label}: already has {n_existing} unique seeds "
                          f"(target={target_seeds}).")
                    continue
            else:
                n_to_run = n_starts

            # Diagnostics for failed / non-converged solves (this spec).
            diag_rows: list[dict] = []

            try:
                # ── Stage 1: multi-start ──────────────────────────────────────
                print(f"\nSolving ({n_to_run} starts): {label}, "
                      f"base_seed={base_seed}, existing={n_existing}/{target_seeds}")
                starts, stage1_failures = run_multistart(
                    product_data, agent_data, x2, demos, n_starts=n_to_run, base_seed=base_seed,
                )
                for seed, exc in stage1_failures:          # starts whose solve raised
                    _record_diag(diag_rows, exc, stage="stage1_solve",
                                 seed=seed, label=label, always=True)

                if starts:
                    raw_detail = compare_multistart_results({label: starts})
                    _append_csv(raw_detail, RAW_ALL_CSV, index=False)
                    print(f"Saved: blp_multistart_raw_all.csv  [{label}]")

                for sr in starts:                          # non-converged stage-1 starts
                    _record_diag(diag_rows, sr.result, stage="stage1_solve",
                                 seed=sr.seed, label=label)
                _flush_diag(diag_rows, DIAG_CSV, label)    # stage-1 diags survive a stage-2 crash

                # ── Stage 2: optimal instruments ─────────────────────────────
                print(f"\nApplying optimal instruments: {label} ({len(starts)} start(s))")
                opt_starts = []
                for sr in starts:
                    ok, reason = is_usable_start(sr)            # Option 2: pre-filter
                    if not ok:
                        print(f"  Skipping seed {sr.seed}: {reason} (pre-filter).")
                        _record_diag(diag_rows, sr.result, stage="stage2_prefilter",
                                     seed=sr.seed, label=label, note=reason, always=True)
                        continue
                    try:                                        # Option 1: backstop
                        opt_sr = apply_optimal_instruments(sr)
                        opt_starts.append(opt_sr)
                        _record_diag(diag_rows, opt_sr.result, stage="stage2_opt_iv",
                                     seed=sr.seed, label=label)   # logged only if not converged
                    except Exception as e:  # noqa: BLE001 — one bad start must not kill the batch
                        print(f"  Dropping seed {sr.seed}: optimal instruments failed "
                              f"({type(e).__name__}: {e}).")
                        _record_diag(diag_rows, e, stage="stage2_opt_iv",
                                     seed=sr.seed, label=label, always=True)

                _flush_diag(diag_rows, DIAG_CSV, label)

                if not opt_starts:
                    print(f"  No usable starts for {label} after optimal instruments; "
                          f"skipping spec.")
                    continue
                opt_starts = sorted(opt_starts, key=lambda sr: float(sr.result.objective))

                opt_detail = compare_multistart_results({label: opt_starts})
                print("\n=== All Starts ===")
                print(opt_detail.to_string(index=False))
                _append_csv(opt_detail, OPT_ALL_CSV, index=False, dedupe='all')

                opt_best = opt_detail[opt_detail['best']].drop(columns='best').set_index('spec')
                print("\n=== Best per Specification ===")
                print(opt_best.to_string())
                _append_csv(opt_best, OPT_BEST_CSV, dedupe='best')

                post = summarise_post_estimation({label: opt_starts}, product_data)
                print("\n=== Post-Estimation: Elasticities, Markups & Merger Simulation ===")
                print(post.to_string())
                _append_csv(post, POST_CSV)

                elast = export_elasticities({label: opt_starts}, product_data)
                _append_csv(elast, ELAST_CSV, index=False)

                print(f"Saved: blp_multistart_all.csv, blp_multistart_best.csv, "
                      f"blp_post_estimation_summary.csv, blp_elasticities_detail.csv  [{label}]")
            except Exception as exc:  # noqa: BLE001 — one bad spec must not kill the sweep
                print(f"  [spec failed] {label}: {type(exc).__name__}: {exc}")
                _record_diag(diag_rows, exc, stage="spec_fatal",
                             seed=None, label=label, always=True)
                _flush_diag(diag_rows, DIAG_CSV, label)


if __name__ == '__main__':
    main()
