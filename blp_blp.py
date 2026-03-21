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
from pathlib import Path
from typing import NamedTuple, Optional

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
_BLP_SIGMA = np.diag([3.612, 0, 4.628, 1.818, 1.050, 2.056])
_BLP_PI    = np.array([[0], [-43.501], [0], [0], [0], [0]], dtype=float)


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
    if not force_random and x2_vars == _BLP_X2 and demo_vars == _BLP_DEMOS:
        return _BLP_SIGMA.copy(), _BLP_PI.copy()

    K2 = 2 + len(x2_vars)
    rng = np.random.default_rng(seed)
    sigma_init = np.diag(rng.uniform(0, 1, K2))
    if demo_vars is not None:
        # Price heterogeneity is captured via pi (price × demographics);
        # sigma[1,1] must be 0 to match the agent data's integration columns.
        sigma_init[1, 1] = 0.0

    if demo_vars is None:
        return sigma_init, None

    pi_init = rng.standard_normal((K2, len(demo_vars)))

    if n_instruments is not None:
        n_nonzero = K2 + K2 * len(demo_vars)
        if n_nonzero > n_instruments:
            max_pi_nonzero = n_instruments - K2
            if max_pi_nonzero <= 0:
                raise ValueError(
                    f"Specification requires at least {K2} instruments for sigma "
                    f"but only {n_instruments} excluded instruments are available."
                )
            flat = pi_init.flatten()
            flat[max_pi_nonzero:] = 0.0
            pi_init = flat.reshape(K2, len(demo_vars))

    return sigma_init, pi_init


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
) -> list[StartResult]:
    """
    Solve a specification n_starts times from different random starting points.

    Start 0 uses BLP (1995) published values for the full spec, or a seeded
    draw otherwise.  Starts 1..n_starts-1 always draw fresh random values
    (force_random=True) with seed = base_seed + start_index.

    Returns StartResult objects sorted ascending by objective (best = index 0).
    """
    problem = build_problem(
        product_data, agent_data, x2_vars, demo_vars,
        include_supply=include_supply,
    )
    n_instr = len([c for c in product_data.columns if c.startswith('demand_instruments')])

    results = []
    for i in range(n_starts):
        seed = base_seed + i
        force_random = (i > 0)
        sigma_init, pi_init = build_initial_params(
            x2_vars, demo_vars,
            n_instruments=n_instr,
            seed=seed,
            force_random=force_random,
        )
        res = solve_spec(
            problem, sigma_init, pi_init,
            gtol=gtol, method=method, include_supply=include_supply,
        )
        results.append(StartResult(result=res, sigma_init=sigma_init, pi_init=pi_init, seed=seed))

    return sorted(results, key=lambda sr: float(sr.result.objective))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Comparison helpers
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
# 8. Post-estimation summary
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
# 9. Elasticity export
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
# 10. Script entry point — grid over specifications
# ─────────────────────────────────────────────────────────────────────────────

def _append_csv(df: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    """Write df to path, appending below existing rows if the file exists."""
    if path.exists():
        existing = pd.read_csv(path, index_col=0 if index else None)
        df = pd.concat([existing, df])
    df.to_csv(path, index=index)


if __name__ == '__main__':
    OUT_DIR = Path('results/blp')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    product_data, agent_data = load_data()

    N_STARTS = 2  # number of random restarts per specification

    x2_combos = [
        # ['hpwt', 'air', 'mpd', 'space'],  # full BLP (1995) spec
        ['hpwt', 'air', 'mpd'],
        # ['hpwt', 'air'],
    ]
    demo_combos = [
        ['I(1 / income)'],  # only demographic available in BLP agent data
    ]

    multistart_results = {}
    for x2 in x2_combos:
        for demos in demo_combos:
            label = f"x2={x2} | demos={demos}"
            all_csv = OUT_DIR / 'blp_multistart_all.csv'
            base_seed = 0
            if all_csv.exists():
                existing_all = pd.read_csv(all_csv)
                spec_rows = existing_all[
                    (existing_all['spec'] == label) & existing_all['seed'].notna()
                ]
                if not spec_rows.empty:
                    base_seed = int(spec_rows['seed'].max()) + 1
            print(f"\nSolving ({N_STARTS} starts): {label}, base_seed={base_seed}")
            multistart_results[label] = run_multistart(
                product_data, agent_data, x2, demos, n_starts=N_STARTS, base_seed=base_seed,
            )

    detail = compare_multistart_results(multistart_results)
    print("\n=== All Starts ===")
    print(detail.to_string(index=False))
    _append_csv(detail, OUT_DIR / 'blp_multistart_all.csv', index=False)
    print("Saved: blp_multistart_all.csv")

    print("\n=== Best per Specification ===")
    best = detail[detail['best']].drop(columns='best').set_index('spec')
    print(best.to_string())
    _append_csv(best, OUT_DIR / 'blp_multistart_best.csv')
    print("Saved: blp_multistart_best.csv")

    print("\n=== Post-Estimation: Elasticities, Markups & Merger Simulation ===")
    post = summarise_post_estimation(multistart_results, product_data)
    print(post.to_string())
    _append_csv(post, OUT_DIR / 'blp_post_estimation_summary.csv')
    print("Saved: blp_post_estimation_summary.csv")

    print("\n=== Exporting Full Elasticity Matrices ===")
    elast = export_elasticities(multistart_results, product_data)
    _append_csv(elast, OUT_DIR / 'blp_elasticities_detail.csv', index=False)
    print("Saved: blp_elasticities_detail.csv")
