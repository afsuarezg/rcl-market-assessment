#!/usr/bin/env python
# coding: utf-8

"""
nevo_blp.py
-----------
Reusable BLP estimation functions for the Nevo (2000a) fake cereal dataset.
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
    """Load Nevo (2000a) product and agent data."""
    product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
    return product_data, agent_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Problem builder
# ─────────────────────────────────────────────────────────────────────────────

def build_problem(
    product_data: pd.DataFrame,
    agent_data: pd.DataFrame,
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    integration: str = 'product',
    integration_size: int = 5,
    seed: int = 0,
) -> pyblp.Problem:
    """
    Build a pyblp.Problem with configurable nonlinear product and demographic
    characteristics.

    Parameters
    ----------
    product_data, agent_data
        Raw DataFrames from load_data().
    x2_vars
        Additional product characteristics for X2 beyond the constant and prices.
        E.g. ['sugar', 'mushy'].
    demo_vars
        Demographic variables to interact with X2.
        If None, no agent formulation is used and integration nodes are simulated.
    integration
        Integration method when demo_vars is None: 'product' or 'monte_carlo'.
    integration_size
        Size parameter for the integration configuration.
    seed
        Random seed for monte_carlo integration.
    """
    X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('1 + prices + ' + ' + '.join(x2_vars))
    product_formulations = (X1_formulation, X2_formulation)

    if demo_vars is None:
        if integration == 'monte_carlo':
            integration_config = pyblp.Integration(
                'monte_carlo', size=integration_size,
                specification_options={'seed': seed}
            )
        else:
            integration_config = pyblp.Integration('product', size=integration_size)
        return pyblp.Problem(product_formulations, product_data,
                             integration=integration_config)
    else:
        agent_formulation = pyblp.Formulation('0 + ' + ' + '.join(demo_vars))
        return pyblp.Problem(product_formulations, product_data,
                             agent_formulation, agent_data)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Initial parameter builder
# ─────────────────────────────────────────────────────────────────────────────

_NEVO_X2    = ['sugar', 'mushy']
_NEVO_DEMOS = ['income', 'income_squared', 'age', 'child']
_NEVO_SIGMA = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
_NEVO_PI    = np.array([
    [ 5.4819,  0,       0.2037,  0     ],
    [15.8935, -1.2000,  0,       2.6342],
    [-0.2506,  0,       0.0511,  0     ],
    [ 1.2650,  0,      -0.8091,  0     ],
])


def build_initial_params(
    x2_vars: list[str],
    demo_vars: Optional[list[str]] = None,
    n_instruments: Optional[int] = None,
    seed: int = 0,
    force_random: bool = False,   # NEW: skip Nevo-baseline detection
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (sigma_init, pi_init) appropriate for the given specification.

    When x2_vars == ['sugar', 'mushy'] and demo_vars == all four Nevo demographics,
    the Nevo (2000a) published starting values are returned. Otherwise sigma_init
    is a diagonal matrix whose entries are drawn from Uniform(0, 1) using the
    provided seed, and nonzero random pi are returned (zeros would fix params at
    zero in pyblp's sparsity convention).

    K2 = 2 + len(x2_vars)  (constant + prices + x2_vars)

    Parameters
    ----------
    n_instruments
        Number of excluded demand instruments. If provided, an order-condition
        check is performed and a ValueError is raised if the spec is underidentified.
    seed
        RNG seed for random sigma and pi starting values (ensures reproducibility).
    force_random
        If True, skip the Nevo-baseline detection and always draw random values.
        Useful for multistart optimization where subsequent starts need fresh draws.
    """
    # Use Nevo's published starting values for the exact baseline specification
    if not force_random and x2_vars == _NEVO_X2 and demo_vars == _NEVO_DEMOS:
        return _NEVO_SIGMA.copy(), _NEVO_PI.copy()

    K2 = 2 + len(x2_vars)
    rng = np.random.default_rng(seed)
    sigma_init = np.diag(rng.uniform(0, 1, K2))
    if demo_vars is None:
        return sigma_init, None

    # Draw pi — all entries start nonzero so pyblp estimates them freely.
    pi_init = rng.standard_normal((K2, len(demo_vars)))

    # Order-condition guard: if more nonzero params than instruments, zero out
    # trailing pi entries (row-major) until n_nonzero == n_instruments.
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
) -> pyblp.ProblemResults:
    """Solve a pyblp.Problem with BFGS and the given initial parameters."""
    optimization = pyblp.Optimization('bfgs', {'gtol': gtol})
    kwargs = dict(optimization=optimization, method=method)
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
    integration: str = 'product',
    integration_size: int = 5,
    gtol: float = 1e-5,
    method: str = '1s',
    seed: int = 0,
) -> pyblp.ProblemResults:
    """Build the problem, construct initial parameters, and solve — all in one call."""
    problem = build_problem(
        product_data, agent_data, x2_vars, demo_vars,
        integration=integration, integration_size=integration_size,
        seed=seed,
    )
    n_instr = len([c for c in product_data.columns if c.startswith('demand_instruments')])
    sigma_init, pi_init = build_initial_params(
        x2_vars, demo_vars, n_instruments=n_instr, seed=seed
    )
    return solve_spec(problem, sigma_init, pi_init, gtol=gtol, method=method)


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
    integration: str = 'product',
    integration_size: int = 5,
    gtol: float = 1e-5,
    method: str = '1s',
) -> list[StartResult]:
    """
    Solve a specification n_starts times from different random starting points.

    Start 0 uses Nevo (2000a) values for the baseline spec or a seeded draw
    otherwise. Starts 1..n_starts-1 always use fresh random draws (force_random=True)
    with seed = base_seed + start_index to ensure reproducibility.

    Returns StartResult objects sorted ascending by objective (best = index 0).
    Each StartResult exposes .result, .sigma_init, and .pi_init.
    """
    problem = build_problem(
        product_data, agent_data, x2_vars, demo_vars,
        integration=integration, integration_size=integration_size,
        seed=base_seed,
    )
    n_instr = len([c for c in product_data.columns if c.startswith('demand_instruments')])

    results = []
    for i in range(n_starts):
        seed = base_seed + i
        force_random = (i > 0)   # start 0 uses Nevo values when applicable
        sigma_init, pi_init = build_initial_params(
            x2_vars, demo_vars,
            n_instruments=n_instr,
            seed=seed,
            force_random=force_random,
        )
        res = solve_spec(problem, sigma_init, pi_init, gtol=gtol, method=method)
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
    """
    rows = []
    for label, res in results_dict.items():
        row = {
            'label':      label,
            'price_coef': float(res.beta[0]),
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
    """
    rows = []
    for label, sr_list in multistart_dict.items():
        for i, sr in enumerate(sr_list):
            row = {
                'spec':       label,
                'start':      i,
                'seed':       sr.seed,
                'price_coef': float(sr.result.beta[0]),
                'objective':  float(sr.result.objective),
                'best':       (i == 0),
            }
            for k, v in enumerate(np.diag(sr.sigma_init)):
                row[f'init_sigma_{k}'] = float(v)
            if sr.pi_init is not None:
                for r in range(sr.pi_init.shape[0]):
                    for c in range(sr.pi_init.shape[1]):
                        row[f'init_pi_{r}_{c}'] = float(sr.pi_init[r, c])
            for k, v in enumerate(np.diag(sr.result.sigma)):
                row[f'est_sigma_{k}'] = float(v)
            if sr.result.pi is not None:
                for r in range(sr.result.pi.shape[0]):
                    for c in range(sr.result.pi.shape[1]):
                        row[f'est_pi_{r}_{c}'] = float(sr.result.pi[r, c])
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
    markets = np.sort(product_data['market_ids'].unique())
    for label, starts in multistart_results.items():
        best = starts[0]
        res  = best.result
        seed = best.seed
        elasticities = res.compute_elasticities()

        # Accumulate elasticity values per (product_j, product_k) pair across markets
        pair_vals: dict[tuple, list[float]] = {}
        for t, market_id in enumerate(markets):
            mask = product_data['market_ids'] == market_id
            product_ids = product_data.loc[mask, 'product_ids'].values
            E_t = elasticities[t]
            for j, prod_j in enumerate(product_ids):
                for k, prod_k in enumerate(product_ids):
                    pair = (prod_j, prod_k)
                    pair_vals.setdefault(pair, []).append(float(E_t[j, k]))

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
    OUT_DIR = Path('results/nevo')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    product_data, agent_data = load_data()

    N_STARTS = 2  # number of random restarts per specification

    x2_combos = [
        ['sugar'],
        ['mushy'],
        # ['sugar', 'mushy'],
    ]
    demo_combos = [
        ['income', 'age'],
        ['income', 'age', 'child'],
        ['income', 'income_squared', 'age', 'child'],  # Nevo (2000a) baseline
    ]

    multistart_results = {}
    for x2 in x2_combos:
        for demos in demo_combos:
            label = f"x2={x2} | demos={demos}"
            all_csv = OUT_DIR / 'multistart_all.csv'
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
    _append_csv(detail, OUT_DIR / 'multistart_all.csv', index=False)
    print("Saved: multistart_all.csv")

    print("\n=== Best per Specification ===")
    best = detail[detail['best']].drop(columns='best').set_index('spec')
    print(best.to_string())
    _append_csv(best, OUT_DIR / 'multistart_best.csv')
    print("Saved: multistart_best.csv")

    print("\n=== Post-Estimation: Elasticities & Diversion Ratios ===")
    post = summarise_post_estimation(multistart_results, product_data, include_supply=False)
    print(post.to_string())
    _append_csv(post, OUT_DIR / 'post_estimation_summary.csv')
    print("Saved: post_estimation_summary.csv")

    print("\n=== Exporting Full Elasticity Matrices ===")
    elast = export_elasticities(multistart_results, product_data)
    _append_csv(elast, OUT_DIR / 'elasticities_detail.csv', index=False)
    print("Saved: elasticities_detail.csv")
