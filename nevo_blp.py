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
from typing import Optional

pyblp.options.digits = 2
pyblp.options.verbose = False


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
    sigma_scale: float = 1.0,
    seed: int = 0,
    force_random: bool = False,   # NEW: skip Nevo-baseline detection
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (sigma_init, pi_init) appropriate for the given specification.

    When x2_vars == ['sugar', 'mushy'] and demo_vars == all four Nevo demographics,
    the Nevo (2000a) published starting values are returned. Otherwise a scaled
    identity sigma and nonzero random pi are returned (zeros would fix params at zero
    in pyblp's sparsity convention).

    K2 = 2 + len(x2_vars)  (constant + prices + x2_vars)

    Parameters
    ----------
    n_instruments
        Number of excluded demand instruments. If provided, an order-condition
        check is performed and a ValueError is raised if the spec is underidentified.
    seed
        RNG seed for random pi starting values (ensures reproducibility).
    force_random
        If True, skip the Nevo-baseline detection and always draw random values.
        Useful for multistart optimization where subsequent starts need fresh draws.
    """
    # Use Nevo's published starting values for the exact baseline specification
    if not force_random and x2_vars == _NEVO_X2 and demo_vars == _NEVO_DEMOS:
        return _NEVO_SIGMA.copy(), _NEVO_PI.copy()

    K2 = 2 + len(x2_vars)
    sigma_init = np.eye(K2) * sigma_scale
    if demo_vars is None:
        return sigma_init, None

    # Draw pi — all entries start nonzero so pyblp estimates them freely.
    rng = np.random.default_rng(seed)
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
) -> list[pyblp.ProblemResults]:
    """
    Solve a specification n_starts times from different random starting points.

    Start 0 uses Nevo (2000a) values for the baseline spec or a seeded draw
    otherwise. Starts 1..n_starts-1 always use fresh random draws (force_random=True)
    with seed = base_seed + start_index to ensure reproducibility.

    Returns results sorted ascending by objective (best = index 0).
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
        results.append(res)

    return sorted(results, key=lambda r: float(r.objective))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def compare_results(results_dict: dict[str, pyblp.ProblemResults]) -> pd.DataFrame:
    """
    Summarise a collection of solved specifications into a tidy DataFrame.

    Columns: price_coef (alpha hat), objective value.
    """
    rows = []
    for label, res in results_dict.items():
        rows.append({
            'label':      label,
            'price_coef': float(res.beta[0]),
            'objective':  float(res.objective),
        })
    return pd.DataFrame(rows).set_index('label')


def compare_multistart_results(
    multistart_dict: dict[str, list[pyblp.ProblemResults]],
) -> pd.DataFrame:
    """
    Summarise multi-start results into a tidy DataFrame.

    Parameters
    ----------
    multistart_dict
        {label: [results sorted best-first]} as returned by run_multistart.

    Returns a DataFrame with columns:
        spec, start, price_coef, objective, best
    where `best` is True for the lowest-objective run within each spec.
    """
    rows = []
    for label, res_list in multistart_dict.items():
        for i, res in enumerate(res_list):
            rows.append({
                'spec':       label,
                'start':      i,
                'price_coef': float(res.beta[0]),
                'objective':  float(res.objective),
                'best':       (i == 0),   # list is sorted best-first
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Script entry point — grid over specifications
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    product_data, agent_data = load_data()

    N_STARTS = 5   # number of random restarts per specification

    x2_combos = [
        # ['sugar'],
        # ['mushy'],
        ['sugar', 'mushy'],
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
            print(f"\nSolving ({N_STARTS} starts): {label}")
            multistart_results[label] = run_multistart(
                product_data, agent_data, x2, demos, n_starts=N_STARTS
            )

    detail = compare_multistart_results(multistart_results)
    print("\n=== All Starts ===")
    print(detail.to_string(index=False))

    print("\n=== Best per Specification ===")
    best = detail[detail['best']].drop(columns='best').set_index('spec')
    print(best.to_string())
