"""
analyze_results.py -- post-hoc analysis of multistart estimation results.

Reads:
  results/nevo/multistart_all.csv
  results/blp/blp_multistart_all.csv
  (also checks /oak/stanford/groups/polinsky/blp_nevo/... if local files absent)

Outputs analysis tables to stdout covering:
  [Nevo] 1. Objective ranking across specs
         2. Demographic expansion effect (fixed X2)
         3. X2 characteristic comparison (fixed demographics)
         4. Price coefficient sensitivity
         5. Multi-start convergence stability
  [BLP]  6. Convergence audit (valid vs invalid starts)
         7. Global minimum identification
         8. Two-basin analysis
         9. Starting-value sensitivity for valid starts
"""

import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OAK_ROOT = Path('/oak/stanford/groups/polinsky/blp_nevo')
_LOCAL_ROOT = Path(__file__).parent / 'results'

def _find(relative: str) -> Path:
    local = _LOCAL_ROOT / relative
    if local.exists():
        return local
    oak = _OAK_ROOT / 'results' / relative
    if oak.exists():
        return oak
    raise FileNotFoundError(
        f"Cannot find {relative!r} in {_LOCAL_ROOT} or {_OAK_ROOT / 'results'}"
    )

NEVO_CSV = _find('nevo/multistart_all.csv')
BLP_CSV  = _find('blp/blp_multistart_all.csv')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _f(val, digits=4) -> str:
    """Format a numeric string to fixed decimals."""
    try:
        return f'{float(val):>{digits + 8}.{digits}f}'
    except (ValueError, TypeError):
        return str(val).rjust(digits + 8)


def _sep(width=80, char='-'):
    print(char * width)


def _header(title: str):
    _sep(char='=')
    print(f'  {title}')
    _sep(char='=')


def _valid(row: dict) -> bool:
    """Economically valid demand estimate: price coefficient must be negative."""
    try:
        return float(row['price_coef']) < 0
    except (ValueError, TypeError):
        return False


def _best_per_spec(rows: list[dict]) -> list[dict]:
    """Return one row per unique spec: the best (lowest objective) valid start."""
    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], []).append(r)

    result = []
    for spec, spec_rows in by_spec.items():
        valid = [r for r in spec_rows if _valid(r)]
        pool = valid if valid else spec_rows
        best = min(pool, key=lambda r: float(r['objective']))
        result.append(best)
    return sorted(result, key=lambda r: float(r['objective']))

# ---------------------------------------------------------------------------
# NEVO analyses
# ---------------------------------------------------------------------------

def nevo_objective_ranking(rows: list[dict]):
    _header('NEVO 1 -- Objective ranking across specifications')
    best = _best_per_spec(rows)
    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"price_coef":>12}  Specification')
    _sep()
    for i, r in enumerate(best, 1):
        print(f'  {i:>4}  {float(r["objective"]):>10.4f}  {float(r["price_coef"]):>12.4f}  {r["spec"]}')
    print()


def nevo_demographic_expansion(rows: list[dict]):
    _header('NEVO 2 -- Effect of adding demographics (fixed X2 characteristic)')
    best_map = {r['spec']: r for r in _best_per_spec(rows)}

    groups = {
        "sugar": [
            "x2=['sugar'] | demos=['income']",
            "x2=['sugar'] | demos=['income', 'income_squared']",
            "x2=['sugar'] | demos=['income', 'age']",
            "x2=['sugar'] | demos=['income', 'age', 'child']",
            "x2=['sugar'] | demos=['income', 'income_squared', 'age', 'child']",
        ],
        "mushy": [
            "x2=['mushy'] | demos=['income']",
            "x2=['mushy'] | demos=['income', 'income_squared']",
            "x2=['mushy'] | demos=['income', 'age']",
            "x2=['mushy'] | demos=['income', 'age', 'child']",
            "x2=['mushy'] | demos=['income', 'income_squared', 'age', 'child']",
        ],
    }

    for x2_label, spec_list in groups.items():
        print(f'  X2 = {x2_label}')
        print(f'    {"Demographics":<45}  {"GMM obj":>10}  {"price_coef":>12}  {"d_obj":>8}')
        _sep(60)
        prev_obj = None
        for spec in spec_list:
            if spec not in best_map:
                continue
            r = best_map[spec]
            obj = float(r['objective'])
            demo_label = spec.split("demos=")[1]
            delta = f'{obj - prev_obj:+.4f}' if prev_obj is not None else '      -'
            print(f'    {demo_label:<45}  {obj:>10.4f}  {float(r["price_coef"]):>12.4f}  {delta:>8}')
            prev_obj = obj
        print()


def nevo_x2_comparison(rows: list[dict]):
    _header('NEVO 3 -- X2 characteristic comparison (fixed demographics = income)')
    best_map = {r['spec']: r for r in _best_per_spec(rows)}

    specs = [
        "x2=['sugar'] | demos=['income']",
        "x2=['mushy'] | demos=['income']",
        "x2=['sugar', 'mushy'] | demos=['income']",
    ]

    print(f'  {"X2":>20}  {"GMM obj":>10}  {"price_coef":>12}')
    _sep(50)
    for spec in specs:
        if spec not in best_map:
            continue
        r = best_map[spec]
        x2_label = spec.split("x2=")[1].split(" |")[0]
        print(f'  {x2_label:>20}  {float(r["objective"]):>10.4f}  {float(r["price_coef"]):>12.4f}')
    print()


def nevo_price_coef_sensitivity(rows: list[dict]):
    _header('NEVO 4 -- Price coefficient sensitivity across specifications')
    best = _best_per_spec(rows)
    coefs = [float(r['price_coef']) for r in best]
    print(f'  Range: [{min(coefs):.4f}, {max(coefs):.4f}]')
    print(f'  Spread: {max(coefs) - min(coefs):.4f}')
    print()
    print(f'  {"price_coef":>12}  {"implied markup ~= -1/alpha":>22}  Specification')
    _sep()
    for r in sorted(best, key=lambda x: float(x['price_coef'])):
        alpha = float(r['price_coef'])
        markup = -1.0 / alpha if alpha != 0 else float('nan')
        print(f'  {alpha:>12.4f}  {markup:>22.4f}  {r["spec"]}')
    print()


def nevo_multistart_stability(rows: list[dict]):
    _header('NEVO 5 -- Multi-start convergence stability')
    from collections import defaultdict
    by_spec: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_spec[r['spec']].append(float(r['objective']))

    print(f'  {"N starts":>8}  {"Obj spread":>12}  Specification')
    _sep()
    for spec, objs in sorted(by_spec.items(), key=lambda x: max(x[1]) - min(x[1]), reverse=True):
        spread = max(objs) - min(objs)
        print(f'  {len(objs):>8}  {spread:>12.6f}  {spec}')
    print()

# ---------------------------------------------------------------------------
# BLP analyses
# ---------------------------------------------------------------------------

def blp_convergence_audit(rows: list[dict]):
    _header('BLP 6 -- Convergence audit (valid = price_coef < 0)')
    print(f'  {"Seed":>6}  {"Start":>6}  {"GMM obj":>10}  {"price_coef":>12}  {"Valid?":>7}  {"Best flag":>9}')
    _sep()
    for r in rows:
        valid = _valid(r)
        marker = '  OK' if valid else '  ** INVALID (alpha > 0) **'
        print(
            f'  {r["seed"]:>6}  {r["start"]:>6}  {float(r["objective"]):>10.4f}'
            f'  {float(r["price_coef"]):>12.4f}  {str(valid):>7}  {r["best"]:>9}{marker}'
        )
    valid_rows = [r for r in rows if _valid(r)]
    invalid_rows = [r for r in rows if not _valid(r)]
    print()
    print(f'  Valid starts:   {len(valid_rows)} / {len(rows)}')
    print(f'  Invalid starts: {len(invalid_rows)} / {len(rows)} (price_coef > 0 => economically invalid)')
    print()


def blp_global_minimum(rows: list[dict]):
    _header('BLP 7 -- Global minimum identification (valid starts only)')
    valid = [r for r in rows if _valid(r)]
    if not valid:
        print('  No valid starts found.')
        return

    valid_sorted = sorted(valid, key=lambda r: float(r['objective']))
    best = valid_sorted[0]
    print(f'  Global best: seed={best["seed"]}  obj={float(best["objective"]):.4f}'
          f'  price_coef={float(best["price_coef"]):.4f}')
    print()
    print(f'  {"Rank":>4}  {"Seed":>6}  {"GMM obj":>10}  {"price_coef":>12}  {"d_from_best":>12}')
    _sep()
    for i, r in enumerate(valid_sorted, 1):
        delta = float(r['objective']) - float(best['objective'])
        print(f'  {i:>4}  {r["seed"]:>6}  {float(r["objective"]):>10.4f}'
              f'  {float(r["price_coef"]):>12.4f}  {delta:>12.4f}')
    print()


def blp_two_basin_analysis(rows: list[dict]):
    _header('BLP 8 -- Two-basin analysis (valid starts only)')
    valid = [r for r in rows if _valid(r)]
    if not valid:
        print('  No valid starts found.')
        return

    objs = [float(r['objective']) for r in valid]
    global_min = min(objs)

    # Classify: basin A = within 5 units of global min; basin B = rest
    THRESHOLD = 5.0
    basin_a = [r for r in valid if float(r['objective']) - global_min <= THRESHOLD]
    basin_b = [r for r in valid if float(r['objective']) - global_min > THRESHOLD]

    def _basin_summary(basin_rows, label):
        if not basin_rows:
            print(f'  {label}: (none)')
            return
        objs_b = [float(r['objective']) for r in basin_rows]
        coefs_b = [float(r['price_coef']) for r in basin_rows]
        seeds = [r['seed'] for r in basin_rows]
        print(f'  {label} (n={len(basin_rows)}, seeds={seeds}):')
        print(f'    obj:        [{min(objs_b):.4f}, {max(objs_b):.4f}]  spread={max(objs_b)-min(objs_b):.4f}')
        print(f'    price_coef: [{min(coefs_b):.4f}, {max(coefs_b):.4f}]  spread={max(coefs_b)-min(coefs_b):.4f}')

        # Named parameter columns
        est_named = [c for c in basin_rows[0].keys()
                     if c.startswith('est_') and '[' in c and basin_rows[0][c]]
        if est_named:
            print(f'    Estimated parameters (mean +/- range across basin):')
            for col in est_named:
                vals = [float(r[col]) for r in basin_rows if r[col]]
                if vals:
                    mean = sum(vals) / len(vals)
                    spread = max(vals) - min(vals)
                    print(f'      {col:<30}  mean={mean:>10.4f}  spread={spread:.4f}')

    _basin_summary(basin_a, f'Basin A (global, d <= {THRESHOLD})')
    print()
    _basin_summary(basin_b, f'Basin B (local,  d >  {THRESHOLD})')
    print()


def blp_starting_value_sensitivity(rows: list[dict]):
    _header('BLP 9 -- Starting-value sensitivity (valid vs invalid starts)')
    # Compare init_sigma and init_pi columns between valid and invalid starts
    init_named = [c for c in rows[0].keys()
                  if c.startswith('init_') and '[' in c]

    valid   = [r for r in rows if _valid(r)]
    invalid = [r for r in rows if not _valid(r)]

    print(f'  {"Parameter":<30}  {"Valid mean":>12}  {"Invalid mean":>12}  {"Diff":>10}')
    _sep()
    for col in init_named:
        v_vals = [float(r[col]) for r in valid   if r.get(col)]
        i_vals = [float(r[col]) for r in invalid if r.get(col)]
        if not v_vals or not i_vals:
            continue
        v_mean = sum(v_vals) / len(v_vals)
        i_mean = sum(i_vals) / len(i_vals)
        print(f'  {col:<30}  {v_mean:>12.4f}  {i_mean:>12.4f}  {v_mean - i_mean:>10.4f}')
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    nevo_rows = _load(NEVO_CSV)
    blp_rows  = _load(BLP_CSV)

    print(f'\nLoaded Nevo: {len(nevo_rows)} rows from {NEVO_CSV}')
    print(f'Loaded BLP:  {len(blp_rows)} rows from {BLP_CSV}\n')

    nevo_objective_ranking(nevo_rows)
    nevo_demographic_expansion(nevo_rows)
    nevo_x2_comparison(nevo_rows)
    nevo_price_coef_sensitivity(nevo_rows)
    nevo_multistart_stability(nevo_rows)

    blp_convergence_audit(blp_rows)
    blp_global_minimum(blp_rows)
    blp_two_basin_analysis(blp_rows)
    blp_starting_value_sensitivity(blp_rows)
