"""
analyze_results.py -- post-hoc analysis of multistart estimation results.

Reads (prefers nested csv/ subdir, falls back to flat layout):
  results/nevo/csv/multistart_all.csv         or  results/nevo/multistart_all.csv
  results/nevo/csv/elasticities_detail.csv    or  results/nevo/elasticities_detail.csv
  results/blp/csv/blp_multistart_all.csv      or  results/blp/blp_multistart_all.csv
  results/blp/csv/blp_elasticities_detail.csv or  results/blp/blp_elasticities_detail.csv
  (also checks /oak/stanford/groups/polinsky/blp_nevo/... with same precedence)

Outputs analysis tables to stdout covering:
  [Both]  01. Objective ranking across specs
          02. Multi-start convergence stability
          03. Convergence audit (valid vs invalid starts)
          04. Global minimum identification
          05. Two-basin analysis
          07. Price coefficient sensitivity
          20. Own-price elasticity summary by spec (ranked by objective)
          21. Elasticity stability across multi-start seeds
          22. Top substitutes per product (best spec)
          23. Cross-price elasticity asymmetry (best spec)
          26. Own-price elasticity product-level cross-spec stability
          27. Cross-price elasticity cross-spec stability (top substitute pairs)
          28. Pairwise spec agreement matrix (mean absolute deviation of own-price)
          39. Starting-value sensitivity for valid starts
  [Nevo]  24. Cross-spec Spearman rank correlation of own-price elasticities
          25. Within-firm vs between-firm substitution patterns
          37. Demographic expansion effect (fixed X2)
          38. X2 characteristic comparison (fixed demographics)
"""

import contextlib
import csv
import io
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_OAK_ROOT = Path('/oak/stanford/groups/polinsky/blp_nevo')
_LOCAL_ROOT = Path(__file__).parent / 'results'

def _find(relative: str) -> Path:
    """Locate a CSV under <root>/<dataset>/csv/<file> or legacy <root>/<dataset>/<file>.

    `relative` is expected to be '<dataset>/<filename>' (e.g. 'nevo/multistart_all.csv').
    """
    dataset, _, filename = relative.partition('/')
    nested = f'{dataset}/csv/{filename}'
    for root in (_LOCAL_ROOT, _OAK_ROOT / 'results'):
        for candidate in (root / nested, root / relative):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Cannot find {relative!r} (tried 'csv/' subdir and flat layout) "
        f"in {_LOCAL_ROOT} or {_OAK_ROOT / 'results'}"
    )


def _analysis_dir(csv_path: Path) -> Path:
    """Return <dataset_root>/analysis/, regardless of csv/ nesting."""
    parent = csv_path.parent
    dataset_root = parent.parent if parent.name == 'csv' else parent
    return dataset_root / 'analysis'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


MAX_SEEDS_PER_SPEC = 40


def _cap_seeds(rows: list[dict], elas_rows: list[dict],
               max_seeds: int = MAX_SEEDS_PER_SPEC) -> tuple[list[dict], list[dict]]:
    """Keep at most `max_seeds` unique seeds per spec (lowest seed numbers).

    Caps the multistart sample so each spec contributes at most `max_seeds`
    simulations; elasticity rows are restricted to the same kept (spec, seed)
    pairs. Validity is not considered here (the convergence audit still sees
    every valid/invalid start within the cap).
    """
    def _seednum(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return float('inf')

    by_spec: dict[str, set] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], set()).add(r['seed'])
    keep = {spec: set(sorted(seeds, key=_seednum)[:max_seeds])
            for spec, seeds in by_spec.items()}
    capped_rows = [r for r in rows      if r['seed'] in keep.get(r['spec'], ())]
    capped_elas = [r for r in elas_rows if r['seed'] in keep.get(r['spec'], ())]
    return capped_rows, capped_elas


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


def _run_and_save(out_dir: Path, filename: str, func, *args):
    """Call func(*args), tee output to stdout and save to out_dir/filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func(*args)
    text = buf.getvalue()
    print(text, end='')
    (out_dir / filename).write_text(text, encoding='utf-8')


def _valid(row: dict) -> bool:
    """Economically and numerically valid demand estimate.

    Requires a negative price coefficient (economic sign) AND a non-negative
    GMM objective (the canonical g'Wg is PSD; negative values are numerical
    failures from a non-converged inner loop or ill-conditioned weighting
    matrix and would otherwise be picked as spurious 'best' starts).
    """
    try:
        return float(row['price_coef']) < 0 and float(row['objective']) >= 0
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


def _best_seed_map(all_rows: list[dict]) -> dict[str, str]:
    """Return {spec: seed} for the best valid start per spec."""
    return {r['spec']: r['seed'] for r in _best_per_spec(all_rows)}


def _stats(vals: list[float]) -> dict:
    """Return mean, median, std, min, max for a list of floats."""
    n = len(vals)
    if n == 0:
        return dict(mean=float('nan'), median=float('nan'), std=0.0,
                    mn=float('nan'), mx=float('nan'))
    mean = sum(vals) / n
    sorted_v = sorted(vals)
    mid = n // 2
    median = sorted_v[mid] if n % 2 else (sorted_v[mid - 1] + sorted_v[mid]) / 2
    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    return dict(mean=mean, median=median, std=std, mn=sorted_v[0], mx=sorted_v[-1])


def _rank(vals: list[float]) -> list[int]:
    """Assign integer ranks (1 = smallest) to a list of values."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for rank, i in enumerate(order, 1):
        ranks[i] = rank
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / denom if denom else float('nan')


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (rank-transform then Pearson)."""
    return _pearson([float(r) for r in _rank(xs)], [float(r) for r in _rank(ys)])

# ---------------------------------------------------------------------------
# Generic analyses (apply to both Nevo and BLP)
# ---------------------------------------------------------------------------

def objective_ranking(rows: list[dict], label: str = ''):
    _header(f'{label} -- Objective ranking across specifications')
    best = _best_per_spec(rows)
    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"price_coef":>12}  Specification')
    _sep()
    for i, r in enumerate(best, 1):
        print(f'  {i:>4}  {float(r["objective"]):>10.4f}  {float(r["price_coef"]):>12.4f}  {r["spec"]}')
    print()


def price_coef_sensitivity(rows: list[dict], label: str = ''):
    _header(f'{label} -- Price coefficient sensitivity across specifications')
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


def multistart_stability(rows: list[dict], label: str = ''):
    _header(f'{label} -- Multi-start convergence stability (valid starts only)')
    from collections import defaultdict
    by_spec: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if not _valid(r):
            continue
        by_spec[r['spec']].append(float(r['objective']))

    print(f'  {"N starts":>8}  {"Obj spread":>12}  Specification')
    _sep()
    for spec, objs in sorted(by_spec.items(), key=lambda x: max(x[1]) - min(x[1]), reverse=True):
        spread = max(objs) - min(objs)
        print(f'  {len(objs):>8}  {spread:>12.6f}  {spec}')
    print()


def convergence_audit(rows: list[dict], label: str = ''):
    _header(f'{label} -- Convergence audit (valid = price_coef < 0 AND objective >= 0)')
    print(f'  {"Spec":<40}  {"Seed":>6}  {"Start":>6}  {"GMM obj":>10}  {"price_coef":>12}  {"Valid?":>7}  {"Best flag":>9}')
    _sep()

    def _reason(r):
        try:
            bad_alpha = float(r['price_coef']) >= 0
            bad_obj   = float(r['objective']) < 0
        except (ValueError, TypeError):
            return '  ** INVALID (unparseable) **'
        flags = []
        if bad_alpha: flags.append('alpha >= 0')
        if bad_obj:   flags.append('objective < 0')
        return '  ** INVALID (' + ', '.join(flags) + ') **' if flags else ''

    for r in rows:
        valid = _valid(r)
        marker = '  OK' if valid else _reason(r)
        print(
            f'  {r["spec"]:<40}  {r["seed"]:>6}  {r["start"]:>6}  {float(r["objective"]):>10.4f}'
            f'  {float(r["price_coef"]):>12.4f}  {str(valid):>7}  {r["best"]:>9}{marker}'
        )
    valid_rows   = [r for r in rows if _valid(r)]
    invalid_rows = [r for r in rows if not _valid(r)]
    n_bad_alpha  = sum(1 for r in rows if not _valid(r) and float(r.get('price_coef', 0) or 0) >= 0)
    n_bad_obj    = sum(1 for r in rows if not _valid(r) and float(r.get('objective', 0) or 0) < 0)
    print()
    print(f'  Valid starts:   {len(valid_rows)} / {len(rows)}')
    print(f'  Invalid starts: {len(invalid_rows)} / {len(rows)}'
          f'   (alpha >= 0: {n_bad_alpha}, objective < 0: {n_bad_obj})')
    print()


def global_minimum(rows: list[dict], label: str = ''):
    _header(f'{label} -- Global minimum identification (valid starts only)')
    valid = [r for r in rows if _valid(r)]
    if not valid:
        print('  No valid starts found.')
        return

    valid_sorted = sorted(valid, key=lambda r: float(r['objective']))
    best = valid_sorted[0]
    print(f'  Global best: spec={best["spec"]}  seed={best["seed"]}  obj={float(best["objective"]):.4f}'
          f'  price_coef={float(best["price_coef"]):.4f}')
    print()
    print(f'  {"Rank":>4}  {"Spec":<30}  {"Seed":>6}  {"GMM obj":>10}  {"price_coef":>12}  {"d_from_best":>12}')
    _sep()
    for i, r in enumerate(valid_sorted, 1):
        delta = float(r['objective']) - float(best['objective'])
        print(f'  {i:>4}  {r["spec"]:<30}  {r["seed"]:>6}  {float(r["objective"]):>10.4f}'
              f'  {float(r["price_coef"]):>12.4f}  {delta:>12.4f}')
    print()


def two_basin_analysis(rows: list[dict], label: str = ''):
    _header(f'{label} -- Two-basin analysis (valid starts only)')
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

    def _basin_summary(basin_rows, blabel):
        if not basin_rows:
            print(f'  {blabel}: (none)')
            return
        objs_b = [float(r['objective']) for r in basin_rows]
        coefs_b = [float(r['price_coef']) for r in basin_rows]
        seeds = [r['seed'] for r in basin_rows]
        print(f'  {blabel} (n={len(basin_rows)}, seeds={seeds}):')
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


def starting_value_sensitivity(rows: list[dict], label: str = ''):
    _header(f'{label} -- Starting-value sensitivity (valid vs invalid starts)')
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
# Elasticity analyses (generic — apply to both Nevo and BLP)
# ---------------------------------------------------------------------------

def elasticity_own_summary(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 20: own-price elasticity summary per spec, ranked by objective."""
    _header(f'{label} -- Own-price elasticity summary by specification (ranked by objective)')

    best_list = _best_per_spec(all_rows)
    seed_map  = {r['spec']: r['seed'] for r in best_list}
    obj_map   = {r['spec']: float(r['objective']) for r in best_list}

    # Collect own-price elasticities per spec/seed
    own_by_spec: dict[str, list[float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            own_by_spec.setdefault(r['spec'], []).append(float(r['elasticity']))

    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"mean_own":>10}  {"median":>8}'
          f'  {"std":>8}  {"min":>10}  {"max":>10}  Specification')
    _sep()
    for rank, best_row in enumerate(best_list, 1):
        spec = best_row['spec']
        vals = own_by_spec.get(spec, [])
        if not vals:
            continue
        s = _stats(vals)
        print(f'  {rank:>4}  {obj_map[spec]:>10.4f}  {s["mean"]:>10.4f}  {s["median"]:>8.4f}'
              f'  {s["std"]:>8.4f}  {s["mn"]:>10.4f}  {s["mx"]:>10.4f}  {spec}')
    print()


def elasticity_multistart_stability(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 21: per-product own-price elasticity spread across seeds."""
    _header(f'{label} -- Own-price elasticity stability across multi-start seeds')

    # Find specs that have more than one seed in elas_rows
    seeds_per_spec: dict[str, set] = {}
    for r in elas_rows:
        if r['own_price'] == 'True':
            seeds_per_spec.setdefault(r['spec'], set()).add(r['seed'])

    multi_specs = {spec for spec, seeds in seeds_per_spec.items() if len(seeds) > 1}

    if not multi_specs:
        print('  No specifications with multiple seeds found in elasticities data.')
        print()
        return

    # For each multi-seed spec, per product collect elasticity per seed
    for spec in sorted(multi_specs):
        seeds = sorted(seeds_per_spec[spec])
        print(f'  Spec: {spec}  ({len(seeds)} seeds: {seeds})')
        print(f'    {"Product":<12}  {"mean_own":>10}  {"spread":>8}  {"min":>10}  {"max":>10}')
        _sep(60)

        # {product: {seed: elasticity}}
        prod_seed: dict[str, dict[str, float]] = {}
        for r in elas_rows:
            if r['spec'] == spec and r['own_price'] == 'True':
                prod_seed.setdefault(r['product_j'], {})[r['seed']] = float(r['elasticity'])

        for product in sorted(prod_seed):
            vals = list(prod_seed[product].values())
            if len(vals) < 2:
                continue
            mean   = sum(vals) / len(vals)
            spread = max(vals) - min(vals)
            print(f'    {product:<12}  {mean:>10.4f}  {spread:>8.6f}  {min(vals):>10.4f}  {max(vals):>10.4f}')
        print()


def elasticity_top_substitutes(elas_rows: list[dict], all_rows: list[dict],
                                label: str = '', top_k: int = 5):
    """Analysis 22: top-k substitute products by cross-price elasticity (best spec)."""
    _header(f'{label} -- Top-{top_k} substitutes per product (best spec by objective)')

    best_list = _best_per_spec(all_rows)
    if not best_list:
        print('  No valid specs found.')
        return

    best_spec = best_list[0]
    spec      = best_spec['spec']
    seed      = best_spec['seed']
    obj       = float(best_spec['objective'])
    print(f'  Best spec (rank 1): {spec}')
    print(f'  Seed: {seed}  GMM obj: {obj:.4f}')
    print()

    # Collect own and cross elasticities for this spec+seed
    own_elas: dict[str, float]         = {}
    cross_elas: dict[str, list[tuple]] = {}
    for r in elas_rows:
        if r['spec'] != spec or r['seed'] != seed:
            continue
        e = float(r['elasticity'])
        if r['own_price'] == 'True':
            own_elas[r['product_j']] = e
        else:
            cross_elas.setdefault(r['product_j'], []).append((r['product_k'], e))

    # For BLP (large product set) limit to the 10 most elastic products
    all_products = sorted(own_elas, key=lambda p: own_elas[p])
    MAX_PRODUCTS = 10 if len(all_products) > 50 else len(all_products)
    display_products = all_products[:MAX_PRODUCTS]

    if len(all_products) > 50:
        print(f'  ({len(all_products)} products total — showing {MAX_PRODUCTS} most price-elastic)')
        print()

    print(f'  {"Product j":<14}  {"own_elas":>10}  Top-{top_k} substitutes (product_k: e_jk)')
    _sep()
    for pj in display_products:
        subs = sorted(cross_elas.get(pj, []), key=lambda x: x[1], reverse=True)[:top_k]
        sub_str = '  '.join(f'{pk}:{e:+.4f}' for pk, e in subs)
        print(f'  {pj:<14}  {own_elas[pj]:>10.4f}  {sub_str}')
    print()


def elasticity_asymmetry(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 23: distribution of |e_jk - e_kj| for the best spec."""
    _header(f'{label} -- Cross-price elasticity asymmetry |e_jk - e_kj| (best spec)')

    best_list = _best_per_spec(all_rows)
    if not best_list:
        print('  No valid specs found.')
        return

    best_spec = best_list[0]
    spec      = best_spec['spec']
    seed      = best_spec['seed']
    print(f'  Best spec (rank 1): {spec}')
    print(f'  Seed: {seed}  GMM obj: {float(best_spec["objective"]):.4f}')
    print()

    # Build lookup: (j, k) -> elasticity for cross-price pairs
    elas_dict: dict[tuple, float] = {}
    for r in elas_rows:
        if r['spec'] == spec and r['seed'] == seed and r['own_price'] == 'False':
            elas_dict[(r['product_j'], r['product_k'])] = float(r['elasticity'])

    # Compute asymmetry for all unordered pairs where both directions exist
    diffs = []
    seen  = set()
    for (j, k), ejk in elas_dict.items():
        if (k, j) in seen:
            continue
        ekj = elas_dict.get((k, j))
        if ekj is not None:
            diffs.append((j, k, ejk, ekj, abs(ejk - ekj)))
            seen.add((j, k))

    if not diffs:
        print('  No symmetric pairs found.')
        print()
        return

    asym_vals = [d[4] for d in diffs]
    s = _stats(asym_vals)
    threshold = 0.1
    n_above   = sum(1 for v in asym_vals if v > threshold)

    print(f'  Total symmetric pairs: {len(diffs)}')
    print(f'  Mean  |e_jk - e_kj|:  {s["mean"]:.4f}')
    print(f'  Median:                {s["median"]:.4f}')
    print(f'  Max:                   {s["mx"]:.4f}')
    print(f'  Pairs with |diff| > {threshold}: {n_above} / {len(diffs)} ({100 * n_above / len(diffs):.1f}%)')
    print()

    top10 = sorted(diffs, key=lambda x: x[4], reverse=True)[:10]
    print(f'  Top-10 most asymmetric pairs:')
    print(f'    {"j":<14}  {"k":<14}  {"e_jk":>10}  {"e_kj":>10}  {"|diff|":>10}')
    _sep(70)
    for j, k, ejk, ekj, diff in top10:
        print(f'    {j:<14}  {k:<14}  {ejk:>10.4f}  {ekj:>10.4f}  {diff:>10.4f}')
    print()

def elasticity_own_cross_spec_stability(elas_rows: list[dict], all_rows: list[dict],
                                        label: str = ''):
    """Analysis 26: per-product own-price elasticity stability across specifications."""
    _header(f'{label} -- Own-price elasticity product-level cross-spec stability')

    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2:
        print('  Fewer than 2 specifications — cross-spec comparison not available.')
        print()
        return

    # {product: {spec: own_elas}} using best seed per spec
    prod_spec_elas: dict[str, dict[str, float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            prod_spec_elas.setdefault(r['product_j'], {})[r['spec']] = float(r['elasticity'])

    # Keep products present in all specs
    all_products = [p for p, d in prod_spec_elas.items() if len(d) == len(specs)]
    if not all_products:
        print('  No products found in all specifications.')
        print()
        return

    # For large markets (BLP) limit to the 20 most elastic products
    MAX_PRODS = 20 if len(all_products) > 50 else len(all_products)
    all_products_sorted = sorted(all_products,
                                 key=lambda p: sum(prod_spec_elas[p].values()) / len(specs))
    display_products = all_products_sorted[:MAX_PRODS]
    if len(all_products) > 50:
        print(f'  ({len(all_products)} products total — showing {MAX_PRODS} most price-elastic)')
        print()

    # Compute CV per product (std / |mean| across specs)
    def _cv(vals: list[float]) -> float:
        s = _stats(vals)
        return s['std'] / abs(s['mean']) if s['mean'] != 0 else float('nan')

    rows_data = []
    for prod in display_products:
        vals = [prod_spec_elas[prod][sp] for sp in specs]
        s    = _stats(vals)
        cv   = _cv(vals)
        rows_data.append((prod, cv, s, vals))

    # Sort by CV descending (most unstable first)
    rows_data.sort(key=lambda x: x[1] if not math.isnan(x[1]) else -1, reverse=True)

    # Short spec labels for column headers
    short = [sp.replace('x2=', '').replace('demos=', '').replace(' | ', '|') for sp in specs]
    col_w = 9

    # Header
    print(f'  {"Product":<14}  {"CV":>7}  {"std":>8}  {"mean":>9}  {"min":>9}  {"max":>9}  '
          + ''.join(f'{s:>{col_w}}' for s in short))
    _sep(14 + 7 + 8 + 9 + 9 + 9 + col_w * len(specs) + 14)

    for prod, cv, s, vals in rows_data:
        cv_str = f'{cv:>7.4f}' if not math.isnan(cv) else f'{"nan":>7}'
        val_str = ''.join(f'{v:>{col_w}.4f}' for v in vals)
        print(f'  {prod:<14}  {cv_str}  {s["std"]:>8.4f}  {s["mean"]:>9.4f}'
              f'  {s["mn"]:>9.4f}  {s["mx"]:>9.4f}  {val_str}')

    # Aggregate stability
    all_cvs = [cv for _, cv, _, _ in rows_data if not math.isnan(cv)]
    if all_cvs:
        print()
        print(f'  Mean CV across {len(all_cvs)} products: {sum(all_cvs) / len(all_cvs):.4f}')
        print(f'  Max  CV: {max(all_cvs):.4f}')
    print()


def elasticity_cross_cross_spec_stability(elas_rows: list[dict], all_rows: list[dict],
                                          label: str = '', top_k: int = 10):
    """Analysis 27: cross-price elasticity stability for top substitute pairs across specs."""
    _header(f'{label} -- Cross-price elasticity cross-spec stability (top-{top_k} substitute pairs)')

    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2:
        print('  Fewer than 2 specifications — cross-spec comparison not available.')
        print()
        return

    if not best_list:
        print('  No valid specs found.')
        print()
        return

    # Identify top substitute pairs from best spec (rank 1)
    best_spec = best_list[0]['spec']
    best_seed = seed_map[best_spec]
    cross_best: list[tuple[str, str, float]] = []
    for r in elas_rows:
        if r['spec'] == best_spec and r['seed'] == best_seed and r['own_price'] == 'False':
            cross_best.append((r['product_j'], r['product_k'], float(r['elasticity'])))

    cross_best.sort(key=lambda x: x[2], reverse=True)
    top_pairs = [(j, k) for j, k, _ in cross_best[:top_k]]

    if not top_pairs:
        print('  No cross-price elasticities found for best spec.')
        print()
        return

    print(f'  Reference spec (rank 1 by objective): {best_spec}')
    print()

    # For each top pair collect elasticity across all specs
    pair_spec_elas: dict[tuple, dict[str, float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'False' and seed_map.get(r['spec']) == r['seed']:
            key = (r['product_j'], r['product_k'])
            if key in set(top_pairs):
                pair_spec_elas.setdefault(key, {})[r['spec']] = float(r['elasticity'])

    def _cv(vals: list[float]) -> float:
        s = _stats(vals)
        return s['std'] / abs(s['mean']) if s['mean'] != 0 else float('nan')

    rows_data = []
    for (j, k) in top_pairs:
        d = pair_spec_elas.get((j, k), {})
        if len(d) < 2:
            continue
        vals = [d.get(sp, float('nan')) for sp in specs]
        valid_vals = [v for v in vals if not math.isnan(v)]
        if not valid_vals:
            continue
        s  = _stats(valid_vals)
        cv = _cv(valid_vals)
        rows_data.append((j, k, cv, s, vals))

    if not rows_data:
        print('  Insufficient cross-spec data for top substitute pairs.')
        print()
        return

    rows_data.sort(key=lambda x: x[2] if not math.isnan(x[2]) else -1, reverse=True)

    short = [sp.replace('x2=', '').replace('demos=', '').replace(' | ', '|') for sp in specs]
    col_w = 9

    print(f'  {"j":<14}  {"k":<14}  {"CV":>7}  {"std":>8}  {"mean":>9}  '
          + ''.join(f'{s:>{col_w}}' for s in short))
    _sep(14 + 14 + 7 + 8 + 9 + col_w * len(specs) + 14)

    for j, k, cv, s, vals in rows_data:
        cv_str  = f'{cv:>7.4f}' if not math.isnan(cv) else f'{"nan":>7}'
        val_str = ''.join(f'{v:>{col_w}.4f}' if not math.isnan(v) else f'{"n/a":>{col_w}}'
                          for v in vals)
        print(f'  {j:<14}  {k:<14}  {cv_str}  {s["std"]:>8.4f}  {s["mean"]:>9.4f}  {val_str}')

    all_cvs = [cv for _, _, cv, _, _ in rows_data if not math.isnan(cv)]
    if all_cvs:
        print()
        print(f'  Mean CV across {len(all_cvs)} pairs: {sum(all_cvs) / len(all_cvs):.4f}')
        print(f'  Max  CV: {max(all_cvs):.4f}')
    print()


def elasticity_spec_pairwise_mad(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 28: pairwise mean absolute deviation of own-price elasticities between specs."""
    _header(f'{label} -- Pairwise spec agreement: mean absolute deviation of own-price elasticities')

    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2:
        print('  Fewer than 2 specifications — pairwise comparison not available.')
        print()
        return

    # {spec: {product: own_elas}} for best seed per spec
    spec_prod: dict[str, dict[str, float]] = {sp: {} for sp in specs}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            spec_prod[r['spec']][r['product_j']] = float(r['elasticity'])

    # Build S×S MAD matrix
    mad: dict[tuple, float] = {}
    for i, si in enumerate(specs):
        for j, sj in enumerate(specs):
            if i == j:
                mad[(si, sj)] = 0.0
                continue
            common = set(spec_prod[si]) & set(spec_prod[sj])
            if not common:
                mad[(si, sj)] = float('nan')
                continue
            mad[(si, sj)] = sum(abs(spec_prod[si][p] - spec_prod[sj][p])
                                for p in common) / len(common)

    short = [sp.replace('x2=', '').replace('demos=', '').replace(' | ', '|') for sp in specs]
    col_w = max(max(len(s) for s in short) + 2, 9)

    # Matrix header
    pad = ' ' * (col_w + 2)
    print(pad + ''.join(f'{s:>{col_w}}' for s in short))
    _sep(col_w * (len(specs) + 1) + 2)

    for i, si in enumerate(specs):
        row_str = f'  {short[i]:<{col_w}}'
        for sj in specs:
            v = mad[(si, sj)]
            row_str += f'{v:>{col_w}.4f}' if not math.isnan(v) else f'{"nan":>{col_w}}'
        print(row_str)
    print()

    # Sorted pairwise list (excluding self-pairs, upper triangle only)
    pairs = []
    for i, si in enumerate(specs):
        for j, sj in enumerate(specs):
            if j <= i:
                continue
            pairs.append((si, sj, mad[(si, sj)]))
    pairs.sort(key=lambda x: x[2] if not math.isnan(x[2]) else -1, reverse=True)

    print(f'  Pairwise MAD (sorted, most divergent first):')
    print(f'  {"MAD":>8}  Spec A  vs  Spec B')
    _sep(80)
    for sa, sb, v in pairs:
        v_str = f'{v:>8.4f}' if not math.isnan(v) else f'{"nan":>8}'
        print(f'  {v_str}  {sa}  vs  {sb}')
    print()


# ---------------------------------------------------------------------------
# Nevo-specific analyses (hardcoded spec labels or firm encoding)
# ---------------------------------------------------------------------------

def nevo_demographic_expansion(rows: list[dict]):
    _header('NEVO -- Effect of adding demographics (fixed X2 characteristic)')
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
    _header('NEVO -- X2 characteristic comparison (fixed demographics = income)')
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


def nevo_elasticity_cross_spec_correlation(elas_rows: list[dict], all_rows: list[dict]):
    """Analysis 24: Spearman rank correlation of own-price elasticities across specs."""
    _header('NEVO -- Cross-spec Spearman rank correlation of own-price elasticities')

    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    # {spec: {product: own_elasticity}} for best seed per spec
    spec_prod_elas: dict[str, dict[str, float]] = {s: {} for s in specs}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            spec_prod_elas[r['spec']][r['product_j']] = float(r['elasticity'])

    # Products common to all specs
    common = set.intersection(*[set(d) for d in spec_prod_elas.values()])
    common_sorted = sorted(common)
    if not common_sorted:
        print('  No products common to all specifications.')
        print()
        return

    print(f'  Products in common: {len(common_sorted)}')
    print()

    # Short spec labels for table header
    short = [s.replace("x2=", "").replace("demos=", "").replace(" | ", "|") for s in specs]
    col_w = max(len(s) for s in short) + 2

    # Header row
    header_pad = ' ' * (col_w + 2)
    print(header_pad + ''.join(f'{s:>{col_w}}' for s in short))
    _sep(col_w * (len(specs) + 1) + 2)

    for i, si in enumerate(specs):
        vi = [spec_prod_elas[si][p] for p in common_sorted]
        row_str = f'  {short[i]:<{col_w}}'
        for j, sj in enumerate(specs):
            vj = [spec_prod_elas[sj][p] for p in common_sorted]
            corr = _spearman(vi, vj)
            row_str += f'{corr:>{col_w}.3f}'
        print(row_str)
    print()

    # Also print the pairwise table sorted by correlation (excluding self-pairs)
    pairs = []
    for i, si in enumerate(specs):
        vi = [spec_prod_elas[si][p] for p in common_sorted]
        for j, sj in enumerate(specs):
            if j <= i:
                continue
            vj = [spec_prod_elas[sj][p] for p in common_sorted]
            pairs.append((si, sj, _spearman(vi, vj)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f'  Pairwise correlations (sorted):')
    print(f'  {"Corr":>6}  Spec A vs Spec B')
    _sep(80)
    for sa, sb, corr in pairs:
        print(f'  {corr:>6.3f}  {sa}  vs  {sb}')
    print()


def nevo_elasticity_firm_substitution(elas_rows: list[dict], all_rows: list[dict]):
    """Analysis 25: within-firm vs between-firm cross-price elasticities."""
    _header('NEVO -- Within-firm vs between-firm substitution patterns')

    def _firm(product_id: str) -> str:
        """Parse firm from product ID: F1B04 -> F1."""
        b_pos = product_id.index('B')
        return product_id[:b_pos]

    best_list = _best_per_spec(all_rows)
    seed_map  = {r['spec']: r['seed'] for r in best_list}
    obj_map   = {r['spec']: float(r['objective']) for r in best_list}

    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"mean_within":>12}  {"mean_between":>13}  {"ratio":>7}  Specification')
    _sep()

    for rank, best_row in enumerate(best_list, 1):
        spec = best_row['spec']
        seed = seed_map[spec]
        within, between = [], []
        for r in elas_rows:
            if r['spec'] != spec or r['seed'] != seed or r['own_price'] == 'True':
                continue
            fj = _firm(r['product_j'])
            fk = _firm(r['product_k'])
            e  = float(r['elasticity'])
            if fj == fk:
                within.append(e)
            else:
                between.append(e)
        mw = sum(within)  / len(within)  if within  else float('nan')
        mb = sum(between) / len(between) if between else float('nan')
        ratio = mw / mb if mb and mb != 0 else float('nan')
        print(f'  {rank:>4}  {obj_map[spec]:>10.4f}  {mw:>12.4f}  {mb:>13.4f}  {ratio:>7.3f}  {spec}')
    print()

    # Firm × firm mean cross-elasticity matrix for the best spec
    best_spec = best_list[0]['spec']
    best_seed = seed_map[best_spec]
    print(f'  Firm x firm mean cross-price elasticity matrix (best spec):')
    print(f'  {best_spec}')
    print()

    firm_pair_elas: dict[tuple, list[float]] = {}
    for r in elas_rows:
        if r['spec'] != best_spec or r['seed'] != best_seed or r['own_price'] == 'True':
            continue
        key = (_firm(r['product_j']), _firm(r['product_k']))
        firm_pair_elas.setdefault(key, []).append(float(r['elasticity']))

    firms = sorted(set(f for (fj, fk) in firm_pair_elas for f in (fj, fk)))
    col_w = 10
    print('  ' + ' ' * 6 + ''.join(f'{f:>{col_w}}' for f in firms))
    _sep(6 + col_w * len(firms) + 2)
    for fj in firms:
        row_str = f'  {fj:<6}'
        for fk in firms:
            vals = firm_pair_elas.get((fj, fk), [])
            mean = sum(vals) / len(vals) if vals else float('nan')
            row_str += f'{mean:>{col_w}.4f}'
        print(row_str)
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def objective_spec_comparison(rows: list[dict], label: str = ''):
    """Analysis 36 — aggregate and compare objective values across valid starts per spec.

    Mirrors figure 36 (plot_objective_spec_comparison), which plots valid starts
    only: specs with no valid start are dropped and every statistic is computed
    over that spec's valid starts. The N / N valid columns still report total
    starts vs valid so the attrition stays visible.
    """
    _header(f'{label} -- Objective aggregation across specifications (valid starts only)')

    by_spec: dict[str, list[float]] = {}   # valid objectives per spec
    total_by_spec: dict[str, int] = {}     # all starts per spec, for the N column
    for r in rows:
        total_by_spec[r['spec']] = total_by_spec.get(r['spec'], 0) + 1
        if _valid(r):
            by_spec.setdefault(r['spec'], []).append(float(r['objective']))

    if not by_spec:
        print('  No valid starts found.\n')
        return

    # Sort specs by mean objective (valid starts) ascending
    ranked = sorted(by_spec.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))

    all_objs = [v for vals in by_spec.values() for v in vals]
    global_min = min(all_objs) if all_objs else float('nan')

    print(f'  {"Rank":>4}  {"N":>3}  {"N valid":>7}  {"Mean obj":>10}  {"Median":>10}  '
          f'{"Std":>9}  {"Min":>10}  {"Max":>10}  {"Range":>9}  {"CV%":>7}  Specification')
    _sep()

    for rank, (spec, objs) in enumerate(ranked, 1):
        s    = _stats(objs)
        cv   = (s['std'] / s['mean'] * 100) if s['mean'] else float('nan')
        rng  = s['mx'] - s['mn']
        print(f'  {rank:>4}  {total_by_spec[spec]:>3}  {len(objs):>7}  {s["mean"]:>10.4f}  {s["median"]:>10.4f}  '
              f'{s["std"]:>9.4f}  {s["mn"]:>10.4f}  {s["mx"]:>10.4f}  {rng:>9.4f}  {cv:>7.2f}  {spec}')

    _sep()
    # Cross-spec summary (valid starts)
    means   = [sum(v) / len(v) for v in by_spec.values()]
    s_means = _stats(means)
    print(f'\n  Cross-spec summary (N specs with valid starts = {len(by_spec)}):')
    print(f'    Mean of spec-means : {s_means["mean"]:.4f}')
    print(f'    Std  of spec-means : {s_means["std"]:.4f}')
    print(f'    Best spec mean     : {s_means["mn"]:.4f}')
    print(f'    Worst spec mean    : {s_means["mx"]:.4f}')
    print(f'    Global minimum obj : {global_min:.4f}')
    print()


def elasticity_pair_across_sims(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 30 — 4 elasticities for one product pair across all simulations of the best-mean-obj spec."""
    _header(f'{label} -- Elasticity pair behavior across all simulations (best-mean-obj spec)')

    # 1. Spec with lowest mean objective across all its starts
    by_spec_obj: dict[str, list[float]] = {}
    for r in all_rows:
        by_spec_obj.setdefault(r['spec'], []).append(float(r['objective']))
    if not by_spec_obj:
        print('  No data.\n')
        return
    ranked_specs = sorted(by_spec_obj, key=lambda s: sum(by_spec_obj[s]) / len(by_spec_obj[s]))

    # Fall back through specs until one has elasticity data
    elas_specs = {r['spec'] for r in elas_rows}
    best_spec  = next((s for s in ranked_specs if s in elas_specs), None)
    if best_spec is None:
        print('  No spec with elasticity data found.\n')
        return
    spec_objs = by_spec_obj[best_spec]
    mean_obj  = sum(spec_objs) / len(spec_objs)

    print(f'  Spec (lowest mean obj with elasticity data): {best_spec}')
    print(f'  Mean objective: {mean_obj:.4f}   N starts: {len(spec_objs)}\n')

    # 2. Product pair with highest mean cross-price elasticity for that spec
    cross: dict[tuple, list[float]] = {}
    for r in elas_rows:
        if r['spec'] == best_spec and r['own_price'] == 'False':
            cross.setdefault((r['product_j'], r['product_k']), []).append(float(r['elasticity']))
    if not cross:
        print('  No cross-price elasticity rows found for this spec.\n')
        return
    prod_j, prod_k = max(cross, key=lambda p: sum(cross[p]) / len(cross[p]))
    print(f'  Product pair: j={prod_j}, k={prod_k}')
    print(f'  (selected as highest mean cross-price elasticity across seeds)\n')

    # 3. Collect the 4 elasticities per seed
    seeds_data: dict[str, dict[str, float]] = {}
    for r in elas_rows:
        if r['spec'] != best_spec:
            continue
        seed = r['seed']
        pj, pk = r['product_j'], r['product_k']
        val = float(r['elasticity'])
        seeds_data.setdefault(seed, {})
        if pj == prod_j and pk == prod_j:
            seeds_data[seed]['e_jj'] = val
        elif pj == prod_k and pk == prod_k:
            seeds_data[seed]['e_kk'] = val
        elif pj == prod_j and pk == prod_k:
            seeds_data[seed]['e_jk'] = val
        elif pj == prod_k and pk == prod_j:
            seeds_data[seed]['e_kj'] = val

    keys = ['e_jj', 'e_kk', 'e_jk', 'e_kj']
    hdr  = f'  {"Seed":>6}  {"e_jj":>10}  {"e_kk":>10}  {"e_jk":>10}  {"e_kj":>10}'
    print(hdr)
    _sep()
    sorted_seeds = sorted(seeds_data)
    for seed in sorted_seeds:
        d = seeds_data[seed]
        vals = [d.get(k, float('nan')) for k in keys]
        fmt  = '  '.join(f'{v:>10.4f}' if not math.isnan(v) else f'{"n/a":>10}' for v in vals)
        print(f'  {seed:>6}  {fmt}')
    _sep()

    # 4. Summary stats per elasticity series
    for k in keys:
        series = [seeds_data[s][k] for s in sorted_seeds if k in seeds_data[s]]
        if not series:
            continue
        st = _stats(series)
        print(f'  {k}  mean={st["mean"]:>8.4f}  std={st["std"]:>8.4f}  '
              f'min={st["mn"]:>8.4f}  max={st["mx"]:>8.4f}')
    print()


def elasticity_pair_best_sim_across_specs(elas_rows: list[dict], all_rows: list[dict], label: str = ''):
    """Analysis 31 — 4 elasticities for one product pair, one best-simulation per spec, across all specs."""
    _header(f'{label} -- Elasticity pair: best simulation per spec, compared across specifications')

    # Step 1: best seed per spec (sorted by objective ascending)
    best_list = _best_per_spec(all_rows)
    if not best_list:
        print('  No data.\n')
        return

    # Step 2: for each spec's best seed, collect cross-price pairs that have data
    spec_cross: dict[str, dict[tuple, float]] = {}   # spec -> {(j,k): elasticity}
    skipped = 0
    for entry in best_list:
        spec, seed = entry['spec'], entry['seed']
        pairs = {}
        for r in elas_rows:
            if r['spec'] == spec and r['seed'] == seed and r['own_price'] == 'False':
                pairs[(r['product_j'], r['product_k'])] = float(r['elasticity'])
        if pairs:
            spec_cross[spec] = pairs
        else:
            skipped += 1

    if not spec_cross:
        print('  No elasticity data for any spec\'s best seed.\n')
        return

    # Step 3: select pair appearing in most specs; break ties by highest mean cross-elasticity
    pair_specs: dict[tuple, list[float]] = {}
    for pairs in spec_cross.values():
        for pair, val in pairs.items():
            pair_specs.setdefault(pair, []).append(val)

    max_coverage = max(len(v) for v in pair_specs.values())
    candidates   = {p: v for p, v in pair_specs.items() if len(v) == max_coverage}
    prod_j, prod_k = max(candidates, key=lambda p: sum(candidates[p]) / len(candidates[p]))

    n_specs = len(spec_cross)
    print(f'  Product pair selected: j={prod_j}, k={prod_k}')
    print(f'  (pair with highest mean cross-price elasticity across {n_specs} spec(s) with data)')
    if skipped:
        print(f'  Specs without elasticity data for best seed: {skipped} (skipped)')
    print()

    # Step 4: collect 4 values per spec from its best seed
    keys    = ['e_jj', 'e_kk', 'e_jk', 'e_kj']
    records = []   # list of (rank, spec, seed, obj, {key: val})
    for rank, entry in enumerate(best_list, 1):
        spec, seed = entry['spec'], entry['seed']
        if spec not in spec_cross:
            continue
        vals: dict[str, float] = {}
        for r in elas_rows:
            if r['spec'] != spec or r['seed'] != seed:
                continue
            pj, pk, v = r['product_j'], r['product_k'], float(r['elasticity'])
            if pj == prod_j and pk == prod_j:
                vals['e_jj'] = v
            elif pj == prod_k and pk == prod_k:
                vals['e_kk'] = v
            elif pj == prod_j and pk == prod_k:
                vals['e_jk'] = v
            elif pj == prod_k and pk == prod_j:
                vals['e_kj'] = v
        records.append((rank, spec, seed, float(entry['objective']), vals))

    spec_w = 55
    print(f'  {"Rank":>4}  {"Specification":<{spec_w}}  {"Seed":>6}  {"Obj":>10}  '
          f'{"e_jj":>10}  {"e_kk":>10}  {"e_jk":>10}  {"e_kj":>10}')
    _sep()
    for rank, spec, seed, obj, vals in records:
        def _fv(k):
            v = vals.get(k, float('nan'))
            return f'{v:>10.4f}' if not math.isnan(v) else f'{"n/a":>10}'
        print(f'  {rank:>4}  {spec:<{spec_w}}  {seed:>6}  {obj:>10.4f}  '
              f'{_fv("e_jj")}  {_fv("e_kk")}  {_fv("e_jk")}  {_fv("e_kj")}')
    _sep()

    # Summary stats across specs
    for k in keys:
        series = [r[4].get(k) for r in records if k in r[4]]
        if not series:
            continue
        st = _stats(series)
        print(f'  {k}  mean={st["mean"]:>8.4f}  std={st["std"]:>8.4f}  '
              f'min={st["mn"]:>8.4f}  max={st["mx"]:>8.4f}')
    print()


def price_coef_across_sims(rows: list[dict], label: str = ''):
    """Analysis 19 — price coefficients across all simulations of the best-mean-obj spec."""
    _header(f'{label} -- Price coefficient across all simulations (best-mean-obj spec)')

    # Spec with lowest mean objective across all its starts
    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], []).append(r)
    if not by_spec:
        print('  No data.\n')
        return

    best_spec = min(by_spec,
                    key=lambda s: sum(float(r['objective']) for r in by_spec[s]) / len(by_spec[s]))
    spec_rows = sorted(by_spec[best_spec], key=lambda r: r['seed'])
    mean_obj  = sum(float(r['objective']) for r in spec_rows) / len(spec_rows)

    print(f'  Spec (lowest mean obj): {best_spec}')
    print(f'  Mean objective: {mean_obj:.4f}   N simulations: {len(spec_rows)}\n')

    print(f'  {"Seed":>6}  {"Objective":>12}  {"price_coef":>12}  {"Valid":>6}')
    _sep()
    pcoefs = []
    for r in spec_rows:
        obj   = float(r['objective'])
        alpha = float(r['price_coef'])
        valid = _valid(r)
        pcoefs.append(alpha)
        marker = 'yes' if valid else 'no'
        print(f'  {r["seed"]:>6}  {obj:>12.4f}  {alpha:>12.4f}  {marker:>6}')
    _sep()

    st = _stats(pcoefs)
    n_valid = sum(1 for r in spec_rows if _valid(r))
    print(f'\n  N valid (price_coef < 0): {n_valid} / {len(spec_rows)}')
    print(f'  mean  = {st["mean"]:>10.4f}')
    print(f'  std   = {st["std"]:>10.4f}')
    print(f'  min   = {st["mn"]:>10.4f}')
    print(f'  max   = {st["mx"]:>10.4f}')
    print(f'  range = {st["mx"] - st["mn"]:>10.4f}')
    print()


def main():
    completed = []

    # --- BLP analyses ---
    try:
        blp_csv       = _find('blp/blp_multistart_all.csv')
        blp_elas_csv  = _find('blp/blp_elasticities_detail.csv')
        blp_rows      = _load(blp_csv)
        blp_elas_rows = _load(blp_elas_csv)
        blp_analysis_dir = _analysis_dir(blp_csv)
        print(f'Loaded BLP  multistart:   {len(blp_rows)} rows from {blp_csv}')
        print(f'Loaded BLP  elasticities: {len(blp_elas_rows)} rows from {blp_elas_csv}')
        blp_rows, blp_elas_rows = _cap_seeds(blp_rows, blp_elas_rows)
        print(f'  capped to <= {MAX_SEEDS_PER_SPEC} seeds/spec: {len(blp_rows)} starts\n')

        _run_and_save(blp_analysis_dir, '01_objective_ranking.txt',              objective_ranking,                    blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '07_price_coef_sensitivity.txt',         price_coef_sensitivity,               blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '02_multistart_stability.txt',           multistart_stability,                 blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '03_convergence_audit.txt',              convergence_audit,                    blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '04_global_minimum.txt',                 global_minimum,                       blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '05_two_basin_analysis.txt',             two_basin_analysis,                   blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '39_starting_value_sensitivity.txt',     starting_value_sensitivity,           blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '20_elasticity_own_summary.txt',         elasticity_own_summary,               blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '21_elasticity_multistart_stability.txt', elasticity_multistart_stability,     blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '22_elasticity_top_substitutes.txt',     elasticity_top_substitutes,           blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '23_elasticity_asymmetry.txt',           elasticity_asymmetry,                 blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '26_elasticity_own_cross_spec_stability.txt',   elasticity_own_cross_spec_stability,   blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '27_elasticity_cross_cross_spec_stability.txt', elasticity_cross_cross_spec_stability, blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '28_elasticity_spec_pairwise_mad.txt',          elasticity_spec_pairwise_mad,          blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '36_objective_spec_comparison.txt',             objective_spec_comparison,             blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '30_elasticity_pair_across_sims.txt',           elasticity_pair_across_sims,           blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '31_elasticity_pair_best_sim_across_specs.txt', elasticity_pair_best_sim_across_specs, blp_elas_rows, blp_rows, 'BLP')
        _run_and_save(blp_analysis_dir, '19_price_coef_across_sims.txt',               price_coef_across_sims,                blp_rows, 'BLP')
        completed.append(blp_analysis_dir)
    except FileNotFoundError as e:
        print(f'[WARN] Skipping BLP analyses: {e}\n')

    # --- Nevo analyses ---
    try:
        nevo_csv       = _find('nevo/multistart_all.csv')
        nevo_elas_csv  = _find('nevo/elasticities_detail.csv')
        nevo_rows      = _load(nevo_csv)
        nevo_elas_rows = _load(nevo_elas_csv)
        nevo_analysis_dir = _analysis_dir(nevo_csv)
        print(f'Loaded Nevo multistart:   {len(nevo_rows)} rows from {nevo_csv}')
        print(f'Loaded Nevo elasticities: {len(nevo_elas_rows)} rows from {nevo_elas_csv}')
        nevo_rows, nevo_elas_rows = _cap_seeds(nevo_rows, nevo_elas_rows)
        print(f'  capped to <= {MAX_SEEDS_PER_SPEC} seeds/spec: {len(nevo_rows)} starts\n')

        _run_and_save(nevo_analysis_dir, '01_objective_ranking.txt',              objective_ranking,                    nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '37_demographic_expansion.txt',          nevo_demographic_expansion,           nevo_rows)
        _run_and_save(nevo_analysis_dir, '38_x2_comparison.txt',                  nevo_x2_comparison,                   nevo_rows)
        _run_and_save(nevo_analysis_dir, '07_price_coef_sensitivity.txt',         price_coef_sensitivity,               nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '02_multistart_stability.txt',           multistart_stability,                 nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '03_convergence_audit.txt',              convergence_audit,                    nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '04_global_minimum.txt',                 global_minimum,                       nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '05_two_basin_analysis.txt',             two_basin_analysis,                   nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '39_starting_value_sensitivity.txt',     starting_value_sensitivity,           nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '20_elasticity_own_summary.txt',         elasticity_own_summary,               nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '21_elasticity_multistart_stability.txt', elasticity_multistart_stability,     nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '22_elasticity_top_substitutes.txt',     elasticity_top_substitutes,           nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '23_elasticity_asymmetry.txt',           elasticity_asymmetry,                 nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '24_elasticity_cross_spec_correlation.txt', nevo_elasticity_cross_spec_correlation, nevo_elas_rows, nevo_rows)
        _run_and_save(nevo_analysis_dir, '25_elasticity_firm_substitution.txt',   nevo_elasticity_firm_substitution,    nevo_elas_rows, nevo_rows)
        _run_and_save(nevo_analysis_dir, '26_elasticity_own_cross_spec_stability.txt',   elasticity_own_cross_spec_stability,   nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '27_elasticity_cross_cross_spec_stability.txt', elasticity_cross_cross_spec_stability, nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '28_elasticity_spec_pairwise_mad.txt',          elasticity_spec_pairwise_mad,          nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '36_objective_spec_comparison.txt',             objective_spec_comparison,             nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '30_elasticity_pair_across_sims.txt',           elasticity_pair_across_sims,           nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '31_elasticity_pair_best_sim_across_specs.txt', elasticity_pair_best_sim_across_specs, nevo_elas_rows, nevo_rows, 'NEVO')
        _run_and_save(nevo_analysis_dir, '19_price_coef_across_sims.txt',               price_coef_across_sims,                nevo_rows, 'NEVO')
        completed.append(nevo_analysis_dir)
    except FileNotFoundError as e:
        print(f'[WARN] Skipping Nevo analyses: {e}\n')

    if completed:
        print('\nAnalysis saved to:')
        for d in completed:
            print(f'  {d}')
    else:
        print('[ERROR] No data found for either dataset. Nothing was analysed.')


if __name__ == '__main__':
    main()
