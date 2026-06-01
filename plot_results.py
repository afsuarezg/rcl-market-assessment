"""
plot_results.py -- publication-quality figures for BLP/Nevo multistart estimation results.

Reads the same 4 CSVs as analyze_results.py and saves one PNG per analysis to:
  results/nevo/graphs/*.png
  results/blp/graphs/*.png

Run:
  python plot_results.py
"""

import csv
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

sns.set_theme(style='ticks', font_scale=1.05)
PALETTE   = sns.color_palette('Set2')
FIG_DPI   = 150
COL_VALID   = PALETTE[1]   # green
COL_INVALID = PALETTE[2]   # red/orange
COL_BASIN_A = PALETTE[0]   # blue
COL_BASIN_B = PALETTE[3]   # purple

# ---------------------------------------------------------------------------
# Paths  (mirrors analyze_results.py)
# ---------------------------------------------------------------------------

_OAK_ROOT   = Path('/oak/stanford/groups/polinsky/blp_nevo')
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


def _graphs_dir(csv_path: Path) -> Path:
    """Return <dataset_root>/graphs/, regardless of csv/ nesting."""
    parent = csv_path.parent
    dataset_root = parent.parent if parent.name == 'csv' else parent
    return dataset_root / 'graphs'


def _resolve_paths():
    """Resolve all CSV paths at runtime (not import time) so missing BLP files don't break import."""
    nevo_csv      = _find('nevo/multistart_all.csv')
    blp_csv       = _find('blp/blp_multistart_all.csv')
    nevo_elas_csv = _find('nevo/elasticities_detail.csv')
    blp_elas_csv  = _find('blp/blp_elasticities_detail.csv')
    return (nevo_csv, blp_csv, nevo_elas_csv, blp_elas_csv,
            _graphs_dir(nevo_csv), _graphs_dir(blp_csv))

# ---------------------------------------------------------------------------
# Helpers  (mirrors analyze_results.py)
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _valid(row: dict) -> bool:
    """Mirror of analyze_results._valid: require price_coef < 0 AND objective >= 0.

    Negative GMM objectives are numerical failures (g'Wg is PSD); excluding
    them prevents `_best_per_spec` from picking a spurious 'best' start.
    """
    try:
        return float(row['price_coef']) < 0 and float(row['objective']) >= 0
    except (ValueError, TypeError):
        return False


MAX_SEEDS_PER_SPEC = 40


def _cap_seeds(rows: list[dict], elas_rows: list[dict],
               max_seeds: int = MAX_SEEDS_PER_SPEC) -> tuple[list[dict], list[dict]]:
    """Keep at most `max_seeds` unique seeds per spec (lowest seed numbers).

    Caps the multistart sample so each spec contributes at most `max_seeds`
    simulations; elasticity rows are restricted to the same kept (spec, seed)
    pairs. Applied to raw loaded rows before any validity filtering.
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


def _filter_valid(rows: list[dict], elas_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep only valid starts and the elasticity rows belonging to them.

    A spec with no valid start is dropped entirely (it contributes no rows to
    either list). Elasticity rows join multistart rows on (spec, seed).
    """
    valid_rows = [r for r in rows if _valid(r)]
    keys = {(r['spec'], r['seed']) for r in valid_rows}
    valid_elas = [r for r in elas_rows if (r['spec'], r['seed']) in keys]
    return valid_rows, valid_elas


def _best_per_spec(rows: list[dict]) -> list[dict]:
    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], []).append(r)
    result = []
    for spec_rows in by_spec.values():
        valid = [r for r in spec_rows if _valid(r)]
        pool  = valid if valid else spec_rows
        result.append(min(pool, key=lambda r: float(r['objective'])))
    return sorted(result, key=lambda r: float(r['objective']))


def _best_seed_map(all_rows: list[dict]) -> dict[str, str]:
    return {r['spec']: r['seed'] for r in _best_per_spec(all_rows)}


def _stats(vals: list[float]) -> dict:
    n = len(vals)
    if n == 0:
        return dict(mean=float('nan'), median=float('nan'), std=0.0,
                    mn=float('nan'), mx=float('nan'))
    mean      = sum(vals) / n
    sorted_v  = sorted(vals)
    mid       = n // 2
    median    = sorted_v[mid] if n % 2 else (sorted_v[mid-1] + sorted_v[mid]) / 2
    std       = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
    return dict(mean=mean, median=median, std=std, mn=sorted_v[0], mx=sorted_v[-1])


def _short(spec: str, max_len: int = 34) -> str:
    """Strip verbose x2=/demos= prefixes and shorten spec label."""
    s = spec.replace("x2=", "").replace("demos=", "").replace(" | ", "\n")
    return textwrap.fill(s, max_len)


def _numeric_seed_sort(seeds):
    """Return (sorted_seeds, perm).

    `perm[i]` is the original index of the i-th seed in numerically sorted
    order. Non-numeric seeds sort after numeric ones in lexicographic order.
    Callers can use `perm` to reorder aligned data lists (pcoefs, colors, …).
    """
    def key(i):
        try:
            return (0, int(seeds[i]))
        except (ValueError, TypeError):
            return (1, seeds[i])
    perm = sorted(range(len(seeds)), key=key)
    return [seeds[i] for i in perm], perm


def _set_sparse_seed_xaxis(ax, sorted_seeds, *, target_ticks: int = 10,
                           fontsize: int = 8):
    """Configure `ax` with ~target_ticks evenly-spaced numeric seed ticks.

    Caller must have already applied the same seed ordering to the plotted
    data (typically via `_numeric_seed_sort`).
    """
    n = len(sorted_seeds)
    if n == 0:
        return
    step = max(1, math.ceil(n / target_ticks))
    idx  = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([sorted_seeds[i] for i in idx],
                       rotation=0, ha='center', fontsize=fontsize)
    ax.set_xlabel('Seed (numerically ordered; intermediate ticks hidden)',
                  fontsize=9)


def _vshort_spec(spec: str) -> str:
    """Single-line condensed spec label for compact key listings."""
    s = spec.replace("x2=", "").replace("demos=", "").replace(" | ", " · ")
    return s.replace("['", "[").replace("']", "]").replace("', '", ",")


def _spec_key_text(labels: list[str]) -> tuple[str, int]:
    """Format a numbered spec key as multi-column monospaced text.

    Picks 1/2/3 columns based on length so the block stays readable.
    Returns (text, n_rows) so callers can budget figure space.
    """
    n = len(labels)
    if n == 0:
        return '', 0
    n_cols = 1 if n <= 4 else (2 if n <= 12 else 3)
    n_rows = math.ceil(n / n_cols)
    entries = [f'{i + 1:>2}. {labels[i]}' for i in range(n)]
    col_w = max(len(e) for e in entries) + 3
    out = []
    for r in range(n_rows):
        parts = []
        for c in range(n_cols):
            idx = c * n_rows + r
            if idx < n:
                parts.append(f'{entries[idx]:<{col_w}}')
        out.append(''.join(parts).rstrip())
    return '\n'.join(out), n_rows


def _savefig(fig: plt.Figure, out_dir: Path, filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def _footer(fig: plt.Figure, text: str):
    fig.text(0.5, 0.01, text, ha='center', va='bottom',
             fontsize=7, color='grey', style='italic')


def _heatmap_annot_kws(n: int) -> dict:
    """Scale annotation font size with matrix size."""
    return {'size': max(5, 10 - n // 6)}


def _clip_with_outlier_markers(ax, xs, ys, color, marker_size: int = 60, *,
                               colors=None, mean_color=None,
                               add_legend: bool = True):
    """Scatter with y-axis clipped to Tukey 3*IQR bounds.

    Points outside [Q1 - 3*IQR, Q3 + 3*IQR] render as triangles at the panel
    edge with the actual numeric value annotated. The dashed reference line
    shows the mean of inliers only, so it tracks the bulk rather than being
    pulled by outliers. NaN ys are dropped. With fewer than 6 points or a
    zero IQR, no clipping is applied and behavior matches a plain scatter
    plus mean line.

    Optional kwargs:
      colors:     per-point colour list aligned with xs/ys. Overrides
                  `color` for inlier scatter and outlier triangle fills.
      mean_color: colour for the dashed reference line (defaults to `color`).
      add_legend: when False the helper still draws the mean line and sets
                  its label but skips ax.legend(), so callers can compose a
                  custom legend using the returned handle.

    Returns the dashed mean Line2D (or None when there are no inliers).
    """
    pts = [(x, y, c) for x, y, c in zip(xs, ys,
                                        colors if colors is not None else [color] * len(xs))
           if not (isinstance(y, float) and math.isnan(y))]
    if not pts:
        return None

    yvals = [y for _, y, _ in pts]
    n = len(yvals)
    s = sorted(yvals)
    if n >= 6:
        q1  = s[n // 4]
        q3  = s[(3 * n) // 4]
        iqr = q3 - q1
    else:
        q1 = q3 = iqr = 0.0

    if n < 6 or iqr <= 0:
        inliers, outliers = pts, []
        lo, hi = min(yvals), max(yvals)
    else:
        lo = q1 - 3.0 * iqr
        hi = q3 + 3.0 * iqr
        inliers  = [p for p in pts if lo <= p[1] <= hi]
        outliers = [p for p in pts if p[1] < lo or p[1] > hi]

    mean_line = None
    if inliers:
        ix = [p[0] for p in inliers]
        iy = [p[1] for p in inliers]
        ic = [p[2] for p in inliers]
        ax.scatter(ix, iy, c=ic, s=marker_size, zorder=3)
        mean_v = sum(iy) / len(iy)
        label  = (f'mean(inliers, n={len(inliers)}/{n})={mean_v:.4f}'
                  if outliers else f'mean={mean_v:.4f}')
        mean_line = ax.axhline(mean_v, color=(mean_color or color),
                               linewidth=1.2, linestyle='--', alpha=0.85,
                               label=label)
        if add_legend:
            ax.legend(fontsize=8, loc='best')

    if not outliers:
        return mean_line

    rng = hi - lo if hi > lo else max(abs(hi), 1.0)
    pad = 0.05 * rng
    ax.set_ylim(lo - pad, hi + pad)

    y_top    = hi + pad * 0.5
    y_bottom = lo - pad * 0.5
    for x, y, c in outliers:
        if y > hi:
            ax.scatter([x], [y_top], marker='^', color=c,
                       s=marker_size + 20, edgecolor='black', linewidth=0.8,
                       zorder=5, clip_on=False)
            ax.annotate(f'{y:.2f}', xy=(x, y_top),
                        xytext=(0, -10), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='black',
                        zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.75))
        else:
            ax.scatter([x], [y_bottom], marker='v', color=c,
                       s=marker_size + 20, edgecolor='black', linewidth=0.8,
                       zorder=5, clip_on=False)
            ax.annotate(f'{y:.2f}', xy=(x, y_bottom),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, color='black',
                        zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.75))
    return mean_line


# ---------------------------------------------------------------------------
# Generic plot functions (both Nevo and BLP)
# ---------------------------------------------------------------------------

def plot_objective_ranking(rows: list[dict], label: str, out_dir: Path):
    best = _best_per_spec(rows)
    specs   = [_short(r['spec']) for r in best]
    objs    = [float(r['objective']) for r in best]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(best) + 1.5)))
    bars    = ax.barh(range(len(best)), objs, color=COL_VALID, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(specs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('GMM Objective')
    ax.set_title(f'{label} — Objective Ranking by Specification', fontweight='bold')
    for i, obj in enumerate(objs):
        ax.text(obj + max(objs) * 0.005, i, f'{obj:.4g}', va='center', fontsize=7, color='#333')
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 01 · GMM objective for best start per specification')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '01_objective_ranking.png')


def plot_price_coef(rows: list[dict], label: str, out_dir: Path):
    best = sorted(_best_per_spec(rows), key=lambda r: float(r['price_coef']))
    specs  = [_short(r['spec']) for r in best]
    pcoefs = [float(r['price_coef']) for r in best]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(best) + 1.5)))
    ax.barh(range(len(best)), pcoefs, color=COL_VALID, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(specs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Price Coefficient (α)')
    ax.set_title(f'{label} — Price Coefficient Sensitivity', fontweight='bold')
    for i, pc in enumerate(pcoefs):
        # Anchor the label just inside the bar tip, growing toward zero so it
        # always stays within the axes (bars can be long and negative).
        if pc >= 0:
            ax.text(pc, i, f'α≈{pc:.4f}  ', va='center', ha='right', fontsize=7, color='#333')
        else:
            ax.text(pc, i, f'  α≈{pc:.4f}', va='center', ha='left', fontsize=7, color='#333')
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 07 · Price coefficient across specifications')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '07_price_coef.png')


def plot_multistart_stability(rows: list[dict], label: str, out_dir: Path):
    from collections import defaultdict
    by_spec: dict[str, list] = defaultdict(list)
    for r in rows:
        by_spec[r['spec']].append(r)

    specs_sorted = sorted(by_spec, key=lambda s: min(float(r['objective'])
                                                     for r in by_spec[s]))
    short_labels = [_short(s, 28) for s in specs_sorted]
    # Count unique valid seeds (not rows): a seed can appear in more than one
    # row, which would otherwise inflate the per-spec count vs the actual starts.
    objs_by_spec = [list({r['seed']: float(r['objective']) for r in by_spec[s]}.values())
                    for s in specs_sorted]
    counts       = [len(o) for o in objs_by_spec]
    variances    = [sum((v - sum(o) / len(o)) ** 2 for v in o) / len(o) for o in objs_by_spec]

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(specs_sorted) + 2), 5))
    for xi, spec in enumerate(specs_sorted):
        for r in by_spec[spec]:
            mk  = '*' if r.get('best') == 'True' else 'o'
            ax.scatter(xi, float(r['objective']), color=COL_VALID, marker=mk,
                       s=60, zorder=3, edgecolors='white', linewidths=0.4)

    # Headroom so the per-spec count and variance annotations clear the points.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))
    for xi, (c, var) in enumerate(zip(counts, variances)):
        ax.text(xi, 0.99, f'n={c}', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color='#555')
        ax.text(xi, 0.94, f'var={var:.3g}', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6, color='#888')

    ax.set_xticks(range(len(specs_sorted)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('GMM Objective')
    ax.set_title(f'{label} — Multi-start Convergence Stability', fontweight='bold')
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_VALID, markersize=8, label='Valid start'),
              Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',     markersize=10, label='Best start')]
    ax.legend(handles=legend, fontsize=8)
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 02 · Each point = one random start; (*) = best start; '
                 f'n = unique valid seeds per spec; var = variance of GMM objective across them')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '02_multistart_stability.png')


def plot_convergence_scatter(rows: list[dict], label: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        mk  = '*' if r.get('best') == 'True' else 'o'
        ax.scatter(float(r['price_coef']), float(r['objective']),
                   color=COL_VALID, marker=mk, s=70, alpha=0.85,
                   edgecolors='white', linewidths=0.4)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='α = 0 boundary')
    ax.set_xlabel('Price Coefficient (α)')
    ax.set_ylabel('GMM Objective')
    ax.set_title(f'{label} — Convergence Scatter', fontweight='bold')
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COL_VALID, markersize=8, label='Valid start'),
              Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',     markersize=10, label='Best start')]
    ax.legend(handles=legend, fontsize=8)
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 03 · Valid starts: objective vs price coefficient')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '03_convergence_scatter.png')


def plot_global_minimum(rows: list[dict], label: str, out_dir: Path):
    valid = sorted([r for r in rows if _valid(r)], key=lambda r: float(r['objective']))
    if not valid:
        print(f'  [{label}] No valid starts — skipping 04_global_minimum.png')
        return
    best_obj = float(valid[0]['objective'])
    ylabels  = [f'{r["spec"][:28]}\nseed={r["seed"]}' for r in valid]
    deltas   = [float(r['objective']) - best_obj for r in valid]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(valid) + 1.5)))
    colors  = [PALETTE[0] if d == 0 else PALETTE[3] for d in deltas]
    ax.barh(range(len(valid)), deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Δ from Global Minimum (GMM objective)')
    ax.set_title(f'{label} — Distance from Global Minimum', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 04 · Valid starts ranked by Δ from global minimum')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '04_global_minimum.png')


def plot_basin_scatter(rows: list[dict], label: str, out_dir: Path):
    valid = [r for r in rows if _valid(r)]
    if not valid:
        print(f'  [{label}] No valid starts — skipping 05_basin_scatter.png')
        return
    global_min = min(float(r['objective']) for r in valid)
    THRESHOLD  = 5.0

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in valid:
        d   = float(r['objective']) - global_min
        col = COL_BASIN_A if d <= THRESHOLD else COL_BASIN_B
        ax.scatter(float(r['objective']), float(r['price_coef']),
                   color=col, s=70, alpha=0.85, edgecolors='white', linewidths=0.4)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel('GMM Objective')
    ax.set_ylabel('Price Coefficient (α)')
    ax.set_title(f'{label} — Two-Basin Analysis', fontweight='bold')
    from matplotlib.patches import Patch
    legend = [Patch(color=COL_BASIN_A, label=f'Basin A (Δ ≤ {THRESHOLD})'),
              Patch(color=COL_BASIN_B, label=f'Basin B (Δ > {THRESHOLD})')]
    ax.legend(handles=legend, fontsize=8)
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 05 · Valid starts classified by distance from global minimum')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '05_basin_scatter.png')


def plot_own_elas_boxplot(elas_rows: list[dict], all_rows: list[dict],
                          label: str, out_dir: Path):
    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)

    # {spec: [elas values]} for best seed
    own_by_spec: dict[str, list[float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            own_by_spec.setdefault(r['spec'], []).append(float(r['elasticity']))

    specs  = [r['spec'] for r in best_list if r['spec'] in own_by_spec]
    data   = [own_by_spec[s] for s in specs]
    labels = [_short(s, 28) for s in specs]

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(specs) + 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for i, (patch, vals) in enumerate(zip(bp['boxes'], data)):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
        # strip plot overlay
        x_jitter = [i + 1 + (hash(str(v)) % 100 - 50) / 800 for v in vals]
        ax.scatter(x_jitter, vals, color=PALETTE[i % len(PALETTE)],
                   s=12, alpha=0.35, zorder=2)

    ax.set_xticks(range(1, len(specs) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Own-Price Elasticity')
    ax.set_title(f'{label} — Own-Price Elasticity Distribution by Specification', fontweight='bold')
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 20 · Box = IQR; dots = individual products (best start per spec)')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '20_own_elas_boxplot.png')


def plot_multistart_elas_dotplot(elas_rows: list[dict], label: str, out_dir: Path):
    """Analysis 21 — per-product own-price elas across seeds (skip if <2 seeds per spec)."""
    seeds_per_spec: dict[str, set] = {}
    for r in elas_rows:
        if r['own_price'] == 'True':
            seeds_per_spec.setdefault(r['spec'], set()).add(r['seed'])

    multi_specs = {s for s, seeds in seeds_per_spec.items() if len(seeds) > 1}
    if not multi_specs:
        print(f'  [{label}] No multi-seed specs — skipping 21_multistart_elas_dotplot.png')
        return

    for si, spec in enumerate(sorted(multi_specs)):
        seeds = sorted(seeds_per_spec[spec])
        # {product: {seed: elas}}
        prod_seed: dict[str, dict[str, float]] = {}
        for r in elas_rows:
            if r['spec'] == spec and r['own_price'] == 'True':
                prod_seed.setdefault(r['product_j'], {})[r['seed']] = float(r['elasticity'])

        products = sorted(prod_seed)
        fig, ax = plt.subplots(figsize=(max(5, len(seeds) * 1.5 + 1), max(4, len(products) * 0.3 + 1.5)))
        seed_x = {s: xi for xi, s in enumerate(seeds)}
        for prod in products:
            xs = [seed_x[s] for s in seeds if s in prod_seed[prod]]
            ys = [prod_seed[prod][s] for s in seeds if s in prod_seed[prod]]
            ax.plot(xs, ys, '-o', markersize=5, linewidth=1, alpha=0.65,
                    color=PALETTE[hash(prod) % len(PALETTE)])
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([f'seed={s}' for s in seeds], fontsize=9)
        ax.set_ylabel('Own-Price Elasticity')
        short_spec = _short(spec, 50)
        ax.set_title(f'{label} — Multistart Elasticity Stability\n{short_spec}', fontweight='bold', fontsize=10)
        sns.despine(ax=ax)
        _footer(fig, f'{label} · Analysis 21 · Each line = one product across seeds')
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        suffix = f'_spec{si+1}' if len(multi_specs) > 1 else ''
        _savefig(fig, out_dir, f'21_multistart_elas_dotplot{suffix}.png')


def plot_cross_elas_heatmap(elas_rows: list[dict], all_rows: list[dict],
                             label: str, out_dir: Path, max_products: int = 24):
    best_list = _best_per_spec(all_rows)
    if not best_list:
        print(f'  [{label}] No valid specs — skipping 22_cross_elas_heatmap.png')
        return
    spec = best_list[0]['spec']
    seed = best_list[0]['seed']

    elas_dict: dict[tuple, float] = {}
    for candidate in best_list:
        spec = candidate['spec']
        seed = candidate['seed']
        for r in elas_rows:
            if r['spec'] == spec and r['seed'] == seed:
                elas_dict[(r['product_j'], r['product_k'])] = float(r['elasticity'])
        if elas_dict:
            break

    if not elas_dict:
        print(f'  [{label}] No elasticity rows match any best spec — skipping 22_cross_elas_heatmap.png')
        return

    products = sorted({p for (j, k) in elas_dict for p in (j, k)})
    if len(products) > max_products:
        # Keep most elastic own-price products
        own = {j: elas_dict.get((j, j), 0.0) for j in products}
        products = sorted(own, key=lambda p: own[p])[:max_products]

    n = len(products)
    matrix = [[elas_dict.get((pj, pk), float('nan')) for pk in products] for pj in products]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.38 + 2), max(6, n * 0.38 + 1.5)))
    annot   = n <= 30
    annot_kws = _heatmap_annot_kws(n)
    fmt = '.2f' if annot else ''

    # Centre colormap at 0 so own-price (negative) and cross-price (positive) diverge
    flat = [v for row in matrix for v in row if not math.isnan(v)]
    vmax = max(abs(v) for v in flat) if flat else 1.0
    vmin = -vmax

    sns.heatmap(matrix, ax=ax, xticklabels=products, yticklabels=products,
                cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
                annot=annot, fmt=fmt, annot_kws=annot_kws,
                linewidths=0.3 if n <= 30 else 0,
                cbar_kws={'label': 'Elasticity'})
    ax.set_xlabel('Product k')
    ax.set_ylabel('Product j')
    short_spec = _short(best_list[0]['spec'], 60)
    ax.set_title(f'{label} — Cross-Price Elasticity Matrix\n(best spec: {short_spec})',
                 fontweight='bold', fontsize=9)
    ax.tick_params(axis='x', rotation=90, labelsize=max(5, 9 - n // 10))
    ax.tick_params(axis='y', rotation=0,  labelsize=max(5, 9 - n // 10))
    _footer(fig, f'{label} · Analysis 22 · Diagonal = own-price; off-diagonal = cross-price')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '22_cross_elas_heatmap.png')


def plot_asymmetry_scatter(elas_rows: list[dict], all_rows: list[dict],
                            label: str, out_dir: Path):
    best_list = _best_per_spec(all_rows)
    if not best_list:
        print(f'  [{label}] No valid specs — skipping 23_asymmetry_scatter.png')
        return
    spec = best_list[0]['spec']
    seed = best_list[0]['seed']

    elas_dict: dict[tuple, float] = {}
    for r in elas_rows:
        if r['spec'] == spec and r['seed'] == seed and r['own_price'] == 'False':
            elas_dict[(r['product_j'], r['product_k'])] = float(r['elasticity'])

    pairs = []
    seen  = set()
    for (j, k), ejk in elas_dict.items():
        if (k, j) in seen:
            continue
        ekj = elas_dict.get((k, j))
        if ekj is not None:
            pairs.append((ejk, ekj, abs(ejk - ekj)))
            seen.add((j, k))

    if not pairs:
        print(f'  [{label}] No symmetric pairs — skipping 23_asymmetry_scatter.png')
        return

    ejks  = [p[0] for p in pairs]
    ekjs  = [p[1] for p in pairs]
    diffs = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(ejks, ekjs, c=diffs, cmap='YlOrRd', s=18, alpha=0.7,
                    edgecolors='none')
    # 45° symmetry line
    lo = min(min(ejks), min(ekjs))
    hi = max(max(ejks), max(ekjs))
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.9, alpha=0.5, label='Perfect symmetry')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('|e_jk − e_kj|', fontsize=8)
    ax.set_xlabel('e_jk')
    ax.set_ylabel('e_kj')
    ax.set_title(f'{label} — Cross-Price Elasticity Symmetry', fontweight='bold')
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    _footer(fig, f'{label} · Analysis 23 · Points on diagonal = perfectly symmetric substitution')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '23_asymmetry_scatter.png')


def plot_own_stability_heatmap(elas_rows: list[dict], all_rows: list[dict],
                                label: str, out_dir: Path):
    """Analysis 26 — own-price elas × spec heatmap with CV sidebar."""
    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2:
        print(f'  [{label}] <2 specs — skipping 26_own_stability_heatmap.png')
        return

    prod_spec: dict[str, dict[str, float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            prod_spec.setdefault(r['product_j'], {})[r['spec']] = float(r['elasticity'])

    products = [p for p, d in prod_spec.items() if len(d) == len(specs)]
    if not products:
        print(f'  [{label}] No products in all specs — skipping 26_own_stability_heatmap.png')
        return

    MAX_P = 20 if len(products) > 50 else len(products)
    products = sorted(products, key=lambda p: sum(prod_spec[p].values()) / len(specs))[:MAX_P]

    # Compute CV per product
    def _cv(vals):
        s = _stats(vals)
        return s['std'] / abs(s['mean']) if s['mean'] != 0 else float('nan')

    cvs = [_cv([prod_spec[p][s] for s in specs]) for p in products]
    # Sort by CV descending
    order = sorted(range(len(products)), key=lambda i: cvs[i] if not math.isnan(cvs[i]) else -1,
                   reverse=True)
    products = [products[i] for i in order]
    cvs      = [cvs[i] for i in order]

    matrix = [[prod_spec[p][s] for s in specs] for p in products]
    short_specs = [_short(s, 24) for s in specs]

    n_prod = len(products)
    n_spec = len(specs)
    fig_w  = max(7, n_spec * 1.2 + 3)
    fig_h  = max(4, n_prod * 0.4 + 2)
    fig, (ax_main, ax_cv) = plt.subplots(1, 2, figsize=(fig_w + 1.5, fig_h),
                                          gridspec_kw={'width_ratios': [n_spec, 1.2], 'wspace': 0.05})

    annot     = n_prod <= 25 and n_spec <= 12
    annot_kws = _heatmap_annot_kws(max(n_prod, n_spec))
    flat = [v for row in matrix for v in row if not math.isnan(v)]
    vmax = max(abs(v) for v in flat) if flat else 1.0

    sns.heatmap(matrix, ax=ax_main, xticklabels=short_specs, yticklabels=products,
                cmap='coolwarm_r', center=0, vmin=-vmax, vmax=vmax,
                annot=annot, fmt='.2f' if annot else '',
                annot_kws=annot_kws,
                linewidths=0.3 if n_prod <= 30 else 0,
                cbar_kws={'label': 'Own-Price Elas', 'shrink': 0.7})
    ax_main.set_xlabel('Specification')
    ax_main.set_ylabel('Product')
    ax_main.set_title(f'{label} — Own-Price Elasticity Cross-Spec Stability', fontweight='bold', fontsize=10)
    ax_main.tick_params(axis='x', rotation=45, labelsize=max(5, 9 - n_spec // 6))
    ax_main.tick_params(axis='y', rotation=0,  labelsize=max(5, 9 - n_prod // 8))

    # CV sidebar
    cv_clean = [[c if not math.isnan(c) else 0.0] for c in cvs]
    sns.heatmap(cv_clean, ax=ax_cv, xticklabels=['CV'], yticklabels=['' for _ in products],
                cmap='Oranges', vmin=0,
                annot=n_prod <= 25, fmt='.3f' if n_prod <= 25 else '',
                annot_kws={'size': max(5, 8 - n_prod // 8)},
                linewidths=0.3 if n_prod <= 30 else 0,
                cbar_kws={'label': 'CV', 'shrink': 0.7})
    ax_cv.set_title('CV', fontsize=9)
    ax_cv.tick_params(axis='x', labelsize=8)

    _footer(fig, f'{label} · Analysis 26 · CV = std/|mean| across specs; sorted by CV (most unstable first)')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '26_own_stability_heatmap.png')


def plot_cross_stability_heatmap(elas_rows: list[dict], all_rows: list[dict],
                                  label: str, out_dir: Path, top_k: int = 10):
    """Analysis 27 — cross-price elas for top-k pairs × spec heatmap."""
    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2 or not best_list:
        print(f'  [{label}] <2 specs — skipping 27_cross_stability_heatmap.png')
        return

    best_spec = best_list[0]['spec']
    best_seed = seed_map[best_spec]
    top_cross = sorted(
        [(r['product_j'], r['product_k'], float(r['elasticity']))
         for r in elas_rows
         if r['spec'] == best_spec and r['seed'] == best_seed and r['own_price'] == 'False'],
        key=lambda x: x[2], reverse=True
    )[:top_k]
    top_pairs = [(j, k) for j, k, _ in top_cross]

    if not top_pairs:
        print(f'  [{label}] No cross-price data — skipping 27_cross_stability_heatmap.png')
        return

    top_set = set(top_pairs)
    pair_spec: dict[tuple, dict[str, float]] = {}
    for r in elas_rows:
        if r['own_price'] == 'False' and seed_map.get(r['spec']) == r['seed']:
            key = (r['product_j'], r['product_k'])
            if key in top_set:
                pair_spec.setdefault(key, {})[r['spec']] = float(r['elasticity'])

    def _cv(vals):
        s = _stats(vals)
        return s['std'] / abs(s['mean']) if s['mean'] != 0 else float('nan')

    rows_data = []
    for (j, k) in top_pairs:
        d = pair_spec.get((j, k), {})
        valid_vals = [d[s] for s in specs if s in d]
        if len(valid_vals) >= 2:
            rows_data.append(((j, k), _cv(valid_vals), [d.get(s, float('nan')) for s in specs]))

    if not rows_data:
        print(f'  [{label}] Insufficient cross-spec data — skipping 27_cross_stability_heatmap.png')
        return

    rows_data.sort(key=lambda x: x[1] if not math.isnan(x[1]) else -1, reverse=True)

    pair_labels = [f'{j}→{k}' for (j, k), _, _ in rows_data]
    short_specs = [_short(s, 24) for s in specs]
    matrix      = [vals for _, _, vals in rows_data]

    n_pairs = len(pair_labels)
    n_spec  = len(specs)
    fig, ax = plt.subplots(figsize=(max(6, n_spec * 1.2 + 2), max(3, n_pairs * 0.55 + 2)))

    flat = [v for row in matrix for v in row if not math.isnan(v)]
    vmax = max(abs(v) for v in flat) if flat else 1.0

    annot     = n_pairs <= 20 and n_spec <= 12
    annot_kws = _heatmap_annot_kws(max(n_pairs, n_spec))

    sns.heatmap(matrix, ax=ax, xticklabels=short_specs, yticklabels=pair_labels,
                cmap='YlGnBu', vmin=0, vmax=vmax,
                annot=annot, fmt='.3f' if annot else '',
                annot_kws=annot_kws,
                linewidths=0.3, cbar_kws={'label': 'Cross-Price Elas'})
    ax.set_xlabel('Specification')
    ax.set_ylabel('Substitute Pair (j → k)')
    ax.set_title(f'{label} — Cross-Price Elasticity Cross-Spec Stability\n(top-{top_k} pairs by best spec)',
                 fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=max(5, 9 - n_spec // 6))
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    sns.despine(ax=ax, left=True, bottom=True)
    _footer(fig, f'{label} · Analysis 27 · Rows sorted by CV (most unstable first)')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '27_cross_stability_heatmap.png')


def plot_spec_pairwise_mad(elas_rows: list[dict], all_rows: list[dict],
                            label: str, out_dir: Path):
    """Analysis 28 — S×S MAD heatmap."""
    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    if len(specs) < 2:
        print(f'  [{label}] <2 specs — skipping 28_pairwise_mad_heatmap.png')
        return

    spec_prod: dict[str, dict[str, float]] = {s: {} for s in specs}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            spec_prod[r['spec']][r['product_j']] = float(r['elasticity'])

    n = len(specs)
    matrix = []
    for si in specs:
        row = []
        for sj in specs:
            if si == sj:
                row.append(0.0)
            else:
                common = set(spec_prod[si]) & set(spec_prod[sj])
                if common:
                    row.append(sum(abs(spec_prod[si][p] - spec_prod[sj][p])
                                   for p in common) / len(common))
                else:
                    row.append(float('nan'))
        matrix.append(row)

    short_specs = [_short(s, 24) for s in specs]
    import numpy as np
    mask = np.eye(n, dtype=bool)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1 + 1.5), max(4, n * 1.0 + 1.5)))
    annot     = n <= 15
    annot_kws = _heatmap_annot_kws(n)
    sns.heatmap(matrix, ax=ax, xticklabels=short_specs, yticklabels=short_specs,
                cmap='Blues', mask=mask,
                annot=annot, fmt='.3f' if annot else '',
                annot_kws=annot_kws,
                linewidths=0.5, square=True,
                cbar_kws={'label': 'Mean Absolute Deviation'})
    ax.set_title(f'{label} — Pairwise Spec Agreement (MAD of Own-Price Elasticities)',
                 fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=max(5, 9 - n // 5))
    ax.tick_params(axis='y', rotation=0,  labelsize=max(5, 9 - n // 5))
    _footer(fig, f'{label} · Analysis 28 · Cell = mean|ε_jj(A) − ε_jj(B)| across products; diagonal masked')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '28_pairwise_mad_heatmap.png')


# ---------------------------------------------------------------------------
# Nevo-specific plot functions
# ---------------------------------------------------------------------------

def plot_nevo_demographic_expansion(rows: list[dict], out_dir: Path):
    best_map = {r['spec']: r for r in _best_per_spec(rows)}

    groups = {
        'sugar': [
            "x2=['sugar'] | demos=['income']",
            "x2=['sugar'] | demos=['income', 'income_squared']",
            "x2=['sugar'] | demos=['income', 'age']",
            "x2=['sugar'] | demos=['income', 'age', 'child']",
            "x2=['sugar'] | demos=['income', 'income_squared', 'age', 'child']",
        ],
        'mushy': [
            "x2=['mushy'] | demos=['income']",
            "x2=['mushy'] | demos=['income', 'income_squared']",
            "x2=['mushy'] | demos=['income', 'age']",
            "x2=['mushy'] | demos=['income', 'age', 'child']",
            "x2=['mushy'] | demos=['income', 'income_squared', 'age', 'child']",
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (x2_label, spec_list) in zip(axes, groups.items()):
        present = [(s, best_map[s]) for s in spec_list if s in best_map]
        if not present:
            continue
        demo_labels = [s.split("demos=")[1] for s, _ in present]
        objs   = [float(r['objective']) for _, r in present]
        pcoefs = [float(r['price_coef']) for _, r in present]
        xs     = range(len(present))

        color_obj  = PALETTE[0]
        color_pc   = PALETTE[1]
        ax2 = ax.twinx()
        ax.plot(xs, objs,   '-o', color=color_obj,  linewidth=2, markersize=7, label='GMM obj')
        ax2.plot(xs, pcoefs, 's--', color=color_pc, linewidth=2, markersize=7, label='price_coef')
        ax.set_xticks(list(xs))
        ax.set_xticklabels(demo_labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('GMM Objective', color=color_obj)
        ax2.set_ylabel('Price Coef (α)', color=color_pc)
        ax.set_title(f'X2 = {x2_label}', fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color_obj)
        ax2.tick_params(axis='y', labelcolor=color_pc)

    fig.suptitle('NEVO — Effect of Adding Demographics (fixed X2)', fontweight='bold', y=1.02)
    sns.despine(fig=fig)
    _footer(fig, 'NEVO · Analysis 37 · Solid = GMM objective (left axis); dashed = price_coef (right axis)')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '37_demographic_expansion.png')


def plot_nevo_x2_comparison(rows: list[dict], out_dir: Path):
    best_map = {r['spec']: r for r in _best_per_spec(rows)}
    specs = [
        "x2=['sugar'] | demos=['income']",
        "x2=['mushy'] | demos=['income']",
        "x2=['sugar', 'mushy'] | demos=['income']",
    ]
    present = [(s, best_map[s]) for s in specs if s in best_map]
    if not present:
        print('  [NEVO] No matching specs — skipping 38_x2_comparison.png')
        return

    x2_labels = [s.split("x2=")[1].split(" |")[0] for s, _ in present]
    objs      = [float(r['objective'])  for _, r in present]
    pcoefs    = [float(r['price_coef']) for _, r in present]
    xs        = range(len(present))
    width     = 0.35

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    b1 = ax1.bar([x - width/2 for x in xs], objs,  width, color=PALETTE[0], alpha=0.8, label='GMM obj')
    b2 = ax2.bar([x + width/2 for x in xs], pcoefs, width, color=PALETTE[1], alpha=0.8, label='price_coef')
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(x2_labels, fontsize=10)
    ax1.set_ylabel('GMM Objective', color=PALETTE[0])
    ax2.set_ylabel('Price Coefficient (α)', color=PALETTE[1])
    ax1.tick_params(axis='y', labelcolor=PALETTE[0])
    ax2.tick_params(axis='y', labelcolor=PALETTE[1])
    ax1.set_title('NEVO — X2 Characteristic Comparison (demographics = income)', fontweight='bold')
    lines = [b1, b2]
    labels = ['GMM objective', 'Price coef (α)']
    ax1.legend(lines, labels, fontsize=8, loc='upper right')
    sns.despine(ax=ax1)
    _footer(fig, 'NEVO · Analysis 38 · Fixed demographics = [income]; varying X2 characteristics')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '38_x2_comparison.png')


def plot_nevo_spearman_heatmap(elas_rows: list[dict], all_rows: list[dict], out_dir: Path):
    """Analysis 24 — Spearman rank correlation heatmap of own-price elas across specs."""
    import math as _math

    def _rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank, i in enumerate(order, 1):
            ranks[i] = rank
        return ranks

    def _pearson(xs, ys):
        n = len(xs)
        if n < 2:
            return float('nan')
        mx = sum(xs) / n;  my = sum(ys) / n
        num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denom = _math.sqrt(sum((x - mx)**2 for x in xs) * sum((y - my)**2 for y in ys))
        return num / denom if denom else float('nan')

    def _spearman(xs, ys):
        return _pearson([float(r) for r in _rank(xs)], [float(r) for r in _rank(ys)])

    seed_map  = _best_seed_map(all_rows)
    best_list = _best_per_spec(all_rows)
    specs     = [r['spec'] for r in best_list]

    spec_prod: dict[str, dict[str, float]] = {s: {} for s in specs}
    for r in elas_rows:
        if r['own_price'] == 'True' and seed_map.get(r['spec']) == r['seed']:
            spec_prod[r['spec']][r['product_j']] = float(r['elasticity'])

    common = sorted(set.intersection(*[set(d) for d in spec_prod.values()])) if specs else []
    if not common:
        print('  [NEVO] No common products — skipping 24_spearman_corr_heatmap.png')
        return

    n = len(specs)
    matrix = []
    for si in specs:
        vi = [spec_prod[si][p] for p in common]
        row = []
        for sj in specs:
            vj = [spec_prod[sj][p] for p in common]
            row.append(_spearman(vi, vj))
        matrix.append(row)

    short_specs = [_short(s, 22) for s in specs]
    fig, ax = plt.subplots(figsize=(max(5, n * 1.1 + 1.5), max(4, n * 1.0 + 1.5)))
    sns.heatmap(matrix, ax=ax, xticklabels=short_specs, yticklabels=short_specs,
                cmap='RdYlGn', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', annot_kws=_heatmap_annot_kws(n),
                linewidths=0.5, square=True,
                cbar_kws={'label': 'Spearman ρ'})
    ax.set_title('NEVO — Cross-Spec Spearman Rank Correlation of Own-Price Elasticities',
                 fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=max(5, 9 - n // 5))
    ax.tick_params(axis='y', rotation=0,  labelsize=max(5, 9 - n // 5))
    _footer(fig, 'NEVO · Analysis 24 · ρ=1 means perfect rank agreement in own-price elasticities across specs')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '24_spearman_corr_heatmap.png')


def plot_nevo_firm_substitution_heatmap(elas_rows: list[dict], all_rows: list[dict],
                                         out_dir: Path):
    """Analysis 25 — firm×firm cross-price elasticity heatmap."""
    def _firm(pid: str) -> str:
        b = pid.index('B')
        return pid[:b]

    best_list = _best_per_spec(all_rows)
    seed_map  = {r['spec']: r['seed'] for r in best_list}
    if not best_list:
        print('  [NEVO] No valid specs — skipping 25_firm_substitution_heatmap.png')
        return

    firm_pair: dict[tuple, list[float]] = {}
    best_spec = best_list[0]['spec']
    for candidate in best_list:
        best_spec = candidate['spec']
        best_seed = seed_map[best_spec]
        for r in elas_rows:
            if r['spec'] != best_spec or r['seed'] != best_seed or r['own_price'] == 'True':
                continue
            key = (_firm(r['product_j']), _firm(r['product_k']))
            firm_pair.setdefault(key, []).append(float(r['elasticity']))
        if firm_pair:
            break

    if not firm_pair:
        print('  [NEVO] No firm-pair elasticity rows match any best spec — skipping 25_firm_substitution_heatmap.png')
        return

    firms = sorted({f for (fj, fk) in firm_pair for f in (fj, fk)})
    n     = len(firms)
    matrix = [[sum(firm_pair.get((fi, fj), [float('nan')])) /
               max(len(firm_pair.get((fi, fj), [1])), 1)
               if (fi, fj) in firm_pair else float('nan')
               for fj in firms] for fi in firms]

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2 + 1.5), max(4, n * 1.0 + 1.5)))
    flat = [v for row in matrix for v in row if not math.isnan(v)]
    vmax = max(abs(v) for v in flat) if flat else 1.0
    sns.heatmap(matrix, ax=ax, xticklabels=firms, yticklabels=firms,
                cmap='YlOrRd', vmin=0, vmax=vmax,
                annot=True, fmt='.4f', annot_kws=_heatmap_annot_kws(n),
                linewidths=0.5, square=True,
                cbar_kws={'label': 'Mean Cross-Price Elas'})
    short_spec = _short(best_spec, 60)
    ax.set_title(f'NEVO — Firm × Firm Cross-Price Elasticity\n(best spec: {short_spec})',
                 fontweight='bold', fontsize=10)
    ax.set_xlabel('Firm k')
    ax.set_ylabel('Firm j')
    ax.tick_params(axis='x', rotation=0, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    _footer(fig, 'NEVO · Analysis 25 · Mean cross-price elasticity between firm j products and firm k products')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '25_firm_substitution_heatmap.png')


def plot_objective_spec_comparison(rows: list[dict], label: str, out_dir: Path):
    """Analysis 36 — mean ± std of GMM objective across all starts, ranked by mean."""
    by_spec: dict[str, list[float]] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], []).append(float(r['objective']))

    ranked = sorted(by_spec.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    if not ranked:
        print(f'  [{label}] No data — skipping 36_objective_spec_comparison.png')
        return

    specs  = [_short(kv[0]) for kv in ranked]
    means  = [sum(kv[1]) / len(kv[1]) for kv in ranked]
    stds   = [math.sqrt(sum((v - m) ** 2 for v in kv[1]) / len(kv[1]))
              for kv, m in zip(ranked, means)]
    mins   = [min(kv[1]) for kv in ranked]
    maxs   = [max(kv[1]) for kv in ranked]
    counts = [len(kv[1]) for kv in ranked]

    n = len(ranked)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * n + 1.5)))
    ys = list(range(n))

    # Bars for mean
    ax.barh(ys, means, color=COL_VALID, alpha=0.6, edgecolor='white', linewidth=0.5)
    # Error bars (±1 std)
    ax.errorbar(means, ys, xerr=stds, fmt='none', color='#444', linewidth=1.2,
                capsize=3, capthick=1.2, label='±1 std')
    # Min/max range markers
    for y, mn, mx in zip(ys, mins, maxs):
        ax.plot([mn, mx], [y, y], color='#888', linewidth=0.7, linestyle='--')
        ax.plot(mn, y, marker='|', color='#555', markersize=6)
        ax.plot(mx, y, marker='|', color='#555', markersize=6)
    # Annotate counts
    x_max = max(mx for mx in maxs) if maxs else 1
    for y, m, c in zip(ys, means, counts):
        ax.text(x_max * 1.01, y, f'n={c}', va='center', ha='left', fontsize=7, color='#555')

    ax.set_yticks(ys)
    ax.set_yticklabels(specs, fontsize=max(6, 9 - n // 8))
    ax.set_xlabel('GMM Objective')
    ax.set_title(f'{label} — Objective by Specification (mean ± std, range)\nsorted by mean',
                 fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=8)
    _footer(fig, f'{label} · Analysis 36 · Mean±std of GMM objective across valid random starts per spec')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '36_objective_spec_comparison.png')


def plot_elasticity_pair_across_sims(elas_rows: list[dict], all_rows: list[dict],
                                      label: str, out_dir: Path):
    """Analysis 30 — 2×2 panel: 4 elasticities for one product pair across all simulations."""
    # Spec with lowest mean objective
    by_spec_obj: dict[str, list[float]] = {}
    for r in all_rows:
        by_spec_obj.setdefault(r['spec'], []).append(float(r['objective']))
    if not by_spec_obj:
        print(f'  [{label}] No data — skipping 30_elasticity_pair_across_sims.png')
        return
    ranked_specs = sorted(by_spec_obj, key=lambda s: sum(by_spec_obj[s]) / len(by_spec_obj[s]))
    elas_specs   = {r['spec'] for r in elas_rows}
    best_spec    = next((s for s in ranked_specs if s in elas_specs), None)
    if best_spec is None:
        print(f'  [{label}] No spec with elasticity data — skipping 30_elasticity_pair_across_sims.png')
        return

    # Product pair with highest mean cross-price elasticity
    cross: dict[tuple, list[float]] = {}
    for r in elas_rows:
        if r['spec'] == best_spec and r['own_price'] == 'False':
            cross.setdefault((r['product_j'], r['product_k']), []).append(float(r['elasticity']))
    if not cross:
        print(f'  [{label}] No cross-price rows for best spec — skipping 30_elasticity_pair_across_sims.png')
        return
    prod_j, prod_k = max(cross, key=lambda p: sum(cross[p]) / len(cross[p]))

    # Collect 4 elasticities per seed
    seeds_data: dict[str, dict[str, float]] = {}
    for r in elas_rows:
        if r['spec'] != best_spec:
            continue
        seed = r['seed']
        pj, pk, val = r['product_j'], r['product_k'], float(r['elasticity'])
        seeds_data.setdefault(seed, {})
        if pj == prod_j and pk == prod_j:
            seeds_data[seed]['e_jj'] = val
        elif pj == prod_k and pk == prod_k:
            seeds_data[seed]['e_kk'] = val
        elif pj == prod_j and pk == prod_k:
            seeds_data[seed]['e_jk'] = val
        elif pj == prod_k and pk == prod_j:
            seeds_data[seed]['e_kj'] = val

    sorted_seeds, _ = _numeric_seed_sort(list(seeds_data))
    if not sorted_seeds:
        print(f'  [{label}] No seed data — skipping 30_elasticity_pair_across_sims.png')
        return

    panel_info = [
        ('e_jj', f'Own-price: {prod_j}',    COL_BASIN_A),
        ('e_kk', f'Own-price: {prod_k}',    COL_BASIN_B),
        ('e_jk', f'Cross-price: {prod_j}→{prod_k}', PALETTE[4]),
        ('e_kj', f'Cross-price: {prod_k}→{prod_j}', PALETTE[5]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    axes = axes.flatten()

    for ax, (key, title, color) in zip(axes, panel_info):
        vals = [seeds_data[s].get(key, float('nan')) for s in sorted_seeds]
        xs   = list(range(len(sorted_seeds)))
        _clip_with_outlier_markers(ax, xs, vals, color, marker_size=60)

        _set_sparse_seed_xaxis(ax, sorted_seeds)
        ax.set_ylabel('Elasticity', fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    short_spec = _short(best_spec, 70)
    fig.suptitle(f'{label} — Elasticity pair across simulations\n'
                 f'Spec: {short_spec}\nPair  j={prod_j}, k={prod_k}',
                 fontweight='bold', fontsize=9, y=1.01)
    _footer(fig, f'{label} · Analysis 30 · 4 elasticities for selected pair across all seeds of best-mean-obj spec'
                 f' · outliers (outside Q1−3·IQR, Q3+3·IQR) shown as labeled edge triangles')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '30_elasticity_pair_across_sims.png')


def plot_elasticity_pair_best_sim_across_specs(elas_rows: list[dict], all_rows: list[dict],
                                               label: str, out_dir: Path):
    """Analysis 31 — 2×2 panel: 4 elasticities for one pair, one best-simulation per spec."""
    # Step 1: best seed per spec (sorted by objective)
    best_list = _best_per_spec(all_rows)
    if not best_list:
        print(f'  [{label}] No data — skipping 31_elasticity_pair_best_sim_across_specs.png')
        return

    # Step 2: collect cross-price pairs available for each spec's best seed
    spec_cross: dict[str, dict[tuple, float]] = {}
    for entry in best_list:
        spec, seed = entry['spec'], entry['seed']
        pairs = {}
        for r in elas_rows:
            if r['spec'] == spec and r['seed'] == seed and r['own_price'] == 'False':
                pairs[(r['product_j'], r['product_k'])] = float(r['elasticity'])
        if pairs:
            spec_cross[spec] = pairs

    if not spec_cross:
        print(f'  [{label}] No elasticity data for any spec\'s best seed — skipping 31_elasticity_pair_best_sim_across_specs.png')
        return

    # Step 3: select pair by coverage then mean cross-elasticity
    pair_specs: dict[tuple, list[float]] = {}
    for pairs in spec_cross.values():
        for pair, val in pairs.items():
            pair_specs.setdefault(pair, []).append(val)
    max_cov    = max(len(v) for v in pair_specs.values())
    candidates = {p: v for p, v in pair_specs.items() if len(v) == max_cov}
    prod_j, prod_k = max(candidates, key=lambda p: sum(candidates[p]) / len(candidates[p]))

    # Step 4: collect 4 values per spec from its best seed, in objective-rank order
    panel_keys = ['e_jj', 'e_kk', 'e_jk', 'e_kj']
    spec_vals: list[tuple] = []   # (short_spec, {key: val})
    for entry in best_list:
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
        spec_vals.append((spec, _short(spec, 30), vals))

    if not spec_vals:
        print(f'  [{label}] No data after filtering — skipping 31_elasticity_pair_best_sim_across_specs.png')
        return

    full_specs  = [sv[0] for sv in spec_vals]
    short_specs = [sv[1] for sv in spec_vals]
    xs          = list(range(len(spec_vals)))

    panel_info = [
        ('e_jj', f'Own-price: {prod_j}',         COL_BASIN_A),
        ('e_kk', f'Own-price: {prod_k}',         COL_BASIN_B),
        ('e_jk', f'Cross-price: {prod_j}→{prod_k}', PALETTE[4]),
        ('e_kj', f'Cross-price: {prod_k}→{prod_j}', PALETTE[5]),
    ]

    # For >4 specs, swap multi-line spec labels for rank numbers and add a
    # numbered key below the figure so the rank → spec mapping stays visible.
    use_ranks = len(spec_vals) > 4
    if use_ranks:
        key_text, key_rows = _spec_key_text([_vshort_spec(s) for s in full_specs])
        # Reserve vertical space proportional to the key (≈ 0.018 figure-frac per row)
        bottom_rect = min(0.45, 0.06 + 0.022 * key_rows)
        fig_height  = 7 + 0.22 * key_rows
    else:
        key_text, key_rows = '', 0
        bottom_rect = 0.05
        fig_height  = 7

    fig, axes = plt.subplots(2, 2, figsize=(11, fig_height), sharey=False)
    axes = axes.flatten()

    for ax, (key, title, color) in zip(axes, panel_info):
        vals = [sv[2].get(key, float('nan')) for sv in spec_vals]
        _clip_with_outlier_markers(ax, xs, vals, color, marker_size=70)

        ax.set_xticks(xs)
        if use_ranks:
            ax.set_xticklabels([str(i + 1) for i in xs], rotation=0,
                               ha='center', fontsize=9)
            ax.set_xlabel('Spec rank (best objective ascending — see key below)',
                          fontsize=8)
        else:
            ax.set_xticklabels(short_specs, rotation=40, ha='right', fontsize=7)
            ax.set_xlabel('Specification (ranked by best objective)', fontsize=8)
        ax.set_ylabel('Elasticity', fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    fig.suptitle(f'{label} — Elasticity pair: best simulation per spec\n'
                 f'Pair  j={prod_j}, k={prod_k}  ·  one point = best seed of each spec',
                 fontweight='bold', fontsize=9, y=1.01)
    _footer(fig, f'{label} · Analysis 31 · 4 elasticities for selected pair, best-seed per spec, sorted by objective'
                 f' · outliers (outside Q1−3·IQR, Q3+3·IQR) shown as labeled edge triangles')
    if key_text:
        # Position the key just above the footer
        fig.text(0.5, 0.035, key_text, ha='center', va='bottom',
                 fontsize=7, family='monospace', color='black')
    plt.tight_layout(rect=[0, bottom_rect, 1, 1])
    _savefig(fig, out_dir, '31_elasticity_pair_best_sim_across_specs.png')


def plot_price_coef_across_sims(rows: list[dict], label: str, out_dir: Path):
    """Analysis 19 — price coefficients across all simulations of the best-mean-obj spec."""
    # Spec with lowest mean objective
    by_spec: dict[str, list[dict]] = {}
    for r in rows:
        by_spec.setdefault(r['spec'], []).append(r)
    if not by_spec:
        print(f'  [{label}] No data — skipping 19_price_coef_across_sims.png')
        return

    best_spec = min(by_spec,
                    key=lambda s: sum(float(r['objective']) for r in by_spec[s]) / len(by_spec[s]))
    spec_rows = by_spec[best_spec]

    raw_seeds    = [r['seed'] for r in spec_rows]
    raw_pcoefs   = [float(r['price_coef']) for r in spec_rows]
    raw_colors   = [COL_VALID for _ in spec_rows]

    sorted_seeds, perm = _numeric_seed_sort(raw_seeds)
    pcoefs       = [raw_pcoefs[i]  for i in perm]
    point_colors = [raw_colors[i]  for i in perm]
    xs           = list(range(len(sorted_seeds)))

    short_spec = _short(best_spec, 70)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(7, 5))
    mean_line = _clip_with_outlier_markers(
        ax, xs, pcoefs, color=COL_VALID, marker_size=70,
        colors=point_colors, mean_color='#444', add_legend=False)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

    _set_sparse_seed_xaxis(ax, sorted_seeds)
    ax.set_ylabel('Price coefficient (α)', fontsize=9)
    ax.set_title('Price coefficient per simulation', fontweight='bold', fontsize=9)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    legend_handles = [Patch(color=COL_VALID, label='valid start (α < 0)')]
    if mean_line is not None:
        legend_handles.insert(0, mean_line)
    ax.legend(handles=legend_handles, fontsize=8, loc='best')

    fig.suptitle(f'{label} — Price coefficient across simulations\nSpec: {short_spec}',
                 fontweight='bold', fontsize=9, y=1.01)
    _footer(fig, f'{label} · Analysis 19 · Price coef per simulation for best-mean-obj spec'
                 f' · outliers (outside Q1−3·IQR, Q3+3·IQR) shown as labeled edge triangles')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    _savefig(fig, out_dir, '19_price_coef_across_sims.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    completed = []

    # --- BLP figures ---
    try:
        blp_csv       = _find('blp/blp_multistart_all.csv')
        blp_elas_csv  = _find('blp/blp_elasticities_detail.csv')
        blp_rows      = _load(blp_csv)
        blp_elas_rows = _load(blp_elas_csv)
        blp_graphs_dir = _graphs_dir(blp_csv)
        print(f'Loaded BLP  multistart:   {len(blp_rows)} rows from {blp_csv}')
        print(f'Loaded BLP  elasticities: {len(blp_elas_rows)} rows from {blp_elas_csv}')
        _n0, _e0 = len(blp_rows), len(blp_elas_rows)
        blp_rows, blp_elas_rows = _cap_seeds(blp_rows, blp_elas_rows)
        blp_rows, blp_elas_rows = _filter_valid(blp_rows, blp_elas_rows)
        print(f'  valid starts: {len(blp_rows)}/{_n0}; '
              f'elasticity rows kept: {len(blp_elas_rows)}/{_e0}\n')

        print('=== BLP figures ===')
        plot_objective_ranking(blp_rows,      'BLP',  blp_graphs_dir)
        plot_price_coef(blp_rows,             'BLP',  blp_graphs_dir)
        plot_multistart_stability(blp_rows,   'BLP',  blp_graphs_dir)
        plot_convergence_scatter(blp_rows,    'BLP',  blp_graphs_dir)
        plot_global_minimum(blp_rows,         'BLP',  blp_graphs_dir)
        plot_basin_scatter(blp_rows,          'BLP',  blp_graphs_dir)
        plot_own_elas_boxplot(blp_elas_rows,  blp_rows, 'BLP', blp_graphs_dir)
        plot_multistart_elas_dotplot(blp_elas_rows,      'BLP', blp_graphs_dir)
        plot_cross_elas_heatmap(blp_elas_rows, blp_rows, 'BLP', blp_graphs_dir, max_products=24)
        plot_asymmetry_scatter(blp_elas_rows, blp_rows, 'BLP', blp_graphs_dir)
        plot_own_stability_heatmap(blp_elas_rows,  blp_rows, 'BLP', blp_graphs_dir)
        plot_cross_stability_heatmap(blp_elas_rows, blp_rows, 'BLP', blp_graphs_dir)
        plot_spec_pairwise_mad(blp_elas_rows, blp_rows, 'BLP', blp_graphs_dir)
        plot_objective_spec_comparison(blp_rows,         'BLP',  blp_graphs_dir)
        plot_elasticity_pair_across_sims(blp_elas_rows, blp_rows, 'BLP',  blp_graphs_dir)
        plot_elasticity_pair_best_sim_across_specs(blp_elas_rows, blp_rows, 'BLP',  blp_graphs_dir)
        plot_price_coef_across_sims(blp_rows,                              'BLP',  blp_graphs_dir)
        completed.append(blp_graphs_dir)
    except FileNotFoundError as e:
        print(f'[WARN] Skipping BLP figures: {e}\n')

    # --- Nevo figures ---
    try:
        nevo_csv       = _find('nevo/multistart_all.csv')
        nevo_elas_csv  = _find('nevo/elasticities_detail.csv')
        nevo_rows      = _load(nevo_csv)
        nevo_elas_rows = _load(nevo_elas_csv)
        nevo_graphs_dir = _graphs_dir(nevo_csv)
        print(f'Loaded Nevo multistart:   {len(nevo_rows)} rows from {nevo_csv}')
        print(f'Loaded Nevo elasticities: {len(nevo_elas_rows)} rows from {nevo_elas_csv}')
        _n0, _e0 = len(nevo_rows), len(nevo_elas_rows)
        nevo_rows, nevo_elas_rows = _cap_seeds(nevo_rows, nevo_elas_rows)
        nevo_rows, nevo_elas_rows = _filter_valid(nevo_rows, nevo_elas_rows)
        print(f'  valid starts: {len(nevo_rows)}/{_n0}; '
              f'elasticity rows kept: {len(nevo_elas_rows)}/{_e0}\n')

        print('=== Nevo figures ===')
        plot_objective_ranking(nevo_rows,      'NEVO', nevo_graphs_dir)
        plot_nevo_demographic_expansion(nevo_rows,     nevo_graphs_dir)
        plot_nevo_x2_comparison(nevo_rows,             nevo_graphs_dir)
        plot_price_coef(nevo_rows,             'NEVO', nevo_graphs_dir)
        plot_multistart_stability(nevo_rows,   'NEVO', nevo_graphs_dir)
        plot_convergence_scatter(nevo_rows,    'NEVO', nevo_graphs_dir)
        plot_global_minimum(nevo_rows,         'NEVO', nevo_graphs_dir)
        plot_basin_scatter(nevo_rows,          'NEVO', nevo_graphs_dir)
        plot_own_elas_boxplot(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_multistart_elas_dotplot(nevo_elas_rows,     'NEVO', nevo_graphs_dir)
        plot_cross_elas_heatmap(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_asymmetry_scatter(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_nevo_spearman_heatmap(nevo_elas_rows, nevo_rows, nevo_graphs_dir)
        plot_nevo_firm_substitution_heatmap(nevo_elas_rows, nevo_rows, nevo_graphs_dir)
        plot_own_stability_heatmap(nevo_elas_rows,  nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_cross_stability_heatmap(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_spec_pairwise_mad(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_objective_spec_comparison(nevo_rows,          'NEVO', nevo_graphs_dir)
        plot_elasticity_pair_across_sims(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_elasticity_pair_best_sim_across_specs(nevo_elas_rows, nevo_rows, 'NEVO', nevo_graphs_dir)
        plot_price_coef_across_sims(nevo_rows,                             'NEVO', nevo_graphs_dir)
        completed.append(nevo_graphs_dir)
    except FileNotFoundError as e:
        print(f'[WARN] Skipping Nevo figures: {e}\n')

    if completed:
        print('\nFigures saved to:')
        for d in completed:
            print(f'  {d}')
    else:
        print('[ERROR] No data found for either dataset. Nothing was plotted.')


if __name__ == '__main__':
    main()
