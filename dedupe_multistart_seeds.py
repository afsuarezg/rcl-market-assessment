"""One-time cleanup: collapse duplicate (spec, seed) rows in multistart CSVs.

multistart_all.csv / blp_multistart_all.csv are appended across top-up runs
(see _append_csv in nevo_blp.py / blp_blp.py). Before the append step learned to
dedupe, the same seed could be written twice for a spec — an identical re-run, or
a divergent start that reused the seed — which inflated the per-spec row count
(e.g. figure 36 reporting N=41 for a spec that only has 40 unique seeds). One
seed is one simulation, so this script folds each (spec, seed) to a single row:
the best (lowest-objective) start, rebuilding the `best` flag so exactly one row
per spec is best=True. The spec-indexed _best file is folded to one row per spec.

Usage:
    python dedupe_multistart_seeds.py                 # clean results/{nevo,blp}/csv/
    python dedupe_multistart_seeds.py path/to/file.csv [...]   # clean given files

The elasticity / post-estimation CSVs are NOT touched: they legitimately hold
many rows per (spec, seed). A timestamp-free .bak copy is written once before any
file is overwritten. Re-running is a no-op once a file is clean.
"""
import sys
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

# (all-file, best-file) per dataset, relative to results/<dataset>/csv/
DATASETS = {
    "nevo": ("multistart_all.csv", "multistart_best.csv"),
    "blp":  ("blp_multistart_all.csv", "blp_multistart_best.csv"),
}


def _dedupe(df: pd.DataFrame, *, by, reflag_best: bool) -> pd.DataFrame:
    """Keep one row per key (the lowest GMM objective). Mirrors
    nevo_blp._dedupe_multistart. `by` is the list of key columns, or None to key
    on the index (the spec-indexed _best file)."""
    if "objective" not in df.columns:
        return df
    df = df.copy()
    if by is not None:
        df = df.reset_index(drop=True)
    df["__obj"] = pd.to_numeric(df["objective"], errors="coerce")
    df = df.sort_values(["spec", "__obj"] if "spec" in df.columns else ["__obj"])
    if by is None:
        df = df[~df.index.duplicated(keep="first")]
    else:
        df = df.drop_duplicates(subset=by, keep="first")
    if reflag_best and "best" in df.columns and "spec" in df.columns:
        df["best"] = False
        df.loc[df.groupby("spec")["__obj"].idxmin(), "best"] = True
    return df.drop(columns="__obj")


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())
        print(f"      backup -> {bak.name}")


def clean_file(path: Path) -> None:
    is_best = "best" in path.name.lower()
    index = is_best                       # _best is spec-indexed; _all is not
    df = pd.read_csv(path, index_col=0 if index else None)
    before = len(df)
    out = _dedupe(df, by=None if is_best else ["spec", "seed"], reflag_best=not is_best)
    after = len(out)
    if after == before:
        print(f"[ok]   {path}  ({before} rows, no duplicates)")
        return
    _backup(path)
    out.to_csv(path, index=index)
    print(f"[fix]  {path}  {before} -> {after} rows ({before - after} duplicate (spec,seed) removed)")


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = []
        for dataset, (all_name, best_name) in DATASETS.items():
            csv_dir = RESULTS_DIR / dataset / "csv"
            paths += [csv_dir / all_name, csv_dir / best_name]

    found = False
    for p in paths:
        if p.exists():
            found = True
            clean_file(p)
        else:
            print(f"[skip] not found: {p}")
    if not found:
        print("\nNo target CSVs found. Point the script at the live files, e.g.:\n"
              "  python dedupe_multistart_seeds.py results/nevo/csv/multistart_all.csv")


if __name__ == "__main__":
    main()
