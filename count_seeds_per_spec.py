from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

DATASETS = {
    "nevo": ("multistart_all.csv", "multistart_best.csv", "solve_diagnostics.csv"),
    "blp":  ("blp_multistart_all.csv", "blp_multistart_best.csv", "blp_solve_diagnostics.csv"),
}


def summarize(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["spec", "seed"])
    g = df.groupby("spec")
    return pd.DataFrame({
        f"n_unique_seeds_{label}": g["seed"].nunique(),
        f"n_rows_{label}": g.size(),
    })


def summarize_diagnostics(csv_path: Path) -> pd.DataFrame:
    """Per-spec failure counts from solve_diagnostics.csv.

    Counts distinct *seeds*, not rows: one seed can be recorded at several
    stages (e.g. a stage-1 non-converged start is logged again when the
    stage-2 pre-filter rejects it), so row counts would double-count.

    n_seeds_flagged — seeds with any failure/non-convergence record. This is
                      an upper bound on dropped seeds: a start whose stage-2
                      re-solve merely fails to converge is flagged here but
                      still kept in multistart_all.csv.
    n_seeds_error   — seeds where a solve raised an exception
    top_error_class — most frequent error class among the raises ('' if none)
    """
    df = pd.read_csv(csv_path, usecols=["spec", "seed", "outcome", "error_classes"])
    g = df.groupby("spec")
    out = pd.DataFrame({
        "n_seeds_flagged": g["seed"].nunique(),
        "n_seeds_error":   g.apply(
            lambda d: d.loc[d["outcome"] == "error", "seed"].nunique(),
            include_groups=False,
        ),
    })
    errs = df[df["outcome"] == "error"].dropna(subset=["error_classes"])
    if not errs.empty:
        out = out.join(errs.groupby("spec")["error_classes"]
                           .agg(lambda s: s.value_counts().index[0])
                           .rename("top_error_class"))
    else:
        out["top_error_class"] = ""
    out["top_error_class"] = out["top_error_class"].fillna("")
    return out


def process(dataset: str, all_name: str, best_name: str, diag_name: str) -> None:
    csv_dir = RESULTS_DIR / dataset / "csv"
    all_path  = csv_dir / all_name
    best_path = csv_dir / best_name
    diag_path = csv_dir / diag_name
    if not all_path.exists() or not best_path.exists():
        print(f"[skip] {dataset}: missing {all_path.name} or {best_path.name}")
        return

    # Outer joins so a spec whose starts all failed (present only in the
    # diagnostics file) still gets a row instead of disappearing.
    frames = [summarize(all_path, "all"), summarize(best_path, "best")]
    if diag_path.exists():
        frames.append(summarize_diagnostics(diag_path))
    out = frames[0].join(frames[1:], how="outer")

    num_cols = [c for c in out.columns if c != "top_error_class"]
    out[num_cols] = out[num_cols].fillna(0).astype(int)
    if "top_error_class" in out.columns:
        out["top_error_class"] = out["top_error_class"].fillna("")
    out = out.sort_index().reset_index()

    out_path = csv_dir / "seeds_per_spec.csv"
    out.to_csv(out_path, index=False)
    print(f"[{dataset}] Wrote {len(out)} specs to {out_path}")


def main() -> None:
    for dataset, names in DATASETS.items():
        process(dataset, *names)


if __name__ == "__main__":
    main()
