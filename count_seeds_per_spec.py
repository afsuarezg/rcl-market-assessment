from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

DATASETS = {
    "nevo": ("multistart_all.csv", "multistart_best.csv"),
    "blp":  ("blp_multistart_all.csv", "blp_multistart_best.csv"),
}


def summarize(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["spec", "seed"])
    g = df.groupby("spec")
    return pd.DataFrame({
        f"n_unique_seeds_{label}": g["seed"].nunique(),
        f"n_rows_{label}": g.size(),
    })


def process(dataset: str, all_name: str, best_name: str) -> None:
    csv_dir = RESULTS_DIR / dataset / "csv"
    all_path  = csv_dir / all_name
    best_path = csv_dir / best_name
    if not all_path.exists() or not best_path.exists():
        print(f"[skip] {dataset}: missing {all_path.name} or {best_path.name}")
        return

    all_df  = summarize(all_path,  "all")
    best_df = summarize(best_path, "best")
    out = all_df.join(best_df, how="outer").fillna(0).astype(int)
    out = out.sort_index().reset_index()

    out_path = csv_dir / "seeds_per_spec.csv"
    out.to_csv(out_path, index=False)
    print(f"[{dataset}] Wrote {len(out)} specs to {out_path}")


def main() -> None:
    for dataset, (all_name, best_name) in DATASETS.items():
        process(dataset, all_name, best_name)


if __name__ == "__main__":
    main()
