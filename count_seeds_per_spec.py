from pathlib import Path
import pandas as pd

NEVO_DIR = Path(__file__).parent / "results" / "nevo" / "csv"


def summarize(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["spec", "seed"])
    g = df.groupby("spec")
    return pd.DataFrame({
        f"n_unique_seeds_{label}": g["seed"].nunique(),
        f"n_rows_{label}": g.size(),
    })


def main() -> None:
    all_df = summarize(NEVO_DIR / "multistart_all.csv", "all")
    best_df = summarize(NEVO_DIR / "multistart_best.csv", "best")
    out = all_df.join(best_df, how="outer").fillna(0).astype(int)
    out = out.sort_index().reset_index()
    out_path = NEVO_DIR / "seeds_per_spec.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} specs to {out_path}")


if __name__ == "__main__":
    main()
