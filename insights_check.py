import os
import math
import json
import numpy as np
import pandas as pd


def parse_objective(value: float) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_generation_index(row_index_zero_based: int, solutions_per_generation: int) -> int:
    return (row_index_zero_based // solutions_per_generation) + 1


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names used downstream
    rename_map = {
        "keras": "kernel_initializer",
        "train %": "train_split",
        "VAL_RMSE": "VAL_RMSE",
        "Objective": "Objective",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Coerce numeric columns
    numeric_cols = [
        "Layers",
        "batch",
        "epochs",
        "dropout",
        "train_split",
        "RMSE",
        "VAL_RMSE",
        "Objective",
        "mae",
        "val_mae",
        "R2",
        "R2_v",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    # Add generation index if not present (assume constant pop size by reading first gen size from script default 40 or inferable by user input)
    solutions_per_generation = None
    # Heuristic: use 40 if divisible, else try 20, else leave None
    if len(df) % 40 == 0:
        solutions_per_generation = 40
    elif len(df) % 20 == 0:
        solutions_per_generation = 20

    if solutions_per_generation is not None:
        df["generation"] = [
            infer_generation_index(i, solutions_per_generation) for i in range(len(df))
        ]

    return df


def summarize_best_solution(df: pd.DataFrame) -> dict:
    best_idx = df["Objective"].idxmax()
    row = df.loc[best_idx]
    return {
        "Objective": float(row["Objective"]),
        "Architecture": f"{int(row['Layers'])} layers, {row['Neurons']}",
        "batch": int(row["batch"]),
        "optimizer": str(row.get("optimiser", "")),
        "activation": str(row.get("activation", "")),
        "dropout": float(row["dropout"]),
        "epochs": int(row["epochs"]),
        "train_split": float(row["train_split"]),
        "RMSE": float(row["RMSE"]),
        "VAL_RMSE": float(row["VAL_RMSE"]),
        "R2": float(row["R2"]),
        "R2_v": float(row["R2_v"]),
        "MAE": float(row["mae"]),
        "VAL_MAE": float(row["val_mae"]),
    }


def per_generation_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "generation" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("generation")["Objective"].agg(
        best=np.max, avg=np.mean, worst=np.min, std=np.std
    )
    return agg.reset_index()


def hyperparameter_impacts(df: pd.DataFrame) -> dict:
    impacts = {}
    categorical = ["optimiser", "activation", "kernel_initializer", "batch", "epochs", "Layers", "dropout"]
    for col in categorical:
        if col in df.columns:
            grp = df.groupby(col)["Objective"].mean().sort_values(ascending=False)
            impacts[col] = grp.to_dict()
    return impacts


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        c
        for c in [
            "Layers",
            "batch",
            "epochs",
            "dropout",
            "train_split",
            "RMSE",
            "VAL_RMSE",
            "Objective",
            "mae",
            "val_mae",
            "R2",
            "R2_v",
        ]
        if c in df.columns
    ]
    return df[candidate_cols].corr()


def write_report(df: pd.DataFrame, out_path: str) -> None:
    lines = []
    lines.append("INSIGHTS CHECK REPORT")
    lines.append("======================\n")

    lines.append(f"Total rows: {len(df)}")
    if "generation" in df.columns:
        lines.append(f"Generations (inferred): {df['generation'].nunique()}")
        lines.append("")

    # Best solution
    best = summarize_best_solution(df)
    lines.append("Best Solution")
    lines.append("-------------")
    for k, v in best.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Per-generation stats
    gen_stats = per_generation_stats(df)
    if not gen_stats.empty:
        lines.append("Per-Generation Objective Stats")
        lines.append("-------------------------------")
        for _, r in gen_stats.iterrows():
            lines.append(
                f"Gen {int(r['generation'])}: best={r['best']:.6f}, avg={r['avg']:.6f}, worst={r['worst']:.6f}, std={r['std']:.6f}"
            )
        lines.append("")

    # Hyperparameter impacts
    impacts = hyperparameter_impacts(df)
    lines.append("Hyperparameter Impacts (mean Objective)")
    lines.append("---------------------------------------")
    for col, mapping in impacts.items():
        lines.append(f"{col}:")
        # show top 5 entries
        top_items = list(mapping.items())[:5]
        for key, val in top_items:
            lines.append(f"  - {key}: {val:.6f}")
    lines.append("")

    # Correlations
    corr = correlation_matrix(df)
    if not corr.empty:
        lines.append("Correlation with Objective")
        lines.append("--------------------------")
        if "Objective" in corr.columns:
            objective_corr = (
                corr["Objective"].sort_values(ascending=False).to_dict()
            )
            for k, v in objective_corr.items():
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    # Simple consistency checks against known expectations from the generated plots
    expected_best_threshold = 0.97
    measured_best = best["Objective"]
    lines.append("Consistency Checks")
    lines.append("-------------------")
    lines.append(
        f"- Best objective >= {expected_best_threshold}? {'YES' if measured_best >= expected_best_threshold else 'NO'} ({measured_best:.6f})"
    )
    if "generation" in df.columns:
        lines.append(
            f"- Last gen avg > first gen avg? {'YES' if gen_stats['avg'].iloc[-1] > gen_stats['avg'].iloc[0] else 'NO'}"
        )
        lines.append(
            f"- Last gen best >= first gen best? {'YES' if gen_stats['best'].iloc[-1] >= gen_stats['best'].iloc[0] else 'NO'}"
        )
    lines.append("")

    # Write
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(repo_root, "Generated_plots", "results.csv")
    out_path = os.path.join(repo_root, "Generated_plots", "insights_check.txt")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"results.csv not found at {csv_path}")

    df = load_results(csv_path)
    write_report(df, out_path)
    print(f"Insights written to {out_path}")


if __name__ == "__main__":
    main()


