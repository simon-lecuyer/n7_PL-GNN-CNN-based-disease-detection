#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = [
    "slow_spread_short_infection",
    "medium_spread_medium_infection",
    "fast_spread_long_infection",
    "very_slow_isolated",
    "fast_multi_source",
]
SEQS = [3, 5, 10]
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    out_dir = Path("results") / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in EXPERIMENTS:
        for seq in SEQS:
            cnn_path = Path("results") / "cnn_experiments" / exp / f"seq{seq}" / "results_per_simulation.json"
            gnn_path = Path("results") / "gnn_experiments" / exp / f"seq{seq}" / "summary.json"

            if not cnn_path.exists() or not gnn_path.exists():
                continue

            cnn = load_json(cnn_path)
            gnn = load_json(gnn_path)

            cnn_avg = cnn["average"]
            gnn_mse = float(gnn["mean_mse"])
            gnn_mae = float(gnn["mean_mae"])
            gnn_rmse = float(gnn["mean_rmse"])

            rows.append({
                "experiment": exp,
                "sequence_length": seq,
                "cnn_mse": float(cnn_avg["mse"]),
                "cnn_mae": float(cnn_avg["mae"]),
                "cnn_rmse": float(cnn_avg["rmse"]),
                "gnn_mse": gnn_mse,
                "gnn_mae": gnn_mae,
                "gnn_rmse": gnn_rmse,
                "delta_mse_gnn_minus_cnn": gnn_mse - float(cnn_avg["mse"]),
                "delta_mae_gnn_minus_cnn": gnn_mae - float(cnn_avg["mae"]),
                "delta_rmse_gnn_minus_cnn": gnn_rmse - float(cnn_avg["rmse"]),
                "winner_mse": "GNN" if gnn_mse < float(cnn_avg["mse"]) else "CNN",
                "winner_mae": "GNN" if gnn_mae < float(cnn_avg["mae"]) else "CNN",
                "winner_rmse": "GNN" if gnn_rmse < float(cnn_avg["rmse"]) else "CNN",
                "cnn_n": int(len(cnn.get("predictions", []))),
                "gnn_n": int(gnn.get("saved_samples", 0)),
            })

    if not rows:
        raise RuntimeError("No comparable rows found.")

    df = pd.DataFrame(rows).sort_values(["experiment", "sequence_length"])

    df.to_csv(out_dir / "cnn_vs_gnn_comparison.csv", index=False)
    df.to_json(out_dir / "cnn_vs_gnn_comparison.json", orient="records", indent=2)

    summary = {
        "num_rows": int(len(df)),
        "wins_mse": df["winner_mse"].value_counts().to_dict(),
        "wins_mae": df["winner_mae"].value_counts().to_dict(),
        "wins_rmse": df["winner_rmse"].value_counts().to_dict(),
        "avg_delta_mse_gnn_minus_cnn": float(df["delta_mse_gnn_minus_cnn"].mean()),
        "avg_delta_mae_gnn_minus_cnn": float(df["delta_mae_gnn_minus_cnn"].mean()),
        "avg_delta_rmse_gnn_minus_cnn": float(df["delta_rmse_gnn_minus_cnn"].mean()),
        "note": "GNN metrics come from visualization summaries (saved_samples), not full test set.",
    }
    with open(out_dir / "cnn_vs_gnn_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for metric in ["mae", "rmse"]:
        fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(5 * len(EXPERIMENTS), 4), sharey=True)
        if len(EXPERIMENTS) == 1:
            axes = [axes]

        for i, exp in enumerate(EXPERIMENTS):
            ax = axes[i]
            sub = df[df["experiment"] == exp].sort_values("sequence_length")
            x = np.arange(len(sub))
            w = 0.35
            ax.bar(x - w / 2, sub[f"cnn_{metric}"], width=w, label="CNN", color="#1f77b4")
            ax.bar(x + w / 2, sub[f"gnn_{metric}"], width=w, label="GNN", color="#ff7f0e")
            ax.set_xticks(x)
            ax.set_xticklabels([f"seq{s}" for s in sub["sequence_length"].tolist()])
            ax.set_title(exp)
            ax.grid(axis="y", alpha=0.2)
            if i == 0:
                ax.set_ylabel(metric.upper())

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"CNN vs GNN ({metric.upper()})", y=1.03)
        fig.tight_layout()
        fig.savefig(out_dir / f"cnn_vs_gnn_{metric}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print("Generated:")
    print(f"- {out_dir / 'cnn_vs_gnn_comparison.csv'}")
    print(f"- {out_dir / 'cnn_vs_gnn_comparison.json'}")
    print(f"- {out_dir / 'cnn_vs_gnn_summary.json'}")
    print(f"- {out_dir / 'cnn_vs_gnn_mae.png'}")
    print(f"- {out_dir / 'cnn_vs_gnn_rmse.png'}")


if __name__ == "__main__":
    main()
