from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment specific
    raise SystemExit(
        "Both pandas and matplotlib are required. "
        "Install them with `pip install pandas matplotlib`."
    ) from exc


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results.csv and coerce training timesteps to numeric."""
    df = pd.read_csv(csv_path)
    df["training_timesteps"] = pd.to_numeric(df["training_timesteps"], errors="coerce")
    df = df.dropna(subset=["training_timesteps"])
    if df.empty:
        raise SystemExit(
            f"No valid training_timesteps entries found in {csv_path}. "
            "Ensure the CSV has numeric values."
        )
    return df


def compute_normalized_steps_by_reward(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Normalize training steps per reward for each algorithm."""
    normalized: dict[str, pd.DataFrame] = {}
    for algorithm, group in df.groupby("algorithm"):
        algo_df = group.copy()
        max_steps = algo_df["training_timesteps"].max()
        if pd.isna(max_steps) or max_steps == 0:
            continue
        algo_df["normalized_training_steps"] = (
            algo_df["training_timesteps"] / max_steps
        )
        algo_df.sort_values(
            "normalized_training_steps", inplace=True, ascending=True
        )
        normalized[algorithm] = algo_df
    if not normalized:
        raise SystemExit(
            "No algorithm groups with valid training_timesteps were found."
        )
    return normalized


def plot_normalized_steps(
    algo_frames: dict[str, pd.DataFrame], output_path: Path | None
) -> None:
    """Create per-algorithm horizontal bar plots of normalized training steps."""
    algorithms = sorted(algo_frames.keys())
    fig, axes = plt.subplots(
        1, len(algorithms), figsize=(6 * len(algorithms), 6), dpi=120, sharex=True
    )
    if len(algorithms) == 1:
        axes = [axes]

    for ax, algorithm in zip(axes, algorithms):
        algo_df = algo_frames[algorithm]
        ax.barh(
            algo_df["reward"],
            algo_df["normalized_training_steps"],
            color="#1f77b4",
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalized Training Step")
        ax.set_title(f"{algorithm}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        for y, (_, row) in enumerate(algo_df.iterrows()):
            normalized_value = row["normalized_training_steps"]
            steps = row["training_timesteps"]
            label = f"{normalized_value:.2f} ({steps:,.0f})"
            ax.text(
                normalized_value + 0.01,
                y,
                label,
                va="center",
                ha="left",
                fontsize=8,
            )

    fig.suptitle("Normalized Training Steps per Reward", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:  # pragma: no cover - interactive use
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot normalized training steps per algorithm from results.csv."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(__file__).resolve().parent / "results.csv",
        help="Path to results.csv (default: ./results.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "normalized_training_steps.png",
        help="Where to store the plot (default: ./normalized_training_steps.png). "
        "Pass an empty string to display instead of saving.",
    )
    args = parser.parse_args()

    df = load_results(args.csv_path)
    algo_frames = compute_normalized_steps_by_reward(df)
    output_path: Path | None
    if args.output and str(args.output).strip():
        output_path = args.output
    else:
        output_path = None
    plot_normalized_steps(algo_frames, output_path)


if __name__ == "__main__":
    main()
