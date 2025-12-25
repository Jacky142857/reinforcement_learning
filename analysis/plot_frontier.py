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

ALGO_COLORS = {
    "PPO": "#E69F00",
    "SAC": "#56B4E9",
    "TD3": "#009E73",
}

# HIGHLIGHTS = [
#     ("SAC", "Information Ratio Reward", "SAC Info Ratio"),
#     ("TD3", "Turnover-Weighted Return", "TD3 Turnover"),
#     ("PPO", "Return minus Drawdown Penalty", "PPO Return-DD"),
# ]


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results and keep rows with numeric value/drawdown columns."""
    df = pd.read_csv(csv_path)
    for column in ("final_portfolio_value", "max_drawdown_pct"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["final_portfolio_value", "max_drawdown_pct"])
    if df.empty:
        raise SystemExit(
            f"No valid final_portfolio_value/max_drawdown_pct entries in {csv_path}."
        )
    return df


def plot_frontier(df: pd.DataFrame, output_path: Path | None) -> None:
    """Scatter final value vs. drawdown to approximate an drawdown frontier."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    for algorithm, group in df.groupby("algorithm"):
        color = ALGO_COLORS.get(algorithm, "#7f7f7f")
        ax.scatter(
            group["final_portfolio_value"],
            group["max_drawdown_pct"],
            label=algorithm,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
            s=60,
        )

    # for algo, reward, label in HIGHLIGHTS:
    #     subset = df[(df["algorithm"] == algo) & (df["reward"] == reward)]
    #     if subset.empty:
    #         continue
    #     row = subset.iloc[0]
    #     ax.annotate(
    #         label,
    #         (row["final_portfolio_value"], row["max_drawdown_pct"]),
    #         textcoords="offset points",
    #         xytext=(6, -6),
    #         fontsize=9,
    #         color="black",
    #     )

    ax.set_xlabel("Final Portfolio Value ($)")
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("Drawdown Frontier: Final Value vs. Drawdown")
    ax.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(title="Algorithm")
    ax.invert_yaxis()  # less severe drawdown (closer to 0) appears toward the top

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:  # pragma: no cover - interactive use
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final portfolio value against drawdown to visualize an drawdown frontier."
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
        default=Path(__file__).resolve().parent / "drawdown_frontier.png",
        help="Where to store the plot (default: ./drawdown_frontier.png). "
        "Pass an empty string to display instead of saving.",
    )
    args = parser.parse_args()

    df = load_results(args.csv_path)
    output_path: Path | None
    if args.output and str(args.output).strip():
        output_path = args.output
    else:
        output_path = None
    plot_frontier(df, output_path)


if __name__ == "__main__":
    main()

