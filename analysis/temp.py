from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


DATA_PATH = Path("results.csv")
OUTPUT_PATH = Path("training_timesteps_mean.png")
ALGORITHMS = ["SAC", "TD3", "PPO"]
BAR_COLOR = "#9ecae1"
OFF_POLICY_BASELINE = 200_150
PPO_RANGE = (1_550_000, 3_650_000)


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_timesteps(csv_path: Path) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps = parse_float(row.get("training_timesteps"))
            if timesteps is None:
                continue
            grouped[row["algorithm"]].append(timesteps)
    return grouped


def compute_means(grouped: Dict[str, List[float]]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    for algo, values in grouped.items():
        if values:
            means[algo] = sum(values) / len(values)
    return means


def plot_mean_bars(means: Dict[str, float]) -> None:
    algorithms = [algo for algo in ALGORITHMS if algo in means]
    if not algorithms:
        raise SystemExit("No algorithm means available for plotting.")
    values = [means[algo] for algo in algorithms]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=180)
    bars = ax.bar(algorithms, values, color=BAR_COLOR, width=0.55)

    ax.set_ylabel("Mean training timesteps")
    ax.set_title("Average training steps per algorithm", fontsize=14, fontweight="bold")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))

    # ax.axhline(
    #     OFF_POLICY_BASELINE,
    #     color="#2ca02c",
    #     linestyle="--",
    #     linewidth=1.2,
    #     label="Off-policy target (200,150)",
    # )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


def main() -> None:
    grouped = load_timesteps(DATA_PATH)
    means = compute_means(grouped)
    plot_mean_bars(means)


if __name__ == "__main__":
    main()
