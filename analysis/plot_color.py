"""Create per-algorithm plots that summarize cash-color coverage."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend set above)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cash_blue_presence.csv"),
        help="CSV file produced by color_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("color_analysis"),
        help="Directory to store the generated summary plots.",
    )
    return parser.parse_args()


def load_data(csv_path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["match_ratio"] = float(row["match_ratio"])
            grouped[row["algorithm"]].append(row)
    return grouped


def plot_algorithm(algorithm: str, rows: List[dict], output_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: r["match_ratio"], reverse=True)
    rewards = [row["reward"] for row in rows]
    ratios = [row["match_ratio"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(rewards) * 0.4), 4))
    bars = ax.bar(range(len(rewards)), ratios, color="#709dc6")
    ax.set_title(f"{algorithm} Cash Allocation Ratio", fontweight="bold")
    ax.set_ylabel("Mean Allocation Ratio")
    ax.set_xticks(range(len(rewards)), rewards, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, max(ratios) * 1.1 if ratios else 1)
    ax.grid(axis="y", alpha=0.3)

    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.2%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig_path = output_dir / f"{algorithm}_cash_color.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data(args.csv)
    for algorithm, rows in data.items():
        if rows:
            plot_algorithm(algorithm, rows, args.output_dir)
    print(f"Wrote {len(data)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
