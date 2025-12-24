from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - user environment specific
    raise SystemExit(
        "pandas is required for this script. Install it with `pip install pandas`."
    ) from exc


@dataclass
class SummaryMetrics:
    reward: str
    final_portfolio_value: float | None
    max_drawdown_pct: float | None
    outperformance_pct: float | None
    training_timesteps: int | None


def _strip_percent(value: str) -> float:
    """Convert a percentage string such as '-27.34%' to a float."""
    cleaned = value.replace("%", "").replace(",", "").strip()
    return float(cleaned)


def _strip_currency(value: str) -> float:
    """Convert a currency string such as '$137,502.40' to a float."""
    cleaned = value.replace("$", "").replace(",", "").strip()
    return float(cleaned)


def _strip_integer(value: str) -> int:
    """Convert an integer string such as '200,150' to an int."""
    cleaned = value.replace(",", "").strip()
    return int(cleaned)


def parse_summary_file(summary_path: Path) -> SummaryMetrics:
    """Parse a summary file and return the key metrics."""
    reward_name: str | None = None
    final_value: float | None = None
    max_drawdown: float | None = None
    outperformance: float | None = None
    training_timesteps: int | None = None

    with summary_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("RESULTS SUMMARY:"):
                reward_name = line.split(":", 1)[1].strip()
            elif line.startswith("Final Portfolio Value:"):
                value_str = line.split(":", 1)[1].strip()
                final_value = _strip_currency(value_str)
            elif line.startswith("Max Drawdown"):
                value_str = line.split(":", 1)[1].strip()
                max_drawdown = _strip_percent(value_str)
            elif line.startswith("Outperformance:"):
                value_str = line.split(":", 1)[1].strip()
                outperformance = _strip_percent(value_str)
            elif line.startswith("Training Timesteps Used:"):
                value_str = line.split(":", 1)[1].strip()
                try:
                    training_timesteps = _strip_integer(value_str)
                except ValueError:
                    training_timesteps = None

    if reward_name is None:
        reward_name = summary_path.stem.replace("_summary", "")

    return SummaryMetrics(
        reward=reward_name,
        final_portfolio_value=final_value,
        max_drawdown_pct=max_drawdown,
        outperformance_pct=outperformance,
        training_timesteps=training_timesteps,
    )


def collect_algorithm_metrics(
    algorithm: str,
    output_dir: Path,
) -> List[Tuple[Tuple[str, str], Dict[str, float | None]]]:
    """Collect metrics for all reward summaries under the given algorithm directory."""
    records: List[Tuple[Tuple[str, str], Dict[str, float | None]]] = []

    if not output_dir.exists():
        return records

    for reward_dir in sorted(output_dir.iterdir()):
        if not reward_dir.is_dir():
            continue

        summaries = list(reward_dir.glob("*_summary.txt"))
        if not summaries:
            continue

        # Assume first summary file represents the reward.
        summary_metrics = parse_summary_file(summaries[0])
        index_key = (algorithm, summary_metrics.reward)
        value_dict = {
            "final_portfolio_value": summary_metrics.final_portfolio_value,
            "max_drawdown_pct": summary_metrics.max_drawdown_pct,
            "outperformance_pct": summary_metrics.outperformance_pct,
            "training_timesteps": summary_metrics.training_timesteps,
        }
        records.append((index_key, value_dict))

    return records


def build_results_dataframe(base_dir: Path) -> pd.DataFrame:
    """Aggregate PPO, SAC, and TD3 summaries into a multi-index DataFrame."""
    algo_dirs = {
        "PPO": base_dir / "ppo_allocation" / "output",
        "SAC": base_dir / "sac_allocation" / "output",
        "TD3": base_dir / "td3_allocation" / "output",
    }

    index_values: List[Tuple[str, str]] = []
    rows: List[Dict[str, float | None]] = []

    for algorithm, dir_path in algo_dirs.items():
        for index_key, metrics in collect_algorithm_metrics(algorithm, dir_path):
            index_values.append(index_key)
            rows.append(metrics)

    if not rows:
        return pd.DataFrame(
            columns=[
                "final_portfolio_value",
                "max_drawdown_pct",
                "outperformance_pct",
                "training_timesteps",
            ]
        )

    multi_index = pd.MultiIndex.from_tuples(
        index_values, names=["algorithm", "reward"]
    )
    df = pd.DataFrame(rows, index=multi_index)
    df.sort_index(inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate PPO, SAC, and TD3 reward summaries into a MultiIndex DataFrame."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository base directory containing the allocation outputs.",
    )
    parser.add_argument(
        "--to-csv",
        type=Path,
        default=None,
        help="Optional path to store the aggregated results as CSV.",
    )

    args = parser.parse_args()

    df = build_results_dataframe(args.base_dir)
    if df.empty:
        print("No summary files found for PPO, SAC, or TD3.")
        return

    print(df)
    if args.to_csv is not None:
        args.to_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.to_csv)
        print(f"\nSaved aggregated results to {args.to_csv}")


if __name__ == "__main__":
    main()
