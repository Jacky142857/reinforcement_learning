from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover - env specific
    raise SystemExit(
        "Both pandas and matplotlib are required. "
        "Install them with `pip install pandas matplotlib`."
    ) from exc

STOCK_SYMBOLS = [
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WBA",
    "WMT",
    "RTX",
]

TRAIN_SPLIT_DATE = pd.Timestamp("2022-12-29", tz="UTC")
TEST_WARMUP_DAYS = 20


def load_prices(data_dir: Path) -> pd.DataFrame:
    """Load close prices for the Dow 29 constituents (ex-DOW) and align dates."""
    price_frames: list[pd.DataFrame] = []
    for symbol in STOCK_SYMBOLS:
        csv_path = data_dir / f"{symbol}_data.csv"
        if not csv_path.exists():
            raise SystemExit(f"Missing expected CSV for {symbol}: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df.sort_values("Date", inplace=True)
        frame = df[["Date", "Close"]].rename(columns={"Close": symbol})
        frame.set_index("Date", inplace=True)
        price_frames.append(frame)

    prices = pd.concat(price_frames, axis=1).sort_index()
    prices = prices.ffill().dropna()
    if prices.empty:
        raise SystemExit("No overlapping dates across the Dow constituents.")
    return prices


def compute_buy_hold_index(prices: pd.DataFrame) -> pd.Series:
    """Construct an equal-weight buy-and-hold index normalized to 1.0."""
    normalized = prices / prices.iloc[0]
    return normalized.mean(axis=1)


def split_portfolio(portfolio: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split portfolio series into training and testing segments."""
    train = portfolio[portfolio.index < TRAIN_SPLIT_DATE]
    test = portfolio[portfolio.index >= TRAIN_SPLIT_DATE]
    return train, test


def prepare_test_segment(test: pd.Series, last_train_value: float) -> pd.Series:
    """Drop the warm-up period from the test set and normalize to connect with training."""
    if len(test) <= TEST_WARMUP_DAYS:
        raise SystemExit(
            f"Not enough test data ({len(test)} points) for a {TEST_WARMUP_DAYS}-day warm-up."
        )
    trimmed = test.iloc[TEST_WARMUP_DAYS:].copy()
    # Normalize so the first test value equals the last training value
    trimmed = trimmed / trimmed.iloc[0] * last_train_value
    return trimmed


def plot_buy_hold(train: pd.Series, test: pd.Series, output_path: Path | None) -> None:
    """Plot the training and testing phases of the buy-and-hold Dow basket."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)

    ax.plot(train.index, train, label="Training (2015-01-01 to 2022-12-28)", color="#1f77b4")
    ax.plot(
        test.index,
        test,
        label="Testing (2022-12-29 to 2024-12-31)",
        color="#ff7f0e",
    )

    if not train.empty:
        split_date = train.index[-1]
        ax.axvline(split_date, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(
            split_date,
            ax.get_ylim()[1],
            " Train/Test Split",
            va="bottom",
            ha="right",
            color="gray",
            fontsize=9,
        )

    ax.set_title("Equal-Weight Dow Buy-and-Hold: Training vs. Testing")
    ax.set_ylabel("Cumulative Value (Normalized to 1.0)")
    ax.set_xlabel("Date")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:  # pragma: no cover - interactive
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training vs testing phases of an equal-weight Dow buy-and-hold index."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("stock_data"),
        help="Directory containing *_data.csv files for Dow constituents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/dow_buy_hold.png"),
        help="Path to save the generated figure (default: figures/dow_buy_hold.png). "
        "Pass an empty string to display instead of saving.",
    )
    args = parser.parse_args()

    prices = load_prices(args.data_dir)
    portfolio = compute_buy_hold_index(prices)
    train, test = split_portfolio(portfolio)
    last_train_value = train.iloc[-1] if not train.empty else 1.0
    test = prepare_test_segment(test, last_train_value)

    output_path: Path | None
    if args.output and str(args.output).strip():
        output_path = args.output
    else:
        output_path = None
    plot_buy_hold(train, test, output_path)


if __name__ == "__main__":
    main()