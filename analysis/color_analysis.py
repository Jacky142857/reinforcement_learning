"""Scan allocation plots for the cash (blue) color and summarize the coverage.

The script walks through the output folders of the A2C, PPO, and SAC runs,
looks for ``plots/3_allocations.png`` inside each reward directory, and checks
how many pixels match the target RGB color that represents cash.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# Default directories to scan relative to the repo root.
ALGORITHM_DIRS = ["td3_allocation", "ppo_allocation", "sac_allocation"]
# Default target RGB value (taken from the digital color meter reading).
DEFAULT_TARGET_RGB = (112, 157, 198)
# Reasonable default tolerance for Euclidean RGB distance.
DEFAULT_TOLERANCE = 35.0
# Parameters for isolating the plot region inside each figure.
DEFAULT_BACKGROUND_RGB = (255, 255, 255)
DEFAULT_BACKGROUND_THRESHOLD = 10.0
DEFAULT_BBOX_LOWER_QUANTILE = 0.01
DEFAULT_BBOX_UPPER_QUANTILE = 0.99
# Parameters for optionally tightening the bounding box until the plot is fully
# covered by the target color (useful when a single asset dominates cash).
DEFAULT_TIGHTEN_TRIGGER_RATIO = 0.9
DEFAULT_TIGHTEN_TARGET_RATIO = 1.0
DEFAULT_TIGHTEN_LINE_THRESHOLD = 0.999
DEFAULT_TIGHTEN_MAX_TRIM_FRACTION = 0.15


@dataclass
class AnalysisResult:
    algorithm: str
    reward: str
    file: Path
    total_pixels: int
    matching_pixels: int
    match_ratio: float

    def to_row(self) -> dict[str, str | float | int]:
        return {
            "algorithm": self.algorithm,
            "reward": self.reward,
            "file": str(self.file),
            "total_pixels": self.total_pixels,
            "matching_pixels": self.matching_pixels,
            "match_ratio": self.match_ratio,
            "has_cash_color": self.matching_pixels > 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root that contains *_allocation directories.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=ALGORITHM_DIRS,
        help="List of algorithm directories to inspect (relative to root).",
    )
    parser.add_argument(
        "--target-rgb",
        nargs=3,
        type=int,
        default=DEFAULT_TARGET_RGB,
        metavar=("R", "G", "B"),
        help="Target RGB triple that represents the cash color.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Euclidean RGB tolerance for counting a pixel as cash.",
    )
    parser.add_argument(
        "--bbox-background-threshold",
        type=float,
        default=DEFAULT_BACKGROUND_THRESHOLD,
        help=(
            "Absolute RGB delta above which a pixel counts as foreground when "
            "estimating the bounding box."
        ),
    )
    parser.add_argument(
        "--bbox-lower-quantile",
        type=float,
        default=DEFAULT_BBOX_LOWER_QUANTILE,
        help="Lower quantile (0-1) used to trim noise while cropping the plot area.",
    )
    parser.add_argument(
        "--bbox-upper-quantile",
        type=float,
        default=DEFAULT_BBOX_UPPER_QUANTILE,
        help="Upper quantile (0-1) used to trim noise while cropping the plot area.",
    )
    parser.add_argument(
        "--tighten-trigger-ratio",
        type=float,
        default=DEFAULT_TIGHTEN_TRIGGER_RATIO,
        help=(
            "Only attempt additional cropping if the initial target coverage "
            "inside the plot is at least this ratio."
        ),
    )
    parser.add_argument(
        "--tighten-target-ratio",
        type=float,
        default=DEFAULT_TIGHTEN_TARGET_RATIO,
        help="Try to keep trimming boundaries until the target coverage reaches this ratio.",
    )
    parser.add_argument(
        "--tighten-line-threshold",
        type=float,
        default=DEFAULT_TIGHTEN_LINE_THRESHOLD,
        help="Per-edge coverage required to keep a row/column during tightening.",
    )
    parser.add_argument(
        "--tighten-max-trim-fraction",
        type=float,
        default=DEFAULT_TIGHTEN_MAX_TRIM_FRACTION,
        help=(
            "Maximum fraction of height/width that tightening is allowed to "
            "remove from each side to avoid trimming away legitimate data."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cash_blue_presence.csv"),
        help="CSV file to write the summary into (relative to root by default).",
    )
    return parser.parse_args()


def iter_plot_paths(root: Path, algorithms: Iterable[str]) -> Iterable[tuple[str, Path]]:
    """Yield (algorithm_name, plot_path) pairs for every 3_allocations plot."""
    for alg in algorithms:
        alg_dir = root / alg / "output"
        if not alg_dir.exists():
            continue
        for reward_dir in sorted(p for p in alg_dir.iterdir() if p.is_dir()):
            plot_path = reward_dir / "plots" / "3_allocations.png"
            if plot_path.exists():
                yield alg, plot_path


def compute_plot_slices(
    arr: np.ndarray,
    background_rgb: np.ndarray,
    diff_threshold: float,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[slice, slice]:
    """Return slices that bound the main plot content, ignoring legends/labels."""
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("Bounding-box quantiles must satisfy 0 <= lower < upper <= 1.")

    diff = np.abs(arr - background_rgb)
    mask = (diff > diff_threshold).any(axis=2)
    if not np.any(mask):
        h, w = arr.shape[:2]
        return slice(0, h), slice(0, w)

    row_weights = mask.sum(axis=1).astype(np.float64)
    col_weights = mask.sum(axis=0).astype(np.float64)

    def _bounds(weights: np.ndarray) -> tuple[int, int]:
        total = float(weights.sum())
        if total == 0.0:
            return 0, weights.size
        cumsum = np.cumsum(weights)
        lower_target = total * lower_quantile
        upper_target = total * upper_quantile
        lower_idx = int(np.searchsorted(cumsum, lower_target, side="left"))
        upper_idx = int(np.searchsorted(cumsum, upper_target, side="left")) + 1
        upper_idx = max(lower_idx + 1, min(weights.size, upper_idx))
        return lower_idx, upper_idx

    row_start, row_end = _bounds(row_weights)
    col_start, col_end = _bounds(col_weights)
    return slice(row_start, row_end), slice(col_start, col_end)


def tighten_target_region(
    mask: np.ndarray,
    trigger_ratio: float,
    target_ratio: float,
    line_threshold: float,
    max_trim_fraction: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Optional refinement: trim edges until the target coverage is near 100%."""
    if mask.size == 0:
        return mask, (0, mask.shape[0]), (0, mask.shape[1])

    ratio = mask.mean()
    if ratio < trigger_ratio or ratio >= target_ratio:
        return mask, (0, mask.shape[0]), (0, mask.shape[1])

    height, width = mask.shape
    top, bottom = 0, height
    left, right = 0, width
    max_row_trim = max(1, int(height * max_trim_fraction))
    max_col_trim = max(1, int(width * max_trim_fraction))
    trimmed_top = trimmed_bottom = trimmed_left = trimmed_right = 0

    while ratio < target_ratio and (bottom - top) > 1 and (right - left) > 1:
        submask = mask[top:bottom, left:right]
        if submask.size == 0:
            break
        row_ratios = submask.mean(axis=1)
        col_ratios = submask.mean(axis=0)
        changed = False

        if row_ratios[0] < line_threshold and trimmed_top < max_row_trim:
            top += 1
            trimmed_top += 1
            changed = True
        if row_ratios[-1] < line_threshold and trimmed_bottom < max_row_trim:
            bottom -= 1
            trimmed_bottom += 1
            changed = True
        if col_ratios[0] < line_threshold and trimmed_left < max_col_trim:
            left += 1
            trimmed_left += 1
            changed = True
        if col_ratios[-1] < line_threshold and trimmed_right < max_col_trim:
            right -= 1
            trimmed_right += 1
            changed = True

        if not changed:
            break

        submask = mask[top:bottom, left:right]
        ratio = submask.mean()

    return mask[top:bottom, left:right], (top, bottom), (left, right)


def analyze_image(
    path: Path,
    target_rgb: np.ndarray,
    tolerance: float,
    background_rgb: np.ndarray,
    bbox_diff_threshold: float,
    bbox_lower_quantile: float,
    bbox_upper_quantile: float,
    tighten_trigger_ratio: float,
    tighten_target_ratio: float,
    tighten_line_threshold: float,
    tighten_max_trim_fraction: float,
) -> AnalysisResult:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    row_slice, col_slice = compute_plot_slices(
        arr,
        background_rgb=background_rgb,
        diff_threshold=bbox_diff_threshold,
        lower_quantile=bbox_lower_quantile,
        upper_quantile=bbox_upper_quantile,
    )
    cropped = arr[row_slice, col_slice]
    target_diff = np.linalg.norm(cropped - target_rgb, axis=2)
    target_mask = target_diff <= tolerance
    matches = int(target_mask.sum())
    total = int(target_mask.size)
    ratio = matches / total if total else 0.0

    tightened_mask, (row_start_offset, row_end_offset), (col_start_offset, col_end_offset) = (
        tighten_target_region(
            target_mask,
            tighten_trigger_ratio,
            tighten_target_ratio,
            tighten_line_threshold,
            tighten_max_trim_fraction,
        )
    )

    if tightened_mask.size != target_mask.size:
        row_slice = slice(row_slice.start + row_start_offset, row_slice.start + row_end_offset)
        col_slice = slice(col_slice.start + col_start_offset, col_slice.start + col_end_offset)
        cropped = arr[row_slice, col_slice]
        target_diff = np.linalg.norm(cropped - target_rgb, axis=2)
        target_mask = target_diff <= tolerance
        matches = int(target_mask.sum())
        total = int(target_mask.size)
        ratio = matches / total if total else 0.0

    return AnalysisResult(
        algorithm=path.parts[-4],  # e.g., sac_allocation
        reward=path.parts[-3],  # e.g., Calmar-like_Reward
        file=path,
        total_pixels=total,
        matching_pixels=matches,
        match_ratio=ratio,
    )


def main() -> None:
    args = parse_args()
    root = args.root
    target_rgb = np.asarray(args.target_rgb, dtype=np.float32)
    tolerance = args.tolerance
    background_rgb = np.asarray(DEFAULT_BACKGROUND_RGB, dtype=np.float32)
    bbox_diff_threshold = args.bbox_background_threshold
    bbox_lower_quantile = args.bbox_lower_quantile
    bbox_upper_quantile = args.bbox_upper_quantile
    tighten_trigger_ratio = args.tighten_trigger_ratio
    tighten_target_ratio = args.tighten_target_ratio
    tighten_line_threshold = args.tighten_line_threshold
    tighten_max_trim_fraction = args.tighten_max_trim_fraction

    results: list[AnalysisResult] = []
    for algorithm, plot_path in iter_plot_paths(root, args.algorithms):
        result = analyze_image(
            plot_path,
            target_rgb,
            tolerance,
            background_rgb,
            bbox_diff_threshold,
            bbox_lower_quantile,
            bbox_upper_quantile,
            tighten_trigger_ratio,
            tighten_target_ratio,
            tighten_line_threshold,
            tighten_max_trim_fraction,
        )
        # Overwrite algorithm field to avoid relying on path structure alone.
        result.algorithm = algorithm
        results.append(result)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "algorithm",
            "reward",
            "file",
            "total_pixels",
            "matching_pixels",
            "match_ratio",
            "has_cash_color",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(res.to_row())

    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
