#!/usr/bin/env python3
"""Compute CUB-200-2011 part visibility statistics for train/test splits.

This script reads the official CUB metadata files:
  - train_test_split.txt
  - parts/parts.txt
  - parts/part_locs.txt (default) or parts/part_click_locs.txt

and reports, for each part, how often it is marked visible in the training and
test splits.

Example:
  python scripts/cub_part_stats.py --cub-root CUB_200_2011/CUB_200_2011
  python scripts/cub_part_stats.py --cub-root CUB_200_2011/CUB_200_2011 --source part_click_locs
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Tuple


Split = str  # "train" | "test"


@dataclass(frozen=True)
class PartRow:
    """Per-part visibility statistics for one dataset split table row."""

    part_id: int
    part_name: str
    train_visible: int
    train_total: int
    train_frac: float
    test_visible: int
    test_total: int
    test_frac: float
    diff_train_minus_test: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute part visibility statistics for CUB-200-2011.",
    )
    parser.add_argument(
        "--cub-root",
        type=str,
        default="CUB_200_2011/CUB_200_2011",
        help="Path to the CUB_200_2011/CUB_200_2011 directory.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="part_locs",
        choices=["part_locs", "part_click_locs"],
        help="Which part annotations to use.",
    )
    parser.add_argument(
        "--click-agg",
        type=str,
        default="any",
        choices=["any", "majority"],
        help=(
            "Aggregation for part_click_locs: "
            "'any' marks visible if any annotator marked it visible; "
            "'majority' requires >=50%% visible votes."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to write the per-part table as CSV.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write the per-part table + summary as JSON.",
    )
    return parser.parse_args()


def _read_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def load_train_test_split(path: Path) -> Dict[int, Split]:
    """Return mapping {image_id -> split}."""

    splits: Dict[int, Split] = {}
    for line in _read_lines(path):
        image_id_str, is_train_str = line.split()
        image_id = int(image_id_str)
        is_train = int(is_train_str)
        splits[image_id] = "train" if is_train == 1 else "test"
    return splits


def load_parts(path: Path) -> Dict[int, str]:
    """Return mapping {part_id -> part_name}."""

    parts: Dict[int, str] = {}
    for line in _read_lines(path):
        part_id_str, part_name = line.split(maxsplit=1)
        parts[int(part_id_str)] = part_name.strip()
    return parts


def iter_part_locs(path: Path) -> Iterable[Tuple[int, int, int]]:
    """Yield (image_id, part_id, visible) from parts/part_locs.txt."""

    for line in _read_lines(path):
        image_id_str, part_id_str, _x, _y, visible_str = line.split()
        yield int(image_id_str), int(part_id_str), int(visible_str)


def compute_stats(
    *,
    cub_root: Path,
    source: str,
    click_agg: str,
) -> tuple[list[PartRow], dict[str, float | int]]:
    """Compute per-part visibility counts/fractions for train/test.

    Args:
        cub_root: Path to `CUB_200_2011/CUB_200_2011`.
        source: Which annotation file to read (`part_locs` or `part_click_locs`).
        click_agg: Aggregation strategy for `part_click_locs` visibility votes.

    Returns:
        (rows, summary) where `rows` is a list of `PartRow` and `summary` contains
        split totals and basic dataset-level aggregates.
    """

    split_path = cub_root / "train_test_split.txt"
    parts_path = cub_root / "parts" / "parts.txt"
    locs_path = cub_root / "parts" / "part_locs.txt"
    click_path = cub_root / "parts" / "part_click_locs.txt"

    for required in (split_path, parts_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")
    if source == "part_locs" and not locs_path.exists():
        raise FileNotFoundError(f"Missing required file: {locs_path}")
    if source == "part_click_locs" and not click_path.exists():
        raise FileNotFoundError(f"Missing required file: {click_path}")

    splits = load_train_test_split(split_path)
    parts = load_parts(parts_path)

    split_totals: Dict[Split, int] = {"train": 0, "test": 0}
    for split in splits.values():
        split_totals[split] += 1

    visible_counts: Dict[Split, DefaultDict[int, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
    }
    image_visible_parts: Dict[Split, DefaultDict[int, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
    }

    if source == "part_locs":
        for image_id, part_id, visible in iter_part_locs(locs_path):
            split = splits.get(image_id)
            if split is None:
                continue
            visible = 1 if visible else 0
            visible_counts[split][part_id] += visible
            image_visible_parts[split][image_id] += visible
    else:
        # For clicks, aggregate vote statistics. We expose two aggregation modes:
        # - any: at least one annotator marked the part visible
        # - majority: at least half of annotators marked it visible
        votes_sum: Dict[Tuple[int, int], int] = {}
        votes_total: Dict[Tuple[int, int], int] = {}
        for line in _read_lines(click_path):
            image_id_str, part_id_str, _x, _y, visible_str, _time = line.split()
            key = (int(image_id_str), int(part_id_str))
            visible = int(visible_str)
            votes_sum[key] = votes_sum.get(key, 0) + visible
            votes_total[key] = votes_total.get(key, 0) + 1

        for (image_id, part_id), total in votes_total.items():
            split = splits.get(image_id)
            if split is None:
                continue
            vis_sum = votes_sum.get((image_id, part_id), 0)
            if click_agg == "majority":
                visible = 1 if (vis_sum / max(total, 1)) >= 0.5 else 0
            else:
                visible = 1 if vis_sum > 0 else 0
            visible_counts[split][part_id] += visible
            image_visible_parts[split][image_id] += visible

    def _mean_visible(split: Split) -> float:
        ids = [img_id for img_id, sp in splits.items() if sp == split]
        if not ids:
            return 0.0
        total_visible = sum(image_visible_parts[split].get(img_id, 0) for img_id in ids)
        return total_visible / len(ids)

    rows: list[PartRow] = []
    for part_id in sorted(parts):
        name = parts[part_id]
        train_visible = int(visible_counts["train"].get(part_id, 0))
        test_visible = int(visible_counts["test"].get(part_id, 0))
        train_total = int(split_totals["train"])
        test_total = int(split_totals["test"])
        train_frac = train_visible / train_total if train_total else 0.0
        test_frac = test_visible / test_total if test_total else 0.0
        rows.append(
            PartRow(
                part_id=part_id,
                part_name=name,
                train_visible=train_visible,
                train_total=train_total,
                train_frac=train_frac,
                test_visible=test_visible,
                test_total=test_total,
                test_frac=test_frac,
                diff_train_minus_test=train_frac - test_frac,
            )
        )

    summary: dict[str, float | int] = {
        "train_images": split_totals["train"],
        "test_images": split_totals["test"],
        "mean_visible_parts_per_image_train": _mean_visible("train"),
        "mean_visible_parts_per_image_test": _mean_visible("test"),
        "source": source,
        "click_agg": click_agg if source == "part_click_locs" else "",
    }
    return rows, summary


def _format_pct(frac: float) -> str:
    return f"{100.0 * frac:5.1f}%"


def print_report(rows: list[PartRow], summary: dict[str, float | int]) -> None:
    """Print a human-readable table of part visibility statistics."""

    print("CUB part visibility statistics")
    print(f"- source: {summary['source']}")
    if summary.get("click_agg"):
        print(f"- click_agg: {summary['click_agg']}")
    print(
        f"- train images: {summary['train_images']} | "
        f"mean visible parts/image: {summary['mean_visible_parts_per_image_train']:.2f}"
    )
    print(
        f"- test  images: {summary['test_images']} | "
        f"mean visible parts/image: {summary['mean_visible_parts_per_image_test']:.2f}"
    )
    print()

    header = (
        "part_id",
        "part_name",
        "train_visible",
        "train_frac",
        "test_visible",
        "test_frac",
        "diff(train-test)",
    )
    print(
        f"{header[0]:>7}  {header[1]:<20}  {header[2]:>12}  {header[3]:>10}  "
        f"{header[4]:>11}  {header[5]:>9}  {header[6]:>15}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{row.part_id:7d}  {row.part_name:<20}  "
            f"{row.train_visible:12d}  {_format_pct(row.train_frac):>10}  "
            f"{row.test_visible:11d}  {_format_pct(row.test_frac):>9}  "
            f"{(100.0 * row.diff_train_minus_test):+14.1f}pp"
        )


def write_csv(path: Path, rows: list[PartRow]) -> None:
    """Write the per-part table as a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: list[PartRow], summary: dict[str, float | int]) -> None:
    """Write the per-part table + summary to a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "parts": [asdict(row) for row in rows]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """CLI entrypoint."""

    args = _parse_args()
    cub_root = Path(args.cub_root).expanduser()
    rows, summary = compute_stats(
        cub_root=cub_root, source=str(args.source), click_agg=str(args.click_agg)
    )
    print_report(rows, summary)

    if args.out_csv:
        write_csv(Path(args.out_csv).expanduser(), rows)
    if args.out_json:
        write_json(Path(args.out_json).expanduser(), rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
