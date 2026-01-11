#!/usr/bin/env python
"""
Parse grid-search logs and plot metrics versus STEPS.

Default input: runs/gridsearch_history_D1-BA-SIR-batch8-eta40.txt
Default output: runs/gridsearch_history_D1-BA-SIR-batch8-eta40.png

Example:
  python scripts/plot_grid_search.py \\
    --log runs/gridsearch_history_D1-BA-SIR-batch8-eta40.txt \\
    --metrics macro_f1 nrmse \\
    --out runs/gridsearch_history_D1-BA-SIR-batch8-eta40.png
"""
import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


HEADER_RE = re.compile(r"STEPS=(\d+)")


def parse_log(path: Path) -> Dict[int, Dict[str, float]]:
    """
    Return mapping: steps -> averaged metrics dict.
    """
    results: Dict[int, Dict[str, float]] = {}
    current_steps = None

    for line in path.read_text().splitlines():
        if not line:
            continue
        header_match = HEADER_RE.search(line)
        if header_match:
            current_steps = int(header_match.group(1))
            continue
        if line.startswith("Averaged over") and current_steps is not None:
            _, dict_str = line.split(":", 1)
            metrics = ast.literal_eval(dict_str.strip())
            results[current_steps] = metrics
            current_steps = None
    return results


def plot_metrics(
    results: Dict[int, Dict[str, float]],
    metrics: List[str],
    out_path: Path,
) -> None:
    steps = sorted(results.keys())
    plt.figure(figsize=(7, 4))
    for metric in metrics:
        ys = [results[s][metric] for s in steps if metric in results[s]]
        xs = [s for s in steps if metric in results[s]]
        plt.plot(xs, ys, marker="o", label=metric)
    plt.xlabel("STEPS")
    plt.ylabel("metric value")
    plt.title("Grid search results")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot grid-search results from log file.")
    parser.add_argument("--log", type=Path, default=Path("runs/gridsearch_history_D1-BA-SIR-batch8-eta40.txt"))
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["macro_f1", "nrmse"],
        help="Metric names to plot (keys in the averaged metrics dict).",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/gridsearch_history_D1-BA-SIR-batch8-eta40.png"))
    args = parser.parse_args()

    data = parse_log(args.log)
    if not data:
        raise SystemExit(f"No results parsed from {args.log}")
    plot_metrics(data, args.metrics, args.out)


if __name__ == "__main__":
    main()
