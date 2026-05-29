import json
from pathlib import Path
from typing import Literal

from jsonargparse import CLI
from rich import box, print
from rich.markup import escape
from rich.table import Table


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {b["fullname"]: b["stats"] for b in data["benchmarks"]}


def _format_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    if seconds >= 1e-6:
        # Unicode for mu
        return f"{seconds * 1e6:.2f} \u03bcs"
    else:
        return f"{seconds * 1e9:.2f} ns"


def _compare_key(
    key: str,
    baseline_stats: dict,
    current_stats: dict,
    metric: str,
    threshold: float,
):
    base = baseline_stats.get(key)
    curr = current_stats.get(key)

    name = key.split("::")[-1]

    if base is None:
        return (0, name, "N/A", _format_time(curr[metric]), "New")
    if curr is None:
        return (0, name, _format_time(base[metric]), "N/A", "Removed")

    base_value = base[metric]
    curr_value = curr[metric]

    delta_pct = (curr_value - base_value) / base_value * 100

    if delta_pct > threshold:
        change_str = f"[red]Regressed ({delta_pct:+.1f}%)[/red]"
    elif delta_pct < -threshold:
        change_str = f"[green]Improved ({delta_pct:+.1f}%)[/green]"
    else:
        change_str = f"Stable ({delta_pct:+.1f}%)"

    return (
        delta_pct,
        name,
        _format_time(base_value),
        _format_time(curr_value),
        change_str,
    )


def main(
    baseline: Path,
    current: Path,
    metric: Literal["min", "max", "mean", "median"] = "median",
    threshold: float = 5.0,
    sort: bool = False,
):
    """Compare benchmarks between two JSON files from pytest-benchmark.

    Args:
        baseline: Path to the baseline JSON file.
        current: Path to the current JSON file.
        metric: The metric to compare (min, max, mean, median).
        threshold: Percentage threshold for highlighting differences.
        sort: Whether to sort the output by percent change in the metric.
    """
    baseline_stats = _load_json(baseline)
    current_stats = _load_json(current)

    all_keys = list(dict.fromkeys(list(baseline_stats) + list(current_stats)))

    if not all_keys:
        print("No benchmarks found in either file.")
        return

    results = [
        _compare_key(key, baseline_stats, current_stats, metric, threshold)
        for key in all_keys
    ]

    if sort:
        results.sort(key=lambda x: x[0], reverse=True)

    table = Table(show_header=True, box=box.MARKDOWN)
    table.add_column("Benchmark", justify="left")
    table.add_column("Baseline", justify="right", style="dim")
    table.add_column("Current", justify="right", style="dim")
    table.add_column("Delta", justify="left")

    for _, name, b_s, c_s, ind in results:
        table.add_row(escape(name), b_s, c_s, ind)

    print(table)


if __name__ == "__main__":
    CLI(main)
