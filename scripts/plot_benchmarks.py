#!/usr/bin/env python3
"""
plot_benchmarks.py — Visualize Q-NarwhalKnight benchmark results

Usage:
    python3 scripts/plot_benchmarks.py target/benchmark-results/LATEST/
    python3 scripts/plot_benchmarks.py --compare dir1/ dir2/

Reads Criterion output and summary.json to produce:
  - Bar chart of suite pass/fail status
  - Throughput comparison across runs (if --compare)
  - Per-benchmark time distributions
"""

import json
import os
import re
import sys
from pathlib import Path

def parse_bencher_output(filepath: Path) -> list:
    """Parse Criterion --output-format=bencher lines into structured data."""
    results = []
    try:
        text = filepath.read_text()
    except FileNotFoundError:
        return results

    for line in text.splitlines():
        # Format: "test bench_name ... bench: N ns/iter (+/- M)"
        m = re.match(r'^test\s+(\S+)\s+\.\.\.\s+bench:\s+([\d,]+)\s+ns/iter\s+\(\+/- ([\d,]+)\)', line)
        if m:
            name = m.group(1)
            ns = int(m.group(2).replace(',', ''))
            variance = int(m.group(3).replace(',', ''))
            results.append({
                'name': name,
                'ns_per_iter': ns,
                'variance_ns': variance,
                'us_per_iter': ns / 1000.0,
                'ms_per_iter': ns / 1_000_000.0,
            })
    return results


def print_text_report(results_dir: Path):
    """Print a text-based summary report."""
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        print(f"No summary.json found in {results_dir}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print("=" * 60)
    print(f"  Benchmark Report: {summary.get('timestamp', 'unknown')}")
    print("=" * 60)
    print(f"  Suites: {summary['total']}  |  Passed: {summary['passed']}  |  Failed: {summary['failed']}")
    print()

    # Suite status
    print(f"  {'Suite':<25} {'Status':<8} {'Duration':<12}")
    print(f"  {'-'*25} {'-'*8} {'-'*12}")
    for name, info in sorted(summary.get('suites', {}).items()):
        status = info['status']
        duration = info['duration']
        marker = 'PASS' if status == 'pass' else 'FAIL'
        print(f"  {name:<25} {marker:<8} {duration:<12}")

    print()

    # Parse individual benchmark results
    all_benchmarks = []
    for txt_file in sorted(results_dir.glob('*.txt')):
        if txt_file.name in ('summary.txt',):
            continue
        suite_name = txt_file.stem
        benchmarks = parse_bencher_output(txt_file)
        if benchmarks:
            print(f"  [{suite_name}] {len(benchmarks)} benchmarks:")
            for b in benchmarks:
                if b['ms_per_iter'] >= 1.0:
                    time_str = f"{b['ms_per_iter']:.2f} ms"
                elif b['us_per_iter'] >= 1.0:
                    time_str = f"{b['us_per_iter']:.1f} us"
                else:
                    time_str = f"{b['ns_per_iter']} ns"
                print(f"    {b['name']:<45} {time_str:>12}")
            print()
            all_benchmarks.extend(benchmarks)

    if all_benchmarks:
        print(f"  Total individual benchmarks: {len(all_benchmarks)}")
    print()


def try_plot(results_dir: Path):
    """Try to create matplotlib charts. Falls back gracefully if not installed."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping charts")
        print("  Install with: pip3 install matplotlib")
        return

    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        summary = json.load(f)

    suites = summary.get('suites', {})
    if not suites:
        return

    # Bar chart of suite durations
    names = sorted(suites.keys())
    durations = []
    colors = []
    for name in names:
        info = suites[name]
        dur_str = info['duration'].rstrip('s')
        try:
            durations.append(float(dur_str))
        except ValueError:
            durations.append(0)
        colors.append('#10B981' if info['status'] == 'pass' else '#EF4444')

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    bars = ax.barh(names, durations, color=colors)
    ax.set_xlabel('Duration (seconds)')
    ax.set_title(f'Benchmark Suite Results — {summary.get("timestamp", "")}')
    ax.invert_yaxis()

    for bar, dur in zip(bars, durations):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{dur:.0f}s', va='center', fontsize=8)

    plt.tight_layout()
    chart_path = results_dir / 'suite_durations.png'
    plt.savefig(chart_path, dpi=150)
    print(f"  Chart saved: {chart_path}")
    plt.close()


def compare_runs(dir1: Path, dir2: Path):
    """Compare two benchmark runs side by side."""
    print(f"Comparing: {dir1.name} vs {dir2.name}")
    print()

    for txt_file in sorted(dir1.glob('*.txt')):
        suite = txt_file.stem
        other = dir2 / txt_file.name
        if not other.exists():
            continue

        results_a = parse_bencher_output(txt_file)
        results_b = parse_bencher_output(other)

        if not results_a or not results_b:
            continue

        # Match by name
        map_b = {r['name']: r for r in results_b}
        print(f"  [{suite}]")
        print(f"    {'Benchmark':<40} {'Run 1':>12} {'Run 2':>12} {'Change':>10}")
        print(f"    {'-'*40} {'-'*12} {'-'*12} {'-'*10}")

        for a in results_a:
            b = map_b.get(a['name'])
            if not b:
                continue
            ns_a = a['ns_per_iter']
            ns_b = b['ns_per_iter']
            if ns_a > 0:
                pct = ((ns_b - ns_a) / ns_a) * 100
                change = f"{pct:+.1f}%"
            else:
                change = "N/A"

            fmt = lambda ns: f"{ns/1000:.1f}us" if ns >= 1000 else f"{ns}ns"
            print(f"    {a['name']:<40} {fmt(ns_a):>12} {fmt(ns_b):>12} {change:>10}")
        print()


def main():
    if len(sys.argv) < 2:
        # Find latest results
        results_base = Path('target/benchmark-results')
        if results_base.exists():
            dirs = sorted(results_base.iterdir())
            if dirs:
                results_dir = dirs[-1]
                print(f"Using latest results: {results_dir}")
                print()
                print_text_report(results_dir)
                try_plot(results_dir)
                return

        print("Usage:")
        print("  python3 scripts/plot_benchmarks.py target/benchmark-results/DIR/")
        print("  python3 scripts/plot_benchmarks.py --compare DIR1 DIR2")
        sys.exit(1)

    if sys.argv[1] == '--compare' and len(sys.argv) >= 4:
        compare_runs(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        results_dir = Path(sys.argv[1])
        print_text_report(results_dir)
        try_plot(results_dir)


if __name__ == '__main__':
    main()
