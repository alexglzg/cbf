"""
Create violin plots summarizing computation times across all obstacle counts.

This script:
- Walks through results/* subdirectories (e.g., n1, n2, n5, ...)
- Loads all JSON files inside each subdirectory
- Handles JSONs containing multiple start/goal runs per environment
- Aggregates all per-step timing data
- Produces two violin plots:
    1) KKT time
    2) Total computation time
Each plot shows DCBF vs PiPCBF for all obstacle counts on a single figure.
"""

from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use("TkAgg")

def latexify():
    params = {'backend': 'TkAgg',
              'axes.labelsize': 32, #26 for phd
              'axes.titlesize': 32, #26 for phd
              'legend.fontsize': 32, #26 for phd
              'xtick.labelsize': 26,
              'ytick.labelsize': 26,
              'text.usetex': True,
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)

latexify()

color_reform = "#0072BD"
color_decomp = [0.8800, 0.350, 0.0980]#"#D95319"
custom_palette = {'dcbf': color_reform, 'pipcbf': color_decomp}

def load_all_results(results_root: str) -> pd.DataFrame:
    """
    Walk through results_root/* subfolders, load all JSON files,
    and extract per-step computation and KKT times.

    Returns
    -------
    df : pandas.DataFrame
        Long-form dataframe with columns:
        - obstacles        (e.g. 'n1', 'n5')
        - controller       ('dcbf' or 'pipcbf')
        - kkt_time_ms
        - total_time_ms
    """
    rows = []
    results_root = Path(results_root)

    for obstacle_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        obstacle_label = obstacle_dir.name  # e.g. n1, n5, ...

        for json_file in obstacle_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                   payload = json.load(f)

                # Normalize: dict → single-element list
                if isinstance(payload, dict):
                    payload = [payload]

                for result in payload:
                    controller = result.get("controller", "unknown")
                    if controller not in ("dcbf", "pipcbf"):
                        continue

                    for step in result.get("steps", []):
                        kkt = step.get("kkt_time_s")
                        comp = step.get("comp_time_s")

                        if kkt is None or comp is None:
                            continue

                        rows.append({
                            "obstacles": obstacle_label,
                            "Controller": controller,
                            "kkt_time_ms": kkt * 1000.0,
                            "total_time_ms": comp * 1000.0,
                        })

            except Exception as e:
                print(f"Failed to load {json_file}: {e}")

    return pd.DataFrame(rows)


def create_violin_plots(df: pd.DataFrame, output_prefix=None):
    """
    Create two violin plots:
    - KKT time
    - Total computation time
    """
    if df.empty:
        print("No data available for plotting.")
        return

    # sns.set_style("whitegrid")
    # sns.set_palette("Set2")

    # --- KKT time violin plot ---
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="obstacles",
        y="kkt_time_ms",
        hue="Controller",
        split=True,
        gap = 0.05,
        inner="quart",
        palette=custom_palette,
        # cut=0
    )
    plt.xlabel("Number of obstacles")
    plt.ylabel("KKT time (ms)")
    # plt.title("KKT computation time across obstacle counts")
    plt.yscale("log")
    plt.tight_layout()

    if output_prefix:
        plt.savefig(f"{output_prefix}_kkt_violin.png", dpi=150)
    plt.legend(loc='upper left')
    plt.show(block = False)

    # --- Total computation time violin plot ---
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="obstacles",
        y="total_time_ms",
        hue="Controller",
        split=True,
        palette=custom_palette,
        inner="quart",
        gap = 0.05,
        # cut=0
    )
    plt.xlabel("Number of obstacles")
    plt.ylabel("Wall time (ms)")
    # plt.title("Total computation time across obstacle counts")
    plt.yscale("log")
    plt.legend(loc='upper left')
    plt.tight_layout()

    if output_prefix:
        plt.savefig(f"{output_prefix}_total_violin.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create violin plots over all obstacle counts"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory (default: results)"
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for saving figures (optional)"
    )
    args = parser.parse_args()

    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} timing samples")

    create_violin_plots(df, args.output_prefix)


if __name__ == "__main__":
    main()