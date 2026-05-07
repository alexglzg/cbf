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
import numpy as np

from matplotlib.patches import FancyBboxPatch


def latexify():
    params = {'backend': 'TkAgg',
              'axes.labelsize': 32, #26 for phd
              'axes.titlesize': 32, #26 for phd
              'legend.fontsize': 32, #26 for phd
              'xtick.labelsize': 26,
              'ytick.labelsize': 26,
              'text.usetex': True,
            #   'font.family': 'serif'
              'font.family':       'STIXGeneral',
              'mathtext.fontset':  'stix',
              }

    matplotlib.rcParams.update(params)

latexify()

color_reform = "#0072BD"
color_decomp = [0.8800, 0.350, 0.0980]#"#D95319"
custom_palette = {'D-CBF': color_reform, 'PiP-CBF': color_decomp}

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
    start_idx = 0

    for obstacle_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        obstacle_label = obstacle_dir.name[1:]  # e.g. n1, n5, ...
        print("Obstacle number: ", obstacle_dir)
        print("Obstacle label: ", obstacle_label)
        if obstacle_label in [str(i) for i in range(1, 11)]:
            obstacle_n = int(obstacle_label)

        local_rows = []  # <-- per obstacle only
        
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

                    if controller == "dcbf":
                        controller = "D-CBF"
                    elif controller == "pipcbf":
                        controller = "PiP-CBF"

                    for step in result.get("steps", []):
                        kkt = step.get("kkt_time_s")
                        comp = step.get("comp_time_s")
                        iters = step.get("iterations")

                        if kkt is None or comp is None or iters is None:
                            continue
                        
                        entry = {
                            "obstacles": obstacle_n,
                            "Controller": controller,
                            "kkt_time_ms": kkt * 1000.0 / iters,
                            "total_time_ms": comp * 1000.0,
                        }

                        local_rows.append(entry)
                        rows.append(entry)

            except Exception as e:
                print(f"Failed to load {json_file}: {e}")

        
        
        # ---- compute reduction for this obstacle count ----
        if local_rows:
            temp = pd.DataFrame(local_rows)

            dcbf = temp[temp["Controller"] == "D-CBF"]["total_time_ms"]
            pipcbf = temp[temp["Controller"] == "PiP-CBF"]["total_time_ms"]

            print(f"N_obs = {obstacle_n}")

            if not dcbf.empty and not pipcbf.empty:
                factor = np.median(dcbf) / np.median(pipcbf)
                print(f"DCBF Perc. above 10Hz : {len(dcbf[dcbf >= 100.0]) / len(dcbf) * 100}")
                print(f"PiP-CBF Perc. above 10Hz : {len(pipcbf[pipcbf >= 100.0]) / len(pipcbf) * 100}")
                print(f"median reduction = {factor:.2f}×")
            else:
                print(f"missing one controller")

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
    # fig, axs = plt.subplots(2, 1, figsize=(14,8))
    fig, axs = plt.subplots(2, 1, figsize=(26,18))
    ax = axs[0]
    # plt.figure(figsize=(14, 6))
    # sns.violinplot(
    #     data=df,
    #     x="obstacles",
    #     y="kkt_time_ms",
    #     hue="Controller",
    #     split=True,
    #     gap = 0.05,
    #     inner="quart",
    #     palette=custom_palette,
    #     # cut=0
    # )
    sns.boxenplot(
        data=df,
        x="obstacles",
        y="kkt_time_ms",
        hue="Controller",
        palette=custom_palette,
        dodge=True,
        width=0.6,
        ax=ax
    )


    ax.set_xlabel("Number of obstacles")
    ax.set_ylabel("KKT time [ms]")
    # plt.title("KKT computation time across obstacle counts")
    ax.set_yscale("log")
    ax.grid(True, zorder=0, alpha=0.3)
    # plt.tight_layout()

    # if output_prefix:
    ax.legend(loc='upper left')
    # plt.savefig(f"{output_prefix}_kkt_violin.pdf", format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.1)
    
    # plt.show(block = False)

    # --- Total computation time violin plot ---
    # plt.figure(figsize=(14, 6))
    # sns.violinplot(
    #     data=df,
    #     x="obstacles",
    #     y="total_time_ms",
    #     hue="Controller",
    #     split=True,
    #     palette=custom_palette,
    #     inner="quart",
    #     gap = 0.05,
    #     # cut=0
    # )
    ax = axs[1]
    
    sns.boxenplot(
        data=df,
        x="obstacles",
        y="total_time_ms",
        hue="Controller",
        palette=custom_palette,
        dodge=True,
        width=0.6,
        ax=ax
    )

    ax.set_xlabel("Number of obstacles")
    ax.set_ylabel("Wall time [ms]")
    # plt.title("Total computation time across obstacle counts")
    ax.set_yscale("log")
    ax.legend(loc='upper left')
    ax.grid(True, zorder=0, alpha=0.3)
    ax.hlines(y=100, xmin=0, xmax=9, color='k', linestyle='dashdot')
    # plt.tight_layout()

    # if output_prefix:
    plt.savefig("total_kkt_per_iter_violin.pdf", format='pdf', transparent=True, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


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