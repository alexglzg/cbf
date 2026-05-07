"""
Analyze computation times and problem metrics across all environments for a given obstacle count.

Creates boxplots for KKT time and total computation time, and prints summary statistics
for problem size (variables, constraints) and solver iterations.

Usage:
    python analyze_computation_times.py n1
    python analyze_computation_times.py n2 --output results.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tabulate import tabulate
import os
import re


def load_results(obstacle_count_dir):
    """
    Load all JSON result files from the given obstacle count directory.
    
    Parameters:
    -----------
    obstacle_count_dir : str
        Directory like 'results/n1' containing all environment results
    
    Returns:
    --------
    results : list of dict
        List of loaded result dictionaries
    file_names : list of str
        Corresponding file names (for labeling)
    """
    results = []
    file_names = []
    
    dir_path = Path(obstacle_count_dir)
    if not dir_path.exists():
        print(f"Error: Directory {obstacle_count_dir} not found")
        return [], []
    
    # Load all JSON files in the directory
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                payload = json.load(f)
 
            # Normalise: a bare dict counts as a single-element list
            if isinstance(payload, dict):
                payload = [payload]
 
            n_entries = len(payload)
            for idx, result in enumerate(payload):
                results.append(result)
                # Use a suffix only when there are multiple poses per file
                label = json_file.stem if n_entries == 1 else f"{json_file.stem}_{idx}"
                file_names.append(label)
 
            print(f"Loaded {json_file.stem}  ({n_entries} run(s))")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
 
    if not results:
        print(f"No JSON files found in {obstacle_count_dir}")
 
    return results, file_names



def extract_timing_data(results, file_names):
    """
    Extract computation times from all results.
    
    Parameters:
    -----------
    results : list of dict
        List of result dictionaries
    file_names : list of str
        Corresponding file names
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns: environment, controller, kkt_time, total_time
    """
    rows = []

    # for result, fname in zip(results, file_names):
    #     print("File: ", fname)
    # File:  env0_dcbf
    # File:  env0_pipcbf
    # File:  env1_dcbf
    # File:  env1_pipcbf

    feas = {'dcbf': {'feas': 0, 'infeas': 0}, 'pipcbf': {'feas': 0, 'infeas': 0}}
    max_halfplanes_in_run = 0
    
    # Loop over dcbf or pipcbf results
    for result, fname in zip(results, file_names):
        controller = result.get('controller', 'unknown')
        assert controller in ('dcbf', 'pipcbf'), \
        f"{fname}: unexpected or missing controller field '{controller}'"
        
        # Extract KKT times (per-step)
        kkt_stats = result.get('kkt_time_s', {})
        kkt_times = result.get('steps', [])
        
        # Extract individual per-step times
        for step_data in result.get('steps', []):
            kkt_time = step_data.get('kkt_time_s')
            comp_time = step_data.get('comp_time_s')
            infeasible = step_data.get('infeasible')
            if controller == 'pipcbf':
                n_halfplanes = step_data.get('n_halfplanes')
                if n_halfplanes is not None:
                    max_halfplanes_in_run = max(max_halfplanes_in_run, n_halfplanes)

            if infeasible:
                feas[controller]['infeas'] += 1
            else:
                feas[controller]['feas'] += 1
            
            if kkt_time is not None and comp_time is not None:
                rows.append({
                    'environment': fname,
                    'controller': controller,
                    'kkt_time_ms': kkt_time * 1000,  # Convert to milliseconds
                    'total_time_ms': comp_time * 1000,  # Convert to milliseconds
                })

    # n1: index 33 = 19.457 secs (env0 somewhere a bug in multiplying by a 1000)
    # {
    #     "comp_time_s": 19.54844648,
    #     "feval_time_s": 0.0005366139999999999,
    #     "kkt_time_s": 19.547909865999998,
    #     "iterations": 6,
    #     "infeasible": false,
    #     "n_variables": 278,
    #     "n_eq": 0,
    #     "n_ineq": 370,
    #     "clearances": [
    #       2.213999669408094
    #     ],
    #     "min_clearance": 2.213999669408094
    #   },

    
    print(
        f"Max halfplanes in any step: "
        f"{max_halfplanes_in_run}"
    )
    
    df = pd.DataFrame(rows)
    return df, feas



def find_results_by_filename(results, filename, results_folder):
    """
    Find the results dict corresponding to a given output filename,
    restricting the search to files that exist inside results_folder
    (recursively).

    Parameters
    ----------
    results : list of dict
        Parsed result dictionaries (already loaded).
    filename : str
        Result filename, e.g. 'env0_pipcbf.json'.
    results_folder : str
        Root results directory, e.g. 'results/n10'.

    Returns
    -------
    result : dict
        Matching results dictionary.

    Raises
    ------
    ValueError
        If the file does not exist in results_folder or no unique result matches.
    """
    # ---- Normalize paths ----
    results_folder = os.path.abspath(os.path.expanduser(results_folder))
    filename = os.path.basename(filename)

    # ---- 1. Ensure file exists inside results_folder (recursive) ----
    found_on_disk = False
    for root, _, files in os.walk(results_folder):
        if filename in files:
            found_on_disk = True
            break

    if not found_on_disk:
        raise ValueError(
            f"File {filename} not found under {results_folder}"
        )

    # ---- 2. Parse env_idx and controller from filename ----
    basename = filename
    match = re.match(r"env(\d+)_(.+)\.json", basename)

    if not match:
        raise ValueError(f"Cannot parse filename {basename}")

    env_idx = int(match.group(1))
    controller = match.group(2)

    matches = [
        r for r in results
        if r.get("env_idx") == env_idx
        and r.get("controller") == controller
    ]

    if len(matches) == 0:
        raise ValueError(f"No loaded results match {basename}")

    # ✅ RETURN LIST (not a single dict)
    return matches



def extract_single_environment(result):
    """
    Extract raw per-step metrics from a single results dictionary
    corresponding to one JSON output file.

    Parameters
    ----------
    result : dict
        Single result dictionary (one JSON file)

    Returns
    -------
    data : dict
        Dictionary of arrays with all relevant metrics.
    """
    data = {
        'env_idx': result.get('env_idx'),
        'controller': result.get('controller'),
        'problem_size': result.get('problem_size'),
        'n_obstacles': result.get('n_obstacles'),
        'robot_shape': result.get('robot_shape'),

        'comp_time_s': [],
        'feval_time_s': [],
        'kkt_time_s': [],
        'iterations': [],
        'n_variables': [],
        'n_eq': [],
        'n_ineq': [],
        'infeasible': [],
        'min_clearance': [],
        'comp_times': [],
    }

    # ---- Top-level step arrays ----
    data['comp_time_s'].extend(result.get('comp_time_s', []))
    data['feval_time_s'].extend(result.get('feval_time_s', []))
    data['kkt_time_s'].extend(result.get('kkt_time_s', []))
    data['iterations'].extend(result.get('iterations', []))
    data['min_clearance'].extend(result.get('min_clearance', []))

    data['n_variables'].extend(result.get('n_variables_steps', []))
    data['n_eq'].extend(result.get('n_eq_steps', []))
    data['n_ineq'].extend(result.get('n_ineq_steps', []))

    # ---- Per-step fallback / infeasibility flag ----
    for step in result.get('steps', []):
        if 'infeasible' in step:
            data['infeasible'].append(step['infeasible'])
        if 'comp_time_s' in step:
            data['comp_times'].append(step['comp_time_s'])

        # Fill missing metrics if not present at top level
        if not data['comp_time_s'] and 'comp_time_s' in step:
            data['comp_time_s'].append(step['comp_time_s'])
        if not data['feval_time_s'] and 'feval_time_s' in step:
            data['feval_time_s'].append(step['feval_time_s'])
        if not data['kkt_time_s'] and 'kkt_time_s' in step:
            data['kkt_time_s'].append(step['kkt_time_s'])
        if not data['iterations'] and 'iterations' in step:
            data['iterations'].append(step['iterations'])

        if not data['n_variables'] and 'n_variables' in step:
            data['n_variables'].append(step['n_variables'])
        if not data['n_eq'] and 'n_eq' in step:
            data['n_eq'].append(step['n_eq'])
        if not data['n_ineq'] and 'n_ineq' in step:
            data['n_ineq'].append(step['n_ineq'])

    return data

def extract_metrics_data(results):
    """
    Extract problem metrics (variables, constraints, iterations) from all results,
    grouped by controller type.
    
    Parameters:
    -----------
    results : list of dict
        List of result dictionaries
    
    Returns:
    --------
    metrics_by_controller : dict
        Dictionary with controller names as keys, each containing:
        {'n_variables': [...], 'n_eq': [...], 'n_ineq': [...], 'iterations': [...]}
    """
    metrics_by_controller = {}
    
    for result in results:
        controller = result.get('controller', 'unknown')
        
        if controller not in metrics_by_controller:
            metrics_by_controller[controller] = {
                'n_variables': [],
                'n_eq': [],
                'n_ineq': [],
                'iterations': [],
                'clearance': [],
            }
        
        metrics = metrics_by_controller[controller]
        
        # First, try to use top-level arrays if they exist and have data
        n_vars_steps = result.get('n_variables_steps', [])
        metrics['n_variables'].extend([v for v in n_vars_steps if v is not None])
        
        n_eq_steps = result.get('n_eq_steps', [])
        metrics['n_eq'].extend([v for v in n_eq_steps if v is not None])
        
        n_ineq_steps = result.get('n_ineq_steps', [])
        metrics['n_ineq'].extend([v for v in n_ineq_steps if v is not None])
        
        # If top-level arrays are empty, fall back to per-step data
        if not n_eq_steps:
            for step_data in result.get('steps', []):
                n_eq = step_data.get('n_eq')
                if n_eq is not None:
                    metrics['n_eq'].append(n_eq)
        
        if not n_ineq_steps:
            for step_data in result.get('steps', []):
                n_ineq = step_data.get('n_ineq')
                if n_ineq is not None:
                    metrics['n_ineq'].append(n_ineq)
        
        # Per-step iterations
        for step_data in result.get('steps', []):
            iters = step_data.get('iterations')
            clearances = step_data.get('min_clearance')
            if iters is not None:
                metrics['iterations'].append(iters)
            if clearances is not None:
                metrics['clearance'].append(clearances)
    
    return metrics_by_controller


def report_max_halfplanes(results, obstacle_count):
    """
    Report the maximum number of halfplanes (inequality constraints)
    over all environments, runs, and timesteps for a given obstacle count.
    """
    all_n_ineq = []

    for result in results:
        # Prefer top-level per-step array if present
        n_ineq_steps = result.get('n_ineq_steps', [])
        all_n_ineq.extend([v for v in n_ineq_steps if v is not None])

        # Fallback to per-step data
        if not n_ineq_steps:
            for step in result.get('steps', []):
                n_ineq = step.get('n_ineq')
                if n_ineq is not None:
                    all_n_ineq.append(n_ineq)

    print("\n" + "=" * 80)
    print(f"MAX NUMBER OF HALFPLANES (Obstacle count: {obstacle_count})")
    print("=" * 80)

    if all_n_ineq:
        print(f"Max halfplanes over all environments and runs: {max(all_n_ineq)}")
    else:
        print("No inequality constraint data found.")


def print_metrics_table(metrics_by_controller):
    """
    Print a formatted table of metrics statistics for each controller.
    
    Parameters:
    -----------
    metrics_by_controller : dict
        Dictionary with controller names as keys, each containing metric arrays
    """
    print("\n" + "="*80)
    print("PROBLEM METRICS SUMMARY (across all steps and environments)")
    print("="*80)
    
    # Map metric names to descriptive labels
    metric_labels = {
        'n_variables': 'Number of Variables',
        'n_eq': 'Equality Constraints',
        'n_ineq': 'Inequality Constraints',
        'iterations': 'Iterations',
        'clearance': 'Clearance'
    }
    
    # Only print if we have data
    if not metrics_by_controller:
        print("No metrics data available")
        return
    
    # If only one controller, print simple table
    if len(metrics_by_controller) == 1:
        controller = list(metrics_by_controller.keys())[0]
        metrics = metrics_by_controller[controller]
        
        print(f"\nController: {controller.upper()}")
        print("-" * 80)
        
        table_data = []
        for metric_name in ['n_variables', 'n_eq', 'n_ineq', 'iterations', 'clearance']:
            values = metrics[metric_name]
            if not values:
                continue
            
            values = np.array(values)
            table_data.append([
                metric_labels[metric_name],
                f"{np.median(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}",
            ])
        
        print(tabulate(
            table_data,
            headers=['Metric', 'Median', 'Std Dev', 'Min', 'Max'],
            tablefmt='grid'
        ))
    else:
        # Multiple controllers - print comparison table
        for metric_name in ['n_variables', 'n_eq', 'n_ineq', 'iterations', 'clearance']:
            table_data = []
            
            for controller in sorted(metrics_by_controller.keys()):
                metrics = metrics_by_controller[controller]
                values = metrics[metric_name]
                
                if not values:
                    continue
                
                values = np.array(values)
                table_data.append([
                    controller.upper(),
                    f"{np.median(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                ])
            
            if table_data:
                print(f"\n{metric_labels[metric_name]}")
                print("-" * 80)
                print(tabulate(
                    table_data,
                    headers=['Controller', 'Median', 'Std Dev', 'Min', 'Max'],
                    tablefmt='grid'
                ))
    
    print()


def create_boxplots(df, obstacle_count, output_path=None):
    """
    Create boxplots for KKT time and total computation time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timing data
    obstacle_count : str
        Obstacle count label (e.g., 'n1')
    output_path : str, optional
        Path to save figure. If None, displays interactively.
    """
    if df.empty:
        print("No timing data to plot")
        return
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # KKT Time boxplot
    sns.boxplot(data=df, x='controller', y='kkt_time_ms', ax=axes[0])
    axes[0].set_title(f'KKT Time - {obstacle_count}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Controller', fontsize=11)
    axes[0].set_ylabel('KKT Time (ms)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Total Computation Time boxplot
    sns.boxplot(data=df, x='controller', y='total_time_ms', ax=axes[1])
    axes[1].set_title(f'Total Computation Time - {obstacle_count}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Controller', fontsize=11)
    axes[1].set_ylabel('Total Time (ms)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved boxplot figure to {output_path}")
    else:
        plt.show()
    
    return fig, axes


def print_timing_summary(df):
    """
    Print summary statistics of timing data by controller.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with timing data
    """
    print("\n" + "="*80)
    print("COMPUTATION TIME SUMMARY (by controller)")
    print("="*80)
    
    table_data = []
    
    for controller in sorted(df['controller'].unique()):
        controller_data = df[df['controller'] == controller]
        
        kkt_times = controller_data['kkt_time_ms'].values
        total_times = controller_data['total_time_ms'].values
        
        table_data.append([
            controller.upper(),
            'KKT Time (ms)',
            f"{np.median(kkt_times):.3f}",
            f"{np.std(kkt_times):.3f}",
            f"{np.min(kkt_times):.3f}",
            f"{np.max(kkt_times):.3f}",
        ])
        table_data.append([
            '',
            'Total Time (ms)',
            f"{np.median(total_times):.3f}",
            f"{np.std(total_times):.3f}",
            f"{np.min(total_times):.3f}",
            f"{np.max(total_times):.3f}",
        ])
    
    print(tabulate(
        table_data,
        headers=['Controller', 'Metric', 'Median', 'Std Dev', 'Min', 'Max'],
        tablefmt='grid'
    ))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze computation times and problem metrics for a given obstacle count'
    )
    parser.add_argument('obstacle_count', 
                       help='Obstacle count directory (e.g., n1, n2, n3)')
    parser.add_argument('--output', default=None,
                       help='Output file path for boxplot (optional)')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory (default: results)')
    
    args = parser.parse_args()
    
    # Construct full path
    obstacle_dir = f"{args.results_dir}/{args.obstacle_count}"
    
    print(f"\nAnalyzing computation times for {args.obstacle_count}...")
    print(f"Results directory: {obstacle_dir}")
    print()
    
    # Load results
    results, file_names = load_results(obstacle_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    print(f"Loaded {len(results)} result files\n")

    # # Load single environment
    # filename = "env1_dcbf.json"
    # result = find_results_by_filename(results, filename, 'results/n3/')
    # data = extract_single_environment(result)
    # # print(f"Infeasibilities in file {filename}: {data['infeasible']}")
    # print(f"Max comp time [ms]: {max(data['comp_times']), np.argmax(data['comp_times'])}")
    
    # Extract timing data
    df_timing, feas = extract_timing_data(results, file_names)

    # print(f"DCBF feas: {feas['dcbf']['feas']}")
    # print(f"DCBF feas (%): {feas['dcbf']['feas']/(feas['dcbf']['feas'] + feas['dcbf']['infeas']) * 100}")
    # print(f"PiPCBF feas: {feas['pipcbf']['feas']}")
    # print(f"DCBF feas (%): {feas['pipcbf']['feas']/(feas['pipcbf']['feas'] + feas['pipcbf']['infeas']) * 100}")
    
    # Extract metrics data
    metrics = extract_metrics_data(results)

    
    # Report max halfplanes
    report_max_halfplanes(results, args.obstacle_count)

    
    # Print tables
    print_timing_summary(df_timing)
    print_metrics_table(metrics)
    
    # Create boxplots
    output = args.output
    if output is None and len(args.obstacle_count) > 0:
        # Auto-generate output filename
        output = f"timing_boxplot_{args.obstacle_count}.png"

    save = False
    if save:
        create_boxplots(df_timing, args.obstacle_count, output_path=output)
    else:
        create_boxplots(df_timing, args.obstacle_count, output_path=None)
    
    print("Analysis complete!")


if __name__ == '__main__':
    main()
