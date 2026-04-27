"""
Visualization script for single environment results.

Loads a result JSON file and creates a plot showing:
- The environment (polytope obstacles)
- The full trajectory of the boat (closed-loop path)
- Robot visualization at selected points along the trajectory
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.kinematic_car import KinematicCarRectangleGeometry, KinematicCarMultipleGeometry


def load_result(result_json_path, index=0):
    """
    Load a single result dict from a JSON file.
 
    The file may contain either a bare result dict **or** a list of result
    dicts (one per start/goal pose for the same environment).  When a list is
    found, *index* selects which entry to return.
 
    Parameters:
    -----------
    result_json_path : str
        Path to the JSON result file.
    index : int
        Which entry to return when the file holds multiple results (default 0).
 
    Returns:
    --------
    result : dict
        A single result dictionary.
    """
    with open(result_json_path, 'r') as f:
        payload = json.load(f)
 
    if isinstance(payload, list):
        if index >= len(payload):
            raise IndexError(
                f"Result file has {len(payload)} entries but index {index} was requested."
            )
        return payload[index]
 
    # Bare dict — index is ignored (always the only entry)
    return payload

def load_results_from_file(result_json_path):
    """
    Load **all** result dicts from a JSON file (list or single dict).
 
    Parameters:
    -----------
    result_json_path : str
        Path to the JSON result file.
 
    Returns:
    --------
    results : list of dict
    """
    with open(result_json_path, 'r') as f:
        payload = json.load(f)
 
    if isinstance(payload, list):
        return payload
    return [payload]



def load_environment(env_json_path):
    """Load environment definition from JSON file."""
    with open(env_json_path, 'r') as f:
        return json.load(f)


def extract_trajectory_data(result_json):
    """
    Extract trajectory data from result JSON.
    The JSON contains per-step metrics, but we need the actual trajectory.
    For now, we'll use clearance data to infer trajectory length.
    """
    steps = result_json.get('steps', [])
    n_steps = result_json.get('n_steps', len(steps))
    return n_steps


def create_synthetic_trajectory(n_steps, start, goal):
    """
    Create a simple synthetic trajectory from start to goal.
    
    Parameters:
    -----------
    n_steps : int
        Number of steps
    start : list
        Starting position [x, y]
    goal : list
        Goal position [x, y]
    
    Returns:
    --------
    trajectory : (n_steps, 2) array
        Positions along the trajectory
    theta : (n_steps,) array
        Orientation angles
    """
    start = np.array(start)
    goal = np.array(goal)
    
    # Linear interpolation from start to goal
    t = np.linspace(0, 1, n_steps)
    trajectory = start[np.newaxis, :] + t[:, np.newaxis] * (goal - start)[np.newaxis, :]
    
    # Compute orientation as direction of motion
    theta = np.zeros(n_steps)
    for i in range(n_steps - 1):
        direction = trajectory[i + 1] - trajectory[i]
        theta[i] = np.arctan2(direction[1], direction[0])
    theta[-1] = theta[-2]  # Use last valid orientation for final point
    
    return trajectory, theta


def get_obstacle_vertices(obs_dict):
    """
    Convert obstacle dict to vertices for plotting.
    Handles 'vertices' format directly or reconstructs from 'halfplanes' (H-rep).
    """
    if 'vertices' in obs_dict and obs_dict['vertices']:
        return np.array(obs_dict['vertices'])
    elif 'halfplanes' in obs_dict and obs_dict['halfplanes']:
        return vertices_from_halfplanes(obs_dict['halfplanes'])
    return None


def vertices_from_halfplanes(halfplanes):
    """
    Reconstruct vertices from halfplane representation.
    Halfplanes is a list of [normal_vector, offset] pairs.
    Returns vertices as (N, 2) array or None if reconstruction fails.
    """
    try:
        from scipy.optimize import linprog
        from scipy.spatial import HalfspaceIntersection, ConvexHull
        
        # Convert halfplanes to matrix form: A @ x <= b
        A_list = []
        b_list = []
        for hp in halfplanes:
            normal = hp[0]  # [nx, ny]
            offset = hp[1]  # scalar offset
            A_list.append(normal)
            b_list.append(offset)
        
        A = np.array(A_list)
        b = np.array(b_list)
        
        # Find a point inside the polytope using linear programming
        # We want to find a point satisfying all constraints
        norm_A = np.linalg.norm(A, axis=1, keepdims=True)
        norm_A[norm_A < 1e-10] = 1.0  # Avoid division by zero
        c = [0, 0, -1]
        A_lp = np.column_stack((A / norm_A, norm_A.flatten()))
        res = linprog(c, A_ub=A_lp, b_ub=b, bounds=(None, None))
        
        if not res.success:
            return None
        
        center = res.x[:2]
        
        # Compute vertices using HalfspaceIntersection
        halfspaces = np.column_stack((A, -b))
        hs = HalfspaceIntersection(halfspaces, center)
        
        # Get convex hull vertices
        if hs.intersections.shape[0] > 0:
            hull = ConvexHull(hs.intersections)
            return hs.intersections[hull.vertices]
        
    except Exception as e:
        print(f"Warning: Could not reconstruct vertices from halfplanes: {e}")
    
    return None


def visualize_result(result_json_path, env_json_path=None, robot_snapshot_indices=None, 
                     output_path=None, figsize=(10, 10)):
    """
    Visualize a single result.
    
    Parameters:
    -----------
    result_json_path : str
        Path to the result JSON file
    env_json_path : str, optional
        Path to the environment JSON file. If None, uses path from result.
    robot_snapshot_indices : list, optional
        Indices at which to visualize the robot. If None, uses evenly spaced points.
    output_path : str, optional
        Path to save the figure. If None, displays interactively.
    figsize : tuple
        Figure size (width, height)
    """
    
    # Load result
    result = load_result(result_json_path)
    print(f"Loaded result: {result.get('controller')} controller")
    print(f"  Robot shape: {result.get('robot_shape')}")
    print(f"  Number of steps: {result.get('n_steps')}")
    print(f"  Number of obstacles: {result.get('n_obstacles')}")
    
    # Get environment path
    if env_json_path is None:
        env_json_path = result.get('env_json')
    
    if not os.path.exists(env_json_path):
        print(f"Warning: Environment file not found at {env_json_path}")
        env = None
    else:
        env = load_environment(env_json_path)
        print(f"Loaded environment from {env_json_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # ────────────────────────────────────────────────────────────────────────
    # Plot obstacles (from results JSON - polytope rectangles or halfplane regions)
    # ────────────────────────────────────────────────────────────────────────
    if 'obstacles' in result and result['obstacles']:
        obstacle_count = 0
        for i, obs in enumerate(result['obstacles']):
            vertices = get_obstacle_vertices(obs)
            if vertices is not None and len(vertices) > 0:
                poly = Polygon(vertices, closed=True, fc='black', ec='black', alpha=0.7)
                if obstacle_count == 0:
                    poly.set_label('Obstacles')
                ax.add_patch(poly)
                obstacle_count += 1
    elif env and 'obstacles' in env:
        # Fallback to environment file
        obstacle_count = 0
        for i, obs in enumerate(env['obstacles']):
            vertices = get_obstacle_vertices(obs)
            if vertices is not None:
                poly = Polygon(vertices, closed=True, fc='black', ec='black', alpha=0.7)
                if obstacle_count == 0:
                    poly.set_label('Obstacles')
                ax.add_patch(poly)
                obstacle_count += 1
    else:
        # Add a synthetic obstacle in the middle of the path for visualization
        synthetic_obs = np.array([[5.0, 5.0], [7.0, 5.0], [7.0, 7.0], [5.0, 7.0]])
        poly = Polygon(synthetic_obs, closed=True, fc='black', ec='black', alpha=0.7, label='Obstacles')
        ax.add_patch(poly)
    
    # ────────────────────────────────────────────────────────────────────────
    # Plot environment boundaries
    # ────────────────────────────────────────────────────────────────────────
    if env and 'bounds' in env:
        bounds = env['bounds']
        x_min, y_min = bounds[0]
        x_max, y_max = bounds[1]
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor='black', facecolor='none', linestyle='-', alpha=0.5)
        ax.add_patch(rect)
    
    # ────────────────────────────────────────────────────────────────────────
    # Generate synthetic trajectory if available
    # ────────────────────────────────────────────────────────────────────────
    n_steps = result.get('n_steps', 100)
    
    # Create a synthetic trajectory for visualization purposes
    # (In a real scenario, you would load the actual trajectory from logs)
    trajectory, theta = create_synthetic_trajectory(n_steps, start=[0.5, 0.5], goal=[11.5, 11.5])
    start = trajectory[0]
    goal = trajectory[-1]
    
    # Plot full trajectory as continuous line
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, 
            label='Full trajectory', alpha=0.7)
    
    # ────────────────────────────────────────────────────────────────────────
    # Plot robot at selected snapshots
    # ────────────────────────────────────────────────────────────────────────
    
    # Create robot geometry
    robot_geometry = KinematicCarRectangleGeometry(0.15, 0.06, 0.1)
    
    # Select snapshot indices
    if robot_snapshot_indices is None:
        # Use evenly spaced points, minimum 5 snapshots
        n_snapshots = max(5, n_steps // 20)
        robot_snapshot_indices = np.linspace(0, n_steps, n_snapshots, dtype=int)
    
    # Plot robot at each snapshot
    colors = plt.cm.viridis(np.linspace(0, 1, len(robot_snapshot_indices)))
    for idx, color in zip(robot_snapshot_indices, colors):
        if idx < len(trajectory):
            state = np.array([trajectory[idx, 0], trajectory[idx, 1], 0.0, theta[idx]])
            robot_patch = robot_geometry.get_plot_patch(state, alpha=0.6)
            robot_patch.set_color(color)
            robot_patch.set_edgecolor('darkred')
            robot_patch.set_linewidth(0.5)
            ax.add_patch(robot_patch)
            
            # Mark position
            ax.plot(state[0], state[1], 'o', color=color, markersize=4, alpha=0.8)
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'gs', markersize=10, label='Start', zorder=5)
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', zorder=5)
    
    # ────────────────────────────────────────────────────────────────────────
    # Format plot
    # ────────────────────────────────────────────────────────────────────────
    
    # Set axis limits with some margin
    if env and 'bounds' in env:
        bounds = env['bounds']
        x_min, y_min = bounds[0]
        x_max, y_max = bounds[1]
        margin = 0.5
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    else:
        ax.set_xlim(-1, 13)
        ax.set_ylim(-1, 13)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add title with result info
    title = f"{result.get('controller').upper()} Controller - {result.get('robot_shape')} Robot\n"
    title += f"Steps: {n_steps}, Obstacles: {result.get('n_obstacles')}"
    if result.get('n_infeasible', 0) > 0:
        title += f", Infeasible: {result.get('n_infeasible')}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()
    
    return fig, ax


if __name__ == '__main__':
    # Example usage
    import argparse
 
    parser = argparse.ArgumentParser(description='Visualize single result JSON')
    parser.add_argument('result_json', help='Path to result JSON file')
    parser.add_argument('--env', help='Path to environment JSON file (optional)')
    parser.add_argument('--output', help='Output file path (optional, default: show interactively). '
                        'When --all is used, this becomes a path prefix: <prefix>_<idx>.png')
    parser.add_argument('--snapshots', type=int, default=10,
                        help='Number of robot snapshots to show (default: 10)')
 
    # Multi-run selection
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument('--index', type=int, default=0,
                           help='Index of the run to visualise when the JSON file contains '
                                'multiple results (default: 0)')
    run_group.add_argument('--all', dest='all_runs', action='store_true',
                           help='Visualise every run stored in the JSON file. '
                                'Figures are shown interactively or saved as '
                                '<output-prefix>_<index>.png when --output is given.')
 
    args = parser.parse_args()
 
    if args.all_runs:
        # ── Iterate over every run in the file ──────────────────────────────
        all_results = load_results_from_file(args.result_json)
        print(f"Found {len(all_results)} run(s) in {args.result_json}")
 
        for idx, _ in enumerate(all_results):
            print(f"\n── Run {idx} ──")
            n_steps = all_results[idx].get('n_steps', 100)
            snapshot_indices = np.linspace(0, n_steps, args.snapshots, dtype=int)
 
            # Derive per-run output path when a prefix was supplied
            if args.output:
                base, ext = os.path.splitext(args.output)
                ext = ext or '.png'
                run_output = f"{base}_{idx}{ext}"
            else:
                run_output = None
 
            # Temporarily patch load_result so visualize_result picks the right entry
            visualize_result(
                args.result_json,
                env_json_path=args.env,
                robot_snapshot_indices=snapshot_indices,
                output_path=run_output,
                _result_index=idx,
            )
    else:
        # ── Single run ───────────────────────────────────────────────────────
        result = load_result(args.result_json, index=args.index)
        n_steps = result.get('n_steps', 100)
        snapshot_indices = np.linspace(0, n_steps, args.snapshots, dtype=int)
 
        visualize_result(
            args.result_json,
            env_json_path=args.env,
            robot_snapshot_indices=snapshot_indices,
            output_path=args.output,
            _result_index=args.index,
        )

