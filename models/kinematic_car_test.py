import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import statistics as st

import sys
import os

# Add the parent directory (cbf) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from control.dcbf_optimizer import NmpcDcbfOptimizerParam
from control.dcbf_controller import NmpcDcbfController
from control.lse_optimizer import NmpcLseOptimizer
from control.lse_controller import NmpcLseController
from models.dubin_car import (
    DubinCarDynamics,
    DubinCarGeometry,
    DubinCarStates,
    DubinCarSystem,
)
from models.geometry_utils import *
from models.kinematic_car import (
    KinematicCarDynamics,
    KinematicCarRectangleGeometry,
    KinematicCarMultipleGeometry,
    KinematicCarTriangleGeometry,
    KinematicCarPentagonGeometry,
    KinematicCarStates,
    KinematicCarSystem,
)
from planning.path_generator.search_path_generator import (
    AstarLoSPathGenerator,
    AstarPathGenerator,
    ThetaStarPathGenerator,
)
from planning.trajectory_generator.constant_speed_generator import (
    ConstantSpeedTrajectoryGenerator,
)
from sim.simulation import Robot, SingleAgentSimulation

def read_json(dir, file):
    import json
    with open(dir + file, 'r') as f:
        data = json.load(f)
    return data


def plot_world(simulation, snapshot_indexes, figure_name="world", local_traj_indexes=[], maze_type=None):
    # TODO: make this plotting function general applicable to different systems
    if maze_type == "maze":
        fig, ax = plt.subplots(figsize=(8.3, 5.0))
    elif maze_type == "oblique_maze":
        fig, ax = plt.subplots(figsize=(6.7, 5.0))
    # degrees_rot = 90
    # transform = mpl.transforms.Affine2D().rotate_deg(degrees_rot) + ax.transData
    # extract data
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    local_paths = simulation._robot._local_planner_logger._trajs
    optimized_trajs = simulation._robot._controller_logger._xtrajs
    # plot robot
    for index in snapshot_indexes:
        for i in range(simulation._robot._system._geometry._num_geometry):
            polygon_patch = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i, 0.25)
            # polygon_patch.set_transform(transform)
            ax.add_patch(polygon_patch)
    # plot global reference
    ax.plot(global_path[:, 0], global_path[:, 1], "o--", color="grey", linewidth=1.5, markersize=2)
    # plot closed loop trajectory
    # ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=1, markersize=4, transform=transform)
    # plot obstacles
    for obs in simulation._obstacles:
        obs_patch = obs.get_plot_patch()
        # obs_patch.set_transform(transform)
        ax.add_patch(obs_patch)
    # plot local reference and local optimized trajectories
    for index in local_traj_indexes:
        local_path = local_paths[index]
        ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)
        optimized_traj = optimized_trajs[index]
        ax.plot(
            optimized_traj[:, 0],
            optimized_traj[:, 1],
            "-",
            color="gold",
            linewidth=3,
            markersize=4,
        )
    # set figure properties
    plt.tight_layout()
    # save figure
    plt.savefig("figures/" + figure_name + ".eps", format="eps", dpi=500, pad_inches=0)
    plt.savefig("figures/" + figure_name + ".png", format="png", dpi=500, pad_inches=0)


def animate_world(simulation, animation_name="world", maze_type=None):
    # TODO: make this plotting function general applicable to different systems
    if maze_type == "maze":
        fig, ax = plt.subplots(figsize=(8.3, 5.0))
    elif maze_type == "oblique_maze":
        fig, ax = plt.subplots(figsize=(6.7, 5.0))
    global_paths = simulation._robot._global_planner_logger._paths
    global_path = global_paths[0]
    ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1.5, markersize=4)

    local_paths = simulation._robot._local_planner_logger._trajs
    local_path = local_paths[0]
    (reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1], "-", color="blue", linewidth=3, markersize=4)

    optimized_trajs = simulation._robot._controller_logger._xtrajs
    optimized_traj = optimized_trajs[0]
    (optimized_traj_line,) = ax.plot(
        optimized_traj[:, 0],
        optimized_traj[:, 1],
        "-",
        color="gold",
        linewidth=3,
        markersize=4,
    )

    closedloop_traj = np.vstack(simulation._robot._system_logger._xs)
    for obs in simulation._obstacles:
        obs_patch = obs.get_plot_patch()
        ax.add_patch(obs_patch)

    robot_patch = []
    for i in range(simulation._robot._system._geometry._num_geometry):
        robot_patch.append(patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="tab:brown"))
        ax.add_patch(robot_patch[i])
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.tight_layout()

    def update(index):
        local_path = local_paths[index]
        reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
        optimized_traj = optimized_trajs[index]
        optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
        # plt.xlabel(str(index))
        for i in range(simulation._robot._system._geometry._num_geometry):
            polygon_patch_next = simulation._robot._system._geometry.get_plot_patch(closedloop_traj[index, :], i)
            robot_patch[i].set_xy(polygon_patch_next.get_xy())
        if index == len(closedloop_traj) - 1:
            ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=3, markersize=4)

    anim = animation.FuncAnimation(fig, update, frames=len(closedloop_traj), interval=1000 * 0.1)
    anim.save("animations/" + animation_name + ".mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=10))


def kinematic_car_triangle_simulation_test():
    start_pos, goal_pos, grid, obstacles = create_env("maze")
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=KinematicCarTriangleGeometry(np.array([[0.14, 0.00], [-0.03, 0.05], [-0.03, -0.05]])),
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.07
    robot.set_global_planner(ThetaStarPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=NmpcDcbfOptimizerParam))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(20.0)
    plot_world(sim, np.arange(0, 200, 5), figure_name="triangle")
    animate_world(sim, animation_name="triangle")


def kinematic_car_pentagon_simulation_test():
    start_pos, goal_pos, grid, obstacles = create_env("maze")
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=KinematicCarPentagonGeometry(
                np.array([[0.15, 0.00], [0.03, 0.05], [-0.01, 0.02], [-0.01, -0.02], [0.03, -0.05]])
            ),
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.06
    robot.set_global_planner(ThetaStarPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=NmpcDcbfOptimizerParam()))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(20.0)
    plot_world(sim, np.arange(0, 200, 5), figure_name="pentagon")
    animate_world(sim, animation_name="pentagon")


def kinematic_car_all_shapes_simulation_test(maze_type, robot_shape, controller_type, polytopes = []):

    # New code that get environments from input
    start_pos, goal_pos, grid, obstacles = create_env_benchmark(polytopes, scale_factor=0.1)

    # Old code with hard-coded environments
    # start_pos, goal_pos, grid, obstacles = create_env(maze_type)
    geometry_regions = KinematicCarMultipleGeometry()

    if robot_shape == "rectangle":
        geometry_regions.add_geometry(KinematicCarRectangleGeometry(0.15, 0.06, 0.1))
        if maze_type == "maze":
            robot_indexes = [0, 17, 24, 33, 39, 48, 58, 66, 76, 86, 92, 102, 112, 129]
            traj_indexes = [0, 24, 33, 48, 66, 92, 112]
        elif maze_type == "oblique_maze":
            robot_indexes = [1, 9, 22, 26, 34, 41, 47, 56, 64, 70, 79, 88, 93, 124, 131, 143, 164]
            traj_indexes = [1, 19, 47, 70, 93]
    if robot_shape == "pentagon":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                np.array([[0.15, 0.00], [0.03, 0.05], [-0.01, 0.02], [-0.01, -0.02], [0.03, -0.05]])
            )
        )
        if maze_type == "maze":
            robot_indexes = [2, 15, 33, 43, 50, 61, 69, 81, 99, 111, 125, 130, 142, 164]
            traj_indexes = [2, 33, 50, 69, 99, 125, 164]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 11, 23, 41, 53, 62, 69, 78, 83, 92, 110, 113, 138, 151, 164, 185]
            traj_indexes = [0, 23, 62, 78, 92, 110, 151]
    if robot_shape == "triangle":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.75 * np.array([[0.14, 0.00], [-0.03, 0.05], [-0.03, -0.05]]))
        )
        if maze_type == "maze":
            robot_indexes = [0, 10, 18, 27, 35, 40, 46, 51, 61, 72, 86, 91, 102, 113, 199]
            traj_indexes = [0, 18, 35, 46, 61, 86, 102]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 11, 17, 26, 32, 43, 50, 58, 74, 78, 110, 123, 168, 182, 199]
            traj_indexes = [0, 17, 32, 50, 74, 168]
    if robot_shape == "lshape":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.4 * np.array([[0, 0.1], [0.02, 0.08], [-0.2, -0.1], [-0.22, -0.08]]))
        )
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(0.4 * np.array([[0, 0.1], [-0.02, 0.08], [0.2, -0.1], [0.22, -0.08]]))
        )
        if maze_type == "maze":
            robot_indexes = [3, 13, 27, 36, 46, 56, 65, 74, 85, 98, 114, 121, 224, 299]
            traj_indexes = [3, 27, 46, 74, 114, 224]
        elif maze_type == "oblique_maze":
            robot_indexes = [0, 12, 21, 32, 40, 49, 59, 67, 81, 94, 122, 129, 141, 177]
            traj_indexes = [0, 21, 40, 59, 81, 129]
    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=geometry_regions,
            dynamics=KinematicCarDynamics(),
        )
    )
    global_path_margin = 0.45 #Should be dimensions of robot to give local planner more margin - 0.05
    robot.set_global_planner(AstarLoSPathGenerator(grid, quad=False, margin=global_path_margin))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
    opt_param = NmpcDcbfOptimizerParam()
    # TODO: Wrap these parameters
    if robot_shape == "rectangle" or robot_shape == "lshape":
        opt_param.terminal_weight = 10.0
    elif robot_shape == "pentagon":
        if maze_type == 1:
            opt_param.terminal_weight = 2.0
        elif maze_type == 2:
            opt_param.terminal_weight = 5.0
    elif robot_shape == "triangle":
        opt_param.terminal_weight = 2.0
    if controller_type == "dcbf":
        robot.set_controller(NmpcDcbfController(dynamics=KinematicCarDynamics(), opt_param=opt_param))
    elif controller_type == "pipcbf":
        robot.set_controller(NmpcLseController(dynamics=KinematicCarDynamics(), opt_param=opt_param))
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)
    sim.run_navigation(100.0)
    print("median: ", st.median(robot._controller._optimizer.solver_times))
    print("std: ", st.stdev(robot._controller._optimizer.solver_times))
    print("min: ", min(robot._controller._optimizer.solver_times))
    print("max: ", max(robot._controller._optimizer.solver_times))
    print("Simulation finished.")
    name = robot_shape + "_" + maze_type + "_" + controller_type
    plot_world(sim, robot_indexes, figure_name=name, local_traj_indexes=traj_indexes, maze_type=maze_type)
    animate_world(sim, animation_name=name, maze_type=maze_type)

def create_env_benchmark(polytopes, scale_factor=1.0):
    if polytopes:
        start = np.array([0.0, 0.0, 0.0])
        # Goal position must be strictly INSIDE the bounds, not on the boundary
        goal = np.array([11.9, 11.9])

        # Create grid in the format expected by GridMap: (bounds, cell_size)
        # bounds should be ((x_min, y_min), (x_max, y_max))
        # Note: positions must be strictly within bounds (not on boundary)
        bounds = ((0.0, 0.0), (12.0, 12.0))
        cell_size = 0.05
        grid = (bounds, cell_size)

        obstacles = []
        for poly in polytopes:
            mat_A = []
            vec_b = []
            for hp in poly:
                mat_A.append(hp[0:2])
                vec_b.append(hp[2])
            obstacles.append(PolytopeRegion(mat_A=np.array(mat_A), vec_b=np.array(vec_b)))
        return start, goal, grid, obstacles

def create_env(env_type):
    if env_type == "s_path":
        s = 1.0  # scale of environment
        start = np.array([0.0 * s, 0.2 * s, 0.0])
        goal = np.array([1.0 * s, 0.8 * s])
        bounds = ((-0.2 * s, 0.0 * s), (1.2 * s, 1.2 * s))
        cell_size = 0.05 * s
        grid = (bounds, cell_size)
        obstacles = []
        obstacles.append(RectangleRegion(0.0 * s, 1.0 * s, 0.9 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 0.4 * s, 0.4 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.6 * s, 1.0 * s, 0.0 * s, 0.7 * s))
        return start, goal, grid, obstacles
    elif env_type == "maze":
        s = 0.15  # scale of environment
        start = np.array([0.5 * s, 5.5 * s, -math.pi / 2.0])
        goal = np.array([12.5 * s, 0.5 * s])
        bounds = ((0.0 * s, 0.0 * s), (13.0 * s, 6.0 * s))
        cell_size = 0.25 * s
        grid = (bounds, cell_size)
        obstacles = []
        obstacles.append(RectangleRegion(0.0 * s, 3.0 * s, 0.0 * s, 3.0 * s))
        obstacles.append(RectangleRegion(1.0 * s, 2.0 * s, 4.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(2.0 * s, 6.0 * s, 5.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(6.0 * s, 7.0 * s, 4.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(4.0 * s, 5.0 * s, 0.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(5.0 * s, 7.0 * s, 2.0 * s, 3.0 * s))
        obstacles.append(RectangleRegion(6.0 * s, 9.0 * s, 1.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(8.0 * s, 9.0 * s, 2.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(9.0 * s, 12.0 * s, 3.0 * s, 4.0 * s))
        obstacles.append(RectangleRegion(11.0 * s, 12.0 * s, 4.0 * s, 5.0 * s))
        obstacles.append(RectangleRegion(8.0 * s, 10.0 * s, 5.0 * s, 6.0 * s))
        obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(12.0 * s, 13.0 * s, 1.0 * s, 2.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, 6.0 * s, 7.0 * s))
        obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, -1.0 * s, 7.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 13.0 * s, -1.0 * s, 0.0 * s))
        obstacles.append(RectangleRegion(13.0 * s, 14.0 * s, -1.0 * s, 7.0 * s))
        return start, goal, grid, obstacles
    elif env_type == "oblique_maze":
        s = 0.15  # scale of environment
        start = np.array([1.0 * s, 1.5 * s, 0.0])
        goal = np.array([8.5 * s, 6.5 * s])
        bounds = ((0.0 * s, 1.0 * s), (10.0 * s, 7.0 * s))
        cell_size = 0.2 * s
        grid = (bounds, cell_size)
        obstacles = []
        # TODO: Overload the constructor of RectangleRegion() 
        obstacles.append(RectangleRegion(-1.0 * s, 0.0 * s, 0.0 * s, 8.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 10.0 * s, 0.0 * s, 1.0 * s))
        obstacles.append(RectangleRegion(0.0 * s, 8.0 * s, 7.0 * s, 8.0 * s))
        obstacles.append(RectangleRegion(10.0 * s, 11.0 * s, 0.0 * s, 8.0 * s))
        obstacles.append(
            PolytopeRegion.convex_hull(s * np.array([[0.0, 2.0], [1.25, 3.875], [2.875, 3.125], [2.5, 2.25]]))
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[1, 4.75], [0.0, 5.0], [0.875, 7], [1.875, 6.375]])))
        obstacles.append(
            PolytopeRegion.convex_hull(s * np.array([[2.75, 1], [4.2, 3.25], [5.125, 3.75], [6.625, 2.5], [6.5, 1.0]]))
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.0, 7.0], [6, 6], [6.5, 7.0]])))
        obstacles.append(
            PolytopeRegion.convex_hull(
                s * np.array([[2.375, 4.875], [2.875, 5.875], [4.5, 5.875], [4.75, 4], [3.375, 4]])
            )
        )
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[6.75, 1.0], [7.25, 2.375], [8.5, 2.0], [8.5, 1.0]])))
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[8.625, 1.0], [10.0, 2.5], [10.0, 1.0]])))
        obstacles.append(PolytopeRegion.convex_hull(s * np.array([[10.0, 2.875], [9.5, 5.75], [10.0, 5.875]])))
        obstacles.append(
            PolytopeRegion.convex_hull(
                s * np.array([[8.875, 3.125], [8.0, 5.5], [6.875, 6.375], [5.875, 5.875], [6.25, 4.375], [7.125, 3.5]])
            )
        )
        return start, goal, grid, obstacles



# ══════════════════════════════════════════════════════════════════════════════
# §  BENCHMARK RUNNER  —  scalability experiments across saved environments
# ══════════════════════════════════════════════════════════════════════════════

import json as _json
import statistics as _st

# ── folder conventions (mirror claude_polytope_env.py) ────────────────────────
#   environments : ./envs/n{N}/env{I}.json
#   results      : ./results/n{N}/env{I}_{controller}.json
# ENVS_ROOT    = "/home/ttamr/Documents/embeddedcbf/benchmark/envs"
ENVS_ROOT = "/home/u0110021/git_repos/embeddedcbf/benchmark/envs"
RESULTS_ROOT = "./results"


def _results_path(n_obs: int, env_idx: int, controller: str) -> str:
    folder = os.path.join(RESULTS_ROOT, f"n{n_obs}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"env{env_idx}_{controller}.json")


def _collect_problem_size(optimizer) -> dict:
    """Read problem-size fields from an optimizer after its first solve."""
    return {
        "n_variables":    getattr(optimizer, 'nr_variables',   None),
        "n_constraints":  getattr(optimizer, 'nr_constraints', None),
        "n_eq":           getattr(optimizer, '_n_eq',          None),
        "n_ineq":         getattr(optimizer, '_n_ineq',        None),
    }


def run_benchmark_env(
    env_json_path: str,
    n_obs: int,
    env_idx: int,
    robot_shape: str    = "rectangle",
    controller_type: str = "dcbf",
    enable_vis: bool    = True,
):
    """
    Run one environment for one controller and write a results JSON.

    Parameters
    ----------
    env_json_path   : path to the environment JSON produced by claude_polytope_env.py
    n_obs           : number of obstacles (used only for the output path)
    env_idx         : environment index within its n-obstacle folder
    robot_shape     : "rectangle" | "pentagon" | "triangle" | "lshape"
    controller_type : "dcbf" | "pipcbf"
    enable_vis      : set False for headless / fast scalability runs
    """
    out_path = _results_path(n_obs, env_idx, controller_type)
    if os.path.exists(out_path):
        print(f"  [skip] results already exist: {out_path}")
        return
    
    # ── load environment ──────────────────────────────────────────────────
    with open(env_json_path) as f:
        env_data = _json.load(f)
    polytopes = env_data['halfplanes']

    start_pos, goal_pos, grid, obstacles = create_env_benchmark(
        polytopes, scale_factor=0.1)

    # ── build robot geometry ──────────────────────────────────────────────
    geometry_regions = KinematicCarMultipleGeometry()
    if robot_shape == "rectangle":
        geometry_regions.add_geometry(KinematicCarRectangleGeometry(0.9, 0.45, 0.1))
    elif robot_shape == "pentagon":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                np.array([[0.15,0.00],[0.03,0.05],[-0.01,0.02],
                          [-0.01,-0.02],[0.03,-0.05]])
            )
        )
    elif robot_shape == "triangle":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                0.75 * np.array([[0.14,0.00],[-0.03,0.05],[-0.03,-0.05]])
            )
        )
    elif robot_shape == "lshape":
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                0.4 * np.array([[0,0.1],[0.02,0.08],[-0.2,-0.1],[-0.22,-0.08]])
            )
        )
        geometry_regions.add_geometry(
            KinematicCarTriangleGeometry(
                0.4 * np.array([[0,0.1],[-0.02,0.08],[0.2,-0.1],[0.22,-0.08]])
            )
        )

    robot = Robot(
        KinematicCarSystem(
            state=KinematicCarStates(
                x=np.block([start_pos[:2], np.array([0.0, start_pos[2]])])),
            geometry=geometry_regions,
            dynamics=KinematicCarDynamics(),
        )
    )

    robot.set_global_planner(
        AstarLoSPathGenerator(grid, quad=False, margin=0.05))
    robot.set_local_planner(ConstantSpeedTrajectoryGenerator())

    opt_param = NmpcDcbfOptimizerParam()
    if robot_shape in ("rectangle", "lshape"):
        opt_param.terminal_weight = 10.0
    elif robot_shape == "triangle":
        opt_param.terminal_weight = 2.0
    elif robot_shape == "pentagon":
        opt_param.terminal_weight = 5.0

    if controller_type == "dcbf":
        controller = NmpcDcbfController(
            dynamics=KinematicCarDynamics(),
            opt_param=opt_param,
            enable_vis=enable_vis,
        )
    elif controller_type == "pipcbf":
        controller = NmpcLseController(
            dynamics=KinematicCarDynamics(),
            opt_param=opt_param,
            enable_vis=enable_vis,
        )
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")

    robot.set_controller(controller)

    # ── run simulation, collecting per-step metrics ───────────────────────
    sim = SingleAgentSimulation(robot, obstacles, goal_pos)

    # Monkey-patch the simulation step to intercept each control call.
    # We store per-step metric dicts here so we don't need to touch sim.py.
    step_metrics: list = []
    _orig_gen = controller.generate_control_input

    def _patched_gen(system, global_path, local_traj, obs):
        u = _orig_gen(system, global_path, local_traj, obs)
        m = controller.collect_metrics(system, obs)
        step_metrics.append(m)
        return u

    controller.generate_control_input = _patched_gen

    sim.run_navigation(20.0)

    controller._create_gif()

    # ── aggregate results ─────────────────────────────────────────────────
    opt = controller._optimizer

    comp_times   = [m["comp_time_s"]   for m in step_metrics if m["comp_time_s"]   is not None]
    feval_times  = [m["feval_time_s"]  for m in step_metrics if m["feval_time_s"]  is not None]
    kkt_times    = [m["kkt_time_s"]    for m in step_metrics if m["kkt_time_s"]    is not None]
    iters        = [m["iterations"]    for m in step_metrics if m["iterations"]    is not None]
    min_clears   = [m["min_clearance"] for m in step_metrics if m["min_clearance"] is not None]
    n_vars_list  = [m["n_variables"]   for m in step_metrics if m["n_variables"]   is not None]
    n_eq_list    = [m["n_eq"]          for m in step_metrics if m["n_eq"]          is not None]
    n_ineq_list  = [m["n_ineq"]        for m in step_metrics if m["n_ineq"]        is not None]
    n_infeasible = sum(1 for m in step_metrics if m.get("infeasible"))
    n_steps      = len(step_metrics)

    def _safe_stats(vals):
        if not vals:
            return {"median": None, "std": None, "min": None, "max": None}
        return {
            "median": _st.median(vals),
            "std":    _st.stdev(vals) if len(vals) > 1 else 0.0,
            "min":    min(vals),
            "max":    max(vals),
        }

    # ── convert obstacles to serializable format ──────────────────────────
    obstacles_data = []
    for obs in obstacles:
        if hasattr(obs, 'vertices') and obs.vertices is not None:
            obstacles_data.append({"vertices": obs.vertices.tolist()})
        elif hasattr(obs, 'x_min'):
            # RectangleRegion
            verts = [
                [obs.x_min, obs.y_min],
                [obs.x_max, obs.y_min],
                [obs.x_max, obs.y_max],
                [obs.x_min, obs.y_max],
            ]
            obstacles_data.append({"vertices": verts})
        elif hasattr(obs, 'get_convex_rep'):
            # PolytopeRegion - try to get vertices
            try:
                A, b = obs.get_convex_rep()
                # For now, store as halfplanes
                obstacles_data.append({
                    "halfplanes": [[A[i].tolist(), float(b[i])] for i in range(len(A))]
                })
            except:
                obstacles_data.append({"type": "polytope"})

    results = {
        # ── identification ────────────────────────────────────────────────
        "env_json":       env_json_path,
        "n_obstacles":    n_obs,
        "env_idx":        env_idx,
        "controller":     controller_type,
        "robot_shape":    robot_shape,
        # ── problem size (constant per env) ───────────────────────────────
        "problem_size": {
            "n_variables":   getattr(opt, 'nr_variables',   None),
            "n_constraints": getattr(opt, 'nr_constraints', None),
        },
        # ── eq/ineq counts per step (may vary if active obstacle set changes)
        "n_variables_steps": n_vars_list,
        "n_eq_steps":    n_eq_list,
        "n_ineq_steps":  n_ineq_list,
        # ── obstacle data ─────────────────────────────────────────────────
        "obstacles":     obstacles_data,
        # ── per-step timing ───────────────────────────────────────────────
        "comp_time_s":    _safe_stats(comp_times),
        "feval_time_s":   _safe_stats(feval_times),
        "kkt_time_s":     _safe_stats(kkt_times),
        # ── solver behaviour ──────────────────────────────────────────────
        "iterations":     _safe_stats(iters),
        "n_steps":        n_steps,
        "n_infeasible":   n_infeasible,
        "infeasibility_rate": n_infeasible / n_steps if n_steps > 0 else None,
        # ── safety (clearance to obstacles) ──────────────────────────────
        "min_clearance":  _safe_stats(min_clears),
        # ── full per-step log (for detailed analysis) ────────────────────
        "steps": step_metrics,
    }

    with open(out_path, "w") as f:
        _json.dump(results, f, indent=2)

    print(f"\n  Results saved → {out_path}")
    print(f"  Steps: {n_steps}  |  Infeasible: {n_infeasible}"
          f"  |  Median solve: "
          f"{results['comp_time_s']['median']:.4f}s"
          if results['comp_time_s']['median'] is not None else "")
    return results


def run_scalability_benchmark(
    min_obs: int        = 1,
    max_obs: int        = 15,
    envs_per_count: int = 10,
    robot_shape: str    = "rectangle",
    controllers: list   = None,
    enable_vis: bool    = True,   # False = fast headless mode
):
    """
    Run the full scalability benchmark across all saved environments.

    Parameters
    ----------
    min_obs / max_obs   : obstacle-count range to sweep
    envs_per_count      : how many environments per obstacle count to run
    robot_shape         : robot shape to use
    controllers         : list of controller strings, default ["dcbf","pipcbf"]
    enable_vis          : set True to show live matplotlib windows
    """
    if controllers is None:
        controllers = ["dcbf", "pipcbf"]

    print("\n══════════════════════════════════════════════════════════════")
    print(f"  Scalability Benchmark  —  obstacles {min_obs}–{max_obs}")
    print(f"  Controllers : {controllers}")
    print(f"  Robot shape : {robot_shape}")
    print(f"  Visualisation: {'ON' if enable_vis else 'OFF (headless)'}")
    print("══════════════════════════════════════════════════════════════\n")

    for n_obs in range(min_obs, max_obs + 1):
        env_folder = os.path.join(ENVS_ROOT, f"n{n_obs}")
        print(env_folder)
        if not os.path.isdir(env_folder):
            print(f"  [n={n_obs}] No environment folder found, skipping.")
            continue

        # Collect available env JSON files, sorted by index
        import re as _re
        pat = _re.compile(r'^env(\d+)\.json$')
        env_files = sorted(
            [(int(m.group(1)), f)
             for f in os.listdir(env_folder)
             if (m := pat.match(f))],
            key=lambda t: t[0]
        )[:envs_per_count]

        if not env_files:
            print(f"  [n={n_obs}] No env files found, skipping.")
            continue

        print(f"\n── n={n_obs} obstacles  ({len(env_files)} envs) ─────────────")
        for env_idx, fname in env_files:
            env_path = os.path.join(env_folder, fname)
            for ctrl in controllers:
                print(f"  Running env{env_idx}  controller={ctrl} …")
                try:
                    run_benchmark_env(
                        env_json_path=env_path,
                        n_obs=n_obs,
                        env_idx=env_idx,
                        robot_shape=robot_shape,
                        controller_type=ctrl,
                        enable_vis=enable_vis,
                    )
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] env{env_idx} {ctrl}: {e}")
                    print(traceback.print_exc())


if __name__ == "__main__":
    # ── single-env quick test (matches original usage) ────────────────────
    # dir  = '/home/ttamr/Documents/embeddedcbf/benchmark/envs/'
    # file = 'env0.json'
    # env_data = read_json(dir, file)
    # polytopes = env_data['halfplanes']
    # kinematic_car_all_shapes_simulation_test("oblique_maze", "rectangle", "dcbf", polytopes)

    # ── scalability benchmark ─────────────────────────────────────────────
    run_scalability_benchmark(
        min_obs     = 8,
        max_obs     = 8,
        envs_per_count = 1, #10
        robot_shape = "rectangle",
        controllers = ["dcbf"], #["dcbf", "pipcbf"],
        enable_vis  = True,   # <── set True to re-enable live plots
    )

# export PYTHONPATH=$PWD:$PYTHONPATH
