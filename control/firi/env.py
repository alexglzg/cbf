"""
env.py – FIRIEnv: High-level environment wrapper
=================================================

Integrates ``FIRISolver``, ``RobotConfig``, ``build_seed``, and
``build_bbox`` into a single object that mirrors the full ROS node
workflow — but without any ROS dependency.

Supports both:
  • Point-cloud obstacles  (voxel-filtered LiDAR points)
  • Polytope obstacles     (vertex lists, e.g. from firi_polytope_node)

Also provides an optional spatial voxel filter matching the one in
firi_node_sdmn.cpp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .solver import FIRISolver, FIRIResult, HalfPlane
from .robot  import RobotConfig, build_seed, build_bbox


# ── Voxel filter ──────────────────────────────────────────────────────────────

def voxel_filter(points: List[np.ndarray], resolution: float) -> List[np.ndarray]:
    """
    Spatial voxel (grid) downsampling of 2-D obstacle points.

    Ported from the ``voxelFilter`` function in firi_node_sdmn.cpp.
    One representative point is kept per grid cell; the first point
    that falls into a cell is retained.

    Parameters
    ----------
    points     : list of np.ndarray shape (2,)
    resolution : voxel cell side length (m)

    Returns
    -------
    List of filtered points (subset of *points*).
    """
    if not points:
        return []
    seen: Dict[Tuple[int, int], bool] = {}
    filtered: List[np.ndarray] = []
    inv = 1.0 / resolution
    for p in points:
        key = (int(np.floor(p[0] * inv)), int(np.floor(p[1] * inv)))
        if key not in seen:
            seen[key] = True
            filtered.append(p)
    return filtered


# ── FIRIEnv ───────────────────────────────────────────────────────────────────

class FIRIEnv:
    """
    High-level FIRI environment.

    Wraps :class:`~firi.solver.FIRISolver` and handles:
    - voxel filtering of point-cloud obstacles
    - seed polygon construction (symmetric or rear-axle)
    - heading-aligned bounding box construction
    - calling the solver

    Parameters
    ----------
    robot_cfg        : :class:`~firi.robot.RobotConfig`
    max_firi_iter    : outer FIRI loop iteration cap (default 10)
    convergence_rho  : volume-improvement convergence threshold (default 0.02)
    voxel_size       : downsampling resolution for point clouds (default 0.1 m)
                       Set to 0 or None to disable filtering.
    rear_axle_mode   : use rear-axle seed (True) or symmetric seed (False)
    sdmn_seed        : RNG seed for SDMN solver
    """

    def __init__(
        self,
        robot_cfg       : Optional[RobotConfig] = None,
        max_firi_iter   : int   = 10,
        convergence_rho : float = 0.02,
        voxel_size      : float = 0.1,
        rear_axle_mode  : bool  = True,
        sdmn_seed       : int   = 42,
    ):
        self.robot_cfg       = robot_cfg or RobotConfig()
        self.max_iter        = max_firi_iter
        self.rho             = convergence_rho
        self.voxel_size      = voxel_size
        self.rear_axle_mode  = rear_axle_mode
        self._solver         = FIRISolver(sdmn_seed=sdmn_seed)

    # ── Point-cloud interface (firi_node_sdmn style) ──────────────────

    def update_pointcloud(
        self,
        robot_pos : np.ndarray,    # shape (2,)
        robot_yaw : float,
        raw_points: List[np.ndarray],
    ) -> FIRIResult:
        """
        Run FIRI given a raw 2-D point cloud.

        Applies voxel filtering, builds seed + bbox, and returns the
        free-space polytope.

        Parameters
        ----------
        robot_pos  : position in the map frame
        robot_yaw  : heading (radians)
        raw_points : obstacle points in the map frame

        Returns
        -------
        FIRIResult
        """
        if self.voxel_size and self.voxel_size > 0:
            obstacles = voxel_filter(raw_points, self.voxel_size)
        else:
            obstacles = list(raw_points)

        seed  = build_seed(robot_pos, robot_yaw, self.robot_cfg,
                           self.rear_axle_mode)
        bbox  = build_bbox(robot_pos, robot_yaw, self.robot_cfg)

        return self._solver.compute(
            obstacles    = obstacles,
            seed_vertices= seed,
            bbox_planes  = bbox,
            max_iter     = self.max_iter,
            rho          = self.rho,
            polytope_mode= False,
        )

    # ── Polytope interface (firi_polytope_node style) ─────────────────

    def update_polytopes(
        self,
        robot_pos : np.ndarray,
        robot_yaw : float,
        polytopes : List[List[np.ndarray]],
    ) -> FIRIResult:
        """
        Run FIRI given a list of convex polytope obstacles (vertex lists).

        Matches the workflow from firi_polytope_node.cpp.

        Parameters
        ----------
        robot_pos : position in the map frame (rear-axle if rear_axle_mode)
        robot_yaw : heading (radians)
        polytopes : list of obstacle polygons, each a list of 2-D vertices

        Returns
        -------
        FIRIResult
        """
        # Filter empty polytopes
        clean = [poly for poly in polytopes if len(poly) >= 1]

        seed = build_seed(robot_pos, robot_yaw, self.robot_cfg,
                          self.rear_axle_mode)
        bbox = build_bbox(robot_pos, robot_yaw, self.robot_cfg)

        return self._solver.compute(
            obstacles    = clean,
            seed_vertices= seed,
            bbox_planes  = bbox,
            max_iter     = self.max_iter,
            rho          = self.rho,
            polytope_mode= True,
        )
    
    def update_halfplane_polytopes(
        self,
        robot_pos          : np.ndarray,
        robot_yaw          : float,
        obstacle_halfplanes: list,   # your list-of-lists-of-lists
    ) -> FIRIResult:
        """
        Run FIRI with obstacles given as halfplane lists.

        obstacle_halfplanes : List[  List[  (normal, offset)  ]  ]
                            outer = one per obstacle
                            inner = halfplanes of that obstacle
        """
        seed = build_seed(robot_pos, robot_yaw, self.robot_cfg,
                        self.rear_axle_mode)
        bbox = build_bbox(robot_pos, robot_yaw, self.robot_cfg)

        return self._solver.compute_from_halfplanes(
            obstacle_halfplanes = obstacle_halfplanes,
            seed_vertices       = seed,
            bbox_planes         = bbox,
            max_iter            = self.max_iter,
            rho                 = self.rho,
        )

    # ── Convenience: single-step with manual seed / bbox ─────────────

    def compute_raw(
        self,
        obstacles    : list,
        seed_vertices: List[np.ndarray],
        bbox_planes  : List[HalfPlane],
        polytope_mode: bool = False,
    ) -> FIRIResult:
        """
        Direct access to the solver — supply seed and bbox manually.

        Useful for unit-testing or when the caller builds the robot
        geometry independently.
        """
        return self._solver.compute(
            obstacles    = obstacles,
            seed_vertices= seed_vertices,
            bbox_planes  = bbox_planes,
            max_iter     = self.max_iter,
            rho          = self.rho,
            polytope_mode= polytope_mode,
        )
