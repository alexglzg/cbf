"""
robot.py – Robot geometry helpers
==================================

Ports two pieces of geometry logic from the original C++ nodes:

1. ``build_seed``
   Robot footprint as the convex seed polygon.

   • **Symmetric mode** (firi_node_sdmn):
     Centre the footprint at the robot position; yaw rotation applied.

   • **Rear-axle mode** (firi_polytope_node):
     Footprint is offset so that the rear axle is at the origin, matching
     the kinematic bicycle/car model.  Uses ``rear_axle_offset`` to push
     the rectangle backward.

2. ``build_bbox``
   **Heading-aligned** bounding box as 4 halfplanes (firi_node_sdmn).
   The box axes are rotated with the robot heading, so the "ahead" wall
   is always in front of the robot regardless of map orientation.
   (firi_polytope_node used a fixed axis-aligned box; this function
   implements the SDMN version which is strictly better.)

Both functions return types compatible with :class:`~firi.solver.FIRISolver`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .solver import HalfPlane


# ── Robot configuration ───────────────────────────────────────────────────────

@dataclass
class RobotConfig:
    """
    Physical dimensions and bounding-box parameters for a 2-D robot.

    Attributes
    ----------
    length          : total body length  (m)
    width           : total body width   (m)
    rear_axle_offset: distance from rear axle to the BACK of the body (m).
                      Set to ``length / 2`` for a symmetric footprint.
    bbox_ahead      : free-space extent in front of the robot (m)
    bbox_behind     : free-space extent behind the robot (m)
    bbox_side       : free-space extent to each side (m)
    """
    length          : float = 0.9
    width           : float = 0.45
    rear_axle_offset: float = 0.45   # symmetric by default (= length/2)
    bbox_ahead      : float = 6.0
    bbox_behind     : float = 1.0
    bbox_side       : float = 3.0


# ── Seed polygon ──────────────────────────────────────────────────────────────

def build_seed(
    pos           : np.ndarray,    # shape (2,) — robot position (rear axle)
    yaw           : float,         # heading in radians
    cfg           : RobotConfig,
    rear_axle_mode: bool = True,
) -> List[np.ndarray]:
    """
    Build the convex seed polygon (robot footprint).

    Parameters
    ----------
    pos : np.ndarray shape (2,)
        Robot position in the map frame.  In rear-axle mode this is the
        rear-axle location; in symmetric mode it is the body centre.
    yaw : float
        Robot heading (radians).
    cfg : RobotConfig
    rear_axle_mode : bool
        If True (default, matches firi_polytope_node), the seed is built
        from the rear axle: the rectangle extends ``front_dist`` ahead and
        ``rear_axle_offset`` behind.
        If False (matches firi_node_sdmn), a symmetric footprint centred
        at *pos* is used.

    Returns
    -------
    List[np.ndarray]
        Four corner points (CCW order) in the map frame.
    """
    R = _rotation_matrix(yaw)
    hw = cfg.width / 2.0

    if rear_axle_mode:
        # firi_polytope_node convention:
        #   front_dist = length - rear_axle_offset
        #   rear_dist  = rear_axle_offset
        front_dist = cfg.length - cfg.rear_axle_offset
        rear_dist  = cfg.rear_axle_offset
        local_corners = [
            np.array([ front_dist,  hw]),   # Front-left
            np.array([ front_dist, -hw]),   # Front-right
            np.array([-rear_dist,  -hw]),   # Rear-right
            np.array([-rear_dist,   hw]),   # Rear-left
        ]
    else:
        # firi_node_sdmn convention: symmetric about centre
        hl = cfg.length / 2.0
        local_corners = [
            np.array([ hl,  hw]),
            np.array([ hl, -hw]),
            np.array([-hl, -hw]),
            np.array([-hl,  hw]),
        ]

    return [pos + R @ c for c in local_corners]


# ── Heading-aligned bounding box ──────────────────────────────────────────────

def build_bbox(
    pos: np.ndarray,    # shape (2,)
    yaw: float,         # heading in radians
    cfg: RobotConfig,
) -> List[HalfPlane]:
    """
    Build four heading-aligned bounding-box halfplanes.

    Ported from firi_node_sdmn.cpp (scanCb).
    The box is expressed in the map frame but aligned with the robot
    heading, so the search region always points "forward / sideways"
    relative to the robot — not aligned with the world axes.

    Layout (in body frame)::

           ┌─────────────┐  ← front wall  (fwd,   +bbox_ahead)
           │             │
           │      ★      │  ← robot (rear-axle origin)
           │             │
           └─────────────┘  ← rear  wall  (-fwd, +bbox_behind)
        left wall           right wall
      (lft, +bbox_side)   (-lft, +bbox_side)

    Each halfplane is  n^T x ≤ offset, where offset encodes the wall
    distance from the map origin.

    Returns
    -------
    List[HalfPlane]  (4 planes, front / rear / left / right)
    """
    R   = _rotation_matrix(yaw)
    fwd = R[:, 0]   # forward unit vector  (R[:,0])
    lft = R[:, 1]   # left    unit vector  (R[:,1])

    return [
        HalfPlane(normal= fwd,
                  offset= fwd.dot(pos) + cfg.bbox_ahead),   # front wall
        HalfPlane(normal=-fwd,
                  offset=-fwd.dot(pos) + cfg.bbox_behind),  # rear  wall
        HalfPlane(normal= lft,
                  offset= lft.dot(pos) + cfg.bbox_side),    # left  wall
        HalfPlane(normal=-lft,
                  offset=-lft.dot(pos) + cfg.bbox_side),    # right wall
    ]


# ── Helper ────────────────────────────────────────────────────────────────────

def _rotation_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s],
                     [s,  c]])
