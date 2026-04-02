"""
FIRI – Fast Iterative Region Inflation (2-D, standalone Python)
===============================================================

Ported from:
  • firi_node_sdmn.cpp  – SDMN solver, MVIE, full FIRI loop,
                          heading-aligned bounding box
  • firi_polytope_node.cpp – polytope obstacle input,
                             rear-axle kinematic car seed

Reference:
  Wang et al., "Fast Iterative Region Inflation for Computing Large
  2-D/3-D Convex Regions of Obstacle-Free Space",
  IEEE Transactions on Robotics, Vol. 41, 2025.

No ROS, no OSQP.  Requires only numpy + matplotlib.
"""

from .sdmn    import SDMN2D
from .mvie    import MVIE2D
from .solver  import FIRISolver, HalfPlane, FIRIResult
from .robot   import RobotConfig, build_seed, build_bbox
from .env     import FIRIEnv
from .vis     import visualize

__all__ = [
    "SDMN2D", "MVIE2D",
    "FIRISolver", "HalfPlane", "FIRIResult",
    "RobotConfig", "build_seed", "build_bbox",
    "FIRIEnv",
    "visualize",
]
