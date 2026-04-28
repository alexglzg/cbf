"""
solver.py – FIRISolver: Full FIRI algorithm (2-D)
==================================================

Combines SDMN2D + MVIE2D into the complete FIRI loop described in:
  Wang et al. (2025), Algorithm 1.

Outer loop
----------
  1. Transform obstacles & seed into normalised space  (ellipsoid → unit ball)
  2. RsI: for each obstacle point/vertex, call SDMN for separating halfplane
  3. Greedy halfplane selection (closest first, remove separated obstacles)
  4. Transform polytope back to original space
  5. Compute MVIE of the polytope
  6. Check convergence (MVIE volume improvement < rho)

Obstacle types accepted
-----------------------
  • Point cloud  – list of 2-D points (from LiDAR, etc.)
  • Polytopes    – list of vertex lists  (from firi_polytope_node)
    Each polytope is represented by its vertex set; the RsI step
    generates one separating halfplane per polytope (all vertices must
    be excluded), matching firi_polytope_node's per-obstacle QP.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .sdmn import SDMN2D
from .mvie import MVIE2D, Ellipsoid
from .vis import _clip_halfplane


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass
class HalfPlane:
    """
    A halfplane  n^T x ≤ offset  where ||n|| = 1.

    Attributes
    ----------
    normal : np.ndarray shape (2,)  — outward unit normal
    offset : float                  — n^T x ≤ offset
    """
    normal: np.ndarray
    offset: float

    def contains(self, p: np.ndarray) -> bool:
        """True iff *p* satisfies this halfplane (with small tolerance)."""
        # return float(self.normal.dot(p)) <= self.offset + 1e-9
        return -1e-6 <= float(self.normal.dot(p)) - self.offset <= 1e-6

    def point_on_plane(self) -> np.ndarray:
        """A point lying on the boundary hyperplane."""
        return self.normal * self.offset


@dataclass
class FIRIResult:
    """
    Output of :meth:`FIRISolver.compute`.

    Attributes
    ----------
    planes       : list of HalfPlane — the free-space polytope
    ellipsoid    : final inscribed ellipsoid (MVIE)
    iterations   : number of outer FIRI iterations performed
    solve_time_ms: wall-clock time in milliseconds
    """
    planes       : List[HalfPlane]
    ellipsoid    : Optional[Ellipsoid]
    iterations   : int
    solve_time_ms: float

    def contains(self, p: np.ndarray) -> bool:
        """True iff point *p* is inside the free-space polytope."""
        return all(hp.contains(p) for hp in self.planes)

    def as_Ab(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the polytope as (A, b) where A x ≤ b, A shape (m, 2)."""
        m = len(self.planes)
        A = np.stack([hp.normal for hp in self.planes])
        b = np.array([hp.offset for hp in self.planes])
        return A, b


# ── Solver ────────────────────────────────────────────────────────────────────

class FIRISolver:
    """
    Fast Iterative Region Inflation (FIRI) — 2-D standalone.

    Parameters
    ----------
    sdmn_seed : int
        RNG seed for SDMN constraint permutation (reproducibility).
    mvie_outer_iters, mvie_inner_iters, mvie_mu :
        Passed to :class:`MVIE2D`.
    """

    def __init__(
        self,
        sdmn_seed       : int   = 42,
        mvie_outer_iters: int   = 20,
        mvie_inner_iters: int   = 40,
        mvie_mu         : float = 4.0,
    ):
        self._sdmn = SDMN2D(seed=sdmn_seed)
        self._mvie = MVIE2D(
            outer_iters=mvie_outer_iters,
            inner_iters=mvie_inner_iters,
            mu=mvie_mu,
        )

    # ── Main entry point ──────────────────────────────────────────────

    def compute(
        self,
        obstacles   : List[np.ndarray] | List[List[np.ndarray]],
        seed_vertices: List[np.ndarray],
        bbox_planes : List[HalfPlane],
        max_iter    : int   = 10,
        rho         : float = 0.02,
        polytope_mode: bool = False,
    ) -> FIRIResult:
        """
        Run FIRI and return the free-space polytope.

        Parameters
        ----------
        obstacles : list
            • **Point-cloud mode** (``polytope_mode=False``):
              list of ``np.ndarray`` shape (2,), one per obstacle point.
            • **Polytope mode** (``polytope_mode=True``):
              list of vertex lists - each element is a list of
              ``np.ndarray`` shape (2,).  Mirrors firi_polytope_node.
        seed_vertices : list of np.ndarray shape (2,)
            Convex seed polygon (robot footprint).
        bbox_planes : list of HalfPlane
            Initial bounding-box halfplanes (4 planes, heading-aligned).
        max_iter : int
            Maximum FIRI outer iterations (default 10).
        rho : float
            Convergence threshold: stop when MVIE volume improvement
            is less than *rho* relative (default 0.02 = 2%).
        polytope_mode : bool
            Set True when *obstacles* are vertex lists of polytopes.

        Returns
        -------
        FIRIResult
        """
        t0 = time.perf_counter()

        if not obstacles or not seed_vertices:
            return FIRIResult(
                planes=list(bbox_planes),
                ellipsoid=None,
                iterations=0,
                solve_time_ms=0.0,
            )

        # ── Initialise ellipsoid strictly inside seed [Paper §III-B] ──
        seed_arr = [np.asarray(v, dtype=float).flatten() for v in seed_vertices]
        d = np.mean(seed_arr, axis=0)
        r_init = self._inscribed_radius(seed_arr, d)
        r_init = max(r_init * 0.8, 1e-4)
        L = r_init * np.eye(2)

        prev_vol  = r_init ** 2 * math.pi
        best_planes = list(bbox_planes)
        best_ellipsoid: Optional[Ellipsoid] = None
        iters = 0

        for k in range(max_iter):
            iters = k + 1

            # ── RsI step ──────────────────────────────────────────────
            if polytope_mode:
                planes = self._rsi_polytopes(
                    obstacles, seed_arr, L, d, bbox_planes)
            else:
                planes = self._rsi_points(
                    obstacles, seed_arr, L, d, bbox_planes)
            best_planes = planes

            # ── MVIE ──────────────────────────────────────────────────
            # TODO: not necessary so remove
            m = len(planes)
            A_mat = np.stack([hp.normal if hasattr(hp, 'normal') else hp[0:2] for hp in planes])   # (m, 2)
            b_vec = np.array([hp.offset if hasattr(hp, 'offset') else hp[2] for hp in planes])   # (m,)

            ell = self._mvie.solve(A_mat, b_vec, d)
            best_ellipsoid = ell

            new_vol = ell.volume
            if k > 0 and (new_vol - prev_vol) / (prev_vol + 1e-15) < rho:
                break   # converged [Paper §III-B3, Line 23]

            prev_vol = new_vol
            L = ell.L
            d = ell.d

        if k == max_iter - 1:
            print(k)
            print("FIRI MAX ITER REACHED")

        ms = (time.perf_counter() - t0) * 1000.0
        
        # Remove redundant bounding box halfplanes
        best_planes = self._remove_redundant_planes(best_planes, d)
        
        return FIRIResult(
            planes=best_planes,
            ellipsoid=best_ellipsoid,
            iterations=iters,
            solve_time_ms=ms,
        )
    
    def compute_from_halfplanes(
        self,
        obstacle_halfplanes: list,   # List[List[Tuple[np.ndarray, float]]]
        seed_vertices      : list,
        bbox_planes        : list,
        max_iter           : int   = 10,
        rho                : float = 0.02,
    ) -> FIRIResult:
        """
        Identical to compute() but obstacles are given as halfplane lists.

        obstacle_halfplanes : List of obstacles, each obstacle is a list of
                            (normal: np.ndarray, offset: float) pairs.

        Internally converts each obstacle to its vertex representation and
        calls the standard polytope pipeline.
        """

        # Convert H-rep → V-rep once, before the iterative loop
        vertex_obstacles = []
        for hp_list in obstacle_halfplanes:
            verts = self._halfplanes_to_verts(hp_list)
            if len(verts) >= 2:           # skip degenerate/empty obstacles
                vertex_obstacles.append(verts)

        return self.compute(
            obstacles    = vertex_obstacles,
            seed_vertices= seed_vertices,
            bbox_planes  = bbox_planes,
            max_iter     = max_iter,
            rho          = rho,
            polytope_mode= True,          # always polytope mode
        )
    
    def _halfplanes_to_verts(self, halfplanes: list) -> list:
        """
        Convert an obstacle from H-representation to V-representation.

        Each halfplane is either:
        • a HalfPlane object  (normal, offset)
        • a tuple/list        (normal_array, offset_float)

        Returns a list of np.ndarray vertices (CCW), or [] if degenerate.
        """
        CLIP = 1e3  # bounding box for unbounded polytopes
        poly = [
            np.array([-CLIP, -CLIP]),
            np.array([ CLIP, -CLIP]),
            np.array([ CLIP,  CLIP]),
            np.array([-CLIP,  CLIP]),
        ]
        for hp in halfplanes:
            # Accept both HalfPlane objects and raw (n, d) tuples
            if hasattr(hp, 'normal'):
                n, d = hp.normal, hp.offset
            else:
                n, d = np.asarray(hp[0:2]), float(hp[2])
            poly = _clip_halfplane(poly, n, d)
            if not poly:
                return []
        return poly   # list of np.ndarray of all vertices of polytope

    def _halfplanes_to_verts(self, halfplanes: list) -> list:
        """
        Convert an obstacle from H-representation to V-representation.

        Each halfplane is either:
        • a HalfPlane object  (normal, offset)
        • a tuple/list        (normal_array, offset_float)

        Returns a list of np.ndarray vertices (CCW), or [] if degenerate.
        """
        CLIP = 1e3  # bounding box for unbounded polytopes
        poly = [
            np.array([-CLIP, -CLIP]),
            np.array([ CLIP, -CLIP]),
            np.array([ CLIP,  CLIP]),
            np.array([-CLIP,  CLIP]),
        ]
        for hp in halfplanes:
            # Accept both HalfPlane objects and raw (n, d) tuples
            if hasattr(hp, 'normal'):
                n, d = hp.normal, hp.offset
            else:
                n, d = np.asarray(hp[0:2]), float(hp[2])
            poly = _clip_halfplane(poly, n, d)
            if not poly:
                return []
        return poly   # list of np.ndarray of all vertices of polytope

    # ── RsI: point-cloud obstacles ────────────────────────────────────

    def _rsi_points(
        self,
        obstacles   : List[np.ndarray],
        seed_verts  : List[np.ndarray],
        L           : np.ndarray,
        d           : np.ndarray,
        bbox_planes : List[HalfPlane],
    ) -> List[HalfPlane]:
        """
        RsI for point-cloud obstacles.
        One SDMN call per obstacle point. [Paper Alg. 1, Lines 10-11]
        """
        L_inv   = np.linalg.inv(L)
        L_inv_T = L_inv.T

        # Transform seed to normalised space [Paper Eq. 5]
        seed_bar = [L_inv @ (v - d) for v in seed_verts]

        # Transform obstacles [Paper Eq. 6]
        obs_bar = [L_inv @ (u - d) for u in obstacles]

        # Pre-build seed constraints (shared across SDMN calls)
        n_seed = len(seed_bar)
        base_e = list(seed_bar)          # e_i^T y ≤ 1
        base_f = [1.0] * n_seed

        candidates = []   # (b_sol, a, a_norm, obs_idx)

        for i, u_bar in enumerate(obs_bar):
            e = base_e + [-u_bar]        # obstacle: -u^T y ≤ -1  →  u^T y ≥ 1
            f = base_f + [-1.0]

            res = self._sdmn.solve(e, f)
            if res.feasible:
                b_sq = res.y.dot(res.y)
                if b_sq > 1e-10:
                    a = res.y / b_sq
                    candidates.append((res.y, a, np.linalg.norm(a), i))

        return self._greedy_select(
            candidates, obs_bar, L_inv_T, d, bbox_planes)

    # ── RsI: polytope obstacles ───────────────────────────────────────

    def _rsi_polytopes(
        self,
        polytopes   : List[List[np.ndarray]],
        seed_verts  : List[np.ndarray],
        L           : np.ndarray,
        d           : np.ndarray,
        bbox_planes : List[HalfPlane],
    ) -> List[HalfPlane]:
        """
        RsI for polytope obstacles (firi_polytope_node style).

        One SDMN call per polytope — all vertices of the polytope must
        be excluded (one constraint per vertex, plus seed constraints).
        This matches the per-obstacle QP in firi_polytope_node.cpp.
        """
        L_inv   = np.linalg.inv(L)
        L_inv_T = L_inv.T

        # Transform seed [Paper Eq. 5]
        seed_bar = [L_inv @ (v - d) for v in seed_verts]

        # Pre-build seed constraints
        base_e = list(seed_bar)
        base_f = [1.0] * len(seed_bar)

        # Transform ALL polytope vertices.
        # For SDMN we use a single representative point per polytope
        # (the transformed vertex closest to the unit ball — most constraining).
        # Obstacle is "separated" if ALL its transformed vertices satisfy
        # b^T u_bar ≥ 1 for the chosen plane.
        poly_bars: List[List[np.ndarray]] = []
        for poly in polytopes:
            poly_bars.append([L_inv @ (v - d) for v in poly])

        # poly_bars: List[List[np.ndarray]] = []
        # for poly in polytopes:
        #     transformed_poly = []
        #     for v in poly:
        #         v_arr = np.asarray(v, dtype=float).flatten()
        #         if len(v_arr) >= 2:
        #             v_arr = v_arr[:2]
        #         transformed_poly.append(L_inv @ (v_arr - d))
        #     poly_bars.append(transformed_poly)

        candidates = []  # (b_sol, a, a_norm, poly_idx)

        for i, verts_bar in enumerate(poly_bars):
            # Distance filter: skip if all vertices are far from unit ball
            min_dist_sq = min(v.dot(v) for v in verts_bar)
            if min_dist_sq > 25.0:  # > 4 normalised units away
                continue

            # One SDMN call that excludes ALL vertices of this polytope.
            # e_j^T b ≥ 1  →  -(v_j^T) b ≤ -1  for each vertex v_j
            e = base_e + [-v for v in verts_bar]
            f = base_f + [-1.0] * len(verts_bar)

            res = self._sdmn.solve(e, f)
            if res.feasible:
                b_sq = res.y.dot(res.y)
                if b_sq > 1e-10:
                    a = res.y / b_sq
                    candidates.append((res.y, a, np.linalg.norm(a), i))

        # import pdb;pdb.set_trace()

        # For greedy selection we need flat "obs_bar" (one entry per polytope,
        # using all vertices to test separation).
        return self._greedy_select_polytopes(
            candidates, poly_bars, L_inv_T, d, bbox_planes)

    # ── Greedy halfplane selection (point cloud) ──────────────────────

    def _greedy_select(
        self,
        candidates  : list,
        obs_bar     : List[np.ndarray],
        L_inv_T     : np.ndarray,
        d           : np.ndarray,
        bbox_planes : List[HalfPlane],
    ) -> List[HalfPlane]:
        """
        Greedy selection sorted by ||a|| ascending (closest first).
        [Paper Alg. 1, Lines 12-16]
        """
        # Sort by a_norm ascending (closest halfplane first)
        candidates.sort(key=lambda c: c[2])

        separated = [False] * len(obs_bar)
        result_planes = list(bbox_planes)

        for b_sol, a, a_norm, obs_idx in candidates:
            if separated[obs_idx]:
                continue

            # Transform halfplane to original space
            # Normalised space: a^T x_bar ≤ ||a||²
            # Original space:   (L^{-T} a)^T x ≤ ||a||² + (L^{-T} a)^T d
            n_orig = L_inv_T @ a
            d_orig = a.dot(a) + n_orig.dot(d)
            n_len = np.linalg.norm(n_orig)
            if n_len < 1e-15:
                continue

            result_planes.append(
                HalfPlane(normal=n_orig / n_len, offset=d_orig / n_len))

            # Mark all obstacles separated by this plane
            for j, u_bar in enumerate(obs_bar):
                if not separated[j] and b_sol.dot(u_bar) >= 1.0 - 1e-8:
                    separated[j] = True

            if len(result_planes) > 50:   # safety cap
                break

        return result_planes

    # ── Greedy halfplane selection (polytopes) ────────────────────────

    def _greedy_select_polytopes(
        self,
        candidates  : list,
        poly_bars   : List[List[np.ndarray]],
        L_inv_T     : np.ndarray,
        d           : np.ndarray,
        bbox_planes : List[HalfPlane],
    ) -> List[HalfPlane]:
        """
        Greedy selection for polytope obstacles.
        A polytope is "separated" iff ALL its transformed vertices
        satisfy b_sol^T v_bar ≥ 1.  (Matches firi_polytope_node logic.)
        """
        candidates.sort(key=lambda c: c[2])

        separated = [False] * len(poly_bars)
        result_planes = list(bbox_planes)

        for b_sol, a, a_norm, poly_idx in candidates:
            if separated[poly_idx]:
                continue

            n_orig = L_inv_T @ a
            d_orig = a.dot(a) + n_orig.dot(d)
            n_len = np.linalg.norm(n_orig)
            if n_len < 1e-15:
                continue

            result_planes.append(
                HalfPlane(normal=n_orig / n_len, offset=d_orig / n_len))

            # A polytope is separated iff ALL its vertices are excluded
            for j, verts_bar in enumerate(poly_bars):
                if not separated[j]:
                    if all(b_sol.dot(v) >= 1.0 - 1e-8 for v in verts_bar):
                        separated[j] = True

            if len(result_planes) > 50:
                break

        return result_planes

    # ── Helpers ───────────────────────────────────────────────────────

    def _halfplane_contains_point(self, point: List, halfplane: List):
        """True iff *p* satisfies this halfplane (with small tolerance)."""
        return -1e-6 <= float(np.dot(halfplane[0:2], point)) - halfplane[2] <= 1e-6

    def _remove_redundant_planes(self, planes: List[HalfPlane] | List[List], center: np.ndarray) -> List[HalfPlane]:
        """
        Remove redundant halfplanes from the polytope.
        A halfplane is redundant if it's implied by the others
        (i.e., all vertices defined by other planes satisfy it).
        
        Parameters
        ----------
        planes : list of HalfPlane
            The halfplanes to check
        center : np.ndarray
            A point strictly inside the polytope (e.g., ellipsoid center)
        """
        if len(planes) <= 4:
            return planes
        
        from scipy.spatial import ConvexHull, HalfspaceIntersection
        
        # try:
        A_full = np.stack([hp.normal if hasattr(hp, 'normal') else hp[0:2] for hp in planes])
        b_full = np.array([hp.offset if hasattr(hp, 'offset') else hp[2] for hp in planes])
        halfspaces = np.column_stack((A_full, -b_full))
        
        hs = HalfspaceIntersection(halfspaces, center)
        verts = hs.intersections[ConvexHull(hs.intersections).vertices]
        # except Exception as e:
        #     # If we can't compute the full polytope, return as-is
        #     return planes
        
        # Test each plane for redundancy
        non_redundant = []
        for hp_test in planes:
            # Check if all vertices satisfy this halfplane
            if hasattr(hp_test, 'normal'):
                # print([hp_test.contains(v) for v in verts])
                if any([hp_test.contains(v) for v in verts]):
                    # This plane is NOT redundant
                    non_redundant.append(hp_test)
            else:
                # print([self._halfplane_contains_point(v, hp_test) for v in verts])
                if any([self._halfplane_contains_point(v, hp_test) for v in verts]):
                    non_redundant.append(hp_test)
                            
        return non_redundant if non_redundant else planes

    @staticmethod
    def _inscribed_radius(
        verts: List[np.ndarray],
        center: np.ndarray,
    ) -> float:
        """
        Inscribed ball radius of a convex polygon at *center*.
        Minimum distance from *center* to any edge.
        """
        r = 1e18
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i + 1) % n]
            edge = b - a
            le = np.linalg.norm(edge)
            if le < 1e-15:
                continue
            normal = np.array([-edge[1], edge[0]]) / le
            dist = abs(float(normal.dot(center - a)))
            r = min(r, dist)
        return r if r < 1e18 else 1e-4
