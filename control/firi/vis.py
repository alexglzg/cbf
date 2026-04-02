"""
vis.py – Visualisation helpers for FIRI results
================================================

All plotting is done with matplotlib only.  No ROS / decomp_ros needed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


# ── Halfplane polygon intersection (for drawing the polytope region) ──────────

def _clip_halfplane(poly: list, n: np.ndarray, d: float) -> list:
    """Sutherland-Hodgman clip polygon by halfplane  n^T x ≤ d."""
    if not poly:
        return []
    result = []
    num = len(poly)
    for i in range(num):
        cur = np.asarray(poly[i])
        nxt = np.asarray(poly[(i + 1) % num])
        c_in = n.dot(cur) <= d + 1e-9
        n_in = n.dot(nxt) <= d + 1e-9
        if c_in:
            result.append(cur.copy())
        if c_in != n_in:
            diff = nxt - cur
            denom = n.dot(diff)
            if abs(denom) > 1e-12:
                t = (d - n.dot(cur)) / denom
                result.append(cur + t * diff)
    return result


def halfplanes_to_polygon(
    planes,
    clip_box: float = 200.0,
) -> Optional[np.ndarray]:
    """
    Convert a list of HalfPlane objects to a (N, 2) vertex array.
    Returns None if the polytope is empty / degenerate.
    """
    poly = [
        np.array([-clip_box, -clip_box]),
        np.array([ clip_box, -clip_box]),
        np.array([ clip_box,  clip_box]),
        np.array([-clip_box,  clip_box]),
    ]
    for hp in planes:
        if hasattr(hp, 'normal'):
            n, d = hp.normal, hp.offset
        else:
            n, d = np.asarray(hp[0:2]), float(hp[2])
        poly = _clip_halfplane(poly, n, d)
        if not poly:
            return None
    if len(poly) < 3:
        return None
    return np.array(poly)


# ── Ellipse patch ─────────────────────────────────────────────────────────────

def ellipse_patch(
    L: np.ndarray,
    d: np.ndarray,
    n: int = 120,
    **kwargs,
) -> np.ndarray:
    """
    Compute the boundary of the MVIE ellipse  { L θ + d : ||θ|| = 1 }.

    Returns (n, 2) array of boundary points.
    """
    t = np.linspace(0, 2 * math.pi, n)
    unit = np.column_stack([np.cos(t), np.sin(t)])  # (n, 2)
    return (L @ unit.T).T + d                        # (n, 2)


# ── Main visualise function ───────────────────────────────────────────────────

def visualize(
    result,
    robot_pos      : Optional[np.ndarray]        = None,
    robot_yaw      : Optional[float]             = None,
    robot_length   : float                        = 0.9,
    robot_width    : float                        = 0.45,
    obstacles      : Optional[List[np.ndarray]]  = None,
    polytopes      : Optional[List[List[np.ndarray]]] = None,
    seed_verts     : Optional[List[np.ndarray]]  = None,
    ax             : Optional[plt.Axes]          = None,
    show_ellipse   : bool                         = True,
    show_halfplanes: bool                         = False,
    show_robot     : bool                         = True,
    show_seed      : bool                         = True,
    title          : str                          = "FIRI Result",
    xlim           : Optional[Tuple[float, float]] = None,
    ylim           : Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Visualise a :class:`~firi.solver.FIRIResult`.

    Parameters
    ----------
    result          : FIRIResult returned by the solver / env
    robot_pos       : 2-D robot position (for the footprint patch)
    robot_yaw       : robot heading (radians)
    robot_length/width : dimensions for drawing the footprint
    obstacles       : raw point-cloud obstacle points
    polytopes       : obstacle polygons (vertex lists)
    seed_verts      : seed polygon vertices to draw
    ax              : existing Axes (creates a new figure if None)
    show_ellipse    : draw the MVIE ellipse
    show_halfplanes : draw outward-normal arrows for each halfplane
    show_robot      : draw robot rectangle (needs robot_pos + robot_yaw)
    show_seed       : draw the seed polygon
    title           : figure title
    xlim, ylim      : axis limits (auto-computed if None)

    Returns
    -------
    matplotlib Figure
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = ax.figure

    ax.set_facecolor("#111122")
    if standalone:
        fig.patch.set_facecolor("#0a0a14")

    # ── Point-cloud obstacles ─────────────────────────────────────────
    if obstacles:
        xs = [p[0] for p in obstacles]
        ys = [p[1] for p in obstacles]
        ax.scatter(xs, ys, s=6, color="#ff6633", zorder=3,
                   linewidths=0, label="Obstacles (points)")

    # ── Polytope obstacles ────────────────────────────────────────────
    if polytopes:
        cmap = plt.cm.Oranges
        n = max(len(polytopes), 1)
        for i, poly in enumerate(polytopes):
            verts = halfplanes_to_polygon(poly)
            if len(poly) < 2:
                continue
            arr = np.array(poly)
            col = cmap(0.4 + 0.5 * i / n)
            patch = plt.Polygon(
                verts, closed=True,
                facecolor=(*col[:3], 0.55),
                edgecolor="#ff8844",
                linewidth=0.9, zorder=3,
            )
            # patch = plt.Polygon(
            #     arr, closed=True,
            #     facecolor=(*col[:3], 0.55),
            #     edgecolor="#ff8844",
            #     linewidth=0.9, zorder=3,
            # )
            ax.add_patch(patch)

    # ── Free-space polytope ───────────────────────────────────────────
    poly_verts = halfplanes_to_polygon(result.planes)
    if poly_verts is not None:
        patch = plt.Polygon(
            poly_verts, closed=True,
            facecolor="#1e3e5a", edgecolor="#44aaff",
            linewidth=1.5, alpha=0.55, zorder=4,
            label="Free-space polytope",
        )
        ax.add_patch(patch)

    # ── Halfplane outward normals ─────────────────────────────────────
    if show_halfplanes:
        for hp in result.planes:
            pt = hp.point_on_plane()
            ax.annotate(
                "", xy=pt + hp.normal * 0.3, xytext=pt,
                arrowprops=dict(arrowstyle="->", color="#ffcc44",
                                lw=1.0, mutation_scale=10),
                zorder=6,
            )

    # ── MVIE ellipse ──────────────────────────────────────────────────
    if show_ellipse and result.ellipsoid is not None:
        ell = result.ellipsoid
        bdry = ellipse_patch(ell.L, ell.d)
        ax.fill(bdry[:, 0], bdry[:, 1],
                color="#00ddaa", alpha=0.22, zorder=5)
        ax.plot(bdry[:, 0], bdry[:, 1],
                color="#00ffcc", lw=1.4, zorder=6,
                label=f"MVIE  (vol={ell.volume:.3f})")
        ax.plot(*ell.d, "o", color="#00ffcc", ms=5, zorder=7)

    # ── Seed polygon ──────────────────────────────────────────────────
    if show_seed and seed_verts:
        sv = np.array(seed_verts)
        sv_closed = np.vstack([sv, sv[0]])
        ax.fill(sv[:, 0], sv[:, 1],
                color="#ffdd55", alpha=0.22, zorder=7)
        ax.plot(sv_closed[:, 0], sv_closed[:, 1],
                color="#ffdd55", lw=1.4, ls="--", zorder=8,
                label="Seed (footprint)")

    # ── Robot rectangle ───────────────────────────────────────────────
    if show_robot and robot_pos is not None and robot_yaw is not None:
        _draw_robot(ax, robot_pos, robot_yaw, robot_length, robot_width)

    # ── Axes / legend ─────────────────────────────────────────────────
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_aspect("equal")

    s = result
    subtitle = (
        f"iterations={s.iterations}  |  "
        f"halfplanes={len(s.planes)}  |  "
        f"time={s.solve_time_ms:.2f} ms"
    )
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(subtitle, color="#7070aa", fontsize=9)
    ax.tick_params(colors="#44446a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#22224a")

    legend_handles = [
        mpatches.Patch(fc="#1e3e5a", ec="#44aaff",  label="Free-space polytope"),
        Line2D([0], [0], color="#00ffcc", lw=1.4,   label=f"MVIE ellipse"),
        mpatches.Patch(fc="#ffdd55", ec="#ffdd55",   label="Seed polygon"),
        mpatches.Patch(fc="#ff5533", ec="white",     label="Robot"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        facecolor="#1a1a2e", edgecolor="#22224a",
        labelcolor="white", fontsize=8, framealpha=0.85,
    )

    if standalone:
        plt.tight_layout()
    return fig


def _draw_robot(
    ax        : plt.Axes,
    pos       : np.ndarray,
    yaw       : float,
    length    : float,
    width     : float,
):
    """Draw the robot as a rotated rectangle with a heading arrow."""
    import matplotlib.transforms as transforms

    hl, hw = length / 2.0, width / 2.0
    rect = mpatches.Rectangle(
        (-hl, -hw), length, width,
        linewidth=1.4, edgecolor="white",
        facecolor="#ff4444", alpha=0.9, zorder=10,
    )
    tr = (transforms.Affine2D()
          .rotate(yaw)
          .translate(pos[0], pos[1])
          + ax.transData)
    rect.set_transform(tr)
    ax.add_patch(rect)

    # Heading arrow
    arrow_len = length * 0.55
    ax.annotate(
        "", xy=pos + arrow_len * np.array([math.cos(yaw), math.sin(yaw)]),
        xytext=pos,
        arrowprops=dict(arrowstyle="->", color="white",
                        lw=1.5, mutation_scale=12),
        zorder=11,
    )
