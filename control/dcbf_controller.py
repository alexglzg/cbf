import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull, HalfspaceIntersection
import casadi as ca

from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from control.firi_polytope_old import FIRI

# Set to True to enable plotting
DEBUG_VIS = True


# ── clearance helper ──────────────────────────────────────────────────────────

def _closest_point_on_convex_poly(point: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """
    Return the point on the boundary (or interior) of a convex polygon that is
    closest to `point`.  Uses brute-force projection onto every edge.
    `verts` is (N, 2), CCW ordered.
    """
    best_pt   = verts[0].copy()
    best_dist = np.inf
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        ab  = b - a
        ab2 = ab @ ab
        if ab2 < 1e-20:
            proj = a.copy()
        else:
            t    = np.clip((point - a) @ ab / ab2, 0.0, 1.0)
            proj = a + t * ab
        d = np.linalg.norm(point - proj)
        if d < best_dist:
            best_dist = d
            best_pt   = proj
    return best_pt


def _robot_world_verts(system) -> np.ndarray:
    """
    Return all robot vertices in world frame as (M, 2) array,
    correctly rotated and translated by the current state.
    State layout for KinematicCar: [x, y, v, theta].
    """
    state = system._state._x
    x, y, theta = float(state[0]), float(state[1]), float(state[3])
    c, s = np.cos(theta), np.sin(theta)
    R    = np.array([[c, -s], [s, c]])

    if hasattr(system._geometry, 'equiv_rep'):
        components = system._geometry.equiv_rep()
    else:
        components = [system._geometry]

    all_verts = []
    for comp in components:
        if hasattr(comp, 'vertices') and comp.vertices is not None:
            v_local = np.array(comp.vertices)           # (K, 2)
        elif hasattr(comp, 'x_min'):
            v_local = np.array([
                [comp.x_min, comp.y_min], [comp.x_max, comp.y_min],
                [comp.x_max, comp.y_max], [comp.x_min, comp.y_max],
            ])
        elif hasattr(comp, 'get_convex_rep'):
            G, g = comp.get_convex_rep()
            # local vertices: G @ v <= -g  (convention from geometry_utils)
            # fall back to bounding-box approximation if solve fails
            try:
                from scipy.spatial import HalfspaceIntersection, ConvexHull
                from scipy.optimize import linprog
                A_lp = np.column_stack((G, np.linalg.norm(G, axis=1)))
                res  = linprog([0, 0, -1], A_ub=A_lp, b_ub=-g,
                               bounds=[(None,None)]*3)
                if res.success:
                    ctr = res.x[:2]
                    hs  = HalfspaceIntersection(
                              np.column_stack((G, g)), ctr)
                    v_local = hs.intersections[
                                  ConvexHull(hs.intersections).vertices]
                else:
                    continue
            except Exception:
                continue
        else:
            continue

        v_world = (v_local @ R.T) + np.array([x, y])
        all_verts.append(v_world)

    return np.vstack(all_verts) if all_verts else np.array([[x, y]])


def _clearance_robot_to_obstacle(robot_verts: np.ndarray, obs) -> float:
    """
    Minimum distance between the robot polygon (world-frame vertices) and one
    obstacle.  The obstacle can expose either vertices or a halfplane rep.

    Positive  = separated (safe margin).
    Zero      = touching.
    Negative  = overlapping (should never happen with a working CBF).
    """
    # ── get obstacle vertices ─────────────────────────────────────────────
    if hasattr(obs, 'vertices') and obs.vertices is not None:
        obs_verts = np.array(obs.vertices)
    elif hasattr(obs, 'x_min'):
        obs_verts = np.array([
            [obs.x_min, obs.y_min], [obs.x_max, obs.y_min],
            [obs.x_max, obs.y_max], [obs.x_min, obs.y_max],
        ])
    elif hasattr(obs, 'get_convex_rep'):
        A, b = obs.get_convex_rep()
        A, b = np.array(A), np.array(b).flatten()
        try:
            from scipy.spatial import HalfspaceIntersection, ConvexHull
            from scipy.optimize import linprog
            A_lp = np.column_stack((A, np.linalg.norm(A, axis=1)))
            res  = linprog([0, 0, -1], A_ub=A_lp, b_ub=b,
                           bounds=[(None,None)]*3)
            if not res.success:
                return float('inf')
            ctr      = res.x[:2]
            hs       = HalfspaceIntersection(np.column_stack((A, -b)), ctr)
            obs_verts = hs.intersections[ConvexHull(hs.intersections).vertices]
        except Exception:
            return float('inf')
    else:
        return float('inf')

    # ── GJK-lite: min dist over all vertex pairs + edge projections ───────
    min_dist = np.inf
    n_r = len(robot_verts)
    n_o = len(obs_verts)

    # Robot vertex → closest point on each obstacle edge
    for rv in robot_verts:
        cp = _closest_point_on_convex_poly(rv, obs_verts)
        min_dist = min(min_dist, float(np.linalg.norm(rv - cp)))

    # Obstacle vertex → closest point on each robot edge
    for ov in obs_verts:
        cp = _closest_point_on_convex_poly(ov, robot_verts)
        min_dist = min(min_dist, float(np.linalg.norm(ov - cp)))

    return min_dist

class NmpcDcbfController:
    def __init__(self, dynamics=None, opt_param=None, enable_vis=True):
        self._param = opt_param
        self._enable_vis = enable_vis and DEBUG_VIS
        # Original DCBF Optimizer
        self._optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
        
        # FIRI for Visualization ONLY
        self._firi = FIRI()
        self._fig = None
        self._ax = None

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        # --- 1. VISUALIZATION STEP (FIRI) ---
        if self._enable_vis:
            try:
                # Extract Data for FIRI
                obs_verts = self._extract_obstacle_vertices(obstacles)
                seed_poly = self._get_robot_seed(system)
                
                # Local BBox (Match LSE logic: +/- 2.0m)
                rx, ry = system._state._x[0], system._state._x[1]
                bbox = (rx-2.0, rx+2.0, ry-2.0, ry+2.0)
                
                # Compute Polytope
                A_safe, b_safe = self._firi.compute(obs_verts, seed_poly, bbox)
                
                # Draw
                self._visualize(seed_poly, obs_verts, A_safe, b_safe, bbox)
            except Exception as e:
                print(f"[DCBF Viz Error] {e}")

        # --- 2. CONTROL STEP (DCBF) ---
        self._optimizer.setup(self._param, system, local_trajectory, obstacles)
        self._opt_sol = self._optimizer.solve_nlp()
        print(f"Current position: {system._state._x[:2]}")
        if self._opt_sol:
            return self._opt_sol.value(self._optimizer.u[0])
        else:
            return np.zeros(2)

    def collect_metrics(self, system, obstacles):
        """
        Compute and return a dict of per-step metrics for the current state.
        Call this immediately after generate_control_input() at every time step.
        The returned dict is later aggregated by the benchmark runner.
        """
        opt = self._optimizer

        # ── clearance: closest point on robot to closest point on each obstacle
        robot_verts = _robot_world_verts(system)
        clearances  = [_clearance_robot_to_obstacle(robot_verts, obs)
                       for obs in obstacles]

        return {
            "comp_time_s":   opt.solver_times[-1]     if opt.solver_times     else None,
            "feval_time_s":  opt.feval_times[-1]      if opt.feval_times      else None,
            "kkt_time_s":    opt.kkt_times[-1]        if opt.kkt_times        else None,
            "iterations":    opt.iterations[-1]       if opt.iterations       else None,
            "infeasible":    opt.infeasible_steps[-1] if opt.infeasible_steps else None,
            "n_variables":   opt.n_variables_steps[-1] if hasattr(opt, 'n_variables_steps') and opt.n_variables_steps else None,
            "n_eq":          opt.n_eq_steps[-1]       if opt.n_eq_steps       else None,
            "n_ineq":        opt.n_ineq_steps[-1]     if opt.n_ineq_steps     else None,
            "clearances":    clearances,
            "min_clearance": float(min(clearances))   if clearances           else None,
        }

    def logging(self, logger):
        if self._opt_sol:
            # Convert stage-wise variables to trajectory format
            x_values = [self._opt_sol.value(self._optimizer.x[k]).flatten() for k in range(len(self._optimizer.x))]
            u_values = [self._opt_sol.value(self._optimizer.u[k]).flatten() for k in range(len(self._optimizer.u))]
            logger._xtrajs.append(np.column_stack(x_values).T)
            logger._utrajs.append(np.column_stack(u_values).T)

    # --- VISUALIZATION HELPERS ---
    def _visualize(self, seed, obstacles, A, b, bbox):
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        
        self._ax.clear()
        
        # BBox
        self._ax.add_patch(Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], 
                         linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='BBox'))

        # Obstacles
        for obs in obstacles:
            self._ax.add_patch(Polygon(obs, color='black', alpha=0.8))
            
        # Robot Seed
        self._ax.add_patch(Polygon(seed, color='blue', alpha=0.5))
        
        # Computed Polytope (Green)
        if A.shape[0] > 0:
            try:
                center = np.mean(seed, axis=0)
                halfspaces = np.column_stack((A, -b))
                hs = HalfspaceIntersection(halfspaces, center)
                verts = hs.intersections[ConvexHull(hs.intersections).vertices]
                self._ax.add_patch(Polygon(verts, color='green', alpha=0.3, label='FIRI Region'))
            except: pass
            
        self._ax.set_xlim(bbox[0]-0.5, bbox[1]+0.5)
        self._ax.set_ylim(bbox[2]-0.5, bbox[3]+0.5)
        self._ax.set_title("DCBF Controller + FIRI Visualization")
        plt.pause(0.001)

    def _extract_obstacle_vertices(self, obstacles):
        obs_list = []
        for obs in obstacles:
            if hasattr(obs, 'vertices'):
                v = np.array(obs.vertices)
                if v.shape[0] > 0: obs_list.append(v)
            elif hasattr(obs, 'x_min'): 
                v = np.array([
                    [obs.x_min, obs.y_min], [obs.x_max, obs.y_min],
                    [obs.x_max, obs.y_max], [obs.x_min, obs.y_max]
                ])
                obs_list.append(v)
            elif hasattr(obs, 'get_convex_rep'):
                A, b = obs.get_convex_rep()
                v = self._vertices_from_h_rep(A, b)
                if v is not None: obs_list.append(v)
        return obs_list

    def _get_robot_seed(self, system):
        # Extract vertices for current state
        if hasattr(system._geometry, 'equiv_rep'):
            comps = system._geometry.equiv_rep()
        else:
            comps = [system._geometry]
            
        state = system._state._x
        # KinematicCar: [x, y, v, theta] -> theta is index 3
        x, y, theta = state[0], state[1], state[3]
        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        verts = []
        for comp in comps:
            v_local = self._extract_convex_region_vertices(comp)
            if v_local is not None:
                v_global = (v_local @ R.T) + np.array([x, y])
                verts.append(v_global)
        
        if verts:
            return np.vstack(verts)
        return np.array([[x, y]])

    def _extract_convex_region_vertices(self, region):
        if hasattr(region, 'vertices') and region.vertices is not None:
             return np.array(region.vertices)
        if hasattr(region, 'x_min'):
            return np.array([
                [region.x_min, region.y_min], [region.x_max, region.y_min],
                [region.x_max, region.y_max], [region.x_min, region.y_max]
            ])
        if hasattr(region, 'get_convex_rep'):
            A, b = region.get_convex_rep()
            return self._vertices_from_h_rep(A, b)
        return None

    def _vertices_from_h_rep(self, A, b):
        try:
            from scipy.optimize import linprog
            norm_A = np.linalg.norm(A, axis=1)
            c = [0, 0, -1]
            A_lp = np.column_stack((A, norm_A))
            res = linprog(c, A_ub=A_lp, b_ub=b, bounds=(None, None))
            if res.success:
                center = res.x[:2]
                hs = HalfspaceIntersection(np.column_stack((A, -b)), center)
                return hs.intersections[ConvexHull(hs.intersections).vertices]
        except: pass
        return None