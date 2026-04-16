import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull, HalfspaceIntersection

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from control.firi import FIRISolver
from control.lse_optimizer import NmpcLseOptimizer

DEBUG_VIS = True

# ── clearance helpers (identical logic to dcbf_controller) ───────────────────

def _closest_point_on_convex_poly(point: np.ndarray, verts: np.ndarray) -> np.ndarray:
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
    Return all robot vertices in world frame (M, 2).
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
            v_local = np.array(comp.vertices)
        elif hasattr(comp, 'x_min'):
            v_local = np.array([
                [comp.x_min, comp.y_min], [comp.x_max, comp.y_min],
                [comp.x_max, comp.y_max], [comp.x_min, comp.y_max],
            ])
        elif hasattr(comp, 'get_convex_rep'):
            G, g = comp.get_convex_rep()
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
    """Minimum distance between robot polygon and one obstacle polygon."""
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

    min_dist = np.inf
    for rv in robot_verts:
        cp = _closest_point_on_convex_poly(rv, obs_verts)
        min_dist = min(min_dist, float(np.linalg.norm(rv - cp)))
    for ov in obs_verts:
        cp = _closest_point_on_convex_poly(ov, robot_verts)
        min_dist = min(min_dist, float(np.linalg.norm(ov - cp)))
    return min_dist

class NmpcLseController:
    def __init__(self, dynamics, opt_param, enable_vis=True):
        self._param = opt_param
        self._enable_vis = enable_vis and DEBUG_VIS
        self.firi_solver = FIRISolver(
            sdmn_seed=42,
            mvie_outer_iters=20,
            mvie_inner_iters=10,
            mvie_mu=4.0,
        )
        self._optimizer = NmpcLseOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
        self._opt_sol = None
        self._fig = None

    def extract_obstacles_halfplanes(self, obstacles):
        obs_hps_list = []
        for obs in obstacles:
            temp = []
            if hasattr(obs, 'get_convex_rep'):
                A, b = obs.get_convex_rep()
                for i in range(A.shape[0]):
                    temp.append(np.array([A[i][0], A[i][1], b[i][0]]))
            obs_hps_list.append(temp)
        return obs_hps_list

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        obs_verts_list = extract_obstacle_vertices(obstacles)
        obs_hps_list = self.extract_obstacles_halfplanes(obstacles)
        
        if hasattr(system._geometry, 'equiv_rep'):
            robot_components = system._geometry.equiv_rep()
        else:
            robot_components = [system._geometry]

        seed_verts_list = []
        state = system._state._x
        x, y, theta = state[0], state[1], state[3]
        
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        all_local_verts = []
        for comp in robot_components:
            v_local = extract_convex_region_vertices(comp)
            if v_local is not None:
                all_local_verts.append(v_local)
                v_global = (v_local @ R.T) + np.array([x, y])
                seed_verts_list.extend(v_global)
        
        if seed_verts_list:
            seed_poly = np.vstack(seed_verts_list)
        else:
            seed_poly = np.array([[x, y]])

        if all_local_verts:
            robot_local_verts = np.vstack(all_local_verts)
        else:
            robot_local_verts = np.zeros((1, 2))

        # Local BBox (Square 4x4 around robot) as list of halfplanes [normal, offset]
        bbox_x_min, bbox_x_max = x - 2.0, x + 2.0
        bbox_y_min, bbox_y_max = y - 2.0, y + 2.0
        bbox = [
            [-1, 0, -bbox_x_min],   # x >= x_min
            [1, 0, bbox_x_max],      # x <= x_max
            [0, -1, -bbox_y_min],    # y >= y_min
            [0, 1, bbox_y_max]       # y <= y_max
        ]
        
        # Compute best ellipsoid, and polytope
        firi_result = self.firi_solver.compute_from_halfplanes(obs_hps_list, seed_verts_list, bbox, max_iter=5, rho=0.02)
        # firi_result = self.firi_solver.compute(vertex_obstacles, seed_verts_list, bbox, max_iter=5, rho=0.02, polytope_mode=True)

        # Get A and b from the FIRI result
        A_safe = np.vstack([hp.normal if hasattr(hp, 'normal') else hp[0:2] for hp in firi_result.planes])
        b_safe = np.array([hp.offset if hasattr(hp, 'offset') else hp[2] for hp in firi_result.planes])

        # A_safe, b_safe = self._firi.compute(obs_verts_list, seed_poly, bbox)
        
        if self._enable_vis:
            # Convert bbox halfplanes to tuple format for visualization
            bbox_tuple = (bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max)
            self._visualize(seed_poly, obs_verts_list, A_safe, b_safe, bbox_tuple)

        self._optimizer.setup(self._param, system, local_trajectory, (A_safe, b_safe), robot_local_verts)
        self._opt_sol = self._optimizer.solve_nlp()
        
        if self._opt_sol:
            return self._opt_sol.value(self._optimizer.u[0])
        else:
            return np.zeros(2)

    def collect_metrics(self, system, obstacles):
        """
        Compute and return a dict of per-step metrics for the current state.
        Call this immediately after generate_control_input() at every time step.
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

    def _visualize(self, seed, obstacles, A, b, bbox):
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        
        self._ax.clear()
        
        # Draw BBox
        self._ax.add_patch(Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], 
                         linewidth=1, edgecolor='r', facecolor='none', linestyle='--', label='BBox'))

        # Draw Obstacles
        for obs in obstacles:
            self._ax.add_patch(Polygon(obs, color='black', alpha=0.8))
            
        # Draw Seed
        self._ax.add_patch(Polygon(seed, color='blue', alpha=0.5))
        
        # Draw Planes (Red Lines)
        # n^T x = p  =>  nx * x + ny * y = p  =>  y = (p - nx*x) / ny
        if A.shape[0] > 0:
            xs = np.linspace(bbox[0], bbox[1], 10)
            for i in range(A.shape[0]):
                n = A[i]
                p = b[i]
                if abs(n[1]) > 0.01:
                    ys = (p - n[0]*xs) / n[1]
                    # Only plot lines roughly inside view
                    if np.any((ys > bbox[2]) & (ys < bbox[3])):
                        self._ax.plot(xs, ys, 'r-', alpha=0.3, linewidth=1)
                else:
                    # Vertical line x = p/nx
                    x_line = p/n[0]
                    self._ax.vlines(x_line, bbox[2], bbox[3], 'r', alpha=0.3)

        # Draw Polytope (Green)
        if A.shape[0] > 0:
            try:
                center = np.mean(seed, axis=0)
                halfspaces = np.column_stack((A, -b))
                hs = HalfspaceIntersection(halfspaces, center)
                verts = hs.intersections[ConvexHull(hs.intersections).vertices]
                self._ax.add_patch(Polygon(verts, color='green', alpha=0.3))
            except: pass
            
        self._ax.set_xlim(bbox[0]-0.5, bbox[1]+0.5)
        self._ax.set_ylim(bbox[2]-0.5, bbox[3]+0.5)
        plt.pause(0.001)

def extract_obstacle_vertices(simulation_obstacles):
    obs_list = []
    for obs in simulation_obstacles:
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
            v = vertices_from_h_rep(A, b)
            if v is not None: obs_list.append(v)
    return obs_list

def extract_convex_region_vertices(region):
    if hasattr(region, 'vertices') and region.vertices is not None:
         return np.array(region.vertices)
    if hasattr(region, 'x_min'):
        return np.array([
            [region.x_min, region.y_min], [region.x_max, region.y_min],
            [region.x_max, region.y_max], [region.x_min, region.y_max]
        ])
    if hasattr(region, 'get_convex_rep'):
        A, b = region.get_convex_rep()
        return vertices_from_h_rep(A, b)
    return None

def vertices_from_h_rep(A, b):
    try:
        from scipy.optimize import linprog
        norm_A = np.linalg.norm(A, axis=1)
        c = [0, 0, -1]
        A_lp = np.column_stack((A, norm_A))
        res = linprog(c, A_ub=A_lp, b_ub=b, bounds=(None, None))
        if res.success:
            center = res.x[:2]
            from scipy.spatial import HalfspaceIntersection
            hs = HalfspaceIntersection(np.column_stack((A, -b)), center)
            return hs.intersections[ConvexHull(hs.intersections).vertices]
    except: pass
    return None