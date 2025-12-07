import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull, HalfspaceIntersection

from control.dcbf_optimizer import NmpcDbcfOptimizer, NmpcDcbfOptimizerParam
from control.firi_polytope import FIRI

# Set to True to enable plotting
DEBUG_VIS = True

class NmpcDcbfController:
    def __init__(self, dynamics=None, opt_param=None):
        self._param = opt_param
        # Original DCBF Optimizer
        self._optimizer = NmpcDbcfOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
        
        # FIRI for Visualization ONLY
        self._firi = FIRI()
        self._fig = None
        self._ax = None

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        # --- 1. VISUALIZATION STEP (FIRI) ---
        if DEBUG_VIS:
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
        return self._opt_sol.value(self._optimizer.variables["u"][:, 0])

    def logging(self, logger):
        logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
        logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"]).T)

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