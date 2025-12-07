import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import ConvexHull, HalfspaceIntersection

from control.firi_polytope import FIRI
from control.lse_optimizer import NmpcLseOptimizer

DEBUG_VIS = True

class NmpcLseController:
    def __init__(self, dynamics, opt_param):
        self._param = opt_param
        self._firi = FIRI()
        self._optimizer = NmpcLseOptimizer({}, {}, dynamics.forward_dynamics_opt(0.1))
        self._opt_sol = None
        self._fig = None

    def generate_control_input(self, system, global_path, local_trajectory, obstacles):
        obs_verts_list = extract_obstacle_vertices(obstacles)
        
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
                seed_verts_list.append(v_global)
        
        if seed_verts_list:
            seed_poly = np.vstack(seed_verts_list)
        else:
            seed_poly = np.array([[x, y]])

        if all_local_verts:
            robot_local_verts = np.vstack(all_local_verts)
        else:
            robot_local_verts = np.zeros((1, 2))

        # Local BBox (Square 4x4 around robot)
        bbox = (x-2.0, x+2.0, y-2.0, y+2.0)
        
        A_safe, b_safe = self._firi.compute(obs_verts_list, seed_poly, bbox)
        
        if DEBUG_VIS:
            self._visualize(seed_poly, obs_verts_list, A_safe, b_safe, bbox)

        self._optimizer.setup(self._param, system, local_trajectory, (A_safe, b_safe), robot_local_verts)
        self._opt_sol = self._optimizer.solve_nlp()
        
        if self._opt_sol:
            return self._opt_sol.value(self._optimizer.variables["u"][:, 0])
        else:
            return np.zeros(2)

    def logging(self, logger):
        if self._opt_sol:
            logger._xtrajs.append(self._opt_sol.value(self._optimizer.variables["x"]).T)
            logger._utrajs.append(self._opt_sol.value(self._optimizer.variables["u"]).T)

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