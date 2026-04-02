import numpy as np
import casadi as ca

class FIRI:
    def __init__(self):
        # We define the solver creation as a method, not a pre-compiled function
        # to ensure robustness during debugging.
        pass

    def compute(self, obstacles, seed_vertices, bbox, max_iter=3):
        d_k = np.mean(seed_vertices, axis=0)
        radius = np.max(np.linalg.norm(seed_vertices - d_k, axis=1)) + 0.05
        C_k = np.eye(2) * radius 

        xmin, xmax, ymin, ymax = bbox
        # BBox definition n^T x <= p
        bbox_planes = [
            (np.array([-1.0, 0.0]), -xmin),
            (np.array([ 1.0, 0.0]),  xmax),
            (np.array([ 0.0,-1.0]), -ymin),
            (np.array([ 0.0, 1.0]),  ymax)
        ]

        current_planes = bbox_planes
        prev_vol = 0.0

        for k in range(max_iter):
            # 1. Transform points
            try:
                O_bar_list = [self._transform_to_ellipsoid(obs, C_k, d_k) for obs in obstacles]
                Q_bar = self._transform_to_ellipsoid(seed_vertices, C_k, d_k)
            except:
                break
            
            candidate_planes = []
            
            # 2. RsI Step - Solve for EVERY obstacle
            for i, obs_bar in enumerate(O_bar_list):
                # Solve QP on the fly to guarantee correctness
                b_vec = self._solve_rsi_robust(obs_bar, Q_bar)
                
                if b_vec is not None:
                    n, p = self._plane_to_world(b_vec, C_k, d_k)
                    candidate_planes.append({'n': n, 'p': p, 'b_local': b_vec})
            
            # 3. No Pruning - Keep All
            keep_planes = [(c['n'], c['p']) for c in candidate_planes]
            polytope_planes = bbox_planes + keep_planes
            
            # 4. MVIE Step
            try:
                C_new, d_new = self._solve_mvie(polytope_planes, C_k, d_k)
            except:
                break 

            vol_new = np.linalg.det(C_new)
            
            if k > 0 and vol_new < prev_vol * 0.99: break 
            if k > 0 and (vol_new / (prev_vol + 1e-9) < 1.02):
                current_planes = polytope_planes
                break
            
            current_planes = polytope_planes
            C_k, d_k = C_new, d_new
            prev_vol = vol_new

        A = np.array([p[0] for p in current_planes])
        b = np.array([p[1] for p in current_planes])
        return A, b

    def _solve_rsi_robust(self, obs_verts, seed_verts):
        """
        Constructs and solves the QP on the fly.
        Slower but guarantees no memory/shape garbage.
        """
        opti = ca.Opti()
        b = opti.variable(2)
        
        opti.minimize(ca.dot(b, b))
        
        # Constraints for Obstacle Vertices (All >= 1)
        for i in range(obs_verts.shape[0]):
            opti.subject_to(ca.dot(obs_verts[i,:].T, b) >= 1.0)
            
        # Constraints for Seed Vertices (All <= 0.999)
        for i in range(seed_verts.shape[0]):
            opti.subject_to(ca.dot(seed_verts[i,:].T, b) <= 0.999)

        opts = {'print_time': False, 'ipopt': {'print_level': 0, 'sb': 'yes'}}
        opti.solver('ipopt', opts)
        
        try:
            sol = opti.solve()
            return sol.value(b)
        except:
            return None

    def _solve_mvie(self, planes, init_C, init_d):
        opti = ca.Opti()
        L = opti.variable(2, 2)
        d = opti.variable(2)
        
        opti.subject_to(L[0, 1] == 0)
        opti.subject_to(L[0, 0] >= 0.01)
        opti.subject_to(L[1, 1] >= 0.01)
        opti.minimize(-ca.log(L[0,0] * L[1,1]) + 0.1*(L[0,0]+L[1,1]))
        
        for (n, p) in planes:
            opti.subject_to(ca.dot(n, d) + ca.norm_2(ca.mtimes(L.T, n)) <= p - 1e-4)
            
        try:
            L_init = np.linalg.cholesky(init_C)
            opti.set_initial(L, L_init)
            opti.set_initial(d, init_d)
        except: pass

        opts = {'print_time': False, 'ipopt': {'print_level': 0, 'sb': 'yes'}}
        opti.solver('ipopt', opts)
        sol = opti.solve()
        L_val = sol.value(L)
        return L_val @ L_val.T, sol.value(d)

    def _transform_to_ellipsoid(self, pts, C, d):
        return np.linalg.solve(C, (pts - d).T).T

    def _plane_to_world(self, b_vec, C, d):
        b_sq = np.dot(b_vec, b_vec)
        if b_sq < 1e-8: return np.array([1.0,0.0]), 10.0
        
        a_bar = b_vec / b_sq
        n_world = np.linalg.solve(C, a_bar)
        p_world = np.dot(a_bar, a_bar) + np.dot(n_world, d)
        
        norm = np.linalg.norm(n_world)
        return n_world / norm, p_world / norm