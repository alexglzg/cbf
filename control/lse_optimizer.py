import datetime
import casadi as ca
import numpy as np

# --- Helper ---
def smooth_min(e, alpha):
    # smooth_min(x) = - (1/alpha) * log( sum( exp( -alpha * x ) ) )
    y = -e
    y_max = ca.mmax(y)
    smooth_max_y = y_max + (1.0/alpha) * ca.log( ca.sum1( ca.exp( alpha * (y - y_max) ) ) )
    return -smooth_max_y

class NmpcLseOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times     = []   # total wall time per step [s]
        self.feval_times      = []   # f+g evaluation time per step [s]
        self.kkt_times        = []   # total - feval time per step [s]
        self.iterations       = []   # solver iterations per step
        self.infeasible_steps = []   # True when solver raised RuntimeError
        self.warm_infeasible_steps = []   # True when solver raised RuntimeError
        self.n_variables_steps = []  # variable count per step
        self.n_eq_steps       = []   # equality constraint count per step
        self.n_ineq_steps     = []   # inequality constraint count per step
        self.n_halfplanes_steps = []   # number of halfplanes (rows of A) per trajectory
        # Stage-wise storage for Fatrop (block diagonal structure)
        self.x = []      # x_0, x_1, ..., x_N
        self.u = []      # u_0, u_1, ..., u_{N-1}

        self._prev_x = None
        self._prev_u = None

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        """Initialize variables in true stage-wise format."""
        self.x = []
        self.u = []
        
        # Create stage variables: [x_k, u_k] for k=0..N-1
        for k in range(param.horizon):
            x_kp1 = self.opti.variable(4, 1)
            u_k = self.opti.variable(2, 1)
            self.u.append(u_k)
            self.x.append(x_kp1)
        
        self.x.append(self.opti.variable(4, 1))

    def add_initial_condition_constraint(self, x_0):
        """Set initial state value."""
        self.opti.subject_to(x_0 == self.state._x)
        # self.opti.set_value(self.x[0], self.state._x)

    def add_input_constraint(self, param, u_k):
        """Add input box constraints - local to each stage."""
        # amin, amax = -0.5, 0.5
        # omegamin, omegamax = -0.5, 0.5
        amin, amax = -2.0, 2.0
        omegamin, omegamax = -2.0, 2.0
        self.opti.subject_to(amin <= u_k[0])
        self.opti.subject_to(u_k[0] <= amax)
        self.opti.subject_to(omegamin <= u_k[1])
        self.opti.subject_to(u_k[1] <= omegamax)

    def add_input_derivative_constraint(self, param):
        """Add input rate constraints - coupling adjacent stages."""
        jerk_min, jerk_max = -1.0, 1.0
        omegadot_min, omegadot_max = -0.5, 0.5
        
        # First stage relative to previous input
        self.opti.subject_to(self.opti.bounded(jerk_min, self.u[0][0] - self.state._u[0], jerk_max))
        self.opti.subject_to(self.opti.bounded(omegadot_min, self.u[0][1] - self.state._u[1], omegadot_max))
        
        # Rate constraints between stages
        for k in range(len(self.u) - 1):
            self.opti.subject_to(self.opti.bounded(jerk_min, self.u[k+1][0] - self.u[k][0], jerk_max))
            self.opti.subject_to(self.opti.bounded(omegadot_min, self.u[k+1][1] - self.u[k][1], omegadot_max))

    def add_dynamics_constraint(self, param, x_k, u_k, xk1):
        """Add dynamics as stage coupling: x_{k+1} = f(x_k, u_k)."""
        self.opti.subject_to(xk1 == self.dynamics_opt(x_k, u_k))

    def add_obstacle_avoidance_constraint(self, param, system, safe_polytope, robot_local_verts, x_k, u_k):
        """Add obstacle avoidance constraints - local to each stage."""
        A_safe, b_safe = safe_polytope
        self.A = A_safe #to log number of halfplanes
        if A_safe is None or A_safe.shape[0] == 0: 
            return

        max_approx = 5e-3
        verts_local = ca.DM(robot_local_verts.T)
        n_cons = A_safe.shape[0] * verts_local.shape[1]
        alpha = ca.log(n_cons) / max_approx

        # print("shape of A: ", A_safe.shape)

        rob_vertices_xk = self.robot_vertices_ca(x_k, verts_local)
        x_k1 = self.dynamics_opt(x_k, u_k)
        rob_vertices_xkp1 = self.robot_vertices_ca(x_k1, verts_local)
        
        # Local obstacle constraints for this stage
        for j in range(rob_vertices_xk.shape[0]):
            for l in range(A_safe.shape[0]):
                dist_xk = b_safe[l] - ca.dot(ca.MX(A_safe[l]), rob_vertices_xk[j, :].T)
                dist_xkp1 = b_safe[l] - ca.dot(ca.MX(A_safe[l]), rob_vertices_xkp1[j, :].T)
                self.opti.subject_to(dist_xkp1 >= param.gamma * dist_xk + (1 - param.gamma) * param.margin_dist)
        
        # # Add obstacle avoidance constraints for each stage
        # for k in range(min(param.horizon_dcbf, len(self.u))):
        #     x_k = self.x[k]
        #     x_kp1 = self.x[k+1]
            
        #     rob_vertices_xk = self.robot_vertices_ca(x_k, verts_local)
        #     rob_vertices_xkp1 = self.robot_vertices_ca(x_kp1, verts_local)
            
        #     # Local obstacle constraints for this stage
        #     for j in range(rob_vertices_xk.shape[0]):
        #         for l in range(A_safe.shape[0]):
        #             dist_xk = b_safe[l] - ca.dot(ca.MX(A_safe[l]), rob_vertices_xk[j, :].T)
        #             dist_xkp1 = b_safe[l] - ca.dot(ca.MX(A_safe[l]), rob_vertices_xkp1[j, :].T)
        #             self.opti.subject_to(dist_xkp1 >= param.gamma * dist_xk)

    def robot_vertices_ca(self, state, verts_local):
        x = state[0]
        y = state[1]
        theta = state[3]
        c, s = ca.cos(theta), ca.sin(theta)
        R = ca.vertcat(ca.horzcat(c, -s), ca.horzcat(s, c))
        verts_global = ca.mtimes(R, verts_local) + ca.vertcat(x, y)
        return verts_global.T

    def _compute_lse_val(self, state, A, b, verts_local, margin, alpha):
        # FIX: State index for Kinematic Car is [x, y, v, theta]
        x = state[0]
        y = state[1]
        # v = state[2] # Not used for geometry
        theta = state[3] # THIS WAS THE BUG (was state[2])

        c, s = ca.cos(theta), ca.sin(theta)
        R = ca.vertcat(ca.horzcat(c, -s), ca.horzcat(s, c))
        
        verts_global = ca.mtimes(R, verts_local) + ca.vertcat(x, y)
        margins = b - ca.mtimes(A, verts_global)
        
        return smooth_min(ca.vec(margins), alpha)

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        """Add reference tracking costs - each stage independent."""
        self.costs["reference_trajectory_tracking"] = 0

        for k in range(len(self.x) - 1):   # k = 0..N-1
            x_k = self.x[k + 1]            # decision variables x[1..N]
            x_diff = x_k - reference_trajectory[k, :]   # reference[0..N-1]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        
        # Terminal cost
        x_terminal = self.x[-1]
        x_diff = x_terminal - reference_trajectory[-1, :]
        self.costs["reference_trajectory_tracking"] += param.terminal_weight * ca.mtimes(
            x_diff.T, ca.mtimes(param.mat_Q, x_diff)
        )

    def add_input_stage_cost(self, param):
        """Add input costs - each stage independent."""
        self.costs["input_stage"] = 0
        for k in range(len(self.u)):
            u_k = self.u[k]
            self.costs["input_stage"] += ca.mtimes(
                u_k.T, ca.mtimes(param.mat_R, u_k)
            )

    def add_prev_input_cost(self, param):
        """Penalize deviation from previous input - only first stage."""
        if len(self.u) > 0:
            u_0 = self.u[0]
            self.costs["prev_input"] = ca.mtimes(
                (u_0 - self.state._u).T,
                ca.mtimes(param.mat_Rold, (u_0 - self.state._u)),
            )

    def add_input_smoothness_cost(self, param):
        """Add input smoothness costs - couple adjacent stages."""
        self.costs["input_smoothness"] = 0
        for k in range(len(self.u) - 1):
            u_k = self.u[k]
            u_kp1 = self.u[k+1]
            self.costs["input_smoothness"] += ca.mtimes(
                (u_kp1 - u_k).T,
                ca.mtimes(param.mat_dR, (u_kp1 - u_k)),
            )
            
    def add_warm_start(self, param, system, cold_start, local_trajectory = []):
        """Set warm start initial values using stage-wise variables."""
        if self._prev_x is None or self._prev_u is None or cold_start:
            print("COLD START MPC!")
            if not local_trajectory.size == 0:
                self.opti.set_initial(self.x[0], system._state._x)
                for k in range(1, len(self.x)):
                    self.opti.set_initial(self.x[k], local_trajectory[k - 1, :])
                # TODO add dynamics roll-out
                for k in range(len(self.u)):
                    self.opti.set_initial(self.u[k], 0.0)
                return
            else:
                # First step: fall back to nominal controller
                x_ws, u_ws = system._dynamics.nominal_safe_controller(
                    self.state._x, 0.1, -1.0, 1.0
                )
                for k in range(len(self.x)):
                    self.opti.set_initial(self.x[k], x_ws)
                for k in range(len(self.u)):
                    self.opti.set_initial(self.u[k], u_ws)
                return

        N = param.horizon

        # Shift states: x[k] <- prev_x[k+1]  for k = 0..N-1
        # Fill terminal: repeat last state
        for k in range(N):
            self.opti.set_initial(self.x[k], self._prev_x[k + 1])
        self.opti.set_initial(self.x[N], self._prev_x[N])  # hold last

        # Shift inputs: u[k] <- prev_u[k+1]  for k = 0..N-2
        # Fill last input: repeat previous last input
        for k in range(N - 1):
            self.opti.set_initial(self.u[k], self._prev_u[k + 1])
        self.opti.set_initial(self.u[N - 1], self._prev_u[N - 1])  # hold last

    def setup(self, param, system, reference_trajectory, obstacles, robot_local_verts, cold_start = False):
        """Setup optimization problem with proper ordering: variables → constraints → costs → warm start."""
        self.set_state(system._state)
        self.opti = ca.Opti()
        
        # 1. Initialize all variables
        self.initialize_variables(param)
        
        # 2. Add all constraints (in logical order)
        for i in range(len(self.u)):
            self.add_dynamics_constraint(param, self.x[i], self.u[i], self.x[i + 1])
            if i == 0:
                self.add_initial_condition_constraint(self.x[i])
            else:
                self.add_input_constraint(param, self.u[i])
                # self.add_input_derivative_constraint(param)
                if i < param.horizon_dcbf:
                    self.add_obstacle_avoidance_constraint(param, system, obstacles, robot_local_verts, self.x[i], self.u[i])
        
        # # 3. Add all costs
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        
        # 4. Set warm start
        self.add_warm_start(param, system, cold_start, reference_trajectory)

    def solve_nlp(self, warm_start):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"fatrop.print_level": 5, "print_time": 1, "expand": True,
                  "fatrop.max_iter": 250, "fatrop.tol": 1e-4, "fatrop.mu_init": 1e-1,
                  "structure_detection": "auto", "debug": True}
        self.opti.solver("fatrop", option)
        # option = {"ipopt.print_level": 5, "print_time": 1, "expand": True,
        #           "ipopt.max_iter": 250, "ipopt.tol": 1e-4}
        # self.opti.solver("ipopt", option)

        # ── problem-size snapshot ─────────────────────────────────────────
        self.nr_variables   = self.opti.nx
        self.nr_constraints = self.opti.ng
        # print("Nr variables: ",   self.nr_variables)
        # print("Nr constraints: ", self.nr_constraints)

        # ── equality / inequality split (recomputed every step) ───────────
        try:
            lbg = np.array(self.opti.lbg, dtype=float)
            ubg = np.array(self.opti.ubg, dtype=float)
            # Find equality constraints where lower bound equals upper bound
            eq_mask = np.isclose(lbg, ubg, atol=1e-10)
            n_eq   = int(np.sum(eq_mask))
            n_ineq = self.nr_constraints - n_eq
        except Exception as e:
            # Fallback: assume all constraints are inequality
            n_eq   = 0
            n_ineq = self.nr_constraints if self.nr_constraints is not None else None

        try:
            opt_sol = self.opti.solve()
            stats   = opt_sol.stats()

            sd_time = stats['fatrop']['compute_sd_time']
            if sd_time >= 10.0:
                # Seems to be bug in fatrop that it reports is very large
                sol_time = stats.get('t_wall_total') - sd_time + sd_time / 1000
            else:    
                sol_time = stats.get('t_wall_total', float('nan'))
            t_feval  = (stats.get('t_wall_nlp_f', 0.0)
                      + stats.get('t_wall_nlp_g', 0.0)
                      + stats.get('t_wall_nlp_grad_f', 0.0)
                      + stats.get('t_wall_nlp_jac_g', 0.0)
                      + stats.get('t_wall_nlp_hess_l', 0.0))
            # KKT time = total solve time minus pure function evaluation time
            t_kkt    = sol_time - t_feval
            iters    = stats.get('iter_count', 0)

            self._prev_x = [opt_sol.value(xk) for xk in self.x]
            self._prev_u = [opt_sol.value(uk) for uk in self.u]

            # if sol_time > 10.0:
            #     import pdb;pdb.set_trace()

            self.solver_times.append(sol_time)
            self.feval_times.append(t_feval)
            self.kkt_times.append(t_kkt)
            self.iterations.append(iters)
            self.warm_infeasible_steps.append(False)
            self.infeasible_steps.append(False)
            self.n_variables_steps.append(self.nr_variables)
            self.n_eq_steps.append(n_eq)
            self.n_ineq_steps.append(n_ineq)
            self.n_halfplanes_steps.append(int(self.A.shape[0]))
            print(f"solver time: {sol_time:.4f}s  iters: {iters}")
            return opt_sol

        except RuntimeError as e:
            print(f"[LSE] Solver failed: {e}")

            if "time>=0" in str(e):
                print(
                    f"[LSE] Solver failed with time>=0 assertion ")
                self.opti.solve()

            if warm_start:
                stats   = self.opti.stats()
                sol_time = stats.get('t_wall_total', float('nan'))
                t_feval  = (stats.get('t_wall_nlp_f', 0.0)
                        + stats.get('t_wall_nlp_g', 0.0)
                        + stats.get('t_wall_nlp_grad_f', 0.0)
                        + stats.get('t_wall_nlp_jac_g', 0.0)
                        + stats.get('t_wall_nlp_hess_l', 0.0))
                # KKT time = total solve time minus pure function evaluation time
                t_kkt    = sol_time - t_feval
                iters    = stats.get('iter_count', 0)

                self.solver_times.append(sol_time)
                self.feval_times.append(t_feval)
                self.kkt_times.append(t_kkt)
                self.iterations.append(iters)
                self.warm_infeasible_steps.append(True)
                self.infeasible_steps.append(False)
                self.n_variables_steps.append(self.nr_variables)
                self.n_eq_steps.append(n_eq)
                self.n_ineq_steps.append(n_ineq)
                self.n_halfplanes_steps.append(int(self.A.shape[0]))
                return None
            else:
                self.solver_times.append(float('nan'))
                self.feval_times.append(float('nan'))
                self.kkt_times.append(float('nan'))
                self.iterations.append(0)
                self.warm_infeasible_steps.append(True)
                self.infeasible_steps.append(True)
                self.n_variables_steps.append(self.nr_variables)
                self.n_eq_steps.append(n_eq)
                self.n_ineq_steps.append(n_ineq)
                self.n_halfplanes_steps.append(int(self.A.shape[0]))
                return None