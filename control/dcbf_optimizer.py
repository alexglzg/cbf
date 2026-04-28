import datetime

import casadi as ca
import numpy as np

from models.geometry_utils import *


class NmpcDcbfOptimizerParam:
    def __init__(self):
        self.horizon = 30
        self.horizon_dcbf = 20
        self.mat_Q = np.diag([2.0, 2.0, 1.0, 1.0])
        self.mat_R = np.diag([0.0, 0.0])
        self.mat_Rold = np.diag([1.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0
        self.gamma = 0.8
        self.pomega = 10.0
        self.margin_dist = 0.1 # 0.1
        self.terminal_weight = 2.0


class NmpcDbcfOptimizer:
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

        self._prev_x = None  # for warm start: previous solution's state trajectory
        self._prev_u = None  # for warm start: previous solution's input trajectory

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param, system, obstacles):
        """Initialize variables in true stage-wise format."""
        self.x = []
        self.u = []
        self.lamb = []
        self.mu = []
        self.omega = []

        robot_components = system._geometry.equiv_rep()
        robot_comp = robot_components[0]  # Assuming single component for simplicity
        robot_G, robot_g = robot_comp.get_convex_rep()
        
        # Create stage variables: [x_k, u_k] for k=0..N-1
        for k in range(param.horizon):
            x_kp1 = self.opti.variable(4, 1)
            self.x.append(x_kp1)
            if k < param.horizon_dcbf:
                for obs_geo in obstacles:
                    mat_A, _ = obs_geo.get_convex_rep()
                    lamb_k = self.opti.variable(mat_A.shape[0], 1)
                    mu_k = self.opti.variable(robot_G.shape[0], 1)
                    omega_k = self.opti.variable(1, 1)
                    self.lamb.append(lamb_k)
                    self.mu.append(mu_k)
                    self.omega.append(omega_k)
            u_k = self.opti.variable(2, 1)
            self.u.append(u_k)

            # for i in range(obstacles_A.shape[0]):
            #     lamb_k = self.opti.variable(obstacles_A[i].shape[0], 1)
            #     mu_k = self.opti.variable(robot_A.shape[0], 1)
            #     omega_k = self.opti.variable(1, 1)
            #     self.lamb.append(lamb_k)
            #     self.mu.append(mu_k)
            #     self.omega.append(omega_k)
            # u_k = self.opti.variable(2, 1)
            # self.u.append(u_k)
        
        self.x.append(self.opti.variable(4, 1))  # Terminal state variable
            

    def add_initial_condition_constraint(self, x_0):
        """Set initial state value."""
        self.opti.subject_to(x_0 == self.state._x)

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
        self.costs["prev_input"] = 0
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

    def add_point_to_convex_constraint(self, param, obs_geo, safe_dist):
        """Add DCBF constraint for point-to-convex - local to each stage."""
        # get current value of cbf
        mat_A, vec_b = obs_geo.get_convex_rep()
        cbf_curr, lamb_curr = get_dist_point_to_region(self.state._x[0:2], mat_A, vec_b)
        # filter obstacle if it's still far away
        if cbf_curr > safe_dist:
            return
        
        # duality-cbf constraints - one per stage
        for k in range(min(param.horizon_dcbf, len(self.x))):
            lamb = self.opti.variable(mat_A.shape[0], 1)
            omega = self.opti.variable(1, 1)
            
            x_k = self.x[k]
            self.opti.subject_to(lamb >= 0)
            self.opti.subject_to(
                ca.mtimes((ca.mtimes(mat_A, x_k[0:2]) - vec_b).T, lamb)
                >= omega * param.gamma ** k * (cbf_curr - param.margin_dist) + param.margin_dist
            )
            temp = ca.mtimes(mat_A.T, lamb)
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega - 1) ** 2
            # warm start
            self.opti.set_initial(lamb, lamb_curr)
            self.opti.set_initial(omega, 0.1)

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo, safe_dist, x_k, lambda_k, mu_k, omega_k, k, reconfigure):
        """Add DCBF constraint for convex-to-convex - local to each stage."""
        mat_A, vec_b = obs_geo.get_convex_rep()
        robot_G, robot_g = robot_geo.get_convex_rep()

        # get current value of cbf
        cbf_curr, lamb_curr, mu_curr = get_dist_region_to_region(
            mat_A,
            vec_b,
            np.dot(robot_G, self.state.rotation().T),
            np.dot(np.dot(robot_G, self.state.rotation().T), self.state.translation()) + robot_g,
        )

        # filter obstacle if it's still far away
        if reconfigure:
            if cbf_curr > safe_dist:
                return 0
        
        robot_R = ca.hcat(
                [
                    ca.vcat(
                        [
                            ca.cos(x_k[3]),
                            ca.sin(x_k[3]),
                        ]
                    ),
                    ca.vcat(
                        [
                            -ca.sin(x_k[3]),
                            ca.cos(x_k[3]),
                        ]
                    ),
                ]
            )
        robot_T = x_k[0:2]
        self.opti.subject_to(lambda_k >= 0)
        self.opti.subject_to(mu_k >= 0)

        self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu_k) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lambda_k)
                >= omega_k * param.gamma ** k * (cbf_curr - param.margin_dist) + param.margin_dist
            )
        self.opti.subject_to(
            ca.mtimes(robot_G.T, mu_k) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lambda_k) == 0
        )
        temp = ca.mtimes(mat_A.T, lambda_k)
        self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
        self.opti.subject_to(omega_k >= 0)
        self.costs["decay_rate_relaxing"] += param.pomega * (omega_k - 1) ** 2
        
        # # warm start
        self.opti.set_initial(lambda_k, lamb_curr)
        self.opti.set_initial(mu_k, mu_curr)
        self.opti.set_initial(omega_k, 0.1)
        
        # # duality-cbf constraints - one per stage
        # for k in range(min(param.horizon_dcbf, len(self.x))):
        #     lamb = self.opti.variable(mat_A.shape[0], 1)
        #     mu = self.opti.variable(robot_G.shape[0], 1)
        #     omega = self.opti.variable(1, 1)
            
        #     x_k = self.x[k]
        #     robot_R = ca.hcat(
        #         [
        #             ca.vcat(
        #                 [
        #                     ca.cos(x_k[3]),
        #                     ca.sin(x_k[3]),
        #                 ]
        #             ),
        #             ca.vcat(
        #                 [
        #                     -ca.sin(x_k[3]),
        #                     ca.cos(x_k[3]),
        #                 ]
        #             ),
        #         ]
        #     )
        #     robot_T = x_k[0:2]
        #     self.opti.subject_to(lamb >= 0)
        #     self.opti.subject_to(mu >= 0)
        #     self.opti.subject_to(
        #         -ca.mtimes(robot_g.T, mu) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb)
        #         >= omega * param.gamma ** k * (cbf_curr - param.margin_dist) + param.margin_dist
        #     )
        #     self.opti.subject_to(
        #         ca.mtimes(robot_G.T, mu) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb) == 0
        #     )
        #     temp = ca.mtimes(mat_A.T, lamb)
        #     self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
        #     self.opti.subject_to(omega >= 0)
        #     self.costs["decay_rate_relaxing"] += param.pomega * (omega - 1) ** 2
        #     # warm start
        #     self.opti.set_initial(lamb, lamb_curr)
        #     self.opti.set_initial(mu, mu_curr)
        #     self.opti.set_initial(omega, 0.1)

        return 1

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo, x_k, lambd, mu, omega, k, reconfigure):
        self.costs["decay_rate_relaxing"] = 0
        # TODO: wrap params
        # TODO: move safe dist inside attribute `system`
        safe_dist = system._dynamics.safe_dist(system._state._x, 0.1, -1.0, 1.0, param.margin_dist)
        robot_components = system._geometry.equiv_rep()
        self.constr_cnt = 0
        robot_comp = robot_components[0]  # Assuming single component for simplicity

        for idx, obs_geo in enumerate(obstacles_geo):
            # TODO: need to add case for `add_point_convex_constraint()`
            k_idx = k * len(obstacles_geo) + idx
            if isinstance(robot_comp, ConvexRegion2D):
                cnt = self.add_convex_to_convex_constraint(param, robot_comp, obs_geo, safe_dist, x_k, lambd[k_idx], mu[k_idx], omega[k_idx], k, reconfigure)
                self.constr_cnt += cnt
            else:
                raise NotImplementedError()

        # print("Nr DCBF constraints added: ", self.constr_cnt)

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

    def setup(self, param, system, reference_trajectory, obstacles, cold_start = False, reconfigure = True):
        """Setup optimization problem with proper ordering: variables → constraints → costs → warm start."""

        # import pdb;pdb.set_trace()
        self.set_state(system._state)
        self.opti = ca.Opti()
        
        # 1. Initialize all variables
        # print(len(obstacles))
        self.initialize_variables(param, system, obstacles)
        
        # 2. Add all constraints (in logical order)
        for i in range(len(self.u)):
            self.add_dynamics_constraint(param, self.x[i], self.u[i], self.x[i + 1])
            if i == 0:
                self.add_initial_condition_constraint(self.x[i])
            else:
                self.add_input_constraint(param, self.u[i])
                # self.add_input_derivative_constraint(param)
                if i < param.horizon_dcbf:
                    self.add_obstacle_avoidance_constraint(param, system, obstacles, self.x[i], self.lamb, self.mu, self.omega, i, reconfigure)

        # self.set_state(system._state)
        # self.opti = ca.Opti()
        
        # # 1. Initialize all variables
        # self.initialize_variables(param)
        
        # # 2. Add all constraints (in logical order)
        # self.add_initial_condition_constraint()
        # self.add_input_constraint(param)
        # self.add_input_derivative_constraint(param)
        # self.add_dynamics_constraint(param)
        # self.add_obstacle_avoidance_constraint(param, system, obstacles)
        
        # # 3. Add all costs
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        
        # # 4. Set warm start
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
        # CasADi stores all constraints in opti.g; equalities have lbg==ubg.
        # The split can change between steps if active obstacles change, so
        # we record it as a list entry every call rather than caching it.
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
            fatrop_stats = stats.get('fatrop')

            sd_time = fatrop_stats['compute_sd_time']
            if sd_time >= 10.0:
                # Seems to be bug in fatrop that it reports very large sd_time
                sol_time = fatrop_stats.get('time_total') - sd_time + sd_time / 1000
            else:    
                # bug in fatrop/casadi interface that it passes very high comp times through casadi
                sol_time = fatrop_stats.get('time_total', float('nan'))

            # if sol_time >= 5.0:
            #     import pdb;pdb.set_trace()
            t_feval  = (stats.get('t_wall_nlp_f', 0.0)
                      + stats.get('t_wall_nlp_g', 0.0)
                      + stats.get('t_wall_nlp_grad_f', 0.0)
                      + stats.get('t_wall_nlp_jac_g', 0.0)
                      + stats.get('t_wall_nlp_hess_l', 0.0))
            # KKT time = everything that is NOT pure function evaluation:
            # linear algebra, factorisation, line-search bookkeeping, etc.
            t_kkt    = sol_time - t_feval
            iters    = stats.get('iter_count', 0)

            self._prev_x = [opt_sol.value(xk) for xk in self.x]
            self._prev_u = [opt_sol.value(uk) for uk in self.u]

            self.solver_times.append(sol_time)
            self.feval_times.append(t_feval)
            self.kkt_times.append(t_kkt)
            self.iterations.append(iters)
            self.warm_infeasible_steps.append(False)
            self.infeasible_steps.append(False)
            self.n_variables_steps.append(self.nr_variables)
            self.n_eq_steps.append(n_eq)
            self.n_ineq_steps.append(n_ineq)
            self.n_halfplanes_steps.append(0)
            print(f"solver time: {sol_time:.4f}s  iters: {iters}")
            return opt_sol

        except RuntimeError as e:
            print(f"[DCBF] Solver failed: {e}")

            if "time>=0" in str(e):
                print(
                    f"[DCBF] Solver failed with time>=0 assertion ")
                self.opti.solve()

            # self._prev_x = None
            # self._prev_u = None

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
                self.n_halfplanes_steps.append(0)
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
                self.n_halfplanes_steps.append(0)
                return None
