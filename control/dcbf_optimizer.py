import datetime

import casadi as ca
import numpy as np

from models.geometry_utils import *


class NmpcDcbfOptimizerParam:
    def __init__(self):
        self.horizon = 11
        self.horizon_dcbf = 6
        self.mat_Q = np.diag([100.0, 100.0, 1.0, 1.0])
        self.mat_R = np.diag([0.0, 0.0])
        self.mat_Rold = np.diag([1.0, 1.0]) * 0.0
        self.mat_dR = np.diag([1.0, 1.0]) * 0.0
        self.gamma = 0.8
        self.pomega = 10.0
        self.margin_dist = 0.00
        self.terminal_weight = 10.0


class NmpcDbcfOptimizer:
    def __init__(self, variables: dict, costs: dict, dynamics_opt):
        self.opti = None
        self.variables = variables
        self.costs = costs
        self.dynamics_opt = dynamics_opt
        self.solver_times = []
        self.iterations = []
        # Stage-wise storage for Fatrop (block diagonal structure)
        self.x = []      # x_0, x_1, ..., x_N
        self.u = []      # u_0, u_1, ..., u_{N-1}

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        """Initialize variables in true stage-wise format."""
        self.x = []
        self.u = []
        
        # Initial state (fixed)
        self.x.append(self.opti.parameter(4, 1))
        
        # Create stage variables: [x_k, u_k] for k=0..N-1
        for k in range(param.horizon):
            u_k = self.opti.variable(2, 1)
            x_kp1 = self.opti.variable(4, 1)
            self.u.append(u_k)
            self.x.append(x_kp1)

    def add_initial_condition_constraint(self):
        """Set initial state value."""
        self.opti.set_value(self.x[0], self.state._x)

    def add_input_constraint(self, param):
        """Add input box constraints - local to each stage."""
        amin, amax = -0.5, 0.5
        omegamin, omegamax = -0.5, 0.5
        for k in range(len(self.u)):
            u_k = self.u[k]
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

    def add_dynamics_constraint(self, param):
        """Add dynamics as stage coupling: x_{k+1} = f(x_k, u_k)."""
        for k in range(len(self.u)):
            x_k = self.x[k]
            u_k = self.u[k]
            x_kp1_pred = self.dynamics_opt(x_k, u_k)
            # This is the only coupling between stages
            self.opti.subject_to(self.x[k+1] == x_kp1_pred)

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        """Add reference tracking costs - each stage independent."""
        self.costs["reference_trajectory_tracking"] = 0
        for k in range(len(self.x)):
            x_k = self.x[k]
            x_diff = x_k - reference_trajectory[k, :]
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

    def add_convex_to_convex_constraint(self, param, robot_geo, obs_geo, safe_dist):
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
        if cbf_curr > safe_dist:
            return 0
        
        # duality-cbf constraints - one per stage
        for k in range(min(param.horizon_dcbf, len(self.x))):
            lamb = self.opti.variable(mat_A.shape[0], 1)
            mu = self.opti.variable(robot_G.shape[0], 1)
            omega = self.opti.variable(1, 1)
            
            x_k = self.x[k]
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
            self.opti.subject_to(lamb >= 0)
            self.opti.subject_to(mu >= 0)
            self.opti.subject_to(
                -ca.mtimes(robot_g.T, mu) + ca.mtimes((ca.mtimes(mat_A, robot_T) - vec_b).T, lamb)
                >= omega * param.gamma ** k * (cbf_curr - param.margin_dist) + param.margin_dist
            )
            self.opti.subject_to(
                ca.mtimes(robot_G.T, mu) + ca.mtimes(ca.mtimes(robot_R.T, mat_A.T), lamb) == 0
            )
            temp = ca.mtimes(mat_A.T, lamb)
            self.opti.subject_to(ca.mtimes(temp.T, temp) <= 1)
            self.opti.subject_to(omega >= 0)
            self.costs["decay_rate_relaxing"] += param.pomega * (omega - 1) ** 2
            # warm start
            self.opti.set_initial(lamb, lamb_curr)
            self.opti.set_initial(mu, mu_curr)
            self.opti.set_initial(omega, 0.1)

        return 1

    def add_obstacle_avoidance_constraint(self, param, system, obstacles_geo):
        self.costs["decay_rate_relaxing"] = 0
        # TODO: wrap params
        # TODO: move safe dist inside attribute `system`
        safe_dist = system._dynamics.safe_dist(system._state._x, 0.1, -1.0, 1.0, param.margin_dist)
        robot_components = system._geometry.equiv_rep()
        self.constr_cnt = 0

        for obs_geo in obstacles_geo:
            for robot_comp in robot_components:
                # TODO: need to add case for `add_point_convex_constraint()`
                if isinstance(robot_comp, ConvexRegion2D):
                    cnt = self.add_convex_to_convex_constraint(param, robot_comp, obs_geo, safe_dist)
                    self.constr_cnt += cnt
                else:
                    raise NotImplementedError()

        print("Nr DCBF constraints added: ", self.constr_cnt)

    def add_warm_start(self, param, system):
        """Set warm start initial values using stage-wise variables."""
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, -1.0, 1.0)
        for k in range(len(self.x)):
            self.opti.set_initial(self.x[k], x_ws)
        for k in range(len(self.u)):
            self.opti.set_initial(self.u[k], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles):
        """Setup optimization problem with proper ordering: variables → constraints → costs → warm start."""
        self.set_state(system._state)
        self.opti = ca.Opti()
        
        # 1. Initialize all variables
        self.initialize_variables(param)
        
        # 2. Add all constraints (in logical order)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles)
        
        # 3. Add all costs
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        
        # 4. Set warm start
        self.add_warm_start(param, system)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 1, "expand": True, "ipopt.linear_solver": "mumps"}

        self.nr_constraints = self.opti.ng
        self.nr_variables = self.opti.nx
        print("Nr variables: ", self.nr_variables)
        print("Nr constraints: ", self.nr_constraints)

        # ### Plot sparisty pattern
        # opti = self.opti
        # J = ca.jacobian(opti.g, opti.x).sparsity()
        # lag = opti.f + ca.dot(opti.lam_g, opti.g)
        # H = ca.hessian(lag, opti.x)[0].sparsity()
        # import matplotlib.pylab as plt

        # plt.subplots(1, 2, figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.spy(np.array(J))
        # plt.title("Jacobian Sparsity dcbf")

        # plt.subplot(1, 2, 2)
        # plt.spy(np.array(H))
        # plt.title("Hessian Sparsity dcbf")

        # plt.show(block=True)
        
        try:
            # start_timer = datetime.datetime.now()
            self.opti.solver("ipopt", option)
            opt_sol = self.opti.solve()
            sol_time = opt_sol.stats()['t_wall_total']
            iters = opt_sol.stats()['iter_count']
            # import pdb;pdb.set_trace()
            # end_timer = datetime.datetime.now()
            # delta_timer = end_timer - start_timer
            # self.solver_times.append(delta_timer.total_seconds())
            self.iterations.append(iters)
            print("solver time: ", sol_time)
            return opt_sol
        except RuntimeError:
            return None
