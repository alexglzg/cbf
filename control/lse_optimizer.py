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
        self.solver_times = []
        self.iterations = []

    def set_state(self, state):
        self.state = state

    def initialize_variables(self, param):
        # Not different from DCBF as their variables are added in the add_point_to_convex function
        self.variables["x"] = self.opti.variable(4, param.horizon + 1)
        self.variables["u"] = self.opti.variable(2, param.horizon)

    def add_initial_condition_constraint(self):
        self.opti.subject_to(self.variables["x"][:, 0] == self.state._x)

    def add_input_constraint(self, param):
        amin, amax = -0.5, 0.5
        omegamin, omegamax = -0.5, 0.5
        for i in range(param.horizon):
            self.opti.subject_to(self.variables["u"][0, i] <= amax)
            self.opti.subject_to(amin <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= omegamax)
            self.opti.subject_to(omegamin <= self.variables["u"][1, i])

    def add_input_derivative_constraint(self, param):
        jerk_min, jerk_max = -1.0, 1.0
        omegadot_min, omegadot_max = -0.5, 0.5
        for i in range(param.horizon - 1):
            self.opti.subject_to(self.opti.bounded(jerk_min, self.variables["u"][0, i + 1] - self.variables["u"][0, i], jerk_max))
            self.opti.subject_to(self.opti.bounded(omegadot_min, self.variables["u"][1, i + 1] - self.variables["u"][1, i], omegadot_max))
        
        # Initial step
        self.opti.subject_to(self.opti.bounded(jerk_min, self.variables["u"][0, 0] - self.state._u[0], jerk_max))
        self.opti.subject_to(self.opti.bounded(omegadot_min, self.variables["u"][1, 0] - self.state._u[1], omegadot_max))

    def add_dynamics_constraint(self, param):
        for i in range(param.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )

    def add_obstacle_avoidance_constraint(self, param, system, safe_polytope, robot_local_verts):
        
        A_safe, b_safe = safe_polytope # safe_polytope is already in the form of (A, b)
        if A_safe is None or A_safe.shape[0] == 0: 
            return

        max_approx = 5e-3
        verts_local = ca.DM(robot_local_verts.T)
        n_cons = A_safe.shape[0] * verts_local.shape[1]
        alpha = ca.log(n_cons) / max_approx

        for k in range(param.horizon_dcbf):
            h_xk = self._compute_lse_val(self.variables["x"][:, k], A_safe, b_safe, verts_local, max_approx, alpha)
            
            # Predict x_next
            x_next = self.dynamics_opt(self.variables["x"][:, k], self.variables["u"][:, k])
            h_xk1 = self._compute_lse_val(x_next, A_safe, b_safe, verts_local, max_approx, alpha)
            
            self.opti.subject_to(h_xk1 >= param.gamma * h_xk + (1 - param.gamma) * max_approx)

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
        
        return -ca.logsumexp(ca.vec(-margins), margin)
        # return smooth_min(ca.vec(margins), alpha) # Change to casadi version of logsumexp

    def add_reference_trajectory_tracking_cost(self, param, reference_trajectory):
        self.costs["reference_trajectory_tracking"] = 0
        for i in range(param.horizon - 1):
            x_diff = self.variables["x"][:, i] - reference_trajectory[i, :]
            self.costs["reference_trajectory_tracking"] += ca.mtimes(x_diff.T, ca.mtimes(param.mat_Q, x_diff))
        x_diff = self.variables["x"][:, -1] - reference_trajectory[-1, :]
        self.costs["reference_trajectory_tracking"] += param.terminal_weight * ca.mtimes(
            x_diff.T, ca.mtimes(param.mat_Q, x_diff)
        )

    def add_input_stage_cost(self, param):
        self.costs["input_stage"] = 0
        for i in range(param.horizon):
            self.costs["input_stage"] += ca.mtimes(
                self.variables["u"][:, i].T, ca.mtimes(param.mat_R, self.variables["u"][:, i])
            )

    def add_prev_input_cost(self, param):
        self.costs["prev_input"] = ca.mtimes(
            (self.variables["u"][:, 0] - self.state._u).T,
            ca.mtimes(param.mat_Rold, (self.variables["u"][:, 0] - self.state._u)),
        )

    def add_input_smoothness_cost(self, param):
        self.costs["input_smoothness"] = 0
        for i in range(param.horizon - 1):
            self.costs["input_smoothness"] += ca.mtimes(
                (self.variables["u"][:, i + 1] - self.variables["u"][:, i]).T,
                ca.mtimes(param.mat_dR, (self.variables["u"][:, i + 1] - self.variables["u"][:, i])),
            )
            
    def add_warm_start(self, param, system):
        x_ws, u_ws = system._dynamics.nominal_safe_controller(self.state._x, 0.1, -1.0, 1.0)
        for i in range(param.horizon):
            self.opti.set_initial(self.variables["x"][:, i + 1], x_ws)
            self.opti.set_initial(self.variables["u"][:, i], u_ws)

    def setup(self, param, system, reference_trajectory, obstacles, robot_local_verts):
        self.set_state(system._state)
        self.opti = ca.Opti()
        self.initialize_variables(param)
        self.add_initial_condition_constraint()
        self.add_input_constraint(param)
        self.add_input_derivative_constraint(param)
        self.add_dynamics_constraint(param)
        self.add_reference_trajectory_tracking_cost(param, reference_trajectory)
        self.add_input_stage_cost(param)
        self.add_prev_input_cost(param)
        self.add_input_smoothness_cost(param)
        self.add_obstacle_avoidance_constraint(param, system, obstacles, robot_local_verts)
        self.add_warm_start(param, system)

    def solve_nlp(self):
        cost = 0
        for cost_name in self.costs:
            cost += self.costs[cost_name]
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 5, "print_time": 1, "expand": False, "ipopt.linear_solver": "ma27"}

        self.nr_constraints = self.opti.ng
        self.nr_variables = self.opti.nx
        print("Nr variables: ", self.nr_variables)
        print("Nr constraints: ", self.nr_constraints)

        # start_timer = datetime.datetime.now()
        self.opti.solver("ipopt", option)
        opt_sol = self.opti.solve()
        sol_time = opt_sol.stats()['t_wall_total']
        iters = opt_sol.stats()['iter_count']
        import pdb;pdb.set_trace()
        # end_timer = datetime.datetime.now()
        # delta_timer = end_timer - start_timer
        # self.solver_times.append(delta_timer.total_seconds())
        self.iterations.append(iters)
        print("solver time: ", sol_time)
        return opt_sol

        # try:
        #     # start_timer = datetime.datetime.now()
        #     self.opti.solver("ipopt", option)
        #     opt_sol = self.opti.solve()
        #     sol_time = opt_sol.stats()['t_wall_total']
        #     iters = opt_sol.stats()['iter_count']
        #     import pdb;pdb.set_trace()
        #     # end_timer = datetime.datetime.now()
        #     # delta_timer = end_timer - start_timer
        #     # self.solver_times.append(delta_timer.total_seconds())
        #     self.iterations.append(iters)
        #     print("solver time: ", sol_time)
        #     return opt_sol
        # except RuntimeError:
        #     return None
