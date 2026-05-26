import numpy as np


class ConstantSpeedTrajectoryGenerator:
    def __init__(self):
        self._global_path_index = 0
        self._num_waypoint = None
        self._reference_speed = 0.75
        self._num_horizon = 30
        self._local_path_timestep = 0.1
        self._local_trajectory = None
        self._proj_dist_buffer = 0.05

    def generate_trajectory(self, system, global_path):
        if self._num_waypoint is None:
            self._global_path = global_path
            self._num_waypoint = global_path.shape[0]
        pos = system._state._x[0:2]
        return self.generate_trajectory_internal(pos, self._global_path)

    def generate_trajectory_internal(self, pos, global_path):
        local_index = self._global_path_index

        # --- Advance along segments (loop instead of recursion) ---
        while local_index < self._num_waypoint - 1:
            trunc_path = np.vstack([global_path[local_index:, :], global_path[-1, :]])
            curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
            curv_length = np.linalg.norm(curv_vec, axis=1)

            if curv_length[0] == 0.0:
                curv_direct = np.zeros((2,))
            else:
                curv_direct = curv_vec[0, :] / curv_length[0]

            proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)

            if proj_dist >= curv_length[0] - self._proj_dist_buffer:
                self._global_path_index += 1
                local_index = self._global_path_index
            else:
                break

        # Rebuild truncated path from final index
        trunc_path = np.vstack([global_path[local_index:, :], global_path[-1, :]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)

        if curv_length[0] == 0.0:
            curv_direct = np.zeros((2,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]
        proj_dist = np.dot(pos - trunc_path[0, :], curv_direct)

        if proj_dist < 0.0:
            proj_dist = 0.0

        # --- Arc-length parameterisation ---
        cumul_arc = np.cumsum(np.hstack([0.0, curv_length]))

        ds = self._reference_speed * self._local_path_timestep
        s_start = proj_dist + self._proj_dist_buffer
        s_samples = s_start + ds * np.arange(self._num_horizon)
        s_samples = np.clip(s_samples, 0.0, cumul_arc[-1])

        # Interpolate x, y along arc-length
        path = np.column_stack([
            np.interp(s_samples, cumul_arc, trunc_path[:, 0]),
            np.interp(s_samples, cumul_arc, trunc_path[:, 1]),
        ])

        path_vel = self._reference_speed * np.ones((self._num_horizon, 1))

        # Heading from segment direction at each sample
        seg_idx = np.searchsorted(cumul_arc, s_samples, side='right') - 1
        seg_idx = np.clip(seg_idx, 0, len(curv_vec) - 1)
        path_head = np.arctan2(curv_vec[seg_idx, 1], curv_vec[seg_idx, 0])
        path_head = (path_head % (2.0 * np.pi)).reshape(self._num_horizon, 1)

        self._local_trajectory = np.hstack([path, path_vel, path_head])
        return self._local_trajectory

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)