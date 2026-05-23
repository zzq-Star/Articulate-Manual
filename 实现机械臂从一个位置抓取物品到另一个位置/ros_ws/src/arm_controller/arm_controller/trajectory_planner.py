import numpy as np


# WAYPOINTS: joint-space path for simulation extraction.
# The simulation extracts this constant and interpolates between waypoints
# using S-curve interpolation. Modify these to change the robot motion.
WAYPOINTS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # home position
    [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],   # reach / pre-grasp
    [0.5, -0.2, 0.1, 0.0, 0.0, 0.0],   # grasp at target
    [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],   # transfer to destination
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # return home
]


class TrajectoryPlanner:
    """Trajectory planning for the robot arm."""

    def plan_ptp(self, start, goal, max_velocity=1.0, max_acceleration=0.5, dt=0.01):
        """Point-to-point trajectory with trapezoidal velocity profile."""
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        dq = goal - start

        t_accel = max_velocity / max_acceleration
        d_accel = 0.5 * max_acceleration * t_accel ** 2
        total_dist = np.max(np.abs(dq))

        if total_dist < 2 * d_accel:
            t_accel = np.sqrt(total_dist / max_acceleration)
            t_total = 2 * t_accel
        else:
            d_cruise = total_dist - 2 * d_accel
            cruise_time = d_cruise / max_velocity
            t_total = 2 * t_accel + cruise_time

        n_points = max(2, int(t_total / dt))
        trajectory = []

        for i in range(n_points):
            t = i * dt
            if t <= t_accel:
                s = 0.5 * max_acceleration * t ** 2
                v = max_acceleration * t
            elif t <= t_total - t_accel:
                s = d_accel + max_velocity * (t - t_accel)
                v = max_velocity
            else:
                t_dec = t_total - t
                s = total_dist - 0.5 * max_acceleration * t_dec ** 2
                v = max_acceleration * t_dec

            fraction = s / total_dist if total_dist > 0 else 0
            pos = start + dq * fraction
            trajectory.append({
                "time": t,
                "positions": pos.tolist(),
                "velocity": float(v),
            })

        return trajectory

    def smooth_trajectory(self, waypoints, dt=0.01):
        """Smooth a multi-point trajectory using cubic splines."""
        if len(waypoints) < 2:
            return waypoints
        return waypoints  # TODO: implement spline interpolation
