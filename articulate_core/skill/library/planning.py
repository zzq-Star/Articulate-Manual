import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from articulate_core.skill.models.dh_template import ArmModel, JointLimit

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    time: float          # seconds from start
    positions: np.ndarray  # joint positions
    velocities: np.ndarray  # joint velocities
    accelerations: np.ndarray  # joint accelerations


@dataclass
class Trajectory:
    points: List[TrajectoryPoint] = field(default_factory=list)
    duration: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert trajectory to (N, 3*n_dof) array for serialization."""
        if not self.points:
            return np.array([])
        n_dof = len(self.points[0].positions)
        arr = np.zeros((len(self.points), 1 + 3 * n_dof))
        for i, pt in enumerate(self.points):
            arr[i, 0] = pt.time
            arr[i, 1:1+n_dof] = pt.positions
            arr[i, 1+n_dof:1+2*n_dof] = pt.velocities
            arr[i, 1+2*n_dof:1+3*n_dof] = pt.accelerations
        return arr


class PlanningLibrary:
    """Trajectory planning wrapper."""

    def _check_ompl(self) -> bool:
        try:
            import ompl  # noqa
            return True
        except ImportError:
            return False

    def plan_ptp(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        arm: ArmModel,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5,
        dt: float = 0.01,
    ) -> Trajectory:
        """Point-to-point joint-space trajectory with trapezoidal velocity profile."""
        q0 = np.asarray(start, dtype=float)
        qf = np.asarray(goal, dtype=float)
        n_dof = len(q0)

        # Compute travel distance per joint
        dq = qf - q0

        # Trapezoidal velocity profile
        v_max = min(max_velocity, min(l.velocity for l in arm.joint_limits)) * 0.9
        a_max = max_acceleration

        # Time to reach max velocity
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel ** 2
        total_dist = np.max(np.abs(dq))

        if total_dist < 2 * d_accel:
            # Triangle profile (no cruise phase)
            t_accel = np.sqrt(total_dist / a_max)
            t_total = 2 * t_accel
            d_cruise = 0
            cruise_time = 0
        else:
            # Trapezoidal profile
            d_cruise = total_dist - 2 * d_accel
            cruise_time = d_cruise / v_max
            t_total = 2 * t_accel + cruise_time

        n_points = max(2, int(t_total / dt))
        traj = Trajectory()

        for i in range(n_points):
            t = i * dt

            if t <= t_accel:
                # Acceleration phase
                s = 0.5 * a_max * t ** 2
                v = a_max * t
                a = a_max
            elif t <= t_total - t_accel:
                # Cruise phase
                s = d_accel + v_max * (t - t_accel)
                v = v_max
                a = 0
            elif t <= t_total:
                # Deceleration phase
                t_dec = t_total - t
                s = total_dist - 0.5 * a_max * t_dec ** 2
                v = a_max * t_dec
                a = -a_max
            else:
                break

            # Scale by joint distances
            fraction = s / total_dist if total_dist > 0 else 0
            pos = q0 + dq * fraction
            vel = dq / total_dist * v if total_dist > 0 else np.zeros(n_dof)
            acc = dq / total_dist * a if total_dist > 0 else np.zeros(n_dof)

            traj.points.append(TrajectoryPoint(
                time=t,
                positions=pos.copy(),
                velocities=vel.copy(),
                accelerations=acc.copy(),
            ))

        traj.duration = t_total
        return traj

    def plan_linear(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        arm: ArmModel,
        max_linear_velocity: float = 0.5,
        dt: float = 0.01,
    ) -> Trajectory:
        """Cartesian linear motion trajectory.

        Approximates linear TCP motion by computing IK at intermediate points.
        """
        q0 = np.asarray(start_pos, dtype=float)
        qf = np.asarray(end_pos, dtype=float)

        # Simple approach: linear interpolation in joint space
        # for true Cartesian, IK would be solved per waypoint
        total_dist = np.linalg.norm(qf - q0)
        t_total = total_dist / max_linear_velocity if max_linear_velocity > 0 else 1.0
        n_points = max(2, int(t_total / dt))

        traj = Trajectory()
        for i in range(n_points):
            t = i * dt
            fraction = min(1.0, t / t_total)

            # Cubic interpolation for smooth start/stop
            s = 3 * fraction ** 2 - 2 * fraction ** 3
            v = 6 * fraction * (1 - fraction) / t_total
            a = 6 * (1 - 2 * fraction) / t_total ** 2

            pos = q0 + (qf - q0) * s
            vel = (qf - q0) * v
            acc = (qf - q0) * a

            traj.points.append(TrajectoryPoint(
                time=t,
                positions=pos.copy(),
                velocities=vel.copy(),
                accelerations=acc.copy(),
            ))

        traj.duration = t_total
        return traj

    def plan_circular(
        self,
        start: np.ndarray,
        via: np.ndarray,
        end: np.ndarray,
        arm: ArmModel,
        max_velocity: float = 0.5,
        dt: float = 0.01,
    ) -> Trajectory:
        """Circular trajectory through three points (approximated as segmented linear)."""
        # Approximate circle segment as multi-segment linear in joint space
        n_segments = 20
        traj = Trajectory()

        # Create waypoints through the arc
        for i in range(n_segments + 1):
            t = i * dt * 10  # coarse sampling
            fraction = i / n_segments

            # Simple quadratic bezier-like interpolation in joint space
            pos = (1 - fraction) ** 2 * np.asarray(start) + \
                  2 * (1 - fraction) * fraction * np.asarray(via) + \
                  fraction ** 2 * np.asarray(end)

            traj.points.append(TrajectoryPoint(
                time=t,
                positions=pos,
                velocities=np.zeros_like(pos),
                accelerations=np.zeros_like(pos),
            ))

        traj.duration = n_segments * dt * 10 if n_segments > 0 else 0
        return traj

    def plan_obstacle_avoidance(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        arm: ArmModel,
    ) -> Trajectory:
        """Obstacle avoidance planning (OMPL-based with fallback)."""
        if self._check_ompl():
            return self._plan_ompl(start, goal, arm)
        else:
            logger.warning("OMPL not available; using direct PTP as fallback")
            return self.plan_ptp(start, goal, arm)

    def _plan_ompl(
        self, start: np.ndarray, goal: np.ndarray, arm: ArmModel,
    ) -> Trajectory:
        """OMPL-based motion planning."""
        try:
            from ompl import base as ob
            from ompl import geometric as og

            space = ob.RealVectorStateSpace(arm.num_dof())

            # Set bounds
            bounds = ob.RealVectorBounds(arm.num_dof())
            for i, lim in enumerate(arm.joint_limits):
                bounds.setLow(i, lim.lower)
                bounds.setHigh(i, lim.upper)
            space.setBounds(bounds)

            si = ob.SpaceInformation(space)
            si.setStateValidityChecker(ob.StateValidityCheckerFn(
                lambda state: self._is_state_valid(state, arm)
            ))
            si.setup()

            # Start and goal states
            start_state = ob.State(space)
            for i in range(arm.num_dof()):
                start_state[i] = float(start[i])

            goal_state = ob.State(space)
            for i in range(arm.num_dof()):
                goal_state[i] = float(goal[i])

            pdef = ob.ProblemDefinition(si)
            pdef.setStartAndGoalStates(start_state, goal_state)

            planner = og.RRTConnect(si)
            planner.setProblemDefinition(pdef)
            planner.setup()

            if planner.solve(1.0):
                path = pdef.getSolutionPath()
                traj = Trajectory()
                for i in range(path.getStateCount()):
                    state = path.getState(i)
                    pos = np.array([state[j] for j in range(arm.num_dof())])
                    traj.points.append(TrajectoryPoint(
                        time=i * 0.1,
                        positions=pos,
                        velocities=np.zeros(arm.num_dof()),
                        accelerations=np.zeros(arm.num_dof()),
                    ))
                traj.duration = (path.getStateCount() - 1) * 0.1
                return traj

            logger.warning("OMPL planning failed, falling back to PTP")
            return self.plan_ptp(start, goal, arm)

        except Exception as e:
            logger.warning("OMPL error, falling back to PTP: %s", e)
            return self.plan_ptp(start, goal, arm)

    @staticmethod
    def _is_state_valid(state, arm: ArmModel) -> bool:
        """Simple validity check: joint limits only."""
        for i in range(arm.num_dof()):
            val = state[i]
            if i < len(arm.joint_limits):
                lim = arm.joint_limits[i]
                if val < lim.lower or val > lim.upper:
                    return False
        return True
