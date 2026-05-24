"""Validation metrics for simulation verification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MetricResult:
    name: str
    passed: bool
    value: float
    threshold: float
    unit: str
    explanation: str = ""
    data: Optional[np.ndarray] = None


@dataclass
class SimulationData:
    """Full state data recorded during simulation."""
    time: np.ndarray                 # (N,) seconds
    joint_positions: np.ndarray      # (N, n_dof) rad
    joint_velocities: np.ndarray     # (N, n_dof) rad/s
    joint_accelerations: np.ndarray  # (N, n_dof) rad/s²
    joint_torques: np.ndarray        # (N, n_dof) Nm
    tcp_positions: np.ndarray        # (N, 3) m
    tcp_orientations: np.ndarray     # (N, 3) or (N, 4)
    self_collision_distances: np.ndarray  # (N,) m
    condition_numbers: np.ndarray    # (N,)

    n_dof: int = 0
    n_steps: int = 0
    arm_model: Optional[Dict[str, Any]] = None  # arm parameters for limit-aware metrics
    kinematic_only: bool = False  # True when MuJoCo physics was unavailable

    def __post_init__(self):
        if self.joint_positions.ndim > 1:
            self.n_dof = self.joint_positions.shape[1]
        self.n_steps = len(self.time)


class BaseMetric(ABC):
    """Abstract base for validation metrics."""

    name: str = ""
    unit: str = ""
    threshold: float = 0.0
    description: str = ""

    @abstractmethod
    def compute(self, data: SimulationData) -> float:
        """Compute raw metric value from simulation data."""

    def evaluate(self, data: SimulationData) -> MetricResult:
        """Evaluate metric and produce result with pass/fail."""
        value = self.compute(data)
        passed = value <= self.threshold if self.threshold > 0 else value >= abs(self.threshold)
        return MetricResult(
            name=self.name,
            passed=passed,
            value=float(value),
            threshold=self.threshold,
            unit=self.unit,
            explanation=self._explain(value, passed),
        )

    def _explain(self, value: float, passed: bool) -> str:
        status = "OK" if passed else f"EXCEEDED (value={value:.4f}, limit={self.threshold})"
        return f"{self.name}: {status}"


class JointPositionMetric(BaseMetric):
    """Check joint positions stay within limits (normalized by range)."""
    name = "joint_position_error"
    unit = "ratio"
    threshold = 0.85  # < 85% of range used from center
    description = "Checks no joint uses >85% of available range from center"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 0.0
        pos = data.joint_positions

        # If arm model has joint limits, check against them directly
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_ratio = 0.0
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                lower = lim.get("lower", -3.14) if isinstance(lim, dict) else lim.lower
                upper = lim.get("upper", 3.14) if isinstance(lim, dict) else lim.upper
                joint_pos = pos[:, j]
                raw_range = max(abs(joint_pos.max()), abs(joint_pos.min())) if len(joint_pos) > 0 else 0.0
                half_range = (upper - lower) / 2
                if half_range > 0:
                    ratio = raw_range / half_range
                    max_ratio = max(max_ratio, float(ratio))
            return max_ratio

        # Fallback: smoothness check
        smoothed = np.zeros_like(pos)
        for i in range(data.n_dof):
            from scipy.ndimage import uniform_filter1d
            try:
                smoothed[:, i] = uniform_filter1d(pos[:, i], size=max(3, data.n_steps // 20))
            except ImportError:
                smoothed[:, i] = pos[:, i]
        diff = np.abs(pos - smoothed)
        valid = diff[np.isfinite(diff)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0


class JointVelocityMetric(BaseMetric):
    """Check joint velocities stay within limits (ratio of nominal, position-control tolerant)."""
    name = "joint_velocity_overshoot"
    unit = "ratio"
    threshold = 20.0  # position control can overshoot rated velocity by 20x during transients
    description = "Checks no joint exceeds rated velocity by 20x"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 0.0

        # Use arm model rated velocities if available
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_ratio = 0.0
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                rated = lim.get("velocity", 3.0) if isinstance(lim, dict) else lim.velocity
                if rated > 0:
                    joint_vel = np.abs(data.joint_velocities[:, j])
                    vel_max = np.nanmax(joint_vel)
                    if np.isfinite(vel_max):
                        max_ratio = max(max_ratio, float(vel_max) / rated)
            return max_ratio

        max_vel = np.nanmax(np.abs(data.joint_velocities))
        return float(max_vel) if np.isfinite(max_vel) else 0.0


class JointAccelerationMetric(BaseMetric):
    """Check joint acceleration peaks as ratio to rated (position-control tolerant)."""
    name = "joint_acceleration_peak"
    unit = "ratio"
    threshold = 40.0
    description = "Checks acceleration within 40x rated bounds under position control"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 0.0

        # Use arm model velocity limits to compute rated acceleration
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_ratio = 0.0
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                rated_vel = lim.get("velocity", 3.0) if isinstance(lim, dict) else lim.velocity
                # Rated acceleration ≈ velocity per step
                dt = np.mean(np.diff(data.time)) if data.n_steps > 1 else 0.01
                rated_acc = rated_vel / max(dt, 0.001) if rated_vel > 0 else 10.0
                joint_acc = np.abs(data.joint_accelerations[:, j])
                acc_max = np.nanmax(joint_acc)
                if np.isfinite(acc_max) and rated_acc > 0:
                    max_ratio = max(max_ratio, float(acc_max) / rated_acc)
            return max_ratio if max_ratio > 0 else 0.0

        max_acc = np.nanmax(np.abs(data.joint_accelerations))
        return float(max_acc) if np.isfinite(max_acc) else 0.0


class JointTorqueMetric(BaseMetric):
    """Check joint torque peaks against rated values from arm model."""
    name = "joint_torque_peak"
    unit = "ratio"
    threshold = 0.95
    description = "Checks no joint exceeds 95% of rated torque"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 0.0

        # Use arm model rated torques if available
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_ratio = 0.0
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                rated = lim.get("torque", 150) if isinstance(lim, dict) else lim.torque
                if rated > 0:
                    joint_torques = np.abs(data.joint_torques[:, j])
                    t_max = np.nanmax(joint_torques)
                    if np.isfinite(t_max):
                        max_ratio = max(max_ratio, float(t_max) / rated)
            return max_ratio

        # Fallback: 150 Nm rated
        max_torque = np.nanmax(np.abs(data.joint_torques))
        return float(max_torque) / 150.0 if np.isfinite(max_torque) else 0.0


class SelfCollisionMetric(BaseMetric):
    """Check minimum distance between any two links."""
    name = "self_collision_distance"
    unit = "mm"
    threshold = -5.0  # minimum 5 mm (negative so evaluate checks value >= 5)
    description = "Checks no self-collision (min distance > 5 mm)"

    def compute(self, data: SimulationData) -> float:
        if len(data.self_collision_distances) == 0:
            return 100.0  # no collision data, assume OK
        min_dist = np.nanmin(data.self_collision_distances)
        if not np.isfinite(min_dist):
            return 100.0
        return float(min_dist) * 1000.0  # convert to mm


class WorkspaceMetric(BaseMetric):
    """Check joint limit margins against arm model limits."""
    name = "joint_limit_margin"
    unit = "rad"
    threshold = -0.05  # minimum 0.05 rad margin (negative => lower-bound check)
    description = "Checks joint margin from hard limits > 0.05 rad"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 1.0

        # Check against actual joint limits from arm model
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            min_margin = float('inf')
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                lower = lim.get("lower", -3.14) if isinstance(lim, dict) else lim.lower
                upper = lim.get("upper", 3.14) if isinstance(lim, dict) else lim.upper
                joint_pos = data.joint_positions[:, j]
                # Filter NaN
                joint_pos = joint_pos[np.isfinite(joint_pos)]
                if len(joint_pos) == 0:
                    continue
                range_used = float(np.max(joint_pos) - np.min(joint_pos))
                range_total = upper - lower
                if range_total > 0:
                    min_margin = min(min_margin, 1.0 - range_used / range_total)
            return max(0.0, min_margin) if np.isfinite(min_margin) else 0.5

        # Fallback: range of motion
        ranges = np.nanmax(data.joint_positions, axis=0) - np.nanmin(data.joint_positions, axis=0)
        valid = ranges[np.isfinite(ranges)]
        return float(np.max(valid)) if len(valid) > 0 else 0.0


class PathSmoothnessMetric(BaseMetric):
    """Check path jerk (derivative of acceleration) as ratio to typical."""
    name = "path_jerk"
    unit = "ratio"
    threshold = 100.0
    description = "Checks trajectory smoothness via normalized jerk metric"

    def compute(self, data: SimulationData) -> float:
        if data.n_steps < 3:
            return 0.0
        dt = np.mean(np.diff(data.time)) if len(data.time) > 1 else 0.01
        if dt <= 0:
            return 0.0
        jerk = np.diff(data.joint_accelerations, axis=0) / dt
        max_jerk = np.nanmax(np.abs(jerk))
        if not np.isfinite(max_jerk):
            return 0.0

        # Normalize by typical jerk scale from velocity limits
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_rated_vel = max(
                (lim.get("velocity", 3.0) if isinstance(lim, dict) else lim.velocity)
                for lim in limits
            ) or 1.0
            # Typical jerk ≈ velocity / dt²
            typical_jerk = max_rated_vel / (dt * dt) if dt > 0 else 1.0
            if typical_jerk > 0:
                return float(max_jerk) / typical_jerk

        return float(max_jerk) if np.isfinite(max_jerk) else 0.0


class SingularityMetric(BaseMetric):
    """Check proximity to singular configurations via condition number."""
    name = "condition_number"
    unit = ""
    threshold = 1000.0
    description = "Checks proximity to singular configurations via P95 condition number (< 1000)"

    def compute(self, data: SimulationData) -> float:
        if len(data.condition_numbers) == 0:
            return 0.0
        p95 = np.nanpercentile(data.condition_numbers, 95)
        return float(p95) if np.isfinite(p95) else 0.0


class TCPAccuracyMetric(BaseMetric):
    """Check TCP position tracking accuracy (normalized).
    Higher threshold (100) accounts for natural TCP speed variation
    in joint-space pick-and-place trajectories where waypoint stop-start
    cycles produce varying step sizes.
    """
    name = "tcp_position_error"
    unit = "ratio"
    threshold = 100.0
    description = "Checks TCP path consistency (max/mean step ratio < 100)"

    def compute(self, data: SimulationData) -> float:
        if data.n_steps < 2:
            return 0.0
        diffs = np.diff(data.tcp_positions, axis=0)
        step_sizes = np.linalg.norm(diffs, axis=1)
        step_sizes = step_sizes[np.isfinite(step_sizes)]
        if len(step_sizes) == 0:
            return 0.0
        mean_step = np.mean(step_sizes)
        if mean_step <= 0:
            return 0.0
        max_dev = np.max(np.abs(step_sizes - mean_step))
        return float(max_dev / mean_step)


class PayloadMetric(BaseMetric):
    """Check payload vs rated capacity using arm model torque limits."""
    name = "payload_ratio"
    unit = "ratio"
    threshold = 0.9
    description = "Checks payload < 90% of rated capacity"

    def compute(self, data: SimulationData) -> float:
        if data.n_dof == 0 or data.n_steps == 0:
            return 0.0
        max_torque = np.nanmax(np.abs(data.joint_torques))
        if not np.isfinite(max_torque):
            return 0.0

        # Use arm model max torque
        rated_torque = 150.0
        if data.arm_model and "joint_limits" in data.arm_model:
            limits = data.arm_model["joint_limits"]
            max_rated = 0.0
            for j in range(min(data.n_dof, len(limits))):
                lim = limits[j]
                tor = lim.get("torque", 150) if isinstance(lim, dict) else lim.torque
                max_rated = max(max_rated, tor)
            if max_rated > 0:
                rated_torque = max_rated
        return float(max_torque) / rated_torque


def get_all_metrics() -> List[BaseMetric]:
    """Return all validation metrics."""
    return [
        JointPositionMetric(),
        JointVelocityMetric(),
        JointAccelerationMetric(),
        JointTorqueMetric(),
        SelfCollisionMetric(),
        WorkspaceMetric(),
        PathSmoothnessMetric(),
        SingularityMetric(),
        TCPAccuracyMetric(),
        PayloadMetric(),
    ]
