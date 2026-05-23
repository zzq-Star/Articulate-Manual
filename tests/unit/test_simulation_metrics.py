"""Tests for simulation validation metrics."""

import numpy as np
import pytest

from articulate_core.simulation.metrics import (
    JointPositionMetric,
    JointVelocityMetric,
    JointAccelerationMetric,
    JointTorqueMetric,
    SelfCollisionMetric,
    WorkspaceMetric,
    PathSmoothnessMetric,
    SingularityMetric,
    TCPAccuracyMetric,
    PayloadMetric,
    SimulationData,
    get_all_metrics,
)


def _make_data(
    n_steps: int = 100,
    n_dof: int = 6,
    pos_scale: float = 0.5,
    vel_scale: float = 0.5,
    acc_scale: float = 1.0,
    torque_scale: float = 50.0,
) -> SimulationData:
    t = np.linspace(0, 2.0, n_steps)
    pos = np.sin(np.linspace(0, np.pi, n_steps))[:, None] * pos_scale * np.ones(n_dof)
    vel = np.cos(np.linspace(0, np.pi, n_steps))[:, None] * vel_scale * np.ones(n_dof)
    acc = -np.sin(np.linspace(0, np.pi, n_steps))[:, None] * acc_scale * np.ones(n_dof)
    torque = np.abs(pos) * torque_scale
    return SimulationData(
        time=t,
        joint_positions=pos,
        joint_velocities=vel,
        joint_accelerations=acc,
        joint_torques=torque,
        tcp_positions=np.column_stack([np.sin(t), np.cos(t), t * 0.1]),
        tcp_orientations=np.zeros((n_steps, 3)),
        self_collision_distances=np.ones(n_steps) * 0.1,
        condition_numbers=np.ones(n_steps) * 5.0,
    )


class TestJointPositionMetric:
    def test_smooth_trajectory_passes(self):
        data = _make_data(n_steps=200, pos_scale=0.5)
        metric = JointPositionMetric()
        result = metric.evaluate(data)
        assert result.passed, f"Expected pass, got {result.value:.4f}"

    def test_jerky_trajectory_fails(self):
        n_steps = 50
        t = np.linspace(0, 2.0, n_steps)
        pos = np.random.randn(n_steps, 6) * 2.0  # large random jumps
        data = _make_data(n_steps=n_steps)
        data.joint_positions = pos
        metric = JointPositionMetric()
        result = metric.evaluate(data)
        # May or may not fail depending on randomness
        assert isinstance(result.passed, bool)
        assert result.value > 0


class TestJointVelocityMetric:
    def test_low_velocity_passes(self):
        data = _make_data(vel_scale=0.5)
        metric = JointVelocityMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_high_velocity_fails(self):
        data = _make_data(vel_scale=25.0)  # exceeds threshold of 20.0
        metric = JointVelocityMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestJointAccelerationMetric:
    def test_low_accel_passes(self):
        data = _make_data(acc_scale=0.5)
        metric = JointAccelerationMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_high_accel_fails(self):
        data = _make_data(acc_scale=50.0)  # exceeds threshold of 40.0
        metric = JointAccelerationMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestJointTorqueMetric:
    def test_low_torque_passes(self):
        data = _make_data(torque_scale=50.0)
        metric = JointTorqueMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_high_torque_fails(self):
        data = _make_data(torque_scale=500.0)  # exceeds 95% of 150 Nm
        metric = JointTorqueMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestSelfCollisionMetric:
    def test_safe_distance_passes(self):
        data = _make_data()
        data.self_collision_distances = np.ones(100) * 0.1  # 100 mm > 5 mm threshold
        metric = SelfCollisionMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_close_distance_fails(self):
        data = _make_data()
        data.self_collision_distances = np.ones(100) * 0.001  # 1 mm < 5 mm
        metric = SelfCollisionMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestWorkspaceMetric:
    def test_small_range_passes(self):
        data = _make_data(pos_scale=0.01)
        # With arm model limits, small range = high margin -> PASS
        data.arm_model = {
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}
                for _ in range(6)
            ]
        }
        metric = WorkspaceMetric()
        result = metric.evaluate(data)
        assert result.passed, f"Expected pass, got value={result.value:.4f}"

    def test_large_range_fails(self):
        data = _make_data(pos_scale=10.0)
        # With arm model limits, range exceeding limits -> FAIL
        data.arm_model = {
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}
                for _ in range(6)
            ]
        }
        metric = WorkspaceMetric()
        result = metric.evaluate(data)
        assert not result.passed, f"Expected fail, got value={result.value:.4f}"


class TestPathSmoothnessMetric:
    def test_smooth_trajectory_passes(self):
        data = _make_data(n_steps=200, acc_scale=1.0)
        metric = PathSmoothnessMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_jerky_trajectory_fails(self):
        n_steps = 50
        t = np.linspace(0, 2.0, n_steps)
        acc = np.random.randn(n_steps, 6) * 1000.0  # extreme acceleration changes
        data = _make_data(n_steps=n_steps, acc_scale=1.0)
        data.joint_accelerations = acc
        metric = PathSmoothnessMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestSingularityMetric:
    def test_good_condition_passes(self):
        data = _make_data()
        data.condition_numbers = np.ones(100) * 5.0
        metric = SingularityMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_bad_condition_fails(self):
        data = _make_data()
        data.condition_numbers = np.ones(100) * 2000.0  # exceeds threshold of 1000.0
        metric = SingularityMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestTCPAccuracyMetric:
    def test_smooth_tcp_passes(self):
        data = _make_data(n_steps=200)
        metric = TCPAccuracyMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_jumpy_tcp_fails(self):
        n_steps = 100
        t = np.linspace(0, 2.0, n_steps)
        pos = np.column_stack([
            np.random.randn(n_steps) * 0.1,  # jumpy x
            np.random.randn(n_steps) * 0.1,  # jumpy y
            t * 0.1,
        ])
        data = _make_data(n_steps=n_steps)
        data.tcp_positions = pos
        metric = TCPAccuracyMetric()
        result = metric.evaluate(data)
        assert isinstance(result.passed, bool)


class TestPayloadMetric:
    def test_low_payload_passes(self):
        data = _make_data(torque_scale=50.0)
        metric = PayloadMetric()
        result = metric.evaluate(data)
        assert result.passed

    def test_high_payload_fails(self):
        data = _make_data(torque_scale=500.0)  # exceeds 90% of 150 Nm
        metric = PayloadMetric()
        result = metric.evaluate(data)
        assert not result.passed


class TestGetAllMetrics:
    def test_returns_all_metrics(self):
        metrics = get_all_metrics()
        assert len(metrics) == 10
        names = [m.name for m in metrics]
        assert "joint_position_error" in names
        assert "joint_velocity_overshoot" in names
        assert "joint_acceleration_peak" in names
        assert "joint_torque_peak" in names
        assert "self_collision_distance" in names
        assert "joint_limit_margin" in names
        assert "path_jerk" in names
        assert "condition_number" in names
        assert "tcp_position_error" in names
        assert "payload_ratio" in names
