"""Tests for MuJoCo simulator (kinematic fallback and MuJoCo path)."""

import numpy as np
import pytest
from pathlib import Path

from articulate_core.simulation.launch_mujoco import MuJoCoSimulator, TrajectoryCommand


@pytest.fixture
def simulator():
    return MuJoCoSimulator()


@pytest.fixture
def simple_trajectory():
    n_dof, n_steps = 6, 50
    t = np.linspace(0, 1.0, n_steps)
    pos = np.zeros((n_steps, n_dof))
    for j in range(n_dof):
        pos[:, j] = 0.3 * np.sin(t * 2.0 + j * 0.5)
    vel = np.zeros_like(pos)
    if n_steps > 1:
        dt = np.mean(np.diff(t))
        vel[1:] = np.diff(pos, axis=0) / dt
    return TrajectoryCommand(pos, vel, t)


@pytest.mark.asyncio
async def test_kinematic_fallback(simulator, simple_trajectory):
    """Simulation with no MJCF path should use kinematic fallback."""
    data = await simulator.run_trajectory(None, simple_trajectory)
    assert data.n_steps == 50
    assert data.n_dof == 6
    assert data.joint_positions.shape == (50, 6)
    assert data.joint_velocities.shape == (50, 6)
    assert data.joint_accelerations.shape == (50, 6)
    assert data.joint_torques.shape == (50, 6)
    assert data.tcp_positions.shape == (50, 3)
    assert len(data.time) == 50


@pytest.mark.asyncio
async def test_kinematic_fallback_nonexistent_path(simulator, simple_trajectory):
    """Simulation with non-existent MJCF path should fall back."""
    data = await simulator.run_trajectory(Path("/nonexistent/test.mjcf"), simple_trajectory)
    assert data.n_steps == 50
    assert data.n_dof == 6


@pytest.mark.asyncio
async def test_kinematic_single_step(simulator):
    """Single-step trajectory should produce valid data."""
    traj = TrajectoryCommand(
        joint_positions=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        joint_velocities=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        time_steps=np.array([0.0]),
    )
    data = await simulator.run_trajectory(None, traj)
    assert data.n_steps == 1
    assert data.n_dof == 6


@pytest.mark.asyncio
async def test_kinematic_two_steps(simulator):
    """Two-step trajectory should produce valid data."""
    traj = TrajectoryCommand(
        joint_positions=np.array([[0.0, 0.0], [0.5, 0.5]]),
        joint_velocities=np.array([[0.0, 0.0], [0.5, 0.5]]),
        time_steps=np.array([0.0, 1.0]),
    )
    data = await simulator.run_trajectory(None, traj)
    assert data.n_steps == 2
    assert data.n_dof == 2


@pytest.mark.asyncio
async def test_kinematic_tcp_position(simulator):
    """TCP position should be computed from joint positions."""
    n_steps = 10
    traj = TrajectoryCommand(
        joint_positions=np.ones((n_steps, 3)) * 0.5,
        joint_velocities=np.zeros((n_steps, 3)),
        time_steps=np.linspace(0, 1.0, n_steps),
    )
    data = await simulator.run_trajectory(None, traj)
    assert data.tcp_positions.shape == (n_steps, 3)
    # All positions should be same for constant joint angles
    assert np.allclose(data.tcp_positions[0], data.tcp_positions[-1], atol=1e-5)


@pytest.mark.asyncio
async def test_kinematic_collision_default(simulator):
    """Kinematic fallback should report default safe collision distances."""
    traj = TrajectoryCommand(
        joint_positions=np.array([[0.1, 0.1]]),
        joint_velocities=np.array([[0.0, 0.0]]),
        time_steps=np.array([0.0]),
    )
    data = await simulator.run_trajectory(None, traj)
    assert len(data.self_collision_distances) == 1
    assert data.self_collision_distances[0] == 50.0  # default safe value


@pytest.mark.asyncio
async def test_kinematic_condition_default(simulator):
    """Kinematic fallback should report default condition numbers."""
    traj = TrajectoryCommand(
        joint_positions=np.array([[0.1, 0.1]]),
        joint_velocities=np.array([[0.0, 0.0]]),
        time_steps=np.array([0.0]),
    )
    data = await simulator.run_trajectory(None, traj)
    assert data.condition_numbers[0] == 5.0  # default value


@pytest.mark.asyncio
async def test_simulation_data_post_init(simulator, simple_trajectory):
    """SimulationData __post_init__ should compute n_dof and n_steps."""
    data = await simulator.run_trajectory(None, simple_trajectory)
    assert data.n_dof == 6
    assert data.n_steps == 50


@pytest.mark.asyncio
async def test_arm_model_passthrough(simulator, simple_trajectory):
    """SimulationData should accept arm_model for limit-aware metrics."""
    data = await simulator.run_trajectory(None, simple_trajectory)
    arm_model = {
        "joint_limits": [
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}
            for _ in range(6)
        ]
    }
    data.arm_model = arm_model
    assert data.arm_model is not None
    assert len(data.arm_model["joint_limits"]) == 6


@pytest.mark.asyncio
async def test_mujoco_check_import(simulator):
    """_check_import should return bool without error."""
    result = simulator._check_import()
    # This may be True or False depending on environment
    assert isinstance(result, bool)
