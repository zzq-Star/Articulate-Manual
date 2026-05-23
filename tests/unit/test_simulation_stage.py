"""Tests for simulation stage (URDF/MJCF generation, trajectory extraction)."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from articulate_core.pipeline.models import GeneratedCode, StageContext
from articulate_core.pipeline.stage4_simulation import SimulationStage
from articulate_core.simulation.launch_mujoco import TrajectoryCommand
from articulate_core.simulation.validation_engine import ValidationReport


def test_extract_trajectory_with_waypoints():
    """Extract trajectory from generated code containing waypoint arrays."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory.py": (
                'waypoints = [\n'
                '    [0.0, 0.0, 0.3, 0.0, 0.0, 0.0],\n'
                '    [0.3, 0.0, 0.3, 0.0, 0.0, 0.0],\n'
                ']'
            ),
        },
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert traj.joint_positions.shape[1] == 6
    assert len(traj.time_steps) >= 40
    assert traj.joint_velocities.shape == traj.joint_positions.shape


def test_extract_trajectory_no_waypoints():
    """Extract trajectory when no waypoints are found (should use defaults)."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory.py": "# just a comment, no arrays",
        },
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert traj.joint_positions.shape[1] == 6
    # Default trajectory uses S-curve to 0.05
    assert np.allclose(traj.joint_positions[0], 0.0, atol=0.01)
    assert np.allclose(traj.joint_positions[-1], 0.05, atol=0.01)


def test_extract_trajectory_empty_code():
    """Extract trajectory from empty code should not crash."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={},
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert traj.joint_positions.shape[1] == 6


def test_extract_trajectory_partial_waypoints():
    """Extract trajectory from code with fewer than 6 DOF in waypoints."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory.py": "waypoints = [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]",
        },
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert traj.joint_positions.shape[1] == 6  # falls back to n_dof=6


@pytest.mark.asyncio
async def test_stage4_no_code(tmp_path):
    """Stage 4 should fail gracefully with no generated code."""
    from articulate_core.config.settings import ArticulateConfig
    from articulate_core.skill import ArticulateSkill

    config = ArticulateConfig(project_dir=tmp_path, anthropic_api_key="test")
    llm = AsyncMock()
    skill = ArticulateSkill(config, llm=llm)
    ctx = StageContext(project_dir=tmp_path)
    ctx.generated_code = None

    stage = SimulationStage(config, llm, skill)
    result = await stage.execute(ctx)
    assert not result.should_continue
    assert result.simulation_report is None


@pytest.mark.asyncio
async def test_stage4_with_code(tmp_path):
    """Stage 4 should stop when MuJoCo physics is unavailable."""
    from unittest.mock import AsyncMock, patch
    from pathlib import Path
    from articulate_core.config.settings import ArticulateConfig
    from articulate_core.skill import ArticulateSkill
    from articulate_core.pipeline.models import GeneratedCode, StageContext
    from articulate_core.pipeline.stage4_simulation import SimulationStage

    config = ArticulateConfig(project_dir=tmp_path, anthropic_api_key="test")
    llm = AsyncMock()
    skill = ArticulateSkill(config, llm=llm)
    ctx = StageContext(project_dir=tmp_path)
    ctx.current_stage = 3
    ctx.generated_code = GeneratedCode(
        package_structure={
            "src/trajectory.py": "waypoints = [[0.0, 0.0, 0.3], [0.3, 0.0, 0.3]]",
        },
        ros2_package_name="test_arm",
    )

    with patch("click.confirm", return_value=True):
        with patch.object(Path, "exists", return_value=False):
            stage = SimulationStage(config, llm, skill)
            result = await stage.execute(ctx)

    # When MuJoCo is unavailable, kinematic fallback is treated as ERROR
    assert result.simulation_report is None
    assert not result.should_continue


def test_generate_default_urdf_no_arm(tmp_path):
    """Default URDF generation without arm model should not crash."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.urdf"
    stage._generate_default_urdf(path, None)
    assert path.exists()
    content = path.read_text()
    assert "<robot" in content
    assert "base_link" in content


def test_generate_default_urdf_with_arm(tmp_path):
    """Default URDF generation with arm model should use DH params."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.urdf"
    arm = {
        "dh_params": [
            {"a": 0.3, "alpha": 0, "d": 0},
            {"a": 0.3, "alpha": 0, "d": 0},
        ],
        "joint_limits": [
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
        ],
        "dynamics": [
            {"mass": 2.0, "friction": 0.1, "damping": 0.01},
            {"mass": 1.5, "friction": 0.1, "damping": 0.01},
        ],
    }
    stage._generate_default_urdf(path, arm)
    assert path.exists()
    content = path.read_text()
    assert "joint_0" in content
    assert "joint_1" in content
    assert 'value="2.0"' in content
    assert 'value="1.5"' in content


def test_generate_default_mjcf_no_arm(tmp_path):
    """Default MJCF generation without arm model should not crash."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.mjcf"
    stage._generate_default_mjcf(path, None)
    assert path.exists()
    content = path.read_text()
    assert "mujoco" in content
    assert "actuator" in content
    assert "sensor" in content


def test_generate_default_mjcf_with_arm(tmp_path):
    """Default MJCF generation with arm model should include all elements."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.mjcf"
    arm = {
        "dh_params": [
            {"a": 0.3, "alpha": 0, "d": 0},
            {"a": 0.3, "alpha": 0, "d": 0},
        ],
        "joint_limits": [
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
        ],
        "dynamics": [
            {"mass": 2.0, "friction": 0.1, "damping": 0.01},
            {"mass": 1.5, "friction": 0.1, "damping": 0.01},
        ],
    }
    stage._generate_default_mjcf(path, arm)
    assert path.exists()
    content = path.read_text()
    assert "joint_0" in content
    assert 'forcerange' in content
    assert "jtorque_0" in content
    assert "collision" in content


def test_generate_default_mjcf_inertia_correct(tmp_path):
    """MJCF inertia should be physically valid (triangle inequality)."""
    import xml.etree.ElementTree as ET
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.mjcf"
    arm = {
        "dh_params": [{"a": 0.3, "alpha": 0, "d": 0}],
        "joint_limits": [{"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}],
        "dynamics": [{"mass": 1.0}],
    }
    stage._generate_default_mjcf(path, arm)
    # Parse and check inertia values
    tree = ET.parse(str(path))
    root = tree.getroot()
    inertial = root.find(".//inertial")
    assert inertial is not None
    fi = inertial.get("fullinertia", "")
    vals = [float(v) for v in fi.split()]
    # Need: vals[0] + vals[1] >= vals[2] (triangle inequality)
    assert len(vals) >= 3
    assert vals[0] + vals[1] >= vals[2] - 1e-10, f"Inertia violates triangle inequality: {vals}"
    assert vals[0] > 0 and vals[1] > 0 and vals[2] > 0, "Inertia must be positive"


def test_generate_default_mjcf_with_actuator_overrides(tmp_path):
    """MJCF actuator overrides should change kp/kv/forcerange in generated XML."""
    import xml.etree.ElementTree as ET
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    stage = SimulationStage.__new__(SimulationStage)
    path = tmp_path / "test.mjcf"
    arm = {
        "dh_params": [{"a": 0.3, "alpha": 0, "d": 0}, {"a": 0.2, "alpha": 0, "d": 0}],
        "joint_limits": [
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
            {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 150},
        ],
        "dynamics": [{"mass": 1.0}, {"mass": 1.5}],
    }
    overrides = {
        "0": {"kp": 20, "kv": 10},
        "1": {"kp": 30, "forcerange_scale": 1.5},
    }
    stage._generate_default_mjcf(path, arm, actuator_overrides=overrides)
    tree = ET.parse(str(path))
    root = tree.getroot()

    actuators = root.findall(".//actuator/position")
    assert len(actuators) == 2

    # Joint 0: kp=20, kv=10 (from override), forcerange should still be -100 to 100 (no scale)
    a0 = actuators[0]
    assert a0.get("kp") == "20", f"Expected kp=20, got {a0.get('kp')}"
    assert a0.get("kv") == "10", f"Expected kv=10 (from override), got {a0.get('kv')}"
    assert a0.get("forcerange") == "-100.0 100.0", f"Unexpected forcerange: {a0.get('forcerange')}"

    # Joint 1: kp=30, kv=0.5 (default), forcerange=150*1.5=225
    a1 = actuators[1]
    assert a1.get("kp") == "30", f"Expected kp=30, got {a1.get('kp')}"
    assert a1.get("kv") == "0.5", f"Expected kv=0.5, got {a1.get('kv')}"
    assert a1.get("forcerange") == "-225.0 225.0", f"Unexpected forcerange: {a1.get('forcerange')}"


@pytest.mark.asyncio
async def test_auto_repair_attempt(tmp_path):
    """Auto-repair should use LLM to modify code and re-run simulation."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from articulate_core.config.settings import ArticulateConfig
    from articulate_core.skill import ArticulateSkill
    from articulate_core.simulation.metrics import MetricResult
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import SimulationData

    config = ArticulateConfig(project_dir=tmp_path, anthropic_api_key="test")
    llm = AsyncMock()

    # Make complete_structured return a valid object with .files and .explanation
    async def fake_structured(*args, **kwargs):
        result = MagicMock()
        result.files = {"src/trajectory.py": "# fixed code"}
        result.explanation = "Reduced velocity scaling to fix overshoot"
        return result
    llm.complete_structured = fake_structured

    skill = ArticulateSkill(config, llm=llm)
    ctx = StageContext(project_dir=tmp_path)
    ctx.generated_code = GeneratedCode(
        package_structure={"src/trajectory.py": "# simple"},
        ros2_package_name="test_arm",
    )

    stage = SimulationStage(config, llm, skill)
    data = SimulationData(
        time=np.array([0.0, 1.0]),
        joint_positions=np.array([[0.1, 0.1], [0.2, 0.2]]),
        joint_velocities=np.array([[0.0, 0.0], [0.1, 0.1]]),
        joint_accelerations=np.array([[0.0, 0.0], [0.1, 0.1]]),
        joint_torques=np.array([[1.0, 1.0], [2.0, 2.0]]),
        tcp_positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        tcp_orientations=np.zeros((2, 3)),
        self_collision_distances=np.array([0.1, 0.1]),
        condition_numbers=np.array([5.0, 5.0]),
    )

    report = ValidationReport(
        passed=False,
        metrics={
            "joint_velocity_overshoot": MetricResult(
                name="joint_velocity_overshoot", passed=False,
                value=5.0, threshold=1.1, unit="ratio",
            ),
        },
        summary="Test failure",
    )

    with patch.object(stage, "_prepare_models", return_value=(tmp_path / "test.urdf", tmp_path / "test.mjcf")):
        with patch.object(stage, "_extract_trajectory") as mock_extract:
            mock_extract.return_value = TrajectoryCommand(
                joint_positions=np.array([[0.5, 0.5]]),
                joint_velocities=np.array([[0.5, 0.5]]),
                time_steps=np.array([0.0, 1.0]),
            )
            new_report = await stage._attempt_repair(ctx, report, data, attempt=1)

    assert isinstance(new_report, ValidationReport)


def test_autotune_actuators_all_pass():
    """No overrides when all metrics pass."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import MetricResult

    report = ValidationReport(passed=True, metrics={
        "joint_torque_peak": MetricResult(
            name="joint_torque_peak", passed=True,
            value=0.5, threshold=0.95, unit="ratio",
        ),
    }, summary="")
    stage = SimulationStage.__new__(SimulationStage)
    result = stage._autotune_actuators(report, n_dof=6, attempt=1)
    assert result is None


def test_autotune_actuators_torque_fail():
    """Torque failure should reduce kp."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import MetricResult

    report = ValidationReport(passed=False, metrics={
        "joint_torque_peak": MetricResult(
            name="joint_torque_peak", passed=False,
            value=1.5, threshold=0.95, unit="ratio",
        ),
        "joint_velocity_overshoot": MetricResult(
            name="joint_velocity_overshoot", passed=True,
            value=0.5, threshold=20.0, unit="ratio",
        ),
    }, summary="")
    stage = SimulationStage.__new__(SimulationStage)
    result = stage._autotune_actuators(report, n_dof=6, attempt=1)
    assert result is not None
    for j in range(6):
        assert result[str(j)]["kp"] == 12  # 20 / (1 * 1.6) = 12
        assert "kv" not in result[str(j)]


def test_autotune_actuators_velocity_fail():
    """Velocity failure should increase kv."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import MetricResult

    report = ValidationReport(passed=False, metrics={
        "joint_velocity_overshoot": MetricResult(
            name="joint_velocity_overshoot", passed=False,
            value=25.0, threshold=20.0, unit="ratio",
        ),
        "joint_torque_peak": MetricResult(
            name="joint_torque_peak", passed=True,
            value=0.5, threshold=0.95, unit="ratio",
        ),
    }, summary="")
    stage = SimulationStage.__new__(SimulationStage)
    result = stage._autotune_actuators(report, n_dof=6, attempt=2)
    assert result is not None
    for j in range(6):
        assert result[str(j)]["kv"] == 1.0  # 0.5 * 2^(2-1) = 1.0
        assert "kp" not in result[str(j)]


def test_autotune_actuators_both_fail():
    """Both torque and velocity failures should adjust kp and kv."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import MetricResult

    report = ValidationReport(passed=False, metrics={
        "joint_torque_peak": MetricResult(
            name="joint_torque_peak", passed=False,
            value=1.5, threshold=0.95, unit="ratio",
        ),
        "joint_velocity_overshoot": MetricResult(
            name="joint_velocity_overshoot", passed=False,
            value=25.0, threshold=20.0, unit="ratio",
        ),
    }, summary="")
    stage = SimulationStage.__new__(SimulationStage)
    result = stage._autotune_actuators(report, n_dof=6, attempt=3)
    assert result is not None
    for j in range(6):
        assert result[str(j)]["kp"] == 5  # max(5, int(20 / (3 * 1.6))) = 5
        assert result[str(j)]["kv"] == 2.0  # 0.5 * 2^(3-1) = 2.0


def test_autotune_actuators_unrelated_metric():
    """Non-torque/velocity failures should not trigger auto-tuning."""
    from articulate_core.pipeline.stage4_simulation import SimulationStage
    from articulate_core.simulation.validation_engine import ValidationReport
    from articulate_core.simulation.metrics import MetricResult

    report = ValidationReport(passed=False, metrics={
        "self_collision_distance": MetricResult(
            name="self_collision_distance", passed=False,
            value=-10.0, threshold=-5.0, unit="mm",
        ),
        "condition_number": MetricResult(
            name="condition_number", passed=False,
            value=2000.0, threshold=1000.0, unit="",
        ),
    }, summary="")
    stage = SimulationStage.__new__(SimulationStage)
    result = stage._autotune_actuators(report, n_dof=6, attempt=1)
    assert result is None


# ─── Code execution validation tests ───────────────────────────────────────

def test_execute_planner_with_valid_code():
    """Execute generated TrajectoryPlanner with valid code should produce TrajectoryCommand."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal, dt=0.01):\n"
                "        n = max(2, int(2.0 / dt))\n"
                "        traj = []\n"
                "        for i in range(n):\n"
                "            s = i / max(n - 1, 1)\n"
                "            pos = [start[j] + s * (goal[j] - start[j]) for j in range(6)]\n"
                "            traj.append({'time': i * dt, 'positions': pos, 'velocity': 0.5})\n"
                "        return traj\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.2, 0.3, 0.0, 0.0, 0.0]),
    ]
    result = stage._execute_trajectory_planner(code, waypoints)
    assert result is not None
    assert isinstance(result, TrajectoryCommand)
    assert result.joint_positions.shape[1] == 6
    assert len(result.time_steps) >= 2
    assert np.allclose(result.joint_positions[0], waypoints[0], atol=0.01)
    assert np.allclose(result.joint_positions[-1], waypoints[-1], atol=0.01)


def test_execute_planner_fallback_on_syntax_error():
    """Syntax error in trajectory code should return None (S-curve fallback)."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner\n"  # missing colon
            ),
        },
        ros2_package_name="test_arm",
    )
    waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    result = stage._execute_trajectory_planner(code, waypoints)
    assert result is None


def test_execute_planner_no_trajectory_file():
    """No trajectory_planner file should return None."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/some_other_file.py": "x = 1",
        },
        ros2_package_name="test_arm",
    )
    waypoints = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    result = stage._execute_trajectory_planner(code, waypoints)
    assert result is None


def test_execute_planner_with_multi_segment():
    """Multi-waypoint trajectory should concatenate segments correctly."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal, dt=0.01):\n"
                "        n = max(2, int(1.0 / dt))\n"
                "        traj = []\n"
                "        for i in range(n):\n"
                "            s = i / max(n - 1, 1)\n"
                "            pos = [start[j] + s * (goal[j] - start[j]) for j in range(6)]\n"
                "            traj.append({'time': i * dt, 'positions': pos, 'velocity': 0.5})\n"
                "        return traj\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, -0.2, 0.0, 0.0, 0.0, 0.0]),
    ]
    result = stage._execute_trajectory_planner(code, waypoints)
    assert result is not None
    assert len(result.time_steps) >= 2
    assert np.allclose(result.joint_positions[0], waypoints[0], atol=0.01)
    assert np.allclose(result.joint_positions[-1], waypoints[-1], atol=0.01)
    mid_found = any(
        np.allclose(result.joint_positions[i], waypoints[1], atol=0.05)
        for i in range(len(result.joint_positions))
    )
    assert mid_found, "Middle waypoint not found in concatenated trajectory"


def test_execute_planner_cleanup_restores_syspath():
    """After execution, sys.path should be restored to original."""
    import sys
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal, dt=0.01):\n"
                "        n = max(2, int(1.0 / dt))\n"
                "        traj = []\n"
                "        for i in range(n):\n"
                "            s = i / max(n - 1, 1)\n"
                "            pos = [start[j] + s * (goal[j] - start[j]) for j in range(6)]\n"
                "            traj.append({'time': i * dt, 'positions': pos, 'velocity': 0.5})\n"
                "        return traj\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    original_path = sys.path.copy()
    waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    _ = stage._execute_trajectory_planner(code, waypoints)
    assert sys.path == original_path, "sys.path was not restored after cleanup"


def test_validate_code_structure_valid():
    """Code with all required symbols should pass all structure checks."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal):\n"
                "        return []\n"
            ),
            "src/arm_kinematics.py": (
                "import numpy as np\n"
                "class ArmKinematics:\n"
                "    def forward(self, angles):\n"
                "        return np.eye(4)\n"
                "    def inverse(self, pose):\n"
                "        return [0.0]*6, True\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_code_structure(code)
    assert results["code_syntax_valid"].passed
    assert results["code_required_symbols"].passed


def test_validate_code_structure_syntax_error():
    """Code with syntax errors should fail syntax check."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner\n"  # SyntaxError: missing colon
            ),
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_code_structure(code)
    assert not results["code_syntax_valid"].passed
    assert results["code_syntax_valid"].value >= 1


def test_validate_code_structure_missing_symbols():
    """Code missing required classes should fail symbol check."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/something.py": "x = 1\n",
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_code_structure(code)
    assert not results["code_required_symbols"].passed
    assert results["code_required_symbols"].value >= 1


def test_validate_kinematics_no_file():
    """No kinematics file should produce failed metrics."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={},
        ros2_package_name="test_arm",
    )
    results = stage._validate_kinematics(code)
    assert not results["kinematics_fk"].passed
    assert not results["kinematics_ik_roundtrip"].passed


def test_forward_kinematics_valid():
    """Forward kinematics should return valid 4x4 matrix."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/arm_kinematics.py": (
                "import numpy as np\n"
                "import math\n"
                "class ArmKinematics:\n"
                "    def __init__(self):\n"
                "        self.dh_params = [\n"
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                "        ]\n"
                "    def forward(self, q):\n"
                "        T = np.eye(4)\n"
                "        for dh, qi in zip(self.dh_params, q):\n"
                "            ct, st = math.cos(qi), math.sin(qi)\n"
                "            ca, sa = math.cos(dh['alpha']), math.sin(dh['alpha'])\n"
                "            a = dh['a']\n"
                "            d = dh['d']\n"
                "            Ti = np.array([\n"
                "                [ct, -st*ca, st*sa, a*ct],\n"
                "                [st, ct*ca, -ct*sa, a*st],\n"
                "                [0,  sa,     ca,     d   ],\n"
                "                [0,  0,      0,      1   ],\n"
                "            ])\n"
                "            T = T @ Ti\n"
                "        return T\n"
                "    def inverse(self, pose):\n"
                "        return [0.0]*2, True\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_kinematics(code)
    assert results["kinematics_fk"].passed
    assert results["kinematics_fk"].value == 0.0


def test_ik_roundtrip_success():
    """IK round-trip should produce position error < 1e-3."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/arm_kinematics.py": (
                "import numpy as np\n"
                "import math\n"
                "class ArmKinematics:\n"
                "    def __init__(self):\n"
                "        self.dh_params = [\n"
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                "        ]\n"
                "    def forward(self, q):\n"
                "        T = np.eye(4)\n"
                "        for dh, qi in zip(self.dh_params, q):\n"
                "            ct, st = math.cos(qi), math.sin(qi)\n"
                "            a = dh['a']\n"
                "            Ti = np.array([\n"
                "                [ct, -st, 0, a*ct],\n"
                "                [st, ct, 0, a*st],\n"
                "                [0, 0, 1, 0],\n"
                "                [0, 0, 0, 1],\n"
                "            ])\n"
                "            T = T @ Ti\n"
                "        return T\n"
                "    def inverse(self, pose):\n"
                "        x = pose[0, 3]\n"
                "        y = pose[1, 3]\n"
                "        a1, a2 = 0.3, 0.3\n"
                "        c2 = (x*x + y*y - a1*a1 - a2*a2) / (2*a1*a2)\n"
                "        if abs(c2) > 1:\n"
                "            return [0.0, 0.0], False\n"
                "        q2 = math.acos(c2)\n"
                "        q1 = math.atan2(y, x) - math.atan2(a2*math.sin(q2), a1 + a2*math.cos(q2))\n"
                "        return [q1, q2], True\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_kinematics(code)
    assert results["kinematics_fk"].passed
    assert results["kinematics_ik_roundtrip"].passed
    assert results["kinematics_ik_roundtrip"].value < 1e-3


def test_validate_code_execution_aggregates_all():
    """Orchestrator should include structure + kinematics metrics."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal):\n"
                "        return []\n"
            ),
            "src/arm_kinematics.py": (
                "import numpy as np\n"
                "import math\n"
                "class ArmKinematics:\n"
                "    def __init__(self):\n"
                "        self.dh_params = [\n"
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                '            {"a": 0.3, "alpha": 0, "d": 0, "theta": 0},\n'
                "        ]\n"
                "    def forward(self, q):\n"
                "        T = np.eye(4)\n"
                "        for dh, qi in zip(self.dh_params, q):\n"
                "            ct, st = math.cos(qi), math.sin(qi)\n"
                "            ca, sa = math.cos(dh['alpha']), math.sin(dh['alpha'])\n"
                "            a = dh['a']\n"
                "            d = dh['d']\n"
                "            Ti = np.array([\n"
                "                [ct, -st*ca, st*sa, a*ct],\n"
                "                [st, ct*ca, -ct*sa, a*st],\n"
                "                [0,  sa,     ca,     d   ],\n"
                "                [0,  0,      0,      1   ],\n"
                "            ])\n"
                "            T = T @ Ti\n"
                "        return T\n"
                "    def inverse(self, pose):\n"
                "        return [0.0]*2, True\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    results = stage._validate_code_execution(code)
    assert "code_syntax_valid" in results
    assert "code_required_symbols" in results
    assert "kinematics_fk" in results
    assert "kinematics_ik_roundtrip" in results


def test_merge_code_metrics_preserves_existing():
    """Merging code metrics should not overwrite simulation metrics."""
    from articulate_core.simulation.metrics import MetricResult as SimMetricResult

    stage = SimulationStage.__new__(SimulationStage)
    report = ValidationReport(
        passed=False,
        metrics={
            "joint_torque_peak": SimMetricResult(
                name="joint_torque_peak", passed=False,
                value=1.5, threshold=0.95, unit="ratio",
            ),
        },
        summary="Test",
    )
    code_metrics = {
        "code_syntax_valid": SimMetricResult(
            name="code_syntax_valid", passed=True,
            value=0.0, threshold=0.5, unit="errors",
        ),
    }
    merged = stage._merge_code_metrics(report, code_metrics)
    assert "joint_torque_peak" in merged.metrics
    assert merged.metrics["joint_torque_peak"].value == 1.5
    assert "code_syntax_valid" in merged.metrics
    assert not merged.passed  # original pass/fail preserved


def test_extract_trajectory_uses_execution_when_available():
    """Extract trajectory should use TrajectoryPlanner execution when available."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import numpy as np\n"
                "WAYPOINTS = [\n"
                "    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\n"
                "    [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],\n"
                "]\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal, dt=0.01):\n"
                "        n = max(2, int(1.0 / dt))\n"
                "        traj = []\n"
                "        for i in range(n):\n"
                "            s = i / max(n - 1, 1)\n"
                "            pos = [start[j] + s * (goal[j] - start[j]) for j in range(6)]\n"
                "            traj.append({'time': i * dt, 'positions': pos, 'velocity': 0.5})\n"
                "        return traj\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert len(traj.time_steps) >= 50
    assert traj.joint_positions.shape[1] == 6
    assert np.allclose(traj.joint_positions[0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=0.01)
    assert np.allclose(traj.joint_positions[-1], [0.5, -0.2, 0.3, 0.0, 0.0, 0.0], atol=0.01)


def test_extract_trajectory_fallback_on_execution_failure():
    """Extract trajectory should fall back to S-curve when execution fails."""
    stage = SimulationStage.__new__(SimulationStage)
    code = GeneratedCode(
        package_structure={
            "src/trajectory_planner.py": (
                "import nonexistent_module\n"
                "WAYPOINTS = [\n"
                "    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\n"
                "    [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],\n"
                "]\n"
            ),
        },
        ros2_package_name="test_arm",
    )
    traj = stage._extract_trajectory(code)
    assert isinstance(traj, TrajectoryCommand)
    assert traj.joint_positions.shape[1] == 6
    assert len(traj.time_steps) >= 40
