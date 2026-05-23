"""Tests for DeploymentManager."""

import tempfile
from pathlib import Path

import pytest

from articulate_core.pipeline.deployment_manager import DeploymentManager
from articulate_core.pipeline.models import GeneratedCode, KinematicsStrategy, RiskAssessment, ROS2Architecture, TechnicalApproach, TrajectoryType


@pytest.fixture
def sample_code():
    return GeneratedCode(
        package_structure={
            "src/trajectory.py": "waypoints = [[0.0, 0.0, 0.3], [0.3, 0.0, 0.3]]",
            "src/main.py": "def main(): pass",
        },
        ros2_package_name="test_arm",
    )


@pytest.fixture
def sample_code_blank():
    return GeneratedCode(
        package_structure={},
        ros2_package_name="test_arm",
    )


class TestDeploymentManager:
    def test_prepare_ur(self, sample_code):
        manager = DeploymentManager(sample_code)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = manager.prepare("ur", Path(tmpdir) / "deploy")
            assert pkg.target_brand == "ur"
            assert len(pkg.files) >= 1
            assert any("script" in str(f) for f in pkg.files.values())
            assert pkg.guide_path.exists()
            assert pkg.checklist_path.exists()

    def test_prepare_kuka(self, sample_code):
        manager = DeploymentManager(sample_code)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = manager.prepare("kuka", Path(tmpdir) / "deploy")
            assert len(pkg.files) == 2  # .src + .dat
            assert any("src" in str(f) for f in pkg.files)

    def test_prepare_abb(self, sample_code):
        manager = DeploymentManager(sample_code)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = manager.prepare("abb", Path(tmpdir) / "deploy")
            assert len(pkg.files) >= 1
            assert any("mod" in str(f) for f in pkg.files)

    def test_prepare_with_blank_code(self, sample_code_blank):
        """Should handle empty code gracefully by generating default waypoints."""
        manager = DeploymentManager(sample_code_blank)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = manager.prepare("ur", Path(tmpdir) / "deploy")
            assert len(pkg.files) >= 1

    def test_prepare_with_approach(self, sample_code):
        approach = TechnicalApproach(
            arm_parameters={"dh_params": [{"a": 0, "alpha": 0, "d": 0, "theta": 0}] * 6},
            kinematics_strategy=KinematicsStrategy(method="numerical"),
            trajectory_types=[TrajectoryType.ptp, TrajectoryType.lin],
            ros2_architecture=ROS2Architecture(),
            simulation_feasibility=True,
            risk_assessment=RiskAssessment(level="low", items=[], warnings=[]),
            description="Test arm",
        )
        manager = DeploymentManager(sample_code, approach)
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = manager.prepare("ur", Path(tmpdir) / "deploy")
            assert pkg.target_brand == "ur"
