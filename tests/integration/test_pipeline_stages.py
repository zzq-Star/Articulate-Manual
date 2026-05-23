"""Integration tests for pipeline stages (with mock LLM)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.models import (
    GeneratedCode,
    KinematicsStrategy,
    RequirementDocument,
    RiskAssessment,
    ROS2Architecture,
    StageContext,
    TechnicalApproach,
    TaskType,
    TrajectoryType,
    Waypoint,
    Position3D,
    SpeedRequirements,
    PrecisionRequirements,
    EnvironmentDescription,
)
from articulate_core.pipeline.stage3_generation import CodeGenerationStage
from articulate_core.pipeline.stage4_simulation import SimulationStage
from articulate_core.pipeline.stage5_deployment import DeploymentStage
from articulate_core.skill import ArticulateSkill


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.complete_structured = AsyncMock()
    return llm


def _make_ctx(project_dir: Path) -> StageContext:
    """Build a StageContext with stages 1-3 completed."""
    return StageContext(
        project_dir=project_dir,
        user_input="pick and place from (0.1, 0.2, 0.3) to (0.4, 0.5, 0.6)",
        current_stage=3,
        requirement_doc=RequirementDocument(
            task_type=TaskType.pick_and_place,
            key_waypoints=[
                Waypoint(position=Position3D(x=0.1, y=0.2, z=0.3)),
                Waypoint(position=Position3D(x=0.4, y=0.5, z=0.6)),
            ],
            end_effector="gripper",
            speed_requirements=SpeedRequirements(linear=0.5),
            precision_requirements=PrecisionRequirements(position_mm=1.0),
            environment=EnvironmentDescription(),
            missing_information=[],
            summary="Test pick and place",
        ),
        technical_approach=TechnicalApproach(
            arm_parameters={
                "dh_params": [
                    {"a": 0, "alpha": 0, "d": 0, "theta": 0},
                    {"a": 0, "alpha": 0, "d": 0, "theta": 0},
                    {"a": 0, "alpha": 0, "d": 0, "theta": 0},
                ],
                "joint_limits": [
                    {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
                    {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
                    {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100},
                ],
            },
            kinematics_strategy=KinematicsStrategy(method="numerical"),
            trajectory_types=[TrajectoryType.ptp, TrajectoryType.lin],
            ros2_architecture=ROS2Architecture(
                nodes=[{"name": "arm_controller", "type": "position_controller"}],
            ),
            simulation_feasibility=True,
            risk_assessment=RiskAssessment(level="low", items=[], warnings=[]),
            description="Test technical approach",
        ),
    )


@pytest.fixture
def config_and_ctx(tmp_path):
    """Return (config, completed_ctx) using tmp_path as project dir."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    config = ArticulateConfig(
        project_dir=project_dir,
        anthropic_api_key="test-key",
    )
    ctx = _make_ctx(project_dir)
    return config, ctx


@pytest.fixture
def skill(config_and_ctx, mock_llm):
    config, _ = config_and_ctx
    return ArticulateSkill(config, llm=mock_llm)


@pytest.mark.asyncio
async def test_stage3_code_generation(config_and_ctx, mock_llm, skill):
    """Stage 3 should generate code structure from technical approach."""
    config, ctx = config_and_ctx
    ctx.current_stage = 2
    stage = CodeGenerationStage(config, mock_llm, skill)
    with patch("builtins.input", return_value="y"):
        result = await stage.execute(ctx)
    assert result.generated_code is not None
    assert len(result.generated_code.package_structure) > 0
    assert result.should_continue


@pytest.mark.asyncio
async def test_stage4_simulation_fallback(config_and_ctx, mock_llm, skill):
    """Stage 4 should run (kinematic fallback) with generated code."""
    config, ctx = config_and_ctx
    ctx.generated_code = GeneratedCode(
        package_structure={
            "src/trajectory.py": "waypoints = [[0.0, 0.0, 0.3], [0.3, 0.0, 0.3]]",
        },
        ros2_package_name="test_arm",
    )
    ctx.current_stage = 3

    with patch("builtins.input", return_value="y"):
        stage = SimulationStage(config, mock_llm, skill)
        result = await stage.execute(ctx)

    assert result.simulation_report is not None
    assert result.should_continue


@pytest.mark.asyncio
async def test_stage5_deployment(config_and_ctx, mock_llm, skill):
    """Stage 5 should generate a deployment package."""
    config, ctx = config_and_ctx
    ctx.generated_code = GeneratedCode(
        package_structure={
            "src/trajectory.py": "waypoints = [[0.0, 0.0, 0.3], [0.3, 0.0, 0.3]]",
        },
        ros2_package_name="test_arm",
    )
    ctx.current_stage = 4

    with patch("builtins.input", return_value="y"):
        with patch.object(Path, "exists", return_value=True):
            stage = DeploymentStage(config, mock_llm, skill)
            result = await stage.execute(ctx)

    assert result.deployment_package is not None
    assert result.deployment_package.target_brand == "ur"
    assert result.should_continue


@pytest.mark.asyncio
async def test_stage3_no_approach(config_and_ctx, mock_llm, skill):
    """Stage 3 should fail gracefully with no technical approach."""
    config, ctx = config_and_ctx
    ctx.technical_approach = None
    ctx.current_stage = 2
    stage = CodeGenerationStage(config, mock_llm, skill)
    result = await stage.execute(ctx)
    assert not result.should_continue


@pytest.mark.asyncio
async def test_stage4_no_code(config_and_ctx, mock_llm, skill):
    """Stage 4 should fail gracefully with no generated code."""
    config, ctx = config_and_ctx
    ctx.generated_code = None
    ctx.current_stage = 3
    stage = SimulationStage(config, mock_llm, skill)
    result = await stage.execute(ctx)
    assert not result.should_continue


@pytest.mark.asyncio
async def test_stage5_no_code(config_and_ctx, mock_llm, skill):
    """Stage 5 should fail gracefully with no generated code."""
    config, ctx = config_and_ctx
    ctx.generated_code = None
    ctx.current_stage = 4
    stage = DeploymentStage(config, mock_llm, skill)
    result = await stage.execute(ctx)
    assert not result.should_continue
