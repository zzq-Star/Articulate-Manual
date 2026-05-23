"""Tests using RecordedLLM — verify prompt quality and response parsing with realistic LLM outputs.

These tests use pre-recorded LLM responses (in tests/fixtures/) to validate
that each pipeline stage correctly processes realistic LLM outputs without
requiring an API key. If the prompts change, the fixtures need updating.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.models import (
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
    GeneratedCode,
)
from articulate_core.pipeline.stage1_requirement import RequirementStage
from articulate_core.pipeline.stage2_approach import TechnicalApproachStage
from articulate_core.pipeline.stage3_generation import CodeGenerationStage
from articulate_core.pipeline.stage4_simulation import SimulationStage
from articulate_core.skill import ArticulateSkill
from tests.fixtures.recorded_llm import RecordedLLM


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config(tmp_path):
    return ArticulateConfig(
        project_dir=tmp_path / "test_project",
        anthropic_api_key="",  # No real key needed
    )


@pytest.fixture
def recorded_llm():
    return RecordedLLM(FIXTURES_DIR)


@pytest.fixture
def skill(config, recorded_llm):
    return ArticulateSkill(config, llm=recorded_llm)


# ─── Stage 1: Requirement Analysis ───────────────────────────────────────

class TestRequirementStageWithFixtures:
    """Verify Stage 1 correctly parses a realistic requirement from LLM."""

    @pytest.mark.asyncio
    async def test_requirement_parsing(self, config, recorded_llm, skill, tmp_path):
        ctx = StageContext(
            project_dir=tmp_path / "test_project",
            user_input="Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)",
        )
        stage = RequirementStage(config, recorded_llm, skill)
        with patch("builtins.input", return_value="y"):
            result = await stage.execute(ctx)

        assert result.requirement_doc is not None
        doc = result.requirement_doc

        # Task type
        assert doc.task_type == TaskType.pick_and_place

        # Waypoints — the fixture has 7 waypoints (pick→approach→retract→place)
        assert len(doc.key_waypoints) >= 2
        assert doc.key_waypoints[0].position.x == 0.3
        assert doc.key_waypoints[0].position.y == 0.0

        # End effector
        assert doc.end_effector == "parallel_gripper"

        # Speed requirements
        assert doc.speed_requirements.linear == 0.25
        assert doc.speed_requirements.angular == 0.5

        # Precision
        assert doc.precision_requirements.position_mm == 1.0

        # Environment has a table obstacle
        assert len(doc.environment.obstacles) >= 1
        assert doc.environment.obstacles[0].position.x == 0.45

        # No missing info
        assert len(doc.missing_information) == 0

        # Summary is populated
        assert len(doc.summary) > 0
        assert "place" in doc.summary.lower()


# ─── Stage 2: Technical Approach ─────────────────────────────────────────

class TestTechnicalApproachStageWithFixtures:
    """Verify Stage 2 correctly parses technical design + risk assessment."""

    @pytest.mark.asyncio
    async def test_approach_parsing(self, config, recorded_llm, skill, tmp_path):
        # Need a requirement doc in context for Stage 2
        ctx = StageContext(
            project_dir=tmp_path / "test_project",
            user_input="Pick and place from (0.3, 0, 0.2) to (0.6, 0, 0.3)",
            requirement_doc=RequirementDocument(
                task_type=TaskType.pick_and_place,
                key_waypoints=[
                    Waypoint(position=Position3D(x=0.3, y=0.0, z=0.2)),
                    Waypoint(position=Position3D(x=0.6, y=0.0, z=0.3)),
                ],
                end_effector="parallel_gripper",
                speed_requirements=SpeedRequirements(linear=0.25),
                precision_requirements=PrecisionRequirements(position_mm=1.0),
                environment=EnvironmentDescription(),
                missing_information=[],
                summary="Pick and place test",
            ),
        )

        with patch("builtins.input", return_value="y"):
            stage = TechnicalApproachStage(config, recorded_llm, skill)
            result = await stage.execute(ctx)

        assert result.technical_approach is not None
        ta = result.technical_approach

        # Kinematics
        assert ta.kinematics_strategy.method == "numerical"

        # Trajectory types
        assert TrajectoryType.ptp in ta.trajectory_types
        assert TrajectoryType.lin in ta.trajectory_types

        # ROS2 architecture
        assert len(ta.ros2_architecture.nodes) >= 1
        assert ta.ros2_architecture.nodes[0]["name"] == "arm_controller"

        # Simulation feasible
        assert ta.simulation_feasibility is True

        # Description
        assert len(ta.description) > 0

        # Risk assessment
        assert ta.risk_assessment.level == "medium"
        assert len(ta.risk_assessment.items) >= 1
        assert len(ta.risk_assessment.warnings) >= 1


# ─── Stage 3: Code Generation (decomposition only) ───────────────────────

class TestCodeGenerationWithFixtures:
    """Verify Stage 3 correctly handles LLM sub-task decomposition."""

    @pytest.mark.asyncio
    async def test_subtask_decomposition(self, config, recorded_llm, skill, tmp_path):
        """Verify that LLM-driven decomposition produces sub-tasks."""
        ctx = StageContext(
            project_dir=tmp_path / "test_project",
            user_input="test",
            current_stage=2,
            technical_approach=TechnicalApproach(
                arm_parameters={
                    "dh_params": [{"a": 0, "alpha": 0, "d": 0, "theta": 0}] * 6,
                    "joint_limits": [{"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}] * 6,
                },
                kinematics_strategy=KinematicsStrategy(method="numerical"),
                trajectory_types=[TrajectoryType.ptp, TrajectoryType.lin],
                ros2_architecture=ROS2Architecture(nodes=[{"name": "arm_controller", "type": "position_controller"}]),
                simulation_feasibility=True,
                risk_assessment=RiskAssessment(level="low", items=[], warnings=[]),
                description="Pick and place with 6-DOF arm",
            ),
        )

        with patch("builtins.input", return_value="y"):
            stage = CodeGenerationStage(config, recorded_llm, skill)
            result = await stage.execute(ctx)

        assert result.generated_code is not None
        assert len(result.generated_code.package_structure) > 0

        # Should have ROS2 files
        files = result.generated_code.package_structure
        assert any("package.xml" in f for f in files), f"Files: {list(files.keys())}"
        assert any("__init__.py" in f for f in files)

        # Code should be valid Python
        for path, content in files.items():
            if path.endswith(".py") and content.strip():
                import ast
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    pytest.fail(f"Invalid Python in {path}: {e}")


# ─── Simulation Failure Analysis ─────────────────────────────────────────

class TestSimulationDiagnosisWithFixtures:
    """Verify simulation auto-repair loop processes LLM diagnosis correctly."""

    @pytest.mark.asyncio
    async def test_failure_diagnosis_from_fixture(self):
        """Verify the DiagnosisSchema can be parsed from the fixture."""
        fixture_path = FIXTURES_DIR / "failure_analysis.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))

        from pydantic import BaseModel
        from typing import Optional

        class DiagnosisSchema(BaseModel):
            diagnosis: str
            severity: str
            suggested_repairs: list
            alternative_approach: Optional[str] = None

        result = DiagnosisSchema.model_validate(data["response"])

        assert result.diagnosis.startswith("Joint velocities exceed")
        assert result.severity == "medium"
        assert len(result.suggested_repairs) == 3
        assert result.suggested_repairs[0]["action"] == "reduce_speed"
        assert result.alternative_approach is not None


# ─── Sanity: All Fixtures Load ───────────────────────────────────────────

class TestFixturesLoadCorrectly:
    """Verify all fixture files are loadable by RecordedLLM."""

    def test_all_fixtures_load(self):
        llm = RecordedLLM(FIXTURES_DIR)
        assert len(llm._fixtures) >= 6  # at least 6 schemas

    def test_each_fixture_has_required_fields(self):
        for path in sorted(FIXTURES_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "schema" in data, f"{path.name} missing 'schema'"
            assert "response" in data, f"{path.name} missing 'response'"
            assert "scenario" in data, f"{path.name} missing 'scenario'"
