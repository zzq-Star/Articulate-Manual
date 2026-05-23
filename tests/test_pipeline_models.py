import json
from pathlib import Path

import pytest

from articulate_core.pipeline.models import (
    DHParameter,
    JointLimit,
    Position3D,
    RequirementDocument,
    StageContext,
    TaskType,
    Waypoint,
    SpeedRequirements,
    PrecisionRequirements,
    EnvironmentDescription,
)


def test_dh_parameter():
    """DHParameter holds joint parameters."""
    p = DHParameter(a=0.3, alpha=1.57, d=0.1, theta=0.0)
    assert p.a == 0.3
    assert p.alpha == 1.57


def test_waypoint():
    """Waypoint holds position and optional orientation."""
    wp = Waypoint(position=Position3D(x=0.5, y=0.0, z=0.3), label="target")
    assert wp.position.x == 0.5
    assert wp.label == "target"


def test_requirement_document():
    """RequirementDocument stores parsed requirement."""
    doc = RequirementDocument(
        task_type=TaskType.pick_and_place,
        key_waypoints=[
            Waypoint(position=Position3D(0.3, 0.0, 0.2), label="pick"),
            Waypoint(position=Position3D(0.6, 0.0, 0.3), label="place"),
        ],
        end_effector="gripper",
        speed_requirements=SpeedRequirements(linear=0.5),
        precision_requirements=PrecisionRequirements(position_mm=1.0),
        environment=EnvironmentDescription(description="Conveyor belt"),
        missing_information=[],
        summary="Pick from (0.3,0,0.2) and place at (0.6,0,0.3)",
    )
    assert doc.task_type == TaskType.pick_and_place
    assert len(doc.key_waypoints) == 2
    assert doc.end_effector == "gripper"


def test_stage_context_serialization(tmp_path):
    """StageContext serializes and deserializes correctly."""
    ctx = StageContext(
        project_dir=tmp_path,
        user_input="test requirement",
        current_stage=2,
    )
    data = ctx.to_dict()
    assert data["user_input"] == "test requirement"
    assert data["current_stage"] == 2
    assert data["project_dir"] == str(tmp_path)

    restored = StageContext.from_dict(data)
    assert restored.project_dir == tmp_path
    assert restored.current_stage == 2


def test_joint_limit():
    """JointLimit holds limit values."""
    jl = JointLimit(lower=-2.967, upper=2.967, velocity=2.0, torque=150)
    assert jl.lower == -2.967
    assert jl.upper == 2.967
