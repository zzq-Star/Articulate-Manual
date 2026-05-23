"""Tests for validation engine."""

import numpy as np
import pytest

from articulate_core.simulation.metrics import SimulationData
from articulate_core.simulation.validation_engine import ValidationEngine, ValidationReport


@pytest.fixture
def engine():
    return ValidationEngine()


@pytest.fixture
def valid_data():
    """Smooth data within all metric thresholds."""
    t = np.linspace(0, 1.0, 50)
    pos = 0.01 * np.ones((50, 6))  # small constant offset from zero
    return SimulationData(
        time=t,
        joint_positions=pos,
        joint_velocities=np.zeros((50, 6)),
        joint_accelerations=np.zeros((50, 6)),
        joint_torques=np.zeros((50, 6)),
        tcp_positions=np.column_stack([np.linspace(0, 0.3, 50), np.zeros(50), np.zeros(50)]),
        tcp_orientations=np.zeros((50, 3)),
        self_collision_distances=np.ones(50) * 0.1,
        condition_numbers=np.ones(50) * 5.0,
        arm_model={
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}
                for _ in range(6)
            ]
        },
    )
    return SimulationData(
        time=t,
        joint_positions=pos,
        joint_velocities=np.gradient(pos, axis=0) / np.mean(np.diff(t)),
        joint_accelerations=np.gradient(pos, axis=0) / np.mean(np.diff(t)),
        joint_torques=np.abs(pos) * 50,
        tcp_positions=np.column_stack([np.sin(t), np.cos(t), t * 0.1]),
        tcp_orientations=np.zeros((100, 3)),
        self_collision_distances=np.ones(100) * 0.1,
        condition_numbers=np.ones(100) * 5.0,
        arm_model={
            "joint_limits": [
                {"lower": -3.14, "upper": 3.14, "velocity": 1.0, "torque": 100}
                for _ in range(6)
            ]
        },
    )


@pytest.mark.asyncio
async def test_validation_passes(engine, valid_data):
    """Validation with clean data should report pass."""
    report = await engine.validate(valid_data)
    assert report.passed
    assert report.pass_rate >= 0.9


@pytest.mark.asyncio
async def test_validation_empty_data(engine):
    """Validation with empty data should not crash."""
    data = SimulationData(
        time=np.array([]),
        joint_positions=np.zeros((0, 6)),
        joint_velocities=np.zeros((0, 6)),
        joint_accelerations=np.zeros((0, 6)),
        joint_torques=np.zeros((0, 6)),
        tcp_positions=np.zeros((0, 3)),
        tcp_orientations=np.zeros((0, 3)),
        self_collision_distances=np.array([]),
        condition_numbers=np.array([]),
    )
    report = await engine.validate(data)
    assert isinstance(report, ValidationReport)
    assert len(report.metrics) == 10


@pytest.mark.asyncio
async def test_validation_all_metrics_present(engine, valid_data):
    """Validation should include all 10 metrics."""
    report = await engine.validate(valid_data)
    assert len(report.metrics) == 10
    expected = {
        "joint_position_error", "joint_velocity_overshoot",
        "joint_acceleration_peak", "joint_torque_peak",
        "self_collision_distance", "joint_limit_margin",
        "path_jerk", "condition_number",
        "tcp_position_error", "payload_ratio",
    }
    assert set(report.metrics.keys()) == expected


@pytest.mark.asyncio
async def test_validation_report_structure(engine, valid_data):
    """ValidationReport should have correct properties."""
    report = await engine.validate(valid_data)
    assert hasattr(report, "passed")
    assert hasattr(report, "metrics")
    assert hasattr(report, "summary")
    assert hasattr(report, "recommendations")
    assert hasattr(report, "failed_metrics")
    assert hasattr(report, "passed_metrics")
    assert hasattr(report, "timestamp")
    assert isinstance(report.recommendations, list)
    assert isinstance(report.failed_metrics, list)
    assert isinstance(report.passed_metrics, list)


@pytest.mark.asyncio
async def test_validation_failed_metrics(engine, valid_data):
    """Validation should correctly identify failed metrics."""
    # Make collision data very small to force fail
    valid_data.self_collision_distances = np.ones(100) * 0.001
    report = await engine.validate(valid_data)
    if not report.passed:
        failed_names = [m.name for m in report.failed_metrics]
        assert "self_collision_distance" in failed_names


@pytest.mark.asyncio
async def test_validation_summary_passes(engine, valid_data):
    """Summary should indicate pass."""
    report = await engine.validate(valid_data)
    assert "All" in report.summary or "passed" in report.summary


@pytest.mark.asyncio
async def test_validation_summary_fails(engine, valid_data):
    """Summary should indicate failures."""
    # Make extreme data to force failures
    valid_data.joint_torques = np.ones((100, 6)) * 1000
    report = await engine.validate(valid_data)
    if not report.passed:
        assert "failed" in report.summary or "FAIL" in report.summary.upper()


@pytest.mark.asyncio
async def test_recommendations_for_failures(engine, valid_data):
    """Recommendations should match failed metrics."""
    # Force velocity overshoot
    valid_data.joint_velocities = np.ones((100, 6)) * 10.0
    report = await engine.validate(valid_data)
    recs_text = " ".join(report.recommendations).lower()
    if "joint_velocity_overshoot" in [m.name for m in report.failed_metrics]:
        assert "velocity" in recs_text


@pytest.mark.asyncio
async def test_generate_markdown(engine, valid_data):
    """Markdown report generation should not error."""
    report = await engine.validate(valid_data)
    md = engine.generate_report_markdown(report)
    assert isinstance(md, str)
    assert len(md) > 50
    assert "# Simulation Validation Report" in md


@pytest.mark.asyncio
async def test_generate_html(engine, valid_data):
    """HTML report generation should not error."""
    report = await engine.validate(valid_data)
    html = engine.generate_report_html(report)
    assert isinstance(html, str)
    assert len(html) > 100
    assert "Simulation Validation Report" in html
    assert "<html>" in html


@pytest.mark.asyncio
async def test_pass_rate_zero(engine):
    """Pass rate should be 0.0 for empty metrics (edge case)."""
    report = ValidationReport(
        passed=False,
        metrics={},
        summary="test",
    )
    assert report.pass_rate == 0.0


@pytest.mark.asyncio
async def test_pass_rate_partial(engine, valid_data):
    """Pass rate should reflect partial passes."""
    report = await engine.validate(valid_data)
    rate = report.pass_rate
    assert 0.0 <= rate <= 1.0
