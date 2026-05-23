"""Validation engine - runs all metrics and produces reports."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from articulate_core.simulation.metrics import (
    BaseMetric,
    MetricResult,
    SimulationData,
    get_all_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool
    metrics: Dict[str, MetricResult]
    summary: str
    recommendations: List[str] = field(default_factory=list)
    failed_metrics: List[MetricResult] = field(default_factory=list)
    passed_metrics: List[MetricResult] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = datetime.now().isoformat()
        self.failed_metrics = [m for m in self.metrics.values() if not m.passed]
        self.passed_metrics = [m for m in self.metrics.values() if m.passed]

    @property
    def pass_rate(self) -> float:
        total = len(self.metrics)
        if total == 0:
            return 0.0
        return len(self.passed_metrics) / total


class ValidationEngine:
    """Runs all metrics against simulation data and aggregates results."""

    def __init__(self):
        self.metrics: List[BaseMetric] = get_all_metrics()

    async def validate(self, sim_data: SimulationData) -> ValidationReport:
        """Evaluate all metrics against simulation data."""
        logger.info("Validating simulation data (%d steps, %d DOF)", sim_data.n_steps, sim_data.n_dof)

        results: Dict[str, MetricResult] = {}

        for metric in self.metrics:
            try:
                result = metric.evaluate(sim_data)
                results[metric.name] = result
                status = "PASS" if result.passed else "FAIL"
                logger.debug("  %s: %s (%.4f / %.4f)", metric.name, status, result.value, result.threshold)
            except Exception as e:
                logger.warning("Metric '%s' failed to evaluate: %s", metric.name, e)
                results[metric.name] = MetricResult(
                    name=metric.name,
                    passed=False,
                    value=0.0,
                    threshold=metric.threshold,
                    unit=metric.unit,
                    explanation=f"Evaluation error: {e}",
                )

        passed = all(r.passed for r in results.values())
        summary = self._generate_summary(passed, results)
        recommendations = self._generate_recommendations(results)

        return ValidationReport(
            passed=passed,
            metrics=results,
            summary=summary,
            recommendations=recommendations,
        )

    def _generate_summary(self, passed: bool, metrics: Dict[str, MetricResult]) -> str:
        total = len(metrics)
        passed_count = sum(1 for m in metrics.values() if m.passed)
        if passed:
            return f"All {total} metrics passed."
        return f"{passed_count}/{total} metrics passed. {total - passed_count} failure(s) detected."

    def _generate_recommendations(self, metrics: Dict[str, MetricResult]) -> List[str]:
        recommendations = []
        failed = {k: v for k, v in metrics.items() if not v.passed}

        if "joint_velocity_overshoot" in failed:
            recommendations.append("Reduce trajectory velocity scaling factor.")
        if "joint_torque_peak" in failed:
            recommendations.append("Reduce acceleration/deceleration or check payload.")
        if "self_collision_distance" in failed:
            recommendations.append("Add collision avoidance waypoints or adjust path.")
        if "condition_number" in failed:
            recommendations.append("Adjust path to avoid singular configurations.")
        if "path_jerk" in failed:
            recommendations.append("Increase trajectory interpolation points for smoother motion.")

        if not recommendations:
            recommendations.append("All checks passed. Ready for deployment preparation.")

        return recommendations

    def generate_report_markdown(self, report: ValidationReport) -> str:
        """Generate Markdown report."""
        lines = [
            "# Simulation Validation Report",
            f"**Timestamp**: {report.timestamp}",
            f"**Overall**: {'PASS' if report.passed else 'FAIL'}",
            f"**Pass rate**: {report.pass_rate:.0%}",
            "",
            "## Metric Results",
            "| Metric | Status | Value | Threshold | Unit |",
            "|--------|--------|-------|-----------|------|",
        ]

        for name, result in sorted(report.metrics.items()):
            status = ":white_check_mark:" if result.passed else ":x:"
            lines.append(f"| {name} | {status} | {result.value:.4f} | {result.threshold} | {result.unit} |")

        lines.extend([
            "",
            "## Summary",
            report.summary,
            "",
            "## Recommendations",
        ])
        for r in report.recommendations:
            lines.append(f"- {r}")

        if report.failed_metrics:
            lines.extend(["", "## Failed Metrics Details"])
            for m in report.failed_metrics:
                lines.extend([
                    "",
                    f"### {m.name}",
                    f"- Value: {m.value:.4f} {m.unit}",
                    f"- Threshold: {m.threshold} {m.unit}",
                    f"- Explanation: {m.explanation}",
                ])

        lines.append("")
        return "\n".join(lines)

    def generate_report_html(self, report: ValidationReport) -> str:
        """Generate HTML report with color-coded metrics."""
        metric_rows = ""
        for name, result in sorted(report.metrics.items()):
            color = "#4caf50" if result.passed else "#f44336"
            metric_rows += f"""
            <tr>
                <td>{name}</td>
                <td style="color:{color}">{'PASS' if result.passed else 'FAIL'}</td>
                <td>{result.value:.4f}</td>
                <td>{result.threshold}</td>
                <td>{result.unit}</td>
            </tr>"""

        failed_rows = ""
        for m in report.failed_metrics:
            failed_rows += f"""
            <div class="failed-metric">
                <h3>{m.name}</h3>
                <p>Value: {m.value:.4f} {m.unit} (threshold: {m.threshold} {m.unit})</p>
                <p>{m.explanation}</p>
            </div>"""

        recs = "".join(f"<li>{r}</li>" for r in report.recommendations)

        return f"""<!DOCTYPE html>
<html>
<head><title>Validation Report</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
h1 {{ color: #333; }}
.summary {{ font-size: 1.2em; margin: 1em 0; }}
.pass {{ color: #4caf50; }}
.fail {{ color: #f44336; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f5f5f5; }}
.failed-metric {{ background: #fff3f3; border: 1px solid #f44336; padding: 1em; margin: 0.5em 0; }}
</style>
</head>
<body>
<h1>Simulation Validation Report</h1>
<div class="summary {'pass' if report.passed else 'fail'}">
Overall: {'PASS' if report.passed else 'FAIL'} | Pass rate: {report.pass_rate:.0%}
</div>
<p>Timestamp: {report.timestamp}</p>

<h2>Metrics</h2>
<table>
<tr><th>Metric</th><th>Status</th><th>Value</th><th>Threshold</th><th>Unit</th></tr>
{metric_rows}
</table>

<h2>Summary</h2>
<p>{report.summary}</p>

<h2>Recommendations</h2>
<ul>{recs}</ul>

{failed_rows if report.failed_metrics else ''}
</body>
</html>"""
