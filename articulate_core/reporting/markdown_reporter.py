"""Markdown report generator — produces a comprehensive project report."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from articulate_core.pipeline.models import (
    DeploymentPackage,
    GeneratedCode,
    RequirementDocument,
    SimulationReport,
    StageContext,
    TechnicalApproach,
)


class MarkdownReportGenerator:
    """Generates a comprehensive Markdown report from pipeline stage outputs."""

    def generate(self, ctx: StageContext, output_path: Optional[Path] = None) -> str:
        """Generate full Markdown report."""
        sections = [
            "# Articulate Pipeline Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Project:** {ctx.project_dir.name}",
            f"**User Input:** {ctx.user_input or 'N/A'}",
            f"**Target Brand:** {ctx.target_brand.upper() if ctx.target_brand else 'N/A'}",
            "",
            "---",
            "",
            self._overview(ctx),
            self._requirement_section(ctx.requirement_doc),
            self._approach_section(ctx.technical_approach),
            self._code_section(ctx.generated_code),
            self._simulation_section(ctx.simulation_report),
            self._deployment_section(ctx.deployment_package, ctx.target_brand),
        ]
        md = "\n".join(sections)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(md, encoding="utf-8")

        return md

    @staticmethod
    def _overview(ctx: StageContext) -> str:
        stages = []
        if ctx.requirement_doc:
            stages.append("- Requirement: Completed")
        if ctx.technical_approach:
            stages.append("- Technical Approach: Completed")
        if ctx.generated_code:
            stages.append("- Code Generation: Completed")
        if ctx.simulation_report:
            status = "Passed" if ctx.simulation_report.passed else "Failed"
            stages.append(f"- Simulation: {status}")
        if ctx.deployment_package:
            stages.append("- Deployment: Completed")
        return "## Pipeline Overview\n\n" + "\n".join(stages) if stages else "## Pipeline Overview\n\n_No stages completed._"

    @staticmethod
    def _requirement_section(req: Optional[RequirementDocument]) -> str:
        if not req:
            return "## Requirement\n\n_Skipped._\n"
        return f"""## Requirement

- **Task Type:** {req.task_type.value}
- **Summary:** {req.summary}
- **Waypoints:** {len(req.key_waypoints)}
- **End Effector:** {req.end_effector or 'N/A'}
"""

    @staticmethod
    def _approach_section(approach: Optional[TechnicalApproach]) -> str:
        if not approach:
            return "## Technical Approach\n\n_Skipped._\n"
        arm = approach.arm_parameters
        n_dof = len(arm.get("dh_params", [])) if arm else 0
        traj_types = ", ".join(t.value for t in approach.trajectory_types)
        return f"""## Technical Approach

- **DOF:** {n_dof}
- **Kinematics:** {approach.kinematics_strategy.method}
- **Trajectory Types:** {traj_types}
- **Risk Level:** {approach.risk_assessment.level.upper()}
- **Description:** {approach.description}
"""

    @staticmethod
    def _code_section(code: Optional[GeneratedCode]) -> str:
        if not code:
            return "## Code Generation\n\n_Skipped._\n"
        files = "\n".join(
            f"  - `{path}` ({len(content)} chars)"
            for path, content in sorted(code.package_structure.items())
        )
        build_status = "Success" if code.build_successful else "N/A"
        return f"""## Code Generation

- **ROS2 Package:** `{code.ros2_package_name}`
- **Files:** {len(code.package_structure)}
- **Build:** {build_status}

**Files:**

{files}
"""

    @staticmethod
    def _simulation_section(report: Optional[SimulationReport]) -> str:
        if not report:
            return "## Simulation\n\n_Skipped._\n"
        metrics = "\n".join(
            f"  - {('PASS' if m.passed else 'FAIL'):4s} | {name}: {m.value:.4f} (threshold: {m.threshold} {m.unit})"
            for name, m in sorted(report.metrics.items())
        )
        return f"""## Simulation Validation

- **Overall:** {'PASS' if report.passed else 'FAIL'}
- **Pass Rate:** {sum(1 for m in report.metrics.values() if m.passed)}/{len(report.metrics)}
- **Repairs:** {len(report.repair_history)}

**Metrics:**

| Status | Metric | Value | Threshold | Unit |
|--------|--------|-------|-----------|------|
{metrics}
"""

    @staticmethod
    def _deployment_section(pkg: Optional[DeploymentPackage], brand: str) -> str:
        if not pkg:
            return "## Deployment\n\n_Skipped._\n"
        files = "\n".join(
            f"  - `{name}` → {path}" for name, path in sorted(pkg.files.items())
        )
        return f"""## Deployment Package

- **Target Brand:** {brand.upper()}
- **Output Directory:** `{pkg.output_dir}`
- **Files:**
{files}
- **Guide:** `{pkg.guide_path.name}`
- **Checklist:** `{pkg.checklist_path.name}`
"""
