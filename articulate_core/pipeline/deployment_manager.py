"""Deployment package generation."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from articulate_core.pipeline.models import GeneratedCode, DeploymentPackage, TechnicalApproach
from articulate_core.skill.converters.base import Trajectory
from articulate_core.skill.converters.factory import ConverterFactory

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Prepares deployment packages for robot brands."""

    def __init__(self, code: GeneratedCode, approach: Optional[TechnicalApproach] = None):
        self.code = code
        self.approach = approach

    def prepare(
        self,
        brand: str,
        output_dir: Path,
    ) -> DeploymentPackage:
        """Generate a full deployment package for the target brand."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract trajectory from generated code
        trajectory = self._extract_trajectory()

        # 2. Get brand-specific converter
        converter = ConverterFactory.get_converter(brand)

        # 3. Generate script files
        files = converter.convert(trajectory, str(output_dir))

        # 4. Write files to disk
        written: Dict[str, Path] = {}
        for rel_path, content in files.items():
            abs_path = output_dir / rel_path
            abs_path.write_text(content, encoding="utf-8")
            written[rel_path] = abs_path
            logger.info("Wrote %s", abs_path)

        # 5. Generate deployment guide
        guide_content = converter.generate_deployment_guide(trajectory)
        guide_path = output_dir / "DEPLOYMENT_GUIDE.md"
        guide_path.write_text(guide_content, encoding="utf-8")
        logger.info("Wrote %s", guide_path)

        # 6. Generate safety checklist
        checklist = converter.generate_safety_checklist()
        checklist_path = output_dir / "SAFETY_CHECKLIST.md"
        checklist_path.write_text(self._format_checklist(checklist), encoding="utf-8")
        logger.info("Wrote %s", checklist_path)

        # 7. Generate metadata
        meta = {
            "brand": brand,
            "waypoints": len(trajectory.waypoints),
            "tool": trajectory.tool_name,
            "payload_kg": trajectory.payload_kg,
            "files": list(written.keys()),
        }
        (output_dir / "deployment_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return DeploymentPackage(
            output_dir=output_dir,
            files=written,
            guide_path=guide_path,
            checklist_path=checklist_path,
            target_brand=brand,
        )

    def _extract_trajectory(self) -> Trajectory:
        """Extract waypoint data from generated code or technical approach."""
        waypoints = []

        # Try extracting from code content first
        if self.code:
            import re
            all_content = "\n".join(self.code.package_structure.values())
            for match in re.finditer(r"\[([\d.,\s-]+)\]", all_content):
                try:
                    vals = [float(x) for x in match.group(1).split(",") if x.strip()]
                    if len(vals) >= 3:
                        waypoints.append({
                            "positions": vals[:6],
                            "type": "PTP",
                        })
                except ValueError:
                    continue

        # Fall back to approach info
        if not waypoints and self.approach:
            n_dof = len(self.approach.arm_parameters.get("dh_params", range(6)))
            for i in range(3):
                waypoints.append({
                    "positions": [0.3 * i] * min(n_dof, 6),
                    "type": "PTP" if i == 0 else "LIN",
                })

        # Ultimate fallback
        if not waypoints:
            waypoints.append({"positions": [0.0, 0.0, 0.3, 0.0, 0.0, 0.0], "type": "PTP"})
            waypoints.append({"positions": [0.3, 0.0, 0.3, 0.0, 0.0, 0.0], "type": "LIN"})

        trajectory_types = self.approach.trajectory_types if self.approach else None
        for i, wp in enumerate(waypoints):
            if trajectory_types and i < len(trajectory_types):
                wp["type"] = trajectory_types[i].value

        return Trajectory(
            waypoints=waypoints,
            speed=0.25,
            blend=0.0,
            tool_name="tool0",
            payload_kg=0.0,
        )

    @staticmethod
    def _format_checklist(items: List[str]) -> str:
        lines = [
            "# Safety Checklist",
            "",
            "## Pre-Operation",
        ]
        for item in items:
            lines.append(f"- [ ] {item}")
        lines.extend([
            "",
            "## Sign-off",
            "",
            "| Role | Name | Date | Signature |",
            "|------|------|------|-----------|",
            "| Operator | | | |",
            "| Supervisor | | | |",
            "| Safety Officer | | | |",
        ])
        return "\n".join(lines)
