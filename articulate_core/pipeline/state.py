from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from articulate_core.pipeline.models import StageContext


@dataclass
class PipelineState:
    """Persistent state between CLI invocations."""

    project_dir: Path
    current_stage: int = 0
    state_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.state_data is None:
            self.state_data = {}

    def save(self, state_dir: Path):
        """Serialize state to JSON file."""
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / "state.json"
        data = {
            "project_dir": str(self.project_dir),
            "current_stage": self.current_stage,
            "state_data": self.state_data,
        }
        state_path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, state_dir: Path) -> Optional["PipelineState"]:
        """Deserialize state from JSON file. Returns None if not found."""
        state_path = state_dir / "state.json"
        if not state_path.exists():
            return None
        data = json.loads(state_path.read_text())
        return cls(
            project_dir=Path(data["project_dir"]),
            current_stage=data.get("current_stage", 0),
            state_data=data.get("state_data", {}),
        )

    def to_context(self) -> StageContext:
        """Convert to StageContext for pipeline execution."""
        ctx = StageContext(
            project_dir=self.project_dir,
            current_stage=self.current_stage,
        )
        # Restore full state from serialized data if available
        sd = self.state_data or {}
        if sd.get("requirement_doc"):
            from articulate_core.pipeline.models import RequirementDocument
            ctx.requirement_doc = RequirementDocument(**sd["requirement_doc"])
        if sd.get("technical_approach"):
            from articulate_core.pipeline.models import TechnicalApproach
            data = sd["technical_approach"]
            ctx.technical_approach = TechnicalApproach(**data)
        if sd.get("generated_code"):
            from articulate_core.pipeline.models import GeneratedCode
            data = sd["generated_code"]
            ctx.generated_code = GeneratedCode(**data)
        if sd.get("simulation_report"):
            from articulate_core.pipeline.models import SimulationReport
            data = sd["simulation_report"]
            ctx.simulation_report = SimulationReport(**data)
        if sd.get("deployment_package"):
            from articulate_core.pipeline.models import DeploymentPackage
            data = sd["deployment_package"]
            ctx.deployment_package = DeploymentPackage(**data)
        if sd.get("target_brand"):
            ctx.target_brand = sd["target_brand"]
        if sd.get("user_input"):
            ctx.user_input = sd["user_input"]
        return ctx

    @classmethod
    def from_context(cls, ctx: StageContext) -> "PipelineState":
        """Create state from a StageContext."""
        return cls(
            project_dir=ctx.project_dir,
            current_stage=ctx.current_stage,
            state_data=ctx.to_dict(),
        )
