import json
import logging
from dataclasses import dataclass
from typing import List

import yaml

from articulate_core.exceptions import StageError
from articulate_core.pipeline.models import (
    EnvironmentDescription,
    Obstacle,
    Position3D,
    PrecisionRequirements,
    RequirementDocument,
    SpeedRequirements,
    StageContext,
    TaskType,
    Waypoint,
)
from articulate_core.pipeline.orchestrator import BaseStage

logger = logging.getLogger(__name__)


class RequirementStage(BaseStage):
    stage_id: int = 1
    stage_name: str = "requirement_analysis"

    async def execute(self, ctx: StageContext) -> StageContext:
        logger.info("[Stage 1] Analyzing requirement: %s", ctx.user_input[:80])

        # 1. Render prompt with user input
        system_prompt, user_msg = self.skill.prompt_mgr.render(
            "requirement_analysis",
            user_input=ctx.user_input,
        )

        # 2. Call LLM for structured parsing
        from pydantic import BaseModel, Field

        class RequirementSchema(BaseModel):
            task_type: str
            key_waypoints: list
            end_effector: str | None
            speed_requirements: dict
            precision_requirements: dict
            environment: dict
            missing_information: list
            summary: str

        response = await self.llm.complete_structured(
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            output_model=RequirementSchema,
        )

        # 3. Convert to RequirementDocument
        waypoints = []
        for wp in response.key_waypoints:
            pos = wp.get("position") or [0, 0, 0]
            orient = wp.get("orientation")
            waypoints.append(Waypoint(
                position=Position3D(x=pos[0], y=pos[1], z=pos[2]),
                orientation=None,
                label=wp.get("label", ""),
            ))

        obstacles = []
        for obs in response.environment.get("obstacles", []):
            pos = obs.get("position") or [0, 0, 0]
            dims = obs.get("dimensions")
            obstacles.append(Obstacle(
                position=Position3D(x=pos[0], y=pos[1], z=pos[2]),
                dimensions=tuple(dims) if dims else None,
            ))

        doc = RequirementDocument(
            task_type=TaskType(response.task_type),
            key_waypoints=waypoints,
            end_effector=response.end_effector,
            speed_requirements=SpeedRequirements(
                linear=response.speed_requirements.get("linear"),
                angular=response.speed_requirements.get("angular"),
            ),
            precision_requirements=PrecisionRequirements(
                position_mm=response.precision_requirements.get("position_mm"),
                rotation_deg=response.precision_requirements.get("rotation_deg"),
            ),
            environment=EnvironmentDescription(
                description=response.environment.get("description", ""),
                obstacles=obstacles,
            ),
            missing_information=response.missing_information,
            summary=response.summary,
        )

        # 4. Handle missing information
        if doc.missing_information:
            print("\n[Stage 1] Missing information detected:")
            for info in doc.missing_information:
                print(f"  ? {info}")
            print()

        # 5. Display requirement summary
        print("\n" + "=" * 60)
        print("STAGE 1: REQUIREMENT ANALYSIS")
        print("=" * 60)
        print(f"  Task type: {doc.task_type.value}")
        print(f"  Summary: {doc.summary}")
        print(f"  Waypoints: {len(doc.key_waypoints)}")
        print(f"  End effector: {doc.end_effector or 'not specified'}")
        if doc.speed_requirements.linear:
            print(f"  Speed: {doc.speed_requirements.linear} m/s")
        if doc.precision_requirements.position_mm:
            print(f"  Precision: {doc.precision_requirements.position_mm} mm")
        print("=" * 60)

        # 6. User confirmation
        if not await self._confirm():
            logger.info("User cancelled at Stage 1")
            return await self.rollback(ctx)

        ctx.requirement_doc = doc
        logger.info("[Stage 1] Requirement analysis complete")
        return ctx

    async def _confirm(self) -> bool:
        """Prompt user for confirmation."""
        response = input("Does this requirement look correct? (y/n): ").strip().lower()
        return response == "y" or response == "yes"

    async def rollback(self, ctx: StageContext) -> StageContext:
        ctx.requirement_doc = None
        ctx.should_continue = False
        return ctx
