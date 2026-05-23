import json
import logging

from articulate_core.pipeline.models import (
    KinematicsStrategy,
    RiskAssessment,
    ROS2Architecture,
    StageContext,
    TechnicalApproach,
    TrajectoryType,
)
from articulate_core.pipeline.orchestrator import BaseStage
from articulate_core.skill.models.dh_template import arm_from_preset
from articulate_core.skill.models.preset_arms import PRESET_ARMS

logger = logging.getLogger(__name__)


class TechnicalApproachStage(BaseStage):
    stage_id: int = 2
    stage_name: str = "technical_approach"

    async def execute(self, ctx: StageContext) -> StageContext:
        logger.info("[Stage 2] Designing technical approach")
        from pydantic import BaseModel, Field

        # 1. Select arm model
        arm = await self._select_arm(ctx)
        print(f"\n[Stage 2] Selected arm: {arm.name} ({arm.num_dof()}-DOF)")

        # 2. Render prompt
        system_prompt, user_msg = self.skill.prompt_mgr.render(
            "technical_design",
            requirement=ctx.requirement_doc.to_dict(),
            arm_model=arm.to_dict(),
        )

        # 3. Call LLM for approach design
        class ApproachSchema(BaseModel):
            kinematics_strategy: dict
            trajectory_types: list
            ros2_architecture: dict
            simulation_feasibility: bool
            description: str

        response = await self.llm.complete_structured(
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            output_model=ApproachSchema,
        )

        # 4. Risk assessment
        risk_prompt, _ = self.skill.prompt_mgr.render(
            "risk_assessment",
            approach=response.model_dump(),
            arm_model=arm.to_dict(),
        )

        class RiskSchema(BaseModel):
            level: str
            items: list
            warnings: list

        try:
            risk_result = await self.llm.complete_structured(
                system=risk_prompt,
                messages=[{"role": "user", "content": "Assess risks for this approach."}],
                output_model=RiskSchema,
            )
        except Exception as e:
            logger.warning("Risk assessment failed: %s", e)
            risk_result = RiskSchema(level="unknown", items=[], warnings=["Risk assessment unavailable"])

        # 5. Build TechnicalApproach
        approach = TechnicalApproach(
            arm_parameters=arm.to_dict(),
            kinematics_strategy=KinematicsStrategy(
                method=response.kinematics_strategy.get("method", "numerical"),
                redundancy_resolution=response.kinematics_strategy.get("redundancy_resolution"),
            ),
            trajectory_types=[
                TrajectoryType(t) for t in response.trajectory_types
            ],
            ros2_architecture=ROS2Architecture(**response.ros2_architecture),
            simulation_feasibility=response.simulation_feasibility,
            risk_assessment=RiskAssessment(
                level=risk_result.level,
                items=risk_result.items,
                warnings=risk_result.warnings,
            ),
            description=response.description,
        )

        # 6. Display approach
        print("\n" + "=" * 60)
        print("STAGE 2: TECHNICAL APPROACH")
        print("=" * 60)
        print(f"  Arm: {arm.name}")
        print(f"  Kinematics: {approach.kinematics_strategy.method}")
        print(f"  Trajectory types: {[t.value for t in approach.trajectory_types]}")
        print(f"  Simulation feasible: {approach.simulation_feasibility}")
        print(f"  Risk level: {approach.risk_assessment.level}")

        if approach.risk_assessment.warnings:
            print("  Warnings:")
            for w in approach.risk_assessment.warnings:
                print(f"    ! {w}")

        print(f"\n  {approach.description}")
        print("=" * 60)

        # 7. User confirmation
        if not await self._confirm():
            logger.info("User cancelled at Stage 2")
            return await self.rollback(ctx)

        ctx.technical_approach = approach
        logger.info("[Stage 2] Technical approach complete")
        return ctx

    async def _select_arm(self, ctx: StageContext) -> "ArmModel":
        """Select arm model based on requirement.

        Auto-select based on task or let user choose.
        """
        available = list(PRESET_ARMS.keys())

        print(f"\n  Available arms: {', '.join(available)}")

        # Simple heuristic: if task mentions 7-DOF or collaborative, use that
        req_lower = ctx.requirement_doc.summary.lower() if ctx.requirement_doc else ""
        if "7" in req_lower and "dof" in req_lower:
            return PRESET_ARMS["seven_dof_standard"]

        choice = input(f"  Select arm (default: {available[0]}): ").strip()
        if choice and choice in PRESET_ARMS:
            return PRESET_ARMS[choice]
        return PRESET_ARMS[available[0]]

    async def _confirm(self) -> bool:
        """Prompt user for confirmation."""
        response = input("Does this approach look correct? (y/n): ").strip().lower()
        return response == "y" or response == "yes"

    async def rollback(self, ctx: StageContext) -> StageContext:
        ctx.technical_approach = None
        ctx.should_continue = False
        return ctx
