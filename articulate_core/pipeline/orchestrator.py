import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from articulate_core.config.settings import ArticulateConfig
from articulate_core.llm.client import BaseLLMClient
from articulate_core.pipeline.models import StageContext
from articulate_core.pipeline.state import PipelineState
from articulate_core.skill import ArticulateSkill

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """Abstract base for all pipeline stages."""

    stage_id: int
    stage_name: str

    def __init__(self, config: ArticulateConfig, llm: BaseLLMClient, skill: ArticulateSkill):
        self.config = config
        self.llm = llm
        self.skill = skill

    @abstractmethod
    async def execute(self, ctx: StageContext) -> StageContext:
        """Execute this stage and return updated context."""

    async def rollback(self, ctx: StageContext) -> StageContext:
        """Roll back this stage's changes in context."""
        return ctx


class PipelineOrchestrator:
    """Central coordinator for the 5-stage pipeline."""

    def __init__(self, config: ArticulateConfig, llm: BaseLLMClient, skill: ArticulateSkill):
        self.config = config
        self.llm = llm
        self.skill = skill
        self.stages: List[BaseStage] = self._init_stages()

    def _init_stages(self) -> List[BaseStage]:
        # Lazy imports to avoid circular deps
        from articulate_core.pipeline.stage1_requirement import RequirementStage
        from articulate_core.pipeline.stage2_approach import TechnicalApproachStage
        from articulate_core.pipeline.stage3_generation import CodeGenerationStage
        from articulate_core.pipeline.stage4_simulation import SimulationStage
        from articulate_core.pipeline.stage5_deployment import DeploymentStage

        return [
            RequirementStage(self.config, self.llm, self.skill),
            TechnicalApproachStage(self.config, self.llm, self.skill),
            CodeGenerationStage(self.config, self.llm, self.skill),
            SimulationStage(self.config, self.llm, self.skill),
            DeploymentStage(self.config, self.llm, self.skill),
        ]

    async def run(self, ctx: StageContext) -> StageContext:
        """Run all stages sequentially from current_stage."""
        for i in range(ctx.current_stage, len(self.stages)):
            if not ctx.should_continue:
                break
            stage = self.stages[i]
            ctx.current_stage = i
            logger.info("Running stage %d: %s", stage.stage_id, stage.stage_name)
            ctx = await stage.execute(ctx)
            self._persist(ctx)
        return ctx

    async def run_single(self, ctx: StageContext, stage_idx: int) -> StageContext:
        """Run a single stage by index."""
        if 0 <= stage_idx < len(self.stages):
            stage = self.stages[stage_idx]
            ctx.current_stage = stage_idx
            logger.info("Running stage %d: %s", stage.stage_id, stage.stage_name)
            ctx = await stage.execute(ctx)
            self._persist(ctx)
        return ctx

    def _persist(self, ctx: StageContext):
        """Save state to .articulate/state.json."""
        state = PipelineState.from_context(ctx)
        state_dir = self.config.project_dir / self.config.state_dir
        state.save(state_dir)

    @classmethod
    def load_state(cls, project_dir: Path) -> Optional[StageContext]:
        """Load saved state from .articulate/ directory."""
        state_dir = project_dir / ".articulate"
        state = PipelineState.load(state_dir)
        if state is None:
            return None
        return state.to_context()
