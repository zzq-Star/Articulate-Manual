from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.models import StageContext
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.skill import ArticulateSkill

console = create_console()


async def run_plan(requirement: str, config: ArticulateConfig, skill: ArticulateSkill):
    """Run Stage 1-2: requirement analysis and technical approach."""
    orchestrator = PipelineOrchestrator(config, skill.llm, skill) if skill.llm else PipelineOrchestrator(config, None, skill)

    ctx = PipelineOrchestrator.load_state(config.project_dir)
    if ctx is None:
        ctx = StageContext(project_dir=config.project_dir)

    ctx.user_input = requirement
    ctx.current_stage = 0

    # Run stages 1 and 2
    ctx = await orchestrator.run_single(ctx, 0)  # stage 1
    if not ctx.should_continue:
        return
    ctx = await orchestrator.run_single(ctx, 1)  # stage 2
    if not ctx.should_continue:
        return

    console.print("[green]OK[/green] Plan phase complete.")
    console.print(f"  Current stage: {ctx.current_stage + 1}/5")
