from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.skill import ArticulateSkill

console = create_console()


async def run_simulate(config: ArticulateConfig, skill: ArticulateSkill):
    """Run Stage 4: simulation verification."""
    orchestrator = PipelineOrchestrator(config, skill.llm, skill) if skill.llm else PipelineOrchestrator(config, None, skill)

    ctx = PipelineOrchestrator.load_state(config.project_dir)
    if ctx is None:
        console.print("[red]Error:[/red] No project state found.")
        return

    ctx = await orchestrator.run_single(ctx, 3)  # stage 4
    if not ctx.should_continue:
        return
    console.print("[green]OK[/green] Simulation complete.")
