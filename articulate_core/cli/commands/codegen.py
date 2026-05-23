from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.skill import ArticulateSkill

console = create_console()


async def run_codegen(config: ArticulateConfig, skill: ArticulateSkill):
    """Run Stage 3: code generation."""
    orchestrator = PipelineOrchestrator(config, skill.llm, skill) if skill.llm else PipelineOrchestrator(config, None, skill)

    ctx = PipelineOrchestrator.load_state(config.project_dir)
    if ctx is None:
        console.print("[red]Error:[/red] No project state found. Run [bold]articulate plan[/bold] first.")
        return

    ctx = await orchestrator.run_single(ctx, 2)  # stage 3
    if not ctx.should_continue:
        return
    console.print("[green]OK[/green] Code generation complete.")
