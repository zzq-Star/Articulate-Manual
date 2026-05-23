from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.skill import ArticulateSkill

console = create_console()


async def run_deploy(brand: str, config: ArticulateConfig, skill: ArticulateSkill):
    """Run Stage 5: deployment package generation."""
    orchestrator = PipelineOrchestrator(config, skill.llm, skill) if skill.llm else PipelineOrchestrator(config, None, skill)

    ctx = PipelineOrchestrator.load_state(config.project_dir)
    if ctx is None:
        console.print("[red]Error:[/red] No project state found.")
        return

    ctx.target_brand = brand
    ctx = await orchestrator.run_single(ctx, 4)  # stage 5 (0-indexed)
    if not ctx.should_continue:
        return
    console.print(f"[green]OK[/green] Deployment package generated for [bold]{brand}[/bold].")
