import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler

from articulate_core.cli.console import create_console

from articulate_core.config.settings import ArticulateConfig
from articulate_core.llm.client import BaseLLMClient, create_llm_client
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.pipeline.state import PipelineState
from articulate_core.skill import ArticulateSkill

# ── Setup ────────────────────────────────────────────────────────────

console = create_console()
app = typer.Typer(
    name="articulate",
    help="Robotic arm code generation Agent.",
    no_args_is_help=True,
)

_log_configured = False


def _setup_logging(verbose: bool):
    global _log_configured
    if _log_configured:
        return
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    _log_configured = True


def _build_skill(config: ArticulateConfig) -> ArticulateSkill:
    """Build the skill (no LLM if no API key configured)."""
    if config.api_key:
        llm = create_llm_client(config)
        return ArticulateSkill(config, llm=llm)
    return ArticulateSkill(config)


# ── Callback ─────────────────────────────────────────────────────────

@app.callback()
def callback(
    ctx: typer.Context,
    project_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Project directory (default: current or discovered)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable debug logging",
    ),
):
    _setup_logging(verbose)
    if project_dir:
        config = ArticulateConfig.from_project_dir(project_dir.resolve())
    elif ArticulateConfig.discover_project():
        config = ArticulateConfig(
            project_dir=ArticulateConfig.discover_project()
        )
    else:
        config = ArticulateConfig()
    ctx.obj = {
        "config": config,
        "skill": _build_skill(config),
    }


# ── Commands ─────────────────────────────────────────────────────────


@app.command()
def generate(
    requirement: str = typer.Argument(..., help="Natural language requirement"),
    brand: str = typer.Option("ur", "--brand", "-b", help="Target arm brand (ur/kuka/abb)"),
    model: str = typer.Option(None, "--model", "-m", help="LLM model (e.g. deepseek-chat, claude-sonnet-4-20250514)"),
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider (anthropic/deepseek/openai); inferred from --model if omitted"),
    ctx: typer.Context = typer.Context,
):
    """One-shot: run all 5 stages from requirement to deployment."""
    from articulate_core.cli.commands.generate import run_generate
    config: ArticulateConfig = ctx.obj["config"]
    if model:
        config.llm_model = model
    if provider:
        config.llm_provider = provider
    asyncio.run(run_generate(requirement, brand, config))


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to create"),
    ctx: typer.Context = typer.Context,
):
    """Initialize a new Articulate project directory."""
    from articulate_core.cli.commands.init import run_init
    config: ArticulateConfig = ctx.obj["config"]
    run_init(project_name, config)


@app.command()
def plan(
    requirement: str = typer.Argument(..., help="Natural language requirement"),
    ctx: typer.Context = typer.Context,
):
    """Stage 1-2: Analyze requirement and design technical approach."""
    from articulate_core.cli.commands.plan import run_plan
    config: ArticulateConfig = ctx.obj["config"]
    skill: ArticulateSkill = ctx.obj["skill"]
    asyncio.run(run_plan(requirement, config, skill))


@app.command()
def codegen(
    ctx: typer.Context = typer.Context,
):
    """Stage 3: Generate ROS2 code from approved approach."""
    from articulate_core.cli.commands.codegen import run_codegen
    config: ArticulateConfig = ctx.obj["config"]
    skill: ArticulateSkill = ctx.obj["skill"]
    asyncio.run(run_codegen(config, skill))


@app.command()
def simulate(
    ctx: typer.Context = typer.Context,
):
    """Stage 4: Run MuJoCo simulation and validate."""
    from articulate_core.cli.commands.simulate import run_simulate
    config: ArticulateConfig = ctx.obj["config"]
    skill: ArticulateSkill = ctx.obj["skill"]
    asyncio.run(run_simulate(config, skill))


@app.command()
def deploy(
    brand: str = typer.Option("ur", "--brand", "-b", help="Target arm brand"),
    ctx: typer.Context = typer.Context,
):
    """Stage 5: Generate deployment package."""
    from articulate_core.cli.commands.deploy import run_deploy
    config: ArticulateConfig = ctx.obj["config"]
    skill: ArticulateSkill = ctx.obj["skill"]
    asyncio.run(run_deploy(brand, config, skill))


@app.command()
def status(
    ctx: typer.Context = typer.Context,
):
    """Show current project and pipeline status."""
    config: ArticulateConfig = ctx.obj["config"]
    state_dir = config.project_dir / config.state_dir
    state_path = state_dir / "state.json"

    if state_path.exists():
        console.print(f"[green]OK[/green] Project: {config.project_dir}")
        console.print(f"[green]OK[/green] State file: {state_path}")
        import json
        data = json.loads(state_path.read_text())
        stage_names = ["unstarted", "requirement", "approach", "codegen", "simulation", "deploy"]
        stage = data.get("current_stage", 0)
        console.print(f"[cyan]Stage:[/cyan] {stage_names[stage]} ({stage}/5)")
    else:
        console.print("[yellow]No project state found.[/yellow]")
        console.print("Run [bold]articulate init <name>[/bold] first.")


@app.command()
def report(
    fmt: str = typer.Option("md", "--format", "-f", help="Output format: md or html"),
    ctx: typer.Context = typer.Context,
):
    """Generate full pipeline report (HTML or Markdown)."""
    config: ArticulateConfig = ctx.obj["config"]
    state_dir = config.project_dir / config.state_dir
    state = PipelineState.load(state_dir)
    if state is None:
        console.print("[yellow]No state to report.[/yellow]")
        raise typer.Exit()

    ctx_obj = state.to_context()
    output_dir = config.project_dir / config.deploy_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "html":
        from articulate_core.reporting.html_reporter import HTMLReportGenerator
        path = output_dir / "pipeline_report.html"
        HTMLReportGenerator().generate(ctx_obj, output_path=path)
        console.print(f"[green]OK[/green] HTML report: {path}")
    else:
        from articulate_core.reporting.markdown_reporter import MarkdownReportGenerator
        path = output_dir / "pipeline_report.md"
        MarkdownReportGenerator().generate(ctx_obj, output_path=path)
        console.print(f"[green]OK[/green] Markdown report: {path}")


# ── Entry point ──────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
