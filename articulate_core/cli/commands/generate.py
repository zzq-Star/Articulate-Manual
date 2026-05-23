"""One-shot generate command: runs all 5 pipeline stages."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig, save_project_config
from articulate_core.llm.client import (
    BaseLLMClient,
    create_llm_client,
    infer_provider_from_model,
)
from articulate_core.pipeline.models import StageContext
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.skill import ArticulateSkill

console = create_console()


def _ensure_api_key(config: ArticulateConfig) -> str:
    """Prompt user for API key if not already configured."""
    key = config.api_key or os.environ.get("ARTICULATE_API_KEY", "")
    if key:
        return key

    console.print()
    console.print("[bold yellow]API Key Required[/bold yellow]")
    console.print("Enter your API key to proceed.")
    console.print()

    try:
        key = input("Enter your API key: ").strip()
        if key:
            return key
    except (EOFError, OSError):
        pass

    console.print()
    console.print("[red]Error:[/red] No API key provided.")
    console.print("Set ARTICULATE_API_KEY and retry:")
    console.print()
    console.print("  # Linux / macOS")
    console.print("  export ARTICULATE_API_KEY=sk-...")
    console.print()
    console.print("  # Windows (CMD)")
    console.print("  set ARTICULATE_API_KEY=sk-...")
    console.print()
    console.print("  # Windows (PowerShell)")
    console.print("  $env:ARTICULATE_API_KEY=\"sk-...\"")
    sys.exit(1)


def _ensure_project_dir(config: ArticulateConfig, requirement: str) -> Path:
    """Create a project directory from the requirement if none exists."""
    existing = ArticulateConfig.discover_project()
    if existing:
        return existing

    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in requirement.lower())
    words = [w for w in slug.split("_") if w][:4]
    name = "_".join(words) if words else "articulate_project"

    project_dir = Path.cwd() / name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / ".articulate").mkdir(parents=True, exist_ok=True)
    (project_dir / "ros_ws" / "src").mkdir(parents=True, exist_ok=True)
    (project_dir / "deploy").mkdir(parents=True, exist_ok=True)
    (project_dir / "assets").mkdir(parents=True, exist_ok=True)

    import json
    state = {
        "project_dir": str(project_dir.resolve()),
        "current_stage": 0,
        "state_data": {},
    }
    (project_dir / ".articulate" / "state.json").write_text(json.dumps(state, indent=2))

    console.print(f"[green]OK[/green] Created project: {project_dir}")
    return project_dir


async def run_generate(requirement: str, brand: str, config: ArticulateConfig):
    """Run all 5 pipeline stages in sequence."""

    # ── 1. API key ──────────────────────────────────────────────────────
    api_key = _ensure_api_key(config)
    config.api_key = api_key
    os.environ["ARTICULATE_API_KEY"] = api_key

    # ── 2. Project dir ──────────────────────────────────────────────────
    project_dir = _ensure_project_dir(config, requirement)
    config.project_dir = project_dir

    # ── 3. Resolve provider from model if not explicitly set ────────────
    if not config.llm_provider or config.llm_provider == "anthropic":
        inferred = infer_provider_from_model(config.llm_model)
        if inferred:
            config.llm_provider = inferred

    # Persist provider/model to project config for future runs
    save_project_config(project_dir, {
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
    })

    # ── 4. Init LLM + Skill ────────────────────────────────────────────
    llm = create_llm_client(config)
    skill = ArticulateSkill(config, llm=llm)
    orchestrator = PipelineOrchestrator(config, llm, skill)

    # ── 5. Build initial context ─────────────────────────────────────────
    ctx = PipelineOrchestrator.load_state(project_dir)
    if ctx is None:
        ctx = StageContext(project_dir=project_dir)
    ctx.user_input = requirement
    ctx.target_brand = brand
    ctx.current_stage = 0

    console.print()
    console.print("=" * 60)
    console.print("  ARTICULATE — Full Pipeline")
    console.print("=" * 60)
    console.print(f"  Requirement: {requirement}")
    console.print(f"  Target brand: {brand}")
    console.print(f"  Model: {config.llm_model}")
    console.print(f"  Project: {project_dir}")
    console.print("=" * 60)

    # ── 6. Run all 5 stages (auto-confirm) ─────────────────────────────
    stage_labels = [
        "1/5  Requirement Analysis",
        "2/5  Technical Approach",
        "3/5  Code Generation",
        "4/5  Simulation Verification",
        "5/5  Deployment Package",
    ]

    for i in range(5):
        console.print()
        console.print(f"[bold cyan]>>> Stage {stage_labels[i]}[/bold cyan]")

        with patch("builtins.input", return_value="y"):
            ctx.current_stage = i
            ctx = await orchestrator.run_single(ctx, i)

        if not ctx.should_continue:
            console.print(f"\n[bold red]Pipeline stopped at stage {i + 1}[/bold red]")
            break

        console.print(f"[green]OK[/green] Stage {i + 1} complete")

    # ── 7. Summary ──────────────────────────────────────────────────────
    console.print()
    console.print("=" * 60)
    console.print("  PIPELINE RESULTS")
    console.print("=" * 60)

    checks = [
        ("Requirement", ctx.requirement_doc, True),
        ("Technical Approach", ctx.technical_approach, True),
        ("Generated Code", ctx.generated_code, True),
        ("Simulation", ctx.simulation_report,
         ctx.simulation_report and ctx.simulation_report.passed),
        ("Deployment", ctx.deployment_package, True),
    ]

    for label, obj, required in checks:
        if obj is None:
            console.print(f"  [red]FAIL[/red]  {label}")
        elif not required:
            console.print(f"  [yellow]WARN[/yellow] {label}")
        else:
            console.print(f"  [green]OK[/green]   {label}")

    if ctx.generated_code:
        ros_ws = project_dir / "ros_ws"
        console.print(f"\n  [underline]Generated files:[/underline]")
        for path in sorted(ctx.generated_code.package_structure.keys()):
            console.print(f"    {path}")
        console.print(f"\n  [dim]Location: {project_dir / 'ros_ws'}[/dim]")

    if ctx.deployment_package:
        console.print(f"\n  [underline]Deployment:[/underline]")
        for name in sorted(ctx.deployment_package.files.keys()):
            console.print(f"    {name}  [dim]({ctx.deployment_package.target_brand})[/dim]")

    await llm.close()
