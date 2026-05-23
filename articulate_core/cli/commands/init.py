import json
from pathlib import Path

from articulate_core.cli.console import create_console
from articulate_core.config.settings import ArticulateConfig

console = create_console()


def run_init(project_name: str, config: ArticulateConfig):
    """Create a new Articulate project directory."""
    project_dir = Path.cwd() / project_name
    state_dir = project_dir / ".articulate"

    if project_dir.exists():
        console.print(f"[yellow]Warning: directory '{project_name}' already exists.[/yellow]")
        return

    # Create structure
    project_dir.mkdir(parents=True)
    state_dir.mkdir()

    # Write initial state
    state = {
        "project_dir": str(project_dir.resolve()),
        "current_stage": 0,
        "state_data": {},
    }
    (state_dir / "state.json").write_text(json.dumps(state, indent=2))

    # Create empty subdirectories
    (project_dir / "ros_ws" / "src").mkdir(parents=True)
    (project_dir / "deploy").mkdir()
    (project_dir / "assets").mkdir()

    console.print(f"[green]OK[/green] Created project '{project_name}'")
    console.print(f"  [dim]Location:[/dim] {project_dir.resolve()}")
    console.print(f"  [dim].articulate/[/dim]    Pipeline state")
    console.print(f"  [dim]ros_ws/src/[/dim]    ROS2 workspace")
    console.print(f"  [dim]deploy/[/dim]        Deployment outputs")
    console.print(f"  [dim]assets/[/dim]        Models and resources")
    console.print("\nNext: [bold]articulate plan \"your requirement\"[/bold]")
