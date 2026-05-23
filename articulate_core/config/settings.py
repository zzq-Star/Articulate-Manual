import json
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_CONFIG_FILE = ".articulate/config.json"


def _find_project_dir() -> Optional[Path]:
    """Walk parent dirs looking for .articulate/ marker directory."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        marker = parent / ".articulate"
        if marker.is_dir():
            return parent
    return None


def _load_project_config(project_dir: Optional[Path]) -> dict:
    """Load .articulate/config.json from project dir if it exists."""
    if project_dir is None:
        return {}
    cfg_path = project_dir / PROJECT_CONFIG_FILE
    if not cfg_path.exists():
        return {}
    try:
        with open(str(cfg_path), encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_project_config(project_dir: Path, overrides: dict):
    """Merge overrides into .articulate/config.json and save."""
    cfg_path = project_dir / PROJECT_CONFIG_FILE
    existing = _load_project_config(project_dir)
    existing.update(overrides)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(cfg_path), "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


class ArticulateConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARTICULATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    llm_provider: str = "anthropic"  # "anthropic" | "deepseek" | "openai"
    api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 2.0

    # --- Deprecated (backward compat) ---
    anthropic_api_key: str = ""
    anthropic_model: str = ""

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data: dict) -> dict:
        """Apply project config + legacy migration as lowest-priority defaults.

        Priority order: explicit constructor arg / env var > project config > code default.
        """
        # ── 1. Detect project dir from the data ──────────────────────
        project_dir = data.get("project_dir")
        if project_dir is None or project_dir == Path.cwd():
            discovered = _find_project_dir()
            if discovered is not None:
                project_dir = discovered

        # ── 2. Load project config as fallback defaults ──────────────
        if project_dir is not None:
            proj_cfg = _load_project_config(project_dir)
            for key in ("llm_provider", "llm_model", "api_key"):
                if key in proj_cfg and key not in data:
                    data[key] = proj_cfg[key]

        # ── 3. Migrate legacy anthropic_* fields ─────────────────────
        if not data.get("api_key"):
            legacy_key = data.get("anthropic_api_key") or ""
            if legacy_key:
                data["api_key"] = legacy_key
        if not data.get("llm_model"):
            legacy_model = data.get("anthropic_model") or ""
            if legacy_model:
                data["llm_model"] = legacy_model

        return data

    # --- Paths (relative to project dir) ---
    project_dir: Path = Field(default_factory=Path.cwd)
    state_dir: Path = Field(default=Path(".articulate"))
    ros_ws_dir: Path = Field(default=Path("ros_ws"))
    deploy_dir: Path = Field(default=Path("deploy"))

    # --- Prompt directories (relative to articulate_core) ---
    prompts_dir: Path = Field(default=Path("skill/prompts"))
    templates_dir: Path = Field(default=Path("skill/templates"))

    # --- Routing ---
    rules_path: Path = Field(default=Path("skill/router_rules.yaml"))
    confidence_threshold: float = 0.7

    # --- Simulation ---
    sim_max_retries: int = 3
    sim_timeout: float = 120.0

    @classmethod
    def discover_project(cls) -> Optional[Path]:
        return _find_project_dir()

    @classmethod
    def from_project_dir(cls, project_dir: Path) -> "ArticulateConfig":
        return cls(project_dir=project_dir)
