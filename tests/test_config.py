import os
from pathlib import Path

import pytest

from articulate_core.config.settings import ArticulateConfig


def test_default_config():
    """Config can be created with defaults."""
    config = ArticulateConfig()
    assert config.llm_max_retries == 3
    assert config.confidence_threshold == 0.7
    assert config.sim_max_retries == 3


def test_env_override(monkeypatch):
    """Config respects ARTICULATE_ env prefix."""
    monkeypatch.setenv("ARTICULATE_ANTHROPIC_MODEL", "claude-opus-4-20250514")
    monkeypatch.setenv("ARTICULATE_LLM_MAX_RETRIES", "5")
    config = ArticulateConfig()
    assert config.anthropic_model == "claude-opus-4-20250514"
    assert config.llm_max_retries == 5


def test_from_project_dir():
    """from_project_dir sets project_dir correctly."""
    config = ArticulateConfig.from_project_dir(Path("/tmp/test_project"))
    assert config.project_dir == Path("/tmp/test_project")


def test_discover_project_no_marker(tmp_path):
    """discover_project returns None when no .articulate dir exists."""
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        result = ArticulateConfig.discover_project()
        assert result is None
    finally:
        os.chdir(cwd)
