"""End-to-end test with real LLM (requires API key, manual execution).

Run with:
    pytest tests/e2e/test_real_llm.py -x --e2e

Or:
    ARTICULATE_LLM_API_KEY=sk-... pytest tests/e2e/test_real_llm.py -x
"""

import os
from pathlib import Path

import pytest

from articulate_core.config.settings import ArticulateConfig
from articulate_core.llm.client import ClaudeClient
from articulate_core.pipeline.models import StageContext
from articulate_core.pipeline.orchestrator import PipelineOrchestrator
from articulate_core.pipeline.state import PipelineState
from articulate_core.skill import ArticulateSkill

pytestmark = pytest.mark.skipif(
    not os.getenv("ARTICULATE_LLM_API_KEY"),
    reason="Requires ARTICULATE_LLM_API_KEY env var",
)


@pytest.fixture
def e2e_config(tmp_path):
    return ArticulateConfig(
        project_dir=tmp_path / "e2e_test_project",
        anthropic_api_key=os.environ["ARTICULATE_LLM_API_KEY"],
        confidence_threshold=0.6,
    )


@pytest.fixture
def e2e_llm(e2e_config):
    return ClaudeClient(api_key=e2e_config.anthropic_api_key)


@pytest.fixture
def e2e_skill(e2e_config, e2e_llm):
    return ArticulateSkill(e2e_config, llm=e2e_llm)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_full_pipeline_stage1(e2e_config, e2e_llm, e2e_skill):
    """Run Stage 1 only with real LLM. Requires user input for confirmation."""
    orchestrator = PipelineOrchestrator(e2e_config, e2e_llm, e2e_skill)

    ctx = StageContext(
        project_dir=e2e_config.project_dir,
        user_input="pick and place from (0.1, 0.2, 0.3) to (0.4, 0.5, 0.6)",
    )
    ctx = await orchestrator.run_single(ctx, 0)

    assert ctx.requirement_doc is not None
    assert ctx.requirement_doc.task_type is not None
    assert len(ctx.requirement_doc.key_waypoints) >= 1


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_full_pipeline_all_stages(e2e_config, e2e_llm, e2e_skill):
    """Run full pipeline end-to-end (may require multiple user confirmations).

    This test is interactive and requires manual input for confirmation prompts.
    """
    orchestrator = PipelineOrchestrator(e2e_config, e2e_llm, e2e_skill)

    ctx = StageContext(
        project_dir=e2e_config.project_dir,
        user_input="pick and place from (0.1, 0.2, 0.3) to (0.4, 0.5, 0.6)",
    )
    ctx = await orchestrator.run(ctx)

    assert ctx.requirement_doc is not None
    assert ctx.technical_approach is not None
    assert ctx.generated_code is not None
    assert ctx.simulation_report is not None
    assert ctx.deployment_package is not None
    assert ctx.deployment_package.target_brand == "ur"
