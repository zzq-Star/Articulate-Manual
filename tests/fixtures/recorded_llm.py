"""RecordedLLM — plays back pre-recorded LLM responses for deterministic tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class RecordedLLM:
    """Mock LLM that returns pre-recorded responses from fixture files.

    Usage in tests:
        llm = RecordedLLM(Path("tests/fixtures"))
        # Now use llm wherever ClaudeClient is expected
        result = await llm.complete_structured(system=..., messages=..., output_model=MySchema)
    """

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = Path(fixtures_dir)
        self._fixtures: Dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        """Load all JSON fixture files from the fixtures directory."""
        for path in self.fixtures_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                schema = data.get("schema")
                if schema:
                    self._fixtures[schema] = data
            except (json.JSONDecodeError, KeyError) as e:
                raise RuntimeError(f"Invalid fixture {path.name}: {e}")

    async def complete_structured(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        output_model: Type[T],
        **kwargs,
    ) -> T:
        """Return pre-recorded response matching the output_model schema."""
        model_name = output_model.__name__
        fixture = self._fixtures.get(model_name)

        if fixture is None:
            available = ", ".join(sorted(self._fixtures.keys()))
            raise KeyError(
                f"No fixture for schema '{model_name}'. "
                f"Available: {available}"
            )

        raw = fixture["response"]
        return output_model.model_validate(raw)

    async def complete(self, system, messages, **kwargs):
        """Unstructured completion isn't used by stages (only complete_structured)."""
        raise NotImplementedError("RecordedLLM only supports complete_structured")

    @property
    def model(self) -> str:
        return "recorded"

    async def close(self):
        pass
