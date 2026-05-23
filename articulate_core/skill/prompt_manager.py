import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    name: str
    version: str
    description: str = ""
    system_prompt: str = ""
    user_template: str = ""
    output_format: str = "json"
    max_tokens: int = 4096


class PromptManager:
    """Load, cache, and render prompt templates from YAML files."""

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, PromptTemplate] = {}

    def _load_file(self, name: str) -> Optional[PromptTemplate]:
        """Load a single YAML prompt file."""
        # Search in all prompt subdirectories
        for subdir in ["system", "analysis", "context"]:
            path = self.prompts_dir / subdir / f"{name}.yaml"
            if path.exists():
                with open(str(path), "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                return PromptTemplate(
                    name=data.get("name", name),
                    version=data.get("version", "1.0"),
                    description=data.get("description", ""),
                    system_prompt=data.get("system_prompt", ""),
                    user_template=data.get("user_template", ""),
                    output_format=data.get("output_format", "json"),
                    max_tokens=data.get("max_tokens", 4096),
                )

        # Check for context .md files
        context_path = self.prompts_dir / "context" / f"{name}.md"
        if context_path.exists():
            content = context_path.read_text(encoding="utf-8")
            return PromptTemplate(
                name=name,
                version="1.0",
                system_prompt=content,
                output_format="text",
            )

        return None

    def get(self, name: str) -> PromptTemplate:
        """Get a prompt template, loading from cache or file."""
        if name not in self._cache:
            template = self._load_file(name)
            if template is None:
                raise FileNotFoundError(
                    f"Prompt template '{name}' not found in {self.prompts_dir}"
                )
            self._cache[name] = template
        return self._cache[name]

    def render(self, name: str, **context) -> Tuple[str, str]:
        """Render a prompt template with Jinja2 variable substitution.

        Returns:
            Tuple[str, str]: (system_prompt, user_message)
        """
        template = self.get(name)

        system_rendered = Template(template.system_prompt).render(**context)

        user_message = ""
        if template.user_template:
            user_message = Template(template.user_template).render(**context)

        return system_rendered, user_message

    def list_available(self) -> List[str]:
        """List all available prompt template names."""
        names = []
        for subdir in ["system", "analysis", "context"]:
            path = self.prompts_dir / subdir
            if path.exists():
                for f in sorted(path.iterdir()):
                    if f.suffix in (".yaml", ".yml", ".md"):
                        names.append(f.stem)
        return names
