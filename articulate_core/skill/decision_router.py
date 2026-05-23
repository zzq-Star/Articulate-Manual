import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from articulate_core.llm.client import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    pattern: str
    route: str  # "library" | "prompt" | "library_with_prompt"
    confidence: float
    module: Optional[str] = None
    note: str = ""


@dataclass
class RoutingResult:
    sub_task: str
    route: str                     # "library" | "prompt" | "library_with_prompt"
    confidence: float              # 0.0 to 1.0
    module: Optional[str] = None
    explanation: str = ""
    requires_arbitration: bool = False


class DecisionRouter:
    """Hybrid routing engine: rule-based matching first, LLM fallback."""

    def __init__(
        self,
        rules_path: Path,
        confidence_threshold: float = 0.7,
        llm: Optional[BaseLLMClient] = None,
    ):
        self.rules_path = Path(rules_path)
        self.confidence_threshold = confidence_threshold
        self.llm = llm
        self._rules: List[Rule] = []
        self.reload_rules()

    def reload_rules(self):
        """Load rules from YAML file."""
        if self.rules_path.exists():
            with open(str(self.rules_path), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._rules = [Rule(**r) for r in data.get("rules", [])]
            logger.info("Loaded %d routing rules from %s", len(self._rules), self.rules_path)
        else:
            logger.warning("Rules file not found: %s", self.rules_path)
            self._rules = []

    async def route(
        self, sub_task: str, context: Optional[Dict] = None,
    ) -> RoutingResult:
        """Route a sub-task to library or prompt generation.

        1. Match against rules table (longest matching pattern wins).
        2. If no match or partial match and LLM available, call LLM classifier.
        3. Set requires_arbitration if confidence < threshold.
        """
        # Try rule-based matching first
        best_rule = self._match_rule(sub_task)

        if best_rule:
            confidence = best_rule.confidence
            result = RoutingResult(
                sub_task=sub_task,
                route=best_rule.route,
                confidence=confidence,
                module=best_rule.module,
                explanation=best_rule.note,
                requires_arbitration=confidence < self.confidence_threshold,
            )
            logger.debug(
                "Rule match: '%s' -> %s (confidence=%.2f)",
                sub_task, best_rule.route, confidence,
            )
        elif self.llm:
            result = await self._llm_classify(sub_task, context)
        else:
            result = RoutingResult(
                sub_task=sub_task,
                route="prompt",
                confidence=0.5,
                explanation="No rule match and no LLM available, defaulting to prompt",
                requires_arbitration=True,
            )

        return result

    async def batch_route(
        self, sub_tasks: List[str], context: Optional[Dict] = None,
    ) -> List[RoutingResult]:
        """Route multiple sub-tasks, potentially batching LLM calls."""
        results = []
        llm_tasks = []

        for task in sub_tasks:
            best = self._match_rule(task)
            if best and best.confidence >= self.confidence_threshold:
                results.append(RoutingResult(
                    sub_task=task,
                    route=best.route,
                    confidence=best.confidence,
                    module=best.module,
                    explanation=best.note,
                ))
            else:
                llm_tasks.append(task)

        # Batch classify remaining via LLM
        if llm_tasks and self.llm:
            for task in llm_tasks:
                result = await self._llm_classify(task, context)
                results.append(result)
        elif llm_tasks:
            for task in llm_tasks:
                results.append(RoutingResult(
                    sub_task=task,
                    route="prompt",
                    confidence=0.5,
                    explanation="No LLM available",
                    requires_arbitration=True,
                ))

        return results

    def _match_rule(self, sub_task: str) -> Optional[Rule]:
        """Find the best matching rule for a sub-task.

        Uses substring matching (lowercased). Longer matches take priority.
        """
        sub_lower = sub_task.lower()
        best = None
        best_len = 0

        for rule in self._rules:
            pat_lower = rule.pattern.lower()
            if pat_lower in sub_lower:
                if len(pat_lower) > best_len:
                    best = rule
                    best_len = len(pat_lower)

        return best

    async def _llm_classify(
        self, sub_task: str, context: Optional[Dict] = None,
    ) -> RoutingResult:
        """Use LLM to classify an unmatched sub-task."""
        if not self.llm:
            return RoutingResult(
                sub_task=sub_task,
                route="prompt",
                confidence=0.5,
                explanation="No LLM available",
                requires_arbitration=True,
            )

        system_prompt = (
            "You are a routing classifier for a robotic arm code generation system. "
            "Given a sub-task description, choose the best route:\n"
            "- 'library': Use standard library calls (high confidence, predictable output)\n"
            "- 'prompt': Use LLM prompt generation (for novel or complex tasks)\n"
            "- 'library_with_prompt': Mix of both\n\n"
            "Respond with valid JSON only: {\"route\": \"...\", \"confidence\": 0.0-1.0, "
            "\"explanation\": \"...\"}\n"
            "Be conservative: when unsure, prefer 'prompt' with lower confidence."
        )
        user_msg = f"Sub-task: {sub_task}"
        if context:
            user_msg += f"\nContext: {context}"

        from pydantic import BaseModel

        class RouteResponse(BaseModel):
            route: str
            confidence: float
            explanation: str

        try:
            response = await self.llm.complete_structured(
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                output_model=RouteResponse,
                max_tokens=256,
            )

            return RoutingResult(
                sub_task=sub_task,
                route=response.route,
                confidence=response.confidence,
                explanation=response.explanation,
                requires_arbitration=response.confidence < self.confidence_threshold,
            )
        except Exception as e:
            logger.warning("LLM routing failed: %s", e)
            return RoutingResult(
                sub_task=sub_task,
                route="prompt",
                confidence=0.5,
                explanation=f"LLM classification failed: {e}",
                requires_arbitration=True,
            )
