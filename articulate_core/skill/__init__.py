"""Skill facade - domain knowledge entry point."""

import logging
from typing import Optional

from articulate_core.config.settings import ArticulateConfig
from articulate_core.llm.client import BaseLLMClient
from articulate_core.skill.converters.factory import ConverterFactory
from articulate_core.skill.decision_router import DecisionRouter
from articulate_core.skill.library.dynamics import DynamicsLibrary
from articulate_core.skill.library.kinematics import KinematicsLibrary
from articulate_core.skill.library.planning import PlanningLibrary
from articulate_core.skill.library.ros2_gen import ROS2Generator
from articulate_core.skill.models.urdf_loader import URDFLoader
from articulate_core.skill.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class ArticulateSkill:
    """Facade over all domain modules.

    Pipeline stages access skill sub-modules through this object
    held in StageContext.
    """

    def __init__(
        self,
        config: ArticulateConfig,
        llm: Optional[BaseLLMClient] = None,
    ):
        self.config = config
        self.llm = llm

        # Locate prompts and templates relative to this file's parent
        skill_dir = __import__("pathlib").Path(__file__).parent
        prompts_dir = skill_dir / "prompts"
        templates_dir = skill_dir / "templates"
        rules_path = skill_dir / "router_rules.yaml"

        # Initialize sub-modules
        self.prompt_mgr = PromptManager(prompts_dir)
        self.router = DecisionRouter(
            rules_path=rules_path,
            confidence_threshold=config.confidence_threshold,
            llm=llm,
        )
        self.kinematics = KinematicsLibrary()
        self.planning = PlanningLibrary()
        self.ros2_gen = ROS2Generator(templates_dir)
        self.dynamics = DynamicsLibrary()
        self.urdf_loader = URDFLoader()
        self.converters = ConverterFactory

        logger.info(
            "ArticulateSkill initialized (llm=%s, prompts=%d, rules=%d)",
            "available" if llm else "N/A",
            len(self.prompt_mgr.list_available()),
            len(self.router._rules),
        )
