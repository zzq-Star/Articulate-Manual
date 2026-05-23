import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from articulate_core.config.settings import ArticulateConfig
from articulate_core.exceptions import GenError
from articulate_core.llm.client import BaseLLMClient, LLMResponse
from articulate_core.pipeline.models import (
    GeneratedCode,
    SubTask,
    SubTaskResult,
    TechnicalApproach,
)
from articulate_core.skill import ArticulateSkill

logger = logging.getLogger(__name__)


@dataclass
class UserCallbacks:
    """User interaction callbacks for the pipeline.

    These can be overridden for CLI vs TUI vs headless operation.
    """
    confirm: Callable[[str], bool] = lambda msg: True
    prompt_missing_info: Callable[[List[str]], Dict[str, str]] = lambda x: {}
    arbitrate_route: Callable[[str, SubTaskResult, SubTaskResult], str] = None
    render_code_summary: Callable[[GeneratedCode], None] = None


class CodeGenerationEngine:
    """Orchestrates sub-task decomposition, routing, generation, and assembly.

    This is the core intelligence of Stage 3.
    """

    def __init__(
        self,
        skill: ArticulateSkill,
        llm: BaseLLMClient,
        callbacks: UserCallbacks,
        config: ArticulateConfig,
    ):
        self.skill = skill
        self.llm = llm
        self.callbacks = callbacks
        self.config = config

    async def decompose(self, approach: TechnicalApproach) -> List[SubTask]:
        """Break technical approach into individual code generation sub-tasks.

        Uses LLM to intelligently decompose the approach, then maps
        known sub-tasks to canonical names for routing.
        """
        from pydantic import BaseModel

        class DecompositionSchema(BaseModel):
            sub_tasks: list

        system_prompt = (
            "You are a robotics code architect. Decompose a technical approach "
            "into individual code generation sub-tasks. Each sub-task should be "
            "a single, well-defined piece of code that can be generated independently.\n\n"
            "Common sub-tasks include:\n"
            "- forward_kinematics: FK solver code\n"
            "- inverse_kinematics: IK solver code\n"
            "- trajectory_planner: Trajectory planning code\n"
            "- ros2_control_node: Main ROS2 control node\n"
            "- launch_file: ROS2 launch file\n"
            "- package_config: package.xml and setup.py\n"
            "- custom_control_logic: Application-specific control logic\n"
            "- obstacle_avoidance: Collision avoidance logic\n\n"
            "Output JSON: {\"sub_tasks\": [{\"name\": \"...\", \"description\": \"...\", "
            "\"context\": {\"key\": \"value\"}}]}"
        )

        user_msg = (
            f"Technical approach: {approach.description}\n"
            f"Kinematics method: {approach.kinematics_strategy.method}\n"
            f"Trajectory types: {[t.value for t in approach.trajectory_types]}\n"
            f"ROS2 nodes: {[n.get('name') for n in approach.ros2_architecture.nodes]}\n"
        )

        try:
            response = await self.llm.complete_structured(
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                output_model=DecompositionSchema,
                max_tokens=1024,
            )
            tasks = []
            for t in response.sub_tasks:
                tasks.append(SubTask(
                    name=t.get("name", "custom_task"),
                    description=t.get("description", ""),
                    context=t.get("context", {}),
                ))
            return tasks
        except Exception as e:
            logger.warning("LLM decomposition failed, using default: %s", e)
            return self._default_decomposition(approach)

    def _default_decomposition(self, approach: TechnicalApproach) -> List[SubTask]:
        """Default sub-task decomposition when LLM is unavailable.

        Names must match patterns in router_rules.yaml for rule-based routing.
        """
        tasks = [
            SubTask(name="forward_kinematics", description="FK solver using DH parameters"),
            SubTask(name="inverse_kinematics", description="Numerical IK solver"),
            SubTask(name="trajectory_planner", description=f"{', '.join(t.value for t in approach.trajectory_types)} trajectory planning"),
            SubTask(name="ros2_controller_node", description="Main ROS2 control node with trajectory execution"),
            SubTask(name="ros2_launch_file", description="ROS2 launch file"),
            SubTask(name="ros2_package_config", description="package.xml and setup.py"),
        ]
        return tasks

    async def generate_subtask(self, task: SubTask, approach: TechnicalApproach) -> SubTaskResult:
        """Route and generate code for a single sub-task."""
        # 1. Route the sub-task
        route_result = await self.skill.router.route(task.name, task.context)

        # 2. Handle user arbitration if confidence is low
        if route_result.requires_arbitration and self.callbacks.arbitrate_route:
            lib_result = SubTaskResult(
                name=task.name, files={},
                route_used="library", confidence=route_result.confidence,
                success=False,
            )
            prompt_result = SubTaskResult(
                name=task.name, files={},
                route_used="prompt", confidence=route_result.confidence,
                success=False,
            )
            chosen_route = self.callbacks.arbitrate_route(
                task.name, lib_result, prompt_result,
            )
            if chosen_route == "cancel":
                raise GenError(f"User cancelled sub-task: {task.name}")
            route_result.route = chosen_route

        # 3. Generate code via chosen route
        if route_result.route == "library":
            files = await self._generate_via_library(task, approach, route_result.module)
        else:
            files = await self._generate_via_prompt(task, approach)

        # 4. Validate generated code
        valid, errors = self._validate_code(files)
        if not valid and route_result.route == "prompt":
            # Retry prompt generation once
            logger.warning("Code validation failed for '%s', retrying: %s", task.name, errors)
            files = await self._generate_via_prompt(task, approach, retry=True)

        # 5. Fallback: if primary route returned empty, try library template
        if not files:
            logger.warning("Primary route failed for '%s', trying library template", task.name)
            files = await self._generate_via_library(task, approach, route_result.module)

        # 6. Last resort: generate a minimal stub for sub-tasks with known patterns
        if not files:
            stub = self._generate_fallback_stub(task)
            if stub:
                files = stub
                logger.info("Generated fallback stub for '%s' (%d file(s))", task.name, len(files))

        return SubTaskResult(
            name=task.name,
            files=files,
            route_used=route_result.route,
            confidence=route_result.confidence,
            success=len(files) > 0,
        )

    async def _generate_via_library(
        self, task: SubTask, approach: TechnicalApproach, module: Optional[str],
    ) -> Dict[str, str]:
        """Generate code using library calls / template rendering."""
        files = {}

        if module and "ros2_gen" in module:
            # Use ROS2 template generator
            files = self.skill.ros2_gen.generate_package(approach.to_dict())
        elif module and ("kinematics" in module or "fk" in module or "ik" in module):
            # Generate kinematics code via template
            arm_params = approach.arm_parameters
            context = {
                "pkg_name": "arm_controller",
                "arm_name": arm_params.get("name", "arm"),
                "has_kinematics": True,
                "trajectory_types": [t.value for t in approach.trajectory_types],
                "node_class": "ArmController",
                "node_name": "arm_controller_node",
                "publishers": [
                    {"name": "trajectory_pub", "msg_type": "JointTrajectory",
                     "topic": "/arm_controller/command"},
                    {"name": "state_pub", "msg_type": "JointState",
                     "topic": "/arm_controller/state"},
                ],
                "subscribers": [
                    {"name": "command_sub", "msg_type": "JointTrajectory",
                     "topic": "/arm_controller/goal", "callback": "on_goal"},
                ],
            }
            fk_code = self.skill.ros2_gen.render_node(context)
            files[f"ros_ws/src/arm_controller/arm_controller/arm_kinematics.py"] = fk_code
        elif module and "planning" in module:
            files[f"ros_ws/src/arm_controller/arm_controller/trajectory_planner.py"] = (
                "import numpy as np\n\n"
                "class TrajectoryPlanner:\n"
                "    def plan_ptp(self, start, goal, dt=0.01):\n"
                "        return self._trapezoidal_profile(start, goal, dt)\n\n"
                "    def _trapezoidal_profile(self, start, goal, dt):\n"
                '        """Generate trapezoidal velocity profile trajectory."""\n'
                "        start, goal = np.asarray(start), np.asarray(goal)\n"
                "        dq = goal - start\n"
                "        n_points = max(2, int(2.0 / dt))\n"
                "        trajectory = []\n"
                "        for i in range(n_points):\n"
                "            s = 3 * (i/n_points)**2 - 2 * (i/n_points)**3\n"
                "            pos = start + dq * s\n"
                "            trajectory.append(pos.tolist())\n"
                "        return trajectory\n"
            )
        else:
            # Generic library generation: use prompt fallback
            files = await self._generate_via_prompt(task, approach)

        return files

    def _generate_fallback_stub(self, task: SubTask) -> Dict[str, str]:
        """Generate minimal stub for sub-tasks with known patterns when LLM fails."""
        name = task.name.lower()
        pkg = "arm_controller"

        if "custom_control" in name or "control_logic" in name or "application" in name:
            # Minimal pick-and-place state machine stub
            return {
                f"ros_ws/src/{pkg}/{pkg}/control_logic.py": (
                    "#!/usr/bin/env python3\n"
                    '"""Application-specific control logic (auto-generated stub)."""\n'
                    "import rclpy\n"
                    "from rclpy.node import Node\n"
                    "from std_msgs.msg import String\n"
                    "\n\n"
                    "class ControlLogic(Node):\n"
                    '    """State machine for pick-and-place operations."""\n'
                    "\n"
                    "    def __init__(self):\n"
                    '        super().__init__("control_logic")\n'
                    "        self.state = \"idle\"\n"
                    "        self.sub = self.create_subscription(\n"
                    "            String, '/arm_controller/event', self.on_event, 10,\n"
                    "        )\n"
                    "        self.pub = self.create_publisher(\n"
                    "            String, '/arm_controller/command', 10,\n"
                    "        )\n"
                    "        self.get_logger().info(\"ControlLogic stub initialized\")\n"
                    "\n"
                    "    def on_event(self, msg):\n"
                    '        """Handle state transitions."""\n'
                    "        self.get_logger().info(f'Received event: {msg.data}')\n"
                    "\n"
                    "    def pick_and_place(self, pick_pose, place_pose):\n"
                    '        """Execute pick-and-place sequence."""\n'
                    "        self.get_logger().info(\n"
                    "            f'Pick at {pick_pose}, place at {place_pose}'\n"
                    "        )\n"
                    "\n\n"
                    "def main(args=None):\n"
                    "    rclpy.init(args=args)\n"
                    "    node = ControlLogic()\n"
                    "    rclpy.spin(node)\n"
                    "    node.destroy_node()\n"
                    "    rclpy.shutdown()\n"
                    "\n\n"
                    'if __name__ == "__main__":\n'
                    "    main()\n"
                ),
            }

        if "obstacle" in name or "collision" in name:
            return {
                f"ros_ws/src/{pkg}/{pkg}/obstacle_avoidance.py": (
                    "#!/usr/bin/env python3\n"
                    '"""Basic obstacle avoidance stub."""\n'
                    "\n\n"
                    "def check_collision(joint_angles, obstacles):\n"
                    '    """Check if joint angles cause collision (stub)."""\n'
                    "    return False\n"
                    "\n\n"
                    "def avoid_obstacles(waypoints, obstacles):\n"
                    '    """Modify waypoints to avoid obstacles (stub)."""\n'
                    "    return waypoints\n"
                ),
            }

        if "ros2_control" in name or "controller_node" in name:
            pkg = "arm_controller"
            return {
                f"ros_ws/src/{pkg}/{pkg}/arm_controller_node.py": (
                    '#!/usr/bin/env python3\n'
                    '"""Main ROS2 controller node (auto-generated stub)."""\n'
                    'import rclpy\n'
                    'from rclpy.node import Node\n'
                    'from trajectory_msgs.msg import JointTrajectory\n'
                    'from sensor_msgs.msg import JointState\n'
                    '\n\n'
                    'class ArmController(Node):\n'
                    '    """Minimal arm controller node."""\n'
                    '\n'
                    '    def __init__(self):\n'
                    '        super().__init__("arm_controller_node")\n'
                    '        self.trajectory_pub = self.create_publisher(\n'
                    '            JointTrajectory, f"/{pkg}/command", 10,\n'
                    '        )\n'
                    '        self.state_pub = self.create_publisher(\n'
                    '            JointState, f"/{pkg}/state", 10,\n'
                    '        )\n'
                    '        self.get_logger().info("ArmController stub initialized")\n'
                    '\n'
                    '    def execute_trajectory(self, joint_positions, time_from_start):\n'
                    '        """Publish a joint trajectory command."""\n'
                    '        msg = JointTrajectory()\n'
                    '        msg.joint_names = [f"joint_{i}" for i in range(6)]\n'
                    '        self.trajectory_pub.publish(msg)\n'
                    '\n\n'
                    'def main(args=None):\n'
                    '    rclpy.init(args=args)\n'
                    '    node = ArmController()\n'
                    '    rclpy.spin(node)\n'
                    '    node.destroy_node()\n'
                    '    rclpy.shutdown()\n'
                    '\n\n'
                    'if __name__ == "__main__":\n'
                    '    main()\n'
                ),
            }

        return {}

    async def _generate_via_prompt(
        self, task: SubTask, approach: TechnicalApproach, retry: bool = False,
    ) -> Dict[str, str]:
        """Generate code via LLM prompt."""
        # Build context with safety guidelines
        try:
            ros2_context, _ = self.skill.prompt_mgr.render("ros2_humble_context")
        except FileNotFoundError:
            ros2_context = "ROS2 Humble conventions apply."

        try:
            safety_ctx, _ = self.skill.prompt_mgr.render("safety_guidelines")
        except FileNotFoundError:
            safety_ctx = "Follow safety best practices."

        system_prompt = (
            "You are an expert ROS2 robotics engineer. Generate Python code "
            "for the following sub-task of a robotic arm control system.\n\n"
            f"{ros2_context}\n\n"
            f"{safety_ctx}\n\n"
            "Output a JSON object where keys are relative file paths and values "
            "are the complete file contents."
        )

        user_msg = (
            f"Sub-task: {task.name}\n"
            f"Description: {task.description}\n"
            f"Arm: {approach.arm_parameters.get('name', 'unknown')}\n"
            f"Trajectory types: {[t.value for t in approach.trajectory_types]}\n"
            f"Additional context: {task.context}\n"
        )
        if retry:
            user_msg += "\n\nPrevious attempt had syntax errors. Ensure all Python code is syntactically valid."

        from pydantic import BaseModel

        class GenResponse(BaseModel):
            files: dict

        try:
            response = await self.llm.complete_structured(
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
                output_model=GenResponse,
                max_tokens=4096,
                temperature=0.3 if not retry else 0.2,
            )
            # Prefix paths with ros_ws/src/arm_controller/
            files = {}
            for path, content in response.files.items():
                if not path.startswith("ros_ws"):
                    path = f"ros_ws/src/arm_controller/{path.lstrip('/')}"
                files[path] = content
            return files
        except Exception as e:
            logger.error("Prompt generation failed for '%s': %s", task.name, e)
            return {}

    async def assemble(
        self, results: List[SubTaskResult], approach: TechnicalApproach,
    ) -> GeneratedCode:
        """Merge all sub-task results into a coherent package structure."""
        merged: Dict[str, str] = {}

        for result in results:
            if result.success:
                for path, content in result.files.items():
                    # Later results overwrite earlier (more specific > generic)
                    merged[path] = content

        # Ensure essential files exist
        pkg_name = "arm_controller"
        if f"ros_ws/src/{pkg_name}/__init__.py" not in merged:
            merged[f"ros_ws/src/{pkg_name}/__init__.py"] = ""
        if f"ros_ws/src/{pkg_name}/package.xml" not in merged:
            merged[f"ros_ws/src/{pkg_name}/package.xml"] = self.skill.ros2_gen.render_package_xml({
                "pkg_name": pkg_name,
                "arm_name": approach.arm_parameters.get("name", "arm"),
            })

        return GeneratedCode(
            package_structure=merged,
            ros2_package_name=pkg_name,
        )

    async def validate(self, code: GeneratedCode) -> GeneratedCode:
        """Post-generation validation.

        - Python syntax check (ast.parse)
        - Check required files exist
        - Check no dangerous patterns
        """
        errors = []
        warnings = []

        # Check required files
        required = ["package.xml", "__init__.py"]
        for req in required:
            found = any(req in p for p in code.package_structure)
            if not found:
                warnings.append(f"Missing recommended file: {req}")

        # Python syntax check
        for path, content in code.package_structure.items():
            if path.endswith(".py") and content.strip():
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {path}: {e}")

        # Dangerous pattern check
        dangerous = ["exec(", "eval(", "__import__(", "subprocess.call", "os.system"]
        for path, content in code.package_structure.items():
            for pattern in dangerous:
                if pattern in content:
                    warnings.append(f"Potentially dangerous pattern '{pattern}' in {path}")

        code.lint_results = {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
        }

        if errors:
            logger.warning("Code validation found %d error(s)", len(errors))
        if warnings:
            logger.warning("Code validation found %d warning(s)", len(warnings))

        return code

    @staticmethod
    def _validate_code(files: Dict[str, str]) -> tuple:
        """Static validation of generated code files."""
        errors = []
        for path, content in files.items():
            if path.endswith(".py") and content.strip():
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"{path}: {e}")
        return len(errors) == 0, errors
