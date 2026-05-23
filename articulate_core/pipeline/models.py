from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── Enums ───────────────────────────────────────────────────────────

class TaskType(str, Enum):
    pick_and_place = "pick_and_place"
    welding = "welding"
    spraying = "spraying"
    palletizing = "palletizing"
    assembly = "assembly"
    custom = "custom"


class TrajectoryType(str, Enum):
    ptp = "PTP"
    lin = "LIN"
    circ = "CIRC"
    spline = "SPLINE"


# ─── Value Objects ───────────────────────────────────────────────────

@dataclass
class Position3D:
    x: float
    y: float
    z: float


@dataclass
class Orientation3D:
    rx: float
    ry: float
    rz: float


@dataclass
class Waypoint:
    position: Position3D
    orientation: Optional[Orientation3D] = None
    label: str = ""


@dataclass
class SpeedRequirements:
    linear: Optional[float] = None    # m/s
    angular: Optional[float] = None   # rad/s


@dataclass
class PrecisionRequirements:
    position_mm: Optional[float] = None
    rotation_deg: Optional[float] = None


@dataclass
class Obstacle:
    position: Position3D
    dimensions: Optional[Tuple[float, float, float]] = None  # dx, dy, dz


# ─── DH Parameter Models ────────────────────────────────────────────

@dataclass
class DHParameter:
    a: float       # link length (m)
    alpha: float   # link twist (rad)
    d: float       # link offset (m)
    theta: float   # joint angle (rad)


@dataclass
class JointLimit:
    lower: float   # rad
    upper: float   # rad
    velocity: float  # rad/s
    torque: float    # Nm


@dataclass
class DHParameters:
    dh_params: List[DHParameter]
    joint_limits: List[JointLimit]
    name: str = "custom"

    def num_dof(self) -> int:
        return len(self.dh_params)


# ─── Stage 1: RequirementDocument ────────────────────────────────────

@dataclass
class EnvironmentDescription:
    description: str = ""
    obstacles: List[Obstacle] = field(default_factory=list)


@dataclass
class RequirementDocument:
    task_type: TaskType
    key_waypoints: List[Waypoint]
    end_effector: Optional[str]
    speed_requirements: SpeedRequirements
    precision_requirements: PrecisionRequirements
    environment: EnvironmentDescription
    missing_information: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Stage 2: TechnicalApproach ──────────────────────────────────────

@dataclass
class KinematicsStrategy:
    method: str  # "analytical" | "numerical" | "hybrid"
    redundancy_resolution: Optional[str] = None


@dataclass
class ROS2Architecture:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    topics: List[Dict[str, str]] = field(default_factory=list)
    services: List[Dict[str, str]] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class RiskAssessment:
    level: str  # "low" | "medium" | "high"
    items: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TechnicalApproach:
    arm_parameters: Dict[str, Any]
    kinematics_strategy: KinematicsStrategy
    trajectory_types: List[TrajectoryType]
    ros2_architecture: ROS2Architecture
    simulation_feasibility: bool
    risk_assessment: RiskAssessment
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Stage 3: GeneratedCode ──────────────────────────────────────────

@dataclass
class SubTask:
    name: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubTaskResult:
    name: str
    files: Dict[str, str]          # relative_path -> content
    route_used: str                # "library" | "prompt"
    confidence: float
    success: bool


@dataclass
class GeneratedCode:
    package_structure: Dict[str, str]   # relative_path -> content
    ros2_package_name: str
    build_successful: Optional[bool] = None
    lint_results: Optional[Dict[str, Any]] = None


# ─── Stage 4: SimulationReport ───────────────────────────────────────

@dataclass
class MetricResult:
    name: str
    passed: bool
    value: float
    threshold: float
    unit: str
    explanation: str = ""


@dataclass
class RepairAttempt:
    iteration: int
    diagnosis: str
    action_taken: str
    success: bool


@dataclass
class SimulationReport:
    passed: bool
    metrics: Dict[str, MetricResult]
    summary: str
    recommendations: List[str] = field(default_factory=list)
    repair_history: List[RepairAttempt] = field(default_factory=list)
    kinematic_only: bool = False


# ─── Stage 5: DeploymentPackage ──────────────────────────────────────

@dataclass
class DeploymentPackage:
    output_dir: Path
    files: Dict[str, Path]
    guide_path: Path
    checklist_path: Path
    target_brand: str


# ─── Shared: StageContext ────────────────────────────────────────────

@dataclass
class StageContext:
    """Holds all state accumulated across pipeline stages.

    Serialized to .articulate/state.json between CLI invocations.
    """
    project_dir: Path
    user_input: str = ""
    current_stage: int = 0          # 0 = unstarted
    should_continue: bool = True

    requirement_doc: Optional[RequirementDocument] = None
    technical_approach: Optional[TechnicalApproach] = None
    generated_code: Optional[GeneratedCode] = None
    simulation_report: Optional[SimulationReport] = None
    deployment_package: Optional[DeploymentPackage] = None

    target_brand: str = "ur"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_dir": str(self.project_dir),
            "user_input": self.user_input,
            "current_stage": self.current_stage,
            "should_continue": self.should_continue,
            "requirement_doc": asdict(self.requirement_doc) if self.requirement_doc else None,
            "technical_approach": asdict(self.technical_approach) if self.technical_approach else None,
            "generated_code": asdict(self.generated_code) if self.generated_code else None,
            "simulation_report": asdict(self.simulation_report) if self.simulation_report else None,
            "deployment_package": asdict(self.deployment_package) if self.deployment_package else None,
            "target_brand": self.target_brand,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageContext":
        return cls(
            project_dir=Path(data["project_dir"]),
            user_input=data.get("user_input", ""),
            current_stage=data.get("current_stage", 0),
            should_continue=data.get("should_continue", True),
            target_brand=data.get("target_brand", "ur"),
        )
