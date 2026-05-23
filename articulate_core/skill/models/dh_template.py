import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DHParameter:
    a: float       # link length (m)
    alpha: float   # link twist (rad)
    d: float       # link offset (m)
    theta: float   # joint angle (rad)


@dataclass
class JointLimit:
    lower: float    # rad
    upper: float    # rad
    velocity: float  # rad/s
    torque: float    # Nm


@dataclass
class DynamicsData:
    mass: float = 0.0           # kg
    inertia: List[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    friction: float = 0.0
    damping: float = 0.0


@dataclass
class ArmModel:
    name: str
    dh_params: List[DHParameter]
    joint_limits: List[JointLimit]
    dynamics: List[DynamicsData] = field(default_factory=list)
    urdf_path: Optional[str] = None
    mjcf_path: Optional[str] = None

    def num_dof(self) -> int:
        return len(self.dh_params)

    def compute_fk(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint angles.

        Uses standard DH convention. Returns 4x4 transformation matrix
        from base to end-effector.
        """
        if len(joint_angles) != self.num_dof():
            raise ValueError(
                f"Expected {self.num_dof()} joint angles, got {len(joint_angles)}"
            )

        T = np.eye(4)
        for i, (dh, q) in enumerate(zip(self.dh_params, joint_angles)):
            theta = dh.theta + q
            alpha = dh.alpha
            a = dh.a
            d = dh.d

            ct = math.cos(theta)
            st = math.sin(theta)
            ca = math.cos(alpha)
            sa = math.sin(alpha)

            Ti = np.array([
                [ct,  -st * ca,  st * sa,  a * ct],
                [st,   ct * ca,  -ct * sa,  a * st],
                [0,    sa,        ca,        d     ],
                [0,    0,         0,         1     ],
            ])
            T = T @ Ti

        return T

    def sample_joint_angles(self, num_samples: int = 100) -> np.ndarray:
        """Sample random joint angles within limits."""
        samples = np.zeros((num_samples, self.num_dof()))
        for i, lim in enumerate(self.joint_limits):
            samples[:, i] = np.random.uniform(lim.lower, lim.upper, num_samples)
        return samples

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_dof": self.num_dof(),
            "dh_params": [
                {"a": p.a, "alpha": p.alpha, "d": p.d, "theta": p.theta}
                for p in self.dh_params
            ],
            "joint_limits": [
                {"lower": l.lower, "upper": l.upper, "velocity": l.velocity, "torque": l.torque}
                for l in self.joint_limits
            ],
            "dynamics": [
                {"mass": d.mass, "friction": d.friction, "damping": d.damping}
                for d in self.dynamics
            ] if self.dynamics else [],
        }


def arm_from_preset(name: str) -> Optional[ArmModel]:
    """Load a preset arm by name."""
    from articulate_core.skill.models.preset_arms import PRESET_ARMS
    return PRESET_ARMS.get(name)
