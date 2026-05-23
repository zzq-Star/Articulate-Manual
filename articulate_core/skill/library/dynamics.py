import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from articulate_core.skill.models.dh_template import ArmModel

logger = logging.getLogger(__name__)


@dataclass
class PayloadResult:
    within_limits: bool
    max_torque_ratio: float   # ratio of actual / rated
    critical_joints: List[int]
    recommended_payload: float


class DynamicsLibrary:
    """Dynamics parameter computation for payload and torque validation."""

    def compute_torque(
        self,
        arm: ArmModel,
        joint_angles: np.ndarray,
        joint_vel: np.ndarray,
        joint_acc: np.ndarray,
        payload_mass: float = 0.0,
    ) -> np.ndarray:
        """Compute required joint torques using simplified rigid-body dynamics.

        Uses recursive Newton-Euler algorithm (simplified).
        Returns torque for each joint in Nm.
        """
        n_dof = arm.num_dof()
        angles = np.asarray(joint_angles, dtype=float)
        vel = np.asarray(joint_vel, dtype=float)
        acc = np.asarray(joint_acc, dtype=float)

        torques = np.zeros(n_dof)

        # Simplified computed-torque: M(q) * qddot + C(q, qdot) + G(q)
        # Using approximate inertia matrix from link masses

        # Gravity vector (assume g along -z)
        g = np.array([0, 0, -9.81])

        for i in range(n_dof):
            # Simplified: torque = inertia * acceleration + gravity + friction
            mass = 0.0
            if i < len(arm.dynamics):
                mass = arm.dynamics[i].mass

            # Inertia torque (simplified)
            inertia_torque = 0.1 * mass * acc[i]

            # Gravity torque (simplified - depends on configuration)
            gravity_torque = mass * 9.81 * 0.3 * np.sin(angles[i])

            # Friction
            friction = 0.0
            if i < len(arm.dynamics):
                friction = arm.dynamics[i].friction * vel[i]

            # Coriolis/centrifugal (simplified)
            coriolis = 0.0
            for j in range(n_dof):
                if j != i:
                    coriolis += 0.5 * mass * vel[i] * vel[j] * np.sin(angles[i] - angles[j])

            # Payload contribution (to last joint)
            payload_torque = 0.0
            if i == n_dof - 1 and payload_mass > 0:
                payload_torque = payload_mass * 9.81 * 0.5 * np.sin(angles[i])

            torques[i] = inertia_torque + gravity_torque + friction + coriolis + payload_torque

        return torques

    def check_payload(
        self,
        arm: ArmModel,
        payload_mass: float,
        joint_angles: np.ndarray,
    ) -> PayloadResult:
        """Check if payload is within torque limits across a range of configurations."""
        n_dof = arm.num_dof()
        max_ratios = np.zeros(n_dof)
        critical_joints = []

        # Test at multiple configurations
        n_samples = 10
        test_configs = arm.sample_joint_angles(n_samples)

        for angles in test_configs:
            torques = self.compute_torque(
                arm, angles,
                joint_vel=np.zeros(n_dof),
                joint_acc=np.zeros(n_dof),
                payload_mass=payload_mass,
            )

            for j in range(n_dof):
                if j < len(arm.joint_limits):
                    rated = arm.joint_limits[j].torque
                    if rated > 0:
                        ratio = abs(torques[j]) / rated
                        max_ratios[j] = max(max_ratios[j], ratio)

        # Identify critical joints
        for j in range(n_dof):
            if j < len(arm.joint_limits) and max_ratios[j] > 0.95:
                critical_joints.append(j)

        overall_within = all(r <= 0.95 for r in max_ratios)
        max_ratio = float(np.max(max_ratios)) if n_dof > 0 else 0.0

        # Estimate recommended payload
        recommended_payload = payload_mass
        if max_ratio > 0.95:
            recommended_payload = payload_mass * (0.9 / max_ratio)

        return PayloadResult(
            within_limits=overall_within,
            max_torque_ratio=max_ratio,
            critical_joints=critical_joints,
            recommended_payload=round(recommended_payload, 3),
        )
