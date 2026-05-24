import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from articulate_core.skill.models.dh_template import ArmModel

logger = logging.getLogger(__name__)


class KinematicsError(Exception):
    """Kinematics computation error."""


@dataclass
class IKResult:
    solution: np.ndarray   # joint angles
    success: bool
    iterations: int
    residual: float


class KinematicsLibrary:
    """Wrapper around Pinocchio (with pure-numpy fallback)."""

    def __init__(self):
        self._pinocchio_available = self._check_pinocchio()

    @staticmethod
    def _check_pinocchio() -> bool:
        try:
            import pinocchio  # noqa
            return True
        except ImportError:
            return False

    def fk(self, arm: ArmModel, joint_angles: np.ndarray) -> np.ndarray:
        """Forward kinematics. Returns 4x4 transform of end effector."""
        return arm.compute_fk(np.asarray(joint_angles, dtype=float))

    def ik(
        self,
        arm: ArmModel,
        target: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "levenberg",
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> IKResult:
        """Inverse kinematics using numerical methods."""
        if self._pinocchio_available:
            return self._ik_pinocchio(arm, target, initial_guess)
        return self._ik_numerical(arm, target, initial_guess, max_iterations, tolerance)

    @staticmethod
    def _orientation_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
        """Orientation error as 3-vector (cross-product of column pairs)."""
        error = np.zeros(3)
        for i in range(3):
            error += np.cross(R_cur[:, i], R_tgt[:, i])
        return 0.5 * error

    @staticmethod
    def _rotation_vector(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix → rotation vector (axis * angle)."""
        theta = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
        if theta < 1e-10:
            return np.zeros(3)
        w = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        return w / (2.0 * math.sin(theta)) * theta

    def _ik_numerical(
        self, arm: ArmModel, target: np.ndarray,
        initial_guess: Optional[np.ndarray],
        max_iter: int, tol: float,
    ) -> IKResult:
        """Numerical IK using Levenberg-Marquardt (6-DOF: position + orientation)."""
        n_dof = arm.num_dof()
        q = initial_guess.copy() if initial_guess is not None else np.zeros(n_dof)
        lam = 1.0  # damping factor

        # Ensure target is a 4x4 matrix (auto-convert 3-vector to identity orientation)
        if target.shape != (4, 4):
            T_target = np.eye(4)
            T_target[:3, 3] = target.ravel()[:3]
            target = T_target

        for i in range(max_iter):
            T_current = arm.compute_fk(q)
            R_cur = T_current[:3, :3]
            p_cur = T_current[:3, 3]
            R_tgt = target[:3, :3]
            p_tgt = target[:3, 3]

            # 6-DOF error
            pos_err = p_tgt - p_cur
            ori_err = self._orientation_error(R_cur, R_tgt)
            error = np.concatenate([pos_err, ori_err])
            err_norm = np.linalg.norm(error)

            if err_norm < tol:
                return IKResult(solution=q, success=True, iterations=i, residual=float(err_norm))

            # 6×n geometric Jacobian via finite differences
            J = self._numerical_jacobian(arm, q)
            JJT = J @ J.T + lam * lam * np.eye(6)

            try:
                dq = J.T @ np.linalg.solve(JJT, error)
            except np.linalg.LinAlgError:
                dq = J.T @ error * 0.1

            # Apply joint limits
            q_new = q + dq
            for j in range(n_dof):
                if j < len(arm.joint_limits):
                    lim = arm.joint_limits[j]
                    q_new[j] = np.clip(q_new[j], lim.lower, lim.upper)

            q = q_new

            # Reduce damping as we converge
            if err_norm < 1e-3:
                lam *= 0.9

        T_final = arm.compute_fk(q)
        residual = float(np.linalg.norm(target[:3, 3] - T_final[:3, 3]))
        return IKResult(
            solution=q, success=residual < tol * 10,
            iterations=max_iter, residual=residual,
        )

    def _ik_pinocchio(
        self, arm: ArmModel, target: np.ndarray,
        initial_guess: Optional[np.ndarray],
    ) -> IKResult:
        """IK using Pinocchio."""
        try:
            import pinocchio as pin

            # Build Pinocchio model from arm
            model = pin.Model()
            geom_model = pin.GeometryModel()
            # Simplified: create basic model from DH params
            for i, dh in enumerate(arm.dh_params):
                joint_name = f"joint_{i}"
                joint_id = model.addJoint(
                    model.getJointId("universe") if i == 0 else model.getJointId(f"joint_{i-1}"),
                    pin.JointModelRY(),
                    pin.SE3(np.eye(3), np.array([0, 0, dh.d])),
                    joint_name,
                )

            data = model.createData()
            q = initial_guess if initial_guess is not None else pin.neutral(model)
            pin.framesForwardKinematics(model, data, q)

            # Use the built-in IK
            from pinocchio import SE3
            oMdes = SE3(target[:3, :3], target[:3, 3])

            # Simple damped IK
            for i in range(100):
                pin.computeJointJacobians(model, data, q)
                J = pin.getJointJacobian(model, data, model.getJointId(f"joint_{arm.num_dof()-1}"), pin.ReferenceFrame.LOCAL)
                err = pin.log(oMdes.inverse() * data.oMi[model.getJointId(f"joint_{arm.num_dof()-1}")]).vector
                if np.linalg.norm(err) < 1e-6:
                    return IKResult(solution=q, success=True, iterations=i, residual=float(np.linalg.norm(err)))
                v = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
                q = pin.integrate(model, q, v * 0.1)

            return IKResult(solution=q, success=True, iterations=100, residual=float(np.linalg.norm(err)))

        except Exception as e:
            logger.warning("Pinocchio IK failed, falling back to numerical: %s", e)
            return self._ik_numerical(arm, target, initial_guess, 100, 1e-6)

    def _numerical_jacobian(self, arm: ArmModel, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute geometric Jacobian via finite differences (6×n: linear + angular)."""
        n_dof = arm.num_dof()
        T0 = arm.compute_fk(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]

        J = np.zeros((6, n_dof))
        for i in range(n_dof):
            q_eps = q.copy()
            q_eps[i] += eps

            if i < len(arm.joint_limits):
                lim = arm.joint_limits[i]
                q_eps[i] = np.clip(q_eps[i], lim.lower, lim.upper)

            T_eps = arm.compute_fk(q_eps)

            # Linear velocity (position finite difference)
            J[:3, i] = (T_eps[:3, 3] - p0) / eps

            # Angular velocity (rotation finite difference)
            R_eps = T_eps[:3, :3]
            R_diff = R_eps @ R0.T  # rotation from q → q+eps in current frame
            J[3:, i] = self._rotation_vector(R_diff) / eps

        return J

    def jacobian(self, arm: ArmModel, joint_angles: np.ndarray) -> np.ndarray:
        """Compute geometric Jacobian."""
        return self._numerical_jacobian(arm, np.asarray(joint_angles, dtype=float))

    def condition_number(self, arm: ArmModel, joint_angles: np.ndarray) -> float:
        """Compute condition number for singularity detection."""
        J = self.jacobian(arm, joint_angles)
        _, s, _ = np.linalg.svd(J)
        if s[-1] < 1e-10:
            return float("inf")
        return float(s[0] / s[-1])
