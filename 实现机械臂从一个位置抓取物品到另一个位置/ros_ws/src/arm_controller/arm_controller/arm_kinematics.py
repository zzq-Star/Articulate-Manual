import numpy as np
import math


class ArmKinematics:
    """Kinematics solver for the robot arm."""

    def __init__(self, dh_params=None):
        self.dh_params = dh_params or self._default_dh()

    def _default_dh(self):
        return [
            {"a": 0, "alpha": -1.5708, "d": 0.3, "theta": 0},
            {"a": 0.4, "alpha": 0, "d": 0, "theta": 0},
            {"a": 0.35, "alpha": 0, "d": 0, "theta": 0},
            {"a": 0, "alpha": -1.5708, "d": 0.3, "theta": 0},
            {"a": 0, "alpha": 1.5708, "d": 0, "theta": 0},
            {"a": 0, "alpha": 0, "d": 0.1, "theta": 0},
        ]

    def forward(self, joint_angles):
        """Compute forward kinematics."""
        T = np.eye(4)
        for dh, q in zip(self.dh_params, joint_angles):
            theta = dh["theta"] + q
            alpha = dh["alpha"]
            a = dh["a"]
            d = dh["d"]
            ct, st = math.cos(theta), math.sin(theta)
            ca, sa = math.cos(alpha), math.sin(alpha)
            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0,  sa,     ca,     d   ],
                [0,  0,      0,      1   ],
            ])
            T = T @ Ti
        return T

    def inverse(self, target_pose, initial_guess=None):
        """Numerical inverse kinematics."""
        import copy
        n_dof = len(self.dh_params)
        q = initial_guess if initial_guess is not None else np.zeros(n_dof)
        lam = 1.0

        for i in range(200):
            T = self.forward(q)
            error = target_pose[:3, 3] - T[:3, 3]
            if np.linalg.norm(error) < 1e-6:
                return q, True

            J = np.zeros((3, n_dof))
            eps = 1e-6
            T0 = self.forward(q)
            p0 = T0[:3, 3]
            for j in range(n_dof):
                q_eps = q.copy()
                q_eps[j] += eps
                Tj = self.forward(q_eps)
                J[:, j] = (Tj[:3, 3] - p0) / eps

            JJT = J @ J.T + lam * lam * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            q = q + dq

            if np.linalg.norm(error) < 1e-3:
                lam *= 0.9

        return q, False
