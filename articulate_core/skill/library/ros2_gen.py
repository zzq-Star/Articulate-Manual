import logging
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class ROS2Generator:
    """ROS2 code generation via Jinja2 templates."""

    def __init__(self, templates_dir: Path):
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )

    def _render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with context."""
        template = self.env.get_template(template_name)
        return template.render(**context)

    def render_node(self, context: Dict[str, Any]) -> str:
        return self._render("ros2_node.py.j2", context)

    def render_launch(self, context: Dict[str, Any]) -> str:
        return self._render("launch.py.j2", context)

    def render_package_xml(self, context: Dict[str, Any]) -> str:
        return self._render("package_xml.j2", context)

    def render_controllers_yaml(self, context: Dict[str, Any]) -> str:
        return self._render("controllers_yaml.j2", context)

    def render_kinematics_yaml(self, context: Dict[str, Any]) -> str:
        return self._render("kinematics_yaml.j2", context)

    def generate_package(self, approach: Dict[str, Any]) -> Dict[str, str]:
        """Generate complete ROS2 package as dict of path -> content."""
        pkg_name = "arm_controller"
        arm_name = approach.get("arm_parameters", {}).get("name", "arm")
        has_kinematics = True
        trajectory_types = approach.get("trajectory_types", ["PTP"])

        context = {
            "pkg_name": pkg_name,
            "arm_name": arm_name,
            "has_kinematics": has_kinematics,
            "trajectory_types": trajectory_types,
            "node_class": "ArmController",
            "node_name": "arm_controller_node",
            "publishers": [
                {"name": "trajectory_pub", "msg_type": "JointTrajectory",
                 "topic": f"/{pkg_name}/command"},
                {"name": "state_pub", "msg_type": "JointState",
                 "topic": f"/{pkg_name}/state"},
            ],
            "subscribers": [
                {"name": "command_sub", "msg_type": "JointTrajectory",
                 "topic": f"/{pkg_name}/goal", "callback": "on_goal"},
            ],
            "methods": [],
        }

        files = {
            f"ros_ws/src/{pkg_name}/package.xml": self.render_package_xml(context),
            f"ros_ws/src/{pkg_name}/setup.py": self._render_setup_py(context),
            f"ros_ws/src/{pkg_name}/config/controllers.yaml": self.render_controllers_yaml(context),
            f"ros_ws/src/{pkg_name}/config/kinematics.yaml": self.render_kinematics_yaml(context),
            f"ros_ws/src/{pkg_name}/launch/{pkg_name}_bringup.launch.py": self.render_launch(context),
            f"ros_ws/src/{pkg_name}/{pkg_name}/__init__.py": "",
            f"ros_ws/src/{pkg_name}/{pkg_name}/arm_kinematics.py": self._render_kinematics_source(context),
            f"ros_ws/src/{pkg_name}/{pkg_name}/trajectory_planner.py": self._render_trajectory_planner(context),
            f"ros_ws/src/{pkg_name}/{pkg_name}/{pkg_name}_node.py": self.render_node(context),
        }

        return files

    def _render_setup_py(self, context: Dict[str, Any]) -> str:
        pkg = context["pkg_name"]
        return f"""from setuptools import find_packages, setup
import os
from glob import glob

package_name = '{pkg}'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Articulate generated arm controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={{
        'console_scripts': [
            '{pkg}_node = {pkg}.{pkg}_node:main',
        ],
    }},
)
"""

    def _render_kinematics_source(self, context: Dict[str, Any]) -> str:
        return """import numpy as np
import math


class ArmKinematics:
    \"\"\"Kinematics solver for the robot arm.\"\"\"

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
        \"\"\"Compute forward kinematics.\"\"\"
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
        \"\"\"Numerical inverse kinematics.\"\"\"
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
"""

    def _render_trajectory_planner(self, context: Dict[str, Any]) -> str:
        return """import numpy as np


# WAYPOINTS: joint-space path for simulation extraction.
# The simulation extracts this constant and interpolates between waypoints
# using S-curve interpolation. Modify these to change the robot motion.
WAYPOINTS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # home position
    [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],   # reach / pre-grasp
    [0.5, -0.2, 0.1, 0.0, 0.0, 0.0],   # grasp at target
    [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],   # transfer to destination
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # return home
]


class TrajectoryPlanner:
    \"\"\"Trajectory planning for the robot arm.\"\"\"

    def plan_ptp(self, start, goal, max_velocity=1.0, max_acceleration=0.5, dt=0.01):
        \"\"\"Point-to-point trajectory with trapezoidal velocity profile.\"\"\"
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        dq = goal - start

        t_accel = max_velocity / max_acceleration
        d_accel = 0.5 * max_acceleration * t_accel ** 2
        total_dist = np.max(np.abs(dq))

        if total_dist < 2 * d_accel:
            t_accel = np.sqrt(total_dist / max_acceleration)
            t_total = 2 * t_accel
        else:
            d_cruise = total_dist - 2 * d_accel
            cruise_time = d_cruise / max_velocity
            t_total = 2 * t_accel + cruise_time

        n_points = max(2, int(t_total / dt))
        trajectory = []

        for i in range(n_points):
            t = i * dt
            if t <= t_accel:
                s = 0.5 * max_acceleration * t ** 2
                v = max_acceleration * t
            elif t <= t_total - t_accel:
                s = d_accel + max_velocity * (t - t_accel)
                v = max_velocity
            else:
                t_dec = t_total - t
                s = total_dist - 0.5 * max_acceleration * t_dec ** 2
                v = max_acceleration * t_dec

            fraction = s / total_dist if total_dist > 0 else 0
            pos = start + dq * fraction
            trajectory.append({
                "time": t,
                "positions": pos.tolist(),
                "velocity": float(v),
            })

        return trajectory

    def smooth_trajectory(self, waypoints, dt=0.01):
        \"\"\"Smooth a multi-point trajectory using cubic splines.\"\"\"
        if len(waypoints) < 2:
            return waypoints
        return waypoints  # TODO: implement spline interpolation
"""
