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
            "arm_parameters": approach.get("arm_parameters", {}),
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
        # Extract actual DH parameters from context so generated code matches simulation model
        arm_params = context.get("arm_parameters", context)
        dh_params_raw = arm_params.get("dh_params", [])
        joint_limits_raw = arm_params.get("joint_limits", [])

        # Build DH parameter code lines
        if dh_params_raw:
            dh_lines = []
            for dh in dh_params_raw:
                a = dh.get("a", 0) if isinstance(dh, dict) else dh.a
                alpha = dh.get("alpha", 0) if isinstance(dh, dict) else dh.alpha
                d_val = dh.get("d", 0) if isinstance(dh, dict) else dh.d
                theta = dh.get("theta", 0) if isinstance(dh, dict) else dh.theta
                dh_lines.append(
                    f'        {{"a": {a}, "alpha": {alpha}, "d": {d_val}, "theta": {theta}}},'
                )
            dh_code = "\n".join(dh_lines)
        else:
            dh_code = (
                '        {"a": 0, "alpha": -1.5708, "d": 0.3, "theta": 0},\n'
                '        {"a": 0.4, "alpha": 0, "d": 0, "theta": 0},\n'
                '        {"a": 0.35, "alpha": 0, "d": 0, "theta": 0},\n'
                '        {"a": 0, "alpha": -1.5708, "d": 0.3, "theta": 0},\n'
                '        {"a": 0, "alpha": 1.5708, "d": 0, "theta": 0},\n'
                '        {"a": 0, "alpha": 0, "d": 0.1, "theta": 0}'
            )

        # Build joint limits code lines
        if joint_limits_raw:
            jl_lines = []
            for jl in joint_limits_raw:
                lower = jl.get("lower", -3.14) if isinstance(jl, dict) else jl.lower
                upper = jl.get("upper", 3.14) if isinstance(jl, dict) else jl.upper
                velocity = jl.get("velocity", 2.0) if isinstance(jl, dict) else jl.velocity
                torque = jl.get("torque", 100) if isinstance(jl, dict) else jl.torque
                jl_lines.append(
                    f'        {{"lower": {lower}, "upper": {upper}, "velocity": {velocity}, "torque": {torque}}},'
                )
            jl_code = "\n".join(jl_lines)
        else:
            jl_code = (
                '        {"lower": -2.967, "upper": 2.967, "velocity": 2.0, "torque": 150},\n'
                '        {"lower": -2.094, "upper": 2.094, "velocity": 2.0, "torque": 150},\n'
                '        {"lower": -2.967, "upper": 2.967, "velocity": 2.5, "torque": 100},\n'
                '        {"lower": -2.094, "upper": 2.094, "velocity": 3.0, "torque": 80},\n'
                '        {"lower": -2.967, "upper": 2.967, "velocity": 3.0, "torque": 80},\n'
                '        {"lower": -2.094, "upper": 2.094, "velocity": 3.0, "torque": 50}'
            )

        n_dof = len(dh_params_raw) if dh_params_raw else 6

        # Use str.replace to embed DH params into template (avoids f-string {{}} escaping issues)
        template = (
            'import numpy as np\n'
            'import math\n'
            '\n'
            '\n'
            'class ArmKinematics:\n'
            '    """Kinematics solver for the robot arm."""\n'
            '\n'
            '    def __init__(self, dh_params=None):\n'
            '        self.dh_params = dh_params or self._default_dh()\n'
            '        self.joint_limits = self._default_limits()\n'
            '\n'
            '    def _default_dh(self):\n'
            '        return [\n'
            '__DH_PARAMS__\n'
            '        ]\n'
            '\n'
            '    def _default_limits(self):\n'
            '        return [\n'
            '__JOINT_LIMITS__\n'
            '        ]\n'
            '\n'
            '    def num_dof(self):\n'
            f'        return {n_dof}\n'
            '\n'
            '    def forward(self, joint_angles):\n'
            '        """Compute forward kinematics. Returns 4x4 homogenous transform."""\n'
            '        T = np.eye(4)\n'
            '        for dh, q in zip(self.dh_params, joint_angles):\n'
            '            theta = dh["theta"] + q\n'
            '            alpha = dh["alpha"]\n'
            '            a = dh["a"]\n'
            '            d = dh["d"]\n'
            '            ct, st = math.cos(theta), math.sin(theta)\n'
            '            ca, sa = math.cos(alpha), math.sin(alpha)\n'
            '            Ti = np.array([\n'
            '                [ct, -st*ca, st*sa, a*ct],\n'
            '                [st, ct*ca, -ct*sa, a*st],\n'
            '                [0,  sa,     ca,     d   ],\n'
            '                [0,  0,      0,      1   ],\n'
            '            ])\n'
            '            T = T @ Ti\n'
            '        return T\n'
            '\n'
            '    @staticmethod\n'
            '    def _orientation_error(R_cur, R_tgt):\n'
            '        """Orientation error as 3-vector (cross-product of column pairs)."""\n'
            '        error = np.zeros(3)\n'
            '        for i in range(3):\n'
            '            error += np.cross(R_cur[:, i], R_tgt[:, i])\n'
            '        return 0.5 * error\n'
            '\n'
            '    @staticmethod\n'
            '    def _rotation_vector(R):\n'
            '        """Convert rotation matrix to rotation vector (axis * angle)."""\n'
            '        theta = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))\n'
            '        if theta < 1e-10:\n'
            '            return np.zeros(3)\n'
            '        w = np.array([\n'
            '            R[2, 1] - R[1, 2],\n'
            '            R[0, 2] - R[2, 0],\n'
            '            R[1, 0] - R[0, 1],\n'
            '        ])\n'
            '        return w / (2.0 * math.sin(theta)) * theta\n'
            '\n'
            '    def inverse(self, target_pose, initial_guess=None):\n'
            '        """Numerical inverse kinematics using damped Levenberg-Marquardt.\n'
            '\n'
            '        Returns (q, converged) where q is n-vector of joint angles.\n'
            '        """\n'
            '        n_dof = len(self.dh_params)\n'
            '        # Try multiple restarts from different initial guesses\n'
            '        restarts = [\n'
            '            initial_guess if initial_guess is not None else np.zeros(n_dof),\n'
            '            np.zeros(n_dof),\n'
            '        ]\n'
            '        # Add more restarts for redundant manipulators\n'
            '        if n_dof > 3:\n'
            '            restarts.append(np.random.uniform(-0.5, 0.5, n_dof))\n'
            '\n'
            '        best_q = np.zeros(n_dof)\n'
            '        best_err = float("inf")\n'
            '\n'
            '        for guess in restarts:\n'
            '            q = guess.copy()\n'
            '            lam = 0.5\n'
            '            prev_err = float("inf")\n'
            '            stall_count = 0\n'
            '\n'
            '            for i in range(300):\n'
            '                T_cur = self.forward(q)\n'
            '                R_cur = T_cur[:3, :3]\n'
            '                p_cur = T_cur[:3, 3]\n'
            '                R_tgt = target_pose[:3, :3]\n'
            '                p_tgt = target_pose[:3, 3]\n'
            '\n'
            '                # 6-DOF error (position + orientation)\n'
            '                pos_err = p_tgt - p_cur\n'
            '                ori_err = self._orientation_error(R_cur, R_tgt)\n'
            '                error = np.concatenate([pos_err, ori_err])\n'
            '                err_norm = np.linalg.norm(error)\n'
            '\n'
            '                if err_norm < best_err:\n'
            '                    best_err = err_norm\n'
            '                    best_q = q.copy()\n'
            '\n'
            '                if err_norm < 1e-8:\n'
            '                    return q, True\n'
            '\n'
            '                # Stall detection: increase damping if no progress\n'
            '                if err_norm >= prev_err * 0.999:\n'
            '                    stall_count += 1\n'
            '                    if stall_count > 20:\n'
            '                        lam *= 2.0\n'
            '                        stall_count = 0\n'
            '                else:\n'
            '                    stall_count = 0\n'
            '                prev_err = err_norm\n'
            '\n'
            '                # 6xN numerical geometric Jacobian\n'
            '                J = np.zeros((6, n_dof))\n'
            '                eps = 1e-6\n'
            '                for j in range(n_dof):\n'
            '                    q_eps = q.copy()\n'
            '                    q_eps[j] += eps\n'
            '                    Tj = self.forward(q_eps)\n'
            '                    J[:3, j] = (Tj[:3, 3] - p_cur) / eps\n'
            '                    Rj = Tj[:3, :3]\n'
            '                    R_diff_eps = Rj @ R_cur.T\n'
            '                    J[3:, j] = self._rotation_vector(R_diff_eps) / eps\n'
            '\n'
            '                # Damped least squares: dq = J.T * inv(J*J.T + lam^2*I) * e\n'
            '                JJT = J @ J.T + lam * lam * np.eye(6)\n'
            '                try:\n'
            '                    dq = J.T @ np.linalg.solve(JJT, error)\n'
            '                except np.linalg.LinAlgError:\n'
            '                    dq = J.T @ error * 0.01\n'
            '                    break\n'
            '\n'
            '                # Under-relax near convergence\n'
            '                scale = 1.0 if err_norm > 0.01 else 0.5\n'
            '                q = q + dq * scale\n'
            '\n'
            '                # Reduce damping as we converge\n'
            '                if err_norm < 1e-2:\n'
            '                    lam = max(lam * 0.95, 1e-6)\n'
            '\n'
            '        # Return best result found across all restarts\n'
            '        last_T = self.forward(best_q)\n'
            '        last_error = np.linalg.norm(target_pose[:3, 3] - last_T[:3, 3])\n'
            '        return best_q, last_error < 1e-4\n'
        )

        return template.replace("__DH_PARAMS__", dh_code).replace("__JOINT_LIMITS__", jl_code)

    def _render_trajectory_planner(self, context: Dict[str, Any]) -> str:
        # Extract joint limits to generate safe waypoints
        arm_params = context.get("arm_parameters", {})
        jl = arm_params.get("joint_limits", [])
        dh_params = arm_params.get("dh_params", [])

        # Generate waypoints that stay within 15% of joint range from center
        # This prevents joint_limit_margin and joint_torque_peak failures
        if jl and len(jl) >= 6:
            wps = []
            # Home: all zeros (center of range by default)
            home = [0.0] * 6

            # Reach: 15% of range on joints 0,1,2 (the "big" axes)
            reach = []
            for j_idx in range(6):
                lim = jl[j_idx] if isinstance(jl[j_idx], dict) else {"lower": -3.14, "upper": 3.14}
                lower = lim.get("lower", -3.14)
                upper = lim.get("upper", 3.14)
                half_range = (upper - lower) / 2.0
                # Use 12-15% of range in positive/negative direction
                if j_idx == 0:
                    reach.append(half_range * 0.15)  # +15% on joint 0
                elif j_idx == 1:
                    reach.append(half_range * -0.12)  # -12% on joint 1
                elif j_idx == 2:
                    reach.append(half_range * 0.12)  # +12% on joint 2
                else:
                    reach.append(0.0)
            grasp = reach.copy()
            # Grasp: slightly closer (modify joint 2 to negative)
            if grasp[2] != 0:
                grasp[2] = -abs(grasp[2]) * 0.5
            else:
                # Joint 2 not in reach: use joint 1's range as fallback
                grasp[2] = -(reach[1] if reach[1] != 0 else 0.15) * 0.3
            # Transfer: same as grasp but joint 0 = 0
            transfer = grasp.copy()
            transfer[0] = 0.0

            waypoint_values = [home, reach, grasp, transfer, home]
        else:
            waypoint_values = [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, -0.2, 0.3, 0.0, 0.0, 0.0],
                [0.5, -0.2, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]

        # Format waypoints as Python code
        wp_lines = []
        for wp in waypoint_values:
            formatted = "[" + ", ".join(f"{v:.4f}" for v in wp) + "],"
            wp_lines.append(f"    {formatted}")
        wp_code = "\n".join(wp_lines)

        template = (
            'import numpy as np\n'
            '\n'
            '\n'
            '# WAYPOINTS: joint-space path for simulation extraction.\n'
            '# The simulation extracts this constant and interpolates between waypoints\n'
            '# using S-curve interpolation. Modify these to change the robot motion.\n'
            'WAYPOINTS = [\n'
            '__WAYPOINTS__\n'
            ']\n'
            '\n'
            '\n'
            'class TrajectoryPlanner:\n'
            '    """Trajectory planning for the robot arm."""\n'
            '\n'
            '    def plan_ptp(self, start, goal, max_velocity=1.0, max_acceleration=0.5, dt=0.01):\n'
            '        """Point-to-point trajectory with trapezoidal velocity profile."""\n'
            '        start = np.asarray(start, dtype=float)\n'
            '        goal = np.asarray(goal, dtype=float)\n'
            '        dq = goal - start\n'
            '\n'
            '        t_accel = max_velocity / max_acceleration\n'
            '        d_accel = 0.5 * max_acceleration * t_accel ** 2\n'
            '        total_dist = np.max(np.abs(dq))\n'
            '\n'
            '        if total_dist < 2 * d_accel:\n'
            '            t_accel = np.sqrt(total_dist / max_acceleration)\n'
            '            t_total = 2 * t_accel\n'
            '        else:\n'
            '            d_cruise = total_dist - 2 * d_accel\n'
            '            cruise_time = d_cruise / max_velocity\n'
            '            t_total = 2 * t_accel + cruise_time\n'
            '\n'
            '        n_points = max(2, int(t_total / dt))\n'
            '        trajectory = []\n'
            '\n'
            '        for i in range(n_points):\n'
            '            t = i * dt\n'
            '            if t <= t_accel:\n'
            '                s = 0.5 * max_acceleration * t ** 2\n'
            '                v = max_acceleration * t\n'
            '            elif t <= t_total - t_accel:\n'
            '                s = d_accel + max_velocity * (t - t_accel)\n'
            '                v = max_velocity\n'
            '            else:\n'
            '                t_dec = t_total - t\n'
            '                s = total_dist - 0.5 * max_acceleration * t_dec ** 2\n'
            '                v = max_acceleration * t_dec\n'
            '\n'
            '            fraction = s / total_dist if total_dist > 0 else 0\n'
            '            pos = start + dq * fraction\n'
            '            trajectory.append({\n'
            '                "time": t,\n'
            '                "positions": pos.tolist(),\n'
            '                "velocity": float(v),\n'
            '            })\n'
            '\n'
            '        return trajectory\n'
            '\n'
            '    def smooth_trajectory(self, waypoints, dt=0.01):\n'
            '        """Smooth a multi-point trajectory using cubic splines."""\n'
            '        if len(waypoints) < 2:\n'
            '            return waypoints\n'
            '        return waypoints  # TODO: implement spline interpolation\n'
        )

        return template.replace("__WAYPOINTS__", wp_code)
