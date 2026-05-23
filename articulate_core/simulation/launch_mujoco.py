"""MuJoCo integration for simulation execution."""

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from articulate_core.simulation.metrics import SimulationData

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryCommand:
    """A trajectory to execute in simulation."""
    joint_positions: np.ndarray   # (N, n_dof) target positions
    joint_velocities: np.ndarray  # (N, n_dof) target velocities (optional)
    time_steps: np.ndarray        # (N,) time for each step


class MuJoCoSimulator:
    """MuJoCo simulation wrapper for trajectory playback.

    Gracefully handles cases where MuJoCo is not installed.
    """

    def __init__(self):
        self._mujoco_available = self._check_import()

    @staticmethod
    def _check_import() -> bool:
        try:
            import mujoco  # noqa
            return True
        except ImportError:
            logger.warning(
                "MuJoCo not installed. Install with: pip install mujoco>=3.0\n"
                "Falling back to kinematic simulation."
            )
            return False

    async def run_trajectory(
        self,
        mjcf_path: Path,
        trajectory: TrajectoryCommand,
        timeout: float = 30.0,
    ) -> SimulationData:
        """Execute trajectory in MuJoCo and collect state data.

        If MuJoCo is not available, generates synthetic simulation data
        from the trajectory commands.
        """
        if self._mujoco_available:
            return await self._run_mujoco(mjcf_path, trajectory, timeout)
        else:
            return self._run_kinematic(trajectory)

    async def _run_mujoco(
        self, mjcf_path: Path, trajectory: TrajectoryCommand, timeout: float,
    ) -> SimulationData:
        """Run MuJoCo simulation with sensor feedback and collision detection."""
        try:
            import mujoco

            if mjcf_path is None or not mjcf_path.exists():
                logger.warning("MJCF path not provided or not found: %s", mjcf_path)
                return self._run_kinematic(trajectory)

            # Copy MJCF to ASCII-only temp path to avoid Unicode issues
            # (MuJoCo C API may not handle non-ASCII paths on some platforms)
            mjcf_path_str = str(mjcf_path)
            try:
                mjcf_path_str.encode("ascii")
            except (UnicodeEncodeError, UnicodeDecodeError):
                ascii_path = Path(tempfile.mktemp(suffix=".mjcf"))
                shutil.copy2(str(mjcf_path), str(ascii_path))
                mjcf_path_str = str(ascii_path)
                logger.info("Copied MJCF to ASCII temp path: %s", ascii_path)

            model = mujoco.MjModel.from_xml_path(mjcf_path_str)
            data = mujoco.MjData(model)

            n_dof = model.nq if model.nq > 0 else trajectory.joint_positions.shape[1]
            n_steps = len(trajectory.time_steps)

            # Map sensor names to indices
            sensor_names = {model.sensor(i).name: i for i in range(model.nsensor)} if model.nsensor > 0 else {}

            # Pre-allocate
            max_steps = n_steps
            time_arr = np.zeros(max_steps)
            pos_arr = np.zeros((max_steps, n_dof))
            vel_arr = np.zeros((max_steps, n_dof))
            torque_arr = np.zeros((max_steps, n_dof))
            tcp_pos_arr = np.zeros((max_steps, 3))
            tcp_orient_arr = np.zeros((max_steps, 3))
            coll_dist_arr = np.ones(max_steps) * 100.0
            cond_num_arr = np.ones(max_steps) * 5.0

            # Find TCP site
            tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") \
                if hasattr(mujoco, 'mj_name2id') else -1
            if tcp_site_id < 0 and model.nsite > 0:
                tcp_site_id = model.nsite - 1  # last site

            # Pre-compute Jacobian buffer
            jacp = np.zeros((3, model.nv)) if hasattr(mujoco, 'mj_jac') else None
            jacr = np.zeros((3, model.nv)) if hasattr(mujoco, 'mj_jac') else None

            # Per-joint actuator force indices
            actuator_ids = {}
            for j in range(n_dof):
                sensor_name = f"jtorque_{j}"
                if sensor_name in sensor_names:
                    actuator_ids[j] = sensor_names[sensor_name]

            sim_step = 0
            for i in range(n_steps):
                # Set position control target
                if i < len(trajectory.joint_positions):
                    target = trajectory.joint_positions[i]
                    for j in range(min(n_dof, len(target), model.nu)):
                        data.ctrl[j] = target[j]

                # Run enough physics steps to match trajectory dt
                if i > 0:
                    dt_target = trajectory.time_steps[i] - trajectory.time_steps[i - 1]
                elif n_steps > 1:
                    dt_target = trajectory.time_steps[1] - trajectory.time_steps[0]
                else:
                    dt_target = model.opt.timestep
                n_substeps = max(1, int(round(dt_target / model.opt.timestep)))
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

                time_arr[sim_step] = data.time

                # Read qpos/qvel (mapped from free/ball joints to our hinge joints)
                for j in range(min(n_dof, model.nq)):
                    pos_arr[sim_step, j] = data.qpos[j]
                    vel_arr[sim_step, j] = data.qvel[j]

                # Read actuator force / torque from sensors
                for j in range(n_dof):
                    if j in actuator_ids:
                        sid = actuator_ids[j]
                        torque_arr[sim_step, j] = abs(data.sensordata[sid])
                    else:
                        torque_arr[sim_step, j] = abs(data.qfrc_actuator[j]) \
                            if j < len(data.qfrc_actuator) else 0.0

                # TCP position from site
                if tcp_site_id >= 0:
                    tcp_pos_arr[sim_step] = data.site_xpos[tcp_site_id]
                elif model.nbody > 0:
                    tcp_pos_arr[sim_step] = data.xpos[-1]  # last body position

                # Compute condition number from Jacobian at end-effector
                if jacp is not None and tcp_site_id >= 0:
                    try:
                        body_id = model.site(tcp_site_id).bodyid.item()
                        site_pos = data.site_xpos[tcp_site_id].copy()
                        mujoco.mj_jac(model, data, jacp, jacr, site_pos, body_id)
                        # Use Jacobian at TCP: jacp[:3, :n_dof]
                        J = jacp[:3, :n_dof]
                        JJt = J @ J.T
                        if np.any(np.isfinite(JJt)):
                            eigenvalues = np.linalg.eigvalsh(JJt)
                            eigenvalues = eigenvalues[eigenvalues > 1e-10]
                            if len(eigenvalues) > 0:
                                cond_num_arr[sim_step] = np.sqrt(
                                    np.max(eigenvalues) / np.min(eigenvalues)
                                ) if np.min(eigenvalues) > 1e-10 else 100.0
                    except Exception:
                        logger.debug("Jacobian computation failed", exc_info=True)

                # Self-collision: check non-adjacent link geoms only
                if model.ngeom > 1:
                    min_dist = 100.0
                    # Geom layout: 0=ground, then per link: visual+collision
                    # Skip ground, same-link pairs, and adjacent-link pairs
                    for g1 in range(1, model.ngeom):  # skip ground
                        # map from geom index to link index: geom 1,2→link0, 3,4→link1, etc.
                        link1 = (g1 - 1) // 2
                        for g2 in range(g1 + 1, model.ngeom):
                            link2 = (g2 - 1) // 2
                            if abs(link1 - link2) <= 1:
                                continue
                            try:
                                dist = mujoco.mj_geomDistance(
                                    model, data, g1, g2, 100.0, None
                                )
                                if dist < min_dist:
                                    min_dist = dist
                            except Exception:
                                pass
                    coll_dist_arr[sim_step] = min_dist if min_dist < 100.0 else 100.0

                sim_step += 1

                if data.time > timeout:
                    logger.warning("Simulation timed out after %.1fs", timeout)
                    break

            n_actual = sim_step

            # Compute acceleration from velocity difference
            acc_arr = np.zeros((n_actual, n_dof))
            if n_actual > 2:
                dt = np.mean(np.diff(time_arr[:n_actual])) if n_actual > 1 else 0.001
                acc_arr[1:] = np.diff(vel_arr[:n_actual], axis=0) / dt

            return SimulationData(
                time=time_arr[:n_actual],
                joint_positions=pos_arr[:n_actual],
                joint_velocities=vel_arr[:n_actual],
                joint_accelerations=acc_arr[:n_actual],
                joint_torques=torque_arr[:n_actual],
                tcp_positions=tcp_pos_arr[:n_actual],
                tcp_orientations=tcp_orient_arr[:n_actual],
                self_collision_distances=coll_dist_arr[:n_actual],
                condition_numbers=cond_num_arr[:n_actual],
            )

        except Exception as e:
            logger.error("MuJoCo simulation error: %s. Falling back to kinematic.", e)
            return self._run_kinematic(trajectory)

    def _run_kinematic(self, trajectory: TrajectoryCommand) -> SimulationData:
        """Generate simulation data from kinematics only (no physics).

        Used when MuJoCo is not available or as fallback.
        """
        n_dof = trajectory.joint_positions.shape[1] if trajectory.joint_positions.ndim > 1 else 6
        n_steps = len(trajectory.time_steps)

        # Compute velocity and acceleration from position differences
        pos = trajectory.joint_positions
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        torques = np.zeros_like(pos)

        if n_steps > 2:
            dt = np.mean(np.diff(trajectory.time_steps[:min(n_steps, 10)])) if n_steps > 1 else 0.01
            vel[1:] = np.diff(pos, axis=0) / dt if n_steps > 1 else vel[1:]
            if n_steps > 2:
                acc[2:] = np.diff(vel[1:], axis=0) / dt if n_steps > 2 else acc[2:]

        # TCP position from last joint (simplified: cumulative sum)
        tcp_pos = np.zeros((n_steps, 3))
        tcp_pos[:, 0] = np.sum(pos[:, :3], axis=1) if n_dof >= 3 else pos[:, 0]

        # Simple torque estimate
        torques = np.abs(vel) * 5.0 + np.abs(acc) * 2.0

        return SimulationData(
            time=np.asarray(trajectory.time_steps),
            joint_positions=pos,
            joint_velocities=vel,
            joint_accelerations=acc,
            joint_torques=torques,
            tcp_positions=tcp_pos[:, :3],
            tcp_orientations=np.zeros((n_steps, 3)),
            self_collision_distances=np.ones(n_steps) * 50.0,
            condition_numbers=np.ones(n_steps) * 5.0,
            kinematic_only=True,
        )

    def load_mjcf(self, path: Path) -> Optional[object]:
        """Load MJCF model file. Returns None if MuJoCo unavailable."""
        if not self._mujoco_available:
            return None
        try:
            import mujoco
            return mujoco.MjModel.from_xml_path(str(path))
        except Exception as e:
            logger.error("Failed to load MJCF: %s", e)
            return None
