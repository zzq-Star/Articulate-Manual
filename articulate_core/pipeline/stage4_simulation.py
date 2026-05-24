import asyncio
import importlib
import json
import logging
import math
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from articulate_core.pipeline.models import (
    GeneratedCode,
    SimulationReport,
    StageContext,
)
from articulate_core.pipeline.orchestrator import BaseStage
from articulate_core.simulation.launch_mujoco import MuJoCoSimulator, TrajectoryCommand
from articulate_core.simulation.metrics import MetricResult, SimulationData
from articulate_core.simulation.validation_engine import ValidationEngine, ValidationReport

logger = logging.getLogger(__name__)


class SimulationStage(BaseStage):
    stage_id: int = 4
    stage_name: str = "simulation_verification"
    MAX_REPAIR_ATTEMPTS = 3

    async def execute(self, ctx: StageContext) -> StageContext:
        logger.info("[Stage 4] Starting simulation verification")

        if not ctx.generated_code:
            logger.error("No generated code found. Run 'articulate codegen' first.")
            ctx.should_continue = False
            return ctx

        code = ctx.generated_code
        arm = ctx.technical_approach.arm_parameters if ctx.technical_approach else None

        print("\n" + "=" * 60)
        print("STAGE 4: SIMULATION VERIFICATION")
        print("=" * 60)

        # 1. Build and prepare
        print("[Stage 4] Preparing simulation environment...")
        urdf_path, mjcf_path = await self._prepare_models(ctx, code, arm)

        # 2. Extract trajectory from generated code
        print("[Stage 4] Extracting trajectory for simulation...")
        traj_cmd = self._extract_trajectory(code)
        print(f"  Trajectory: {len(traj_cmd.time_steps)} steps, {traj_cmd.joint_positions.shape[1] if traj_cmd.joint_positions.ndim > 1 else 6} joints")

        # 3. Run simulation
        print("[Stage 4] Running simulation...")
        simulator = MuJoCoSimulator()
        sim_data = await simulator.run_trajectory(mjcf_path, traj_cmd)
        sim_data.arm_model = arm or {}
        print(f"  Simulation: {sim_data.n_steps} steps recorded")

        if sim_data.kinematic_only:
            print()
            print("  [bold red]ERROR: MuJoCo physics simulation failed to run.[/bold red]")
            print("  [red]The kinematic fallback produces unreliable synthetic data[/red]")
            print("  [red]and cannot be used for meaningful validation.[/red]")
            print()
            print("  Possible causes:")
            print("    1. MuJoCo not installed:  [yellow]pip install mujoco>=3.0[/yellow]")
            print('    2. MJCF file contains non-ASCII path characters')
            print('    3. Missing system dependencies (MuJoCo library)')
            print()
            print("  Install MuJoCo and retry:")
            print("    [yellow]pip install 'articulate-core[simulation]'[/yellow]")
            print("    [yellow]# or: pip install mujoco>=3.0[/yellow]")
            print()
            ctx.should_continue = False
            return ctx

        # 4. Validate
        print("[Stage 4] Validating simulation results...")
        engine = ValidationEngine()
        report = await engine.validate(sim_data)

        # 4.5 Validate code execution (structure, FK/IK)
        print("[Stage 4] Validating code execution...")
        code_metrics = self._validate_code_execution(code)
        if code_metrics:
            report = self._merge_code_metrics(report, code_metrics)

        # 5. Display results
        self._display_results(report)

        # 6. Auto-repair loop
        # First MAX_REPAIR_ATTEMPTS (3) are automatic; after that, user decides.
        attempt = 0
        while not report.passed:
            attempt += 1
            is_auto = attempt <= self.MAX_REPAIR_ATTEMPTS

            if not is_auto:
                # Interactive: ask user whether to keep trying
                print(f"\n[Stage 4] Repair attempt {attempt-1} did not pass all checks.")
                if not await self._confirm(
                    f"继续第 {attempt} 次修复尝试？"
                ):
                    if await self._confirm("是否继续到 Stage 5？（否则停留在 Stage 4）"):
                        break  # proceed with current (failed) report
                    return await self.rollback(ctx)

            print(
                f"\n[Stage 4] Auto-repair attempt {attempt}"
                f"/{self.MAX_REPAIR_ATTEMPTS}..."
                if is_auto else
                f"\n[Stage 4] Extra repair attempt {attempt}..."
            )
            report = await self._attempt_repair(ctx, report, sim_data, attempt)

            # Auto-tuning fallback: if LLM repair didn't fix torque/velocity failures,
            # automatically adjust actuator parameters independent of LLM output.
            if not report.passed and attempt < self.MAX_REPAIR_ATTEMPTS:
                auto_overrides = self._autotune_actuators(report, sim_data.n_dof, attempt)
                if auto_overrides:
                    kp_val = auto_overrides.get("0", {}).get("kp", "?")
                    kv_val = auto_overrides.get("0", {}).get("kv", "?")
                    print(f"  Auto-tuning actuators (kp≈{kp_val}, kv≈{kv_val})...")
                    arm = ctx.technical_approach.arm_parameters if ctx.technical_approach else None
                    urdf_path, mjcf_path = await self._prepare_models(
                        ctx, code, arm, auto_overrides,
                    )
                    traj_cmd = self._extract_trajectory(code)
                    sim_data = await simulator.run_trajectory(mjcf_path, traj_cmd)
                    sim_data.arm_model = arm or {}
                    if sim_data.kinematic_only:
                        break
                    engine = ValidationEngine()
                    report = await engine.validate(sim_data)
                    # Re-validate code execution after auto-tuning (code unchanged but
                    # re-extracted trajectory uses Phase 1.5 execution)
                    code_metrics = self._validate_code_execution(code)
                    if code_metrics:
                        report = self._merge_code_metrics(report, code_metrics)

            self._display_results(report)

        # 7. Generate report
        ctx.simulation_report = self._to_report(report, sim_data)
        self._save_report(ctx, report)

        # 8. User confirmation (skipped if user already chose to proceed in interactive loop)
        print(f"\n[Stage 4] Overall: {'PASS' if report.passed else 'FAIL'}")
        if report.passed:
            if not await self._confirm("Does this simulation result look acceptable?"):
                return await self.rollback(ctx)
        # If report still FAIL, user already chose to proceed to Stage 5 in interactive loop

        logger.info("[Stage 4] Simulation verification complete (passed=%s)", report.passed)
        return ctx

    async def _prepare_models(
        self, ctx: StageContext, code: GeneratedCode, arm: Optional[dict],
        actuator_overrides: Optional[dict] = None,
    ) -> tuple:
        """Prepare URDF and MJCF model files."""
        project_dir = ctx.project_dir
        assets_dir = project_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        urdf_path = assets_dir / "arm.urdf"
        mjcf_path = assets_dir / "arm.mjcf"

        # Generate a simple URDF if none exists
        if not urdf_path.exists():
            self._generate_default_urdf(urdf_path, arm)

        # Always regenerate MJCF from our generator (not MuJoCo URDF converter)
        # to ensure proper actuator settings (kp, kv, forcerange) and damping
        # Pass actuator_overrides from auto-repair to allow tuning kp/kv/forcerange
        self._generate_default_mjcf(mjcf_path, arm, actuator_overrides)

        return urdf_path, mjcf_path

    def _generate_default_urdf(self, path: Path, arm: Optional[dict]):
        """Generate URDF from actual DH parameters for accurate simulation."""
        import xml.etree.ElementTree as ET
        import math

        robot = ET.Element("robot", name="articulate_arm")

        # Base link
        base_link = ET.SubElement(robot, "link", name="base_link")
        base_vis = ET.SubElement(base_link, "visual")
        base_geom = ET.SubElement(base_vis, "geometry")
        ET.SubElement(base_geom, "box", size="0.2 0.2 0.05")
        base_inertial = ET.SubElement(base_link, "inertial")
        ET.SubElement(base_inertial, "mass", value="5.0")
        ET.SubElement(base_inertial, "inertia",
                      ixx="0.1", iyy="0.1", izz="0.1",
                      ixy="0", ixz="0", iyz="0")

        if arm and "dh_params" in arm:
            dh_list = arm["dh_params"]
            jl = arm.get("joint_limits", [])
            dyn = arm.get("dynamics", [])

            for i, dh in enumerate(dh_list):
                a = dh.get("a", 0)
                alpha = dh.get("alpha", 0)
                d_val = dh.get("d", 0)

                # Link geometry sized by actual link length
                link_len = abs(a) if abs(a) > 0.01 else abs(d_val) * 0.6 if abs(d_val) > 0.01 else 0.1
                radius = 0.03

                link = ET.SubElement(robot, "link", name=f"link_{i}")
                vis = ET.SubElement(link, "visual")
                vis_geom = ET.SubElement(vis, "geometry")
                ET.SubElement(vis_geom, "cylinder", length=str(link_len), radius=str(radius))
                ET.SubElement(vis, "origin", xyz=f"{link_len / 2} 0 0", rpy="0 0 0")

                # Collision geometry
                col = ET.SubElement(link, "collision")
                col_geom = ET.SubElement(col, "geometry")
                ET.SubElement(col_geom, "cylinder", length=str(link_len), radius=str(radius))
                ET.SubElement(col, "origin", xyz=f"{link_len / 2} 0 0", rpy="0 0 0")

                # Inertial from dynamics data (cylinder along x-axis)
                inertial = ET.SubElement(link, "inertial")
                mass = float(dyn[i].get("mass", 0.5)) if i < len(dyn) else 0.5
                ET.SubElement(inertial, "mass", value=str(mass))
                ixx = 0.5 * mass * radius ** 2  # around x (principal axis)
                iyy = mass * link_len ** 2 / 12 + mass * radius ** 2 / 4  # around y
                izz = iyy  # around z (same as y for capsule)
                ET.SubElement(inertial, "inertia",
                              ixx=str(ixx), iyy=str(iyy), izz=str(izz),
                              ixy="0", ixz="0", iyz="0")
                # COM at center of link
                ET.SubElement(inertial, "origin", xyz=f"{link_len / 2} 0 0", rpy="0 0 0")

                # Joint using DH parameters
                joint = ET.SubElement(robot, "joint", name=f"joint_{i}", type="revolute")
                parent_name = f"link_{i - 1}" if i > 0 else "base_link"
                ET.SubElement(joint, "parent", link=parent_name)
                ET.SubElement(joint, "child", link=f"link_{i}")
                ET.SubElement(joint, "origin",
                              xyz=f"{a} 0 {d_val}",
                              rpy=f"{alpha} 0 0")
                ET.SubElement(joint, "axis", xyz="0 0 1")

                if i < len(jl):
                    lim = jl[i]
                    ET.SubElement(joint, "limit",
                                  lower=str(lim.get("lower", -3.14)),
                                  upper=str(lim.get("upper", 3.14)),
                                  velocity=str(lim.get("velocity", 1.0)),
                                  effort=str(lim.get("torque", 100)))

                # Dynamics: friction and damping
                if i < len(dyn):
                    friction = dyn[i].get("friction", 0.0)
                    damping = dyn[i].get("damping", 0.0)
                    if friction > 0 or damping > 0:
                        ET.SubElement(joint, "dynamics",
                                      friction=str(friction),
                                      damping=str(damping))

        tree = ET.ElementTree(robot)
        tree.write(str(path), xml_declaration=True)
        logger.info("Generated DH-parameterized URDF: %s", path)

    def _generate_default_mjcf(self, path: Path, arm: Optional[dict],
                                 actuator_overrides: Optional[dict] = None):
        """Generate MJCF from DH parameters with actuators and sensors.

        Args:
            actuator_overrides: Optional dict of {joint_name_or_index: {kp, kv, forcerange_scale}}
                                to override hardcoded actuator settings. Used by auto-repair.
        """
        import xml.etree.ElementTree as ET
        import math

        mujoco = ET.Element("mujoco", model="articulate_arm")

        # Compiler options
        ET.SubElement(mujoco, "compiler", angle="radian", coordinate="local")

        # Asset definitions
        asset = ET.SubElement(mujoco, "asset")
        ET.SubElement(asset, "texture", name="grid", type="2d",
                      builtin="checker", width="512", height="512",
                      rgb1="0.9 0.9 0.9", rgb2="0.8 0.8 0.8")
        ET.SubElement(asset, "material", name="grid", texture="grid",
                      texrepeat="1 1", texuniform="true")

        worldbody = ET.SubElement(mujoco, "worldbody")
        ET.SubElement(worldbody, "light", pos="0 0 3", directional="true")
        ET.SubElement(worldbody, "geom", name="ground", type="plane",
                      size="1 1 0.01", material="grid")

        n_dof = len(arm.get("dh_params", range(6))) if arm else 6
        dh_list = arm.get("dh_params", []) if arm else []
        jl = arm.get("joint_limits", []) if arm else []
        dyn = arm.get("dynamics", []) if arm else []

        current = worldbody

        for i in range(n_dof):
            a = dh_list[i].get("a", 0) if i < len(dh_list) else 0
            alpha = dh_list[i].get("alpha", 0) if i < len(dh_list) else 0
            d_val = dh_list[i].get("d", 0) if i < len(dh_list) else 0.15

            link_len = abs(a) if abs(a) > 0.01 else abs(d_val) * 0.6 if abs(d_val) > 0.01 else 0.1

            # Body position from DH a and d; orientation from DH alpha (x-axis rotation)
            # This ensures correct kinematic coupling between joints
            body = ET.SubElement(current, "body", name=f"link_{i}",
                                 pos=f"{a} 0 {d_val}",
                                 euler=f"{alpha} 0 0")

            # Joint
            joint_elem = ET.SubElement(body, "joint", name=f"joint_{i}",
                                       type="hinge", axis="0 0 1",
                                       pos="0 0 0")
            if i < len(jl):
                lim = jl[i]
                joint_elem.set("range", f"{lim.get('lower', -3.14)} {lim.get('upper', 3.14)}")

            # Dynamics: friction and damping
            # Note: arm model dynamics data is for ROS2 controllers, not MuJoCo stability.
            # We use conservative defaults for MuJoCo to prevent unstable oscillations.
            friction = float(dyn[i].get("friction", 0.0)) if i < len(dyn) else 0.0
            damping = 5.0  # Fixed stable damping for MuJoCo simulation
            if friction > 0:
                joint_elem.set("frictionloss", str(friction))
            if damping > 0:
                joint_elem.set("damping", str(damping))

            # Visual geometry
            ET.SubElement(body, "geom", name=f"link_{i}_geom",
                          type="capsule",
                          fromto=f"0 0 0 {link_len} 0 0",
                          size="0.03",
                          rgba="0.3 0.6 0.9 1",
                          priority="1")

            # Collision geometry (slightly larger for safety)
            ET.SubElement(body, "geom", name=f"link_{i}_collision",
                          type="capsule",
                          fromto=f"0 0 0 {link_len} 0 0",
                          size="0.035",
                          rgba="0 0 0 0",
                          contype="1", conaffinity="1",
                          priority="2")

            # Site for FK measurement (TCP offset only on last link)
            if i == n_dof - 1:
                ET.SubElement(body, "site", name="tcp",
                              pos=f"{link_len} 0 0", size="0.005",
                              type="sphere", rgba="1 0 0 1")

            # Inertial from dynamics data (capsule along x-axis)
            mass = float(dyn[i].get("mass", 0.5)) if i < len(dyn) else 0.5
            radius = 0.03
            ixx = 0.5 * mass * radius ** 2  # around x (principal axis)
            iyy = mass * link_len ** 2 / 12 + mass * radius ** 2 / 4  # around y
            izz = iyy  # around z (same as y for capsule on x)
            # COM at center of link
            com_x = link_len / 2
            ET.SubElement(body, "inertial", pos=f"{com_x} 0 0",
                          mass=str(mass),
                          fullinertia=f"{ixx} {iyy} {izz} 0 0 0")

            current = body

        # Actuators with torque-based force limits
        # kp=20 provides responsive tracking; kv=0.5 keeps active damping low so
        # actuator force stays within rated torque limits. Passive joint damping
        # (set to 5.0) handles oscillation suppression without affecting actuatorfrc.
        # Auto-repair can override via actuator_overrides for aggressive trajectories.
        actuator = ET.SubElement(mujoco, "actuator")
        for i in range(n_dof):
            torque_limit = float(jl[i].get("torque", 100)) if i < len(jl) else 100
            kp = str(actuator_overrides.get(str(i), {}).get("kp", 20)) if actuator_overrides else "20"
            kv = str(actuator_overrides.get(str(i), {}).get("kv", 0.5)) if actuator_overrides else "0.5"
            scale = float(actuator_overrides.get(str(i), {}).get("forcerange_scale", 1.0)) if actuator_overrides else 1.0
            adj_limit = torque_limit * scale
            ET.SubElement(actuator, "position", name=f"actuator_{i}",
                          joint=f"joint_{i}", kp=kp,
                          kv=kv,
                          forcelimited="true",
                          forcerange=f"-{adj_limit} {adj_limit}")

        # Sensors
        sensor = ET.SubElement(mujoco, "sensor")
        for i in range(n_dof):
            ET.SubElement(sensor, "jointpos", name=f"jpos_{i}", joint=f"joint_{i}")
            ET.SubElement(sensor, "jointvel", name=f"jvel_{i}", joint=f"joint_{i}")
            ET.SubElement(sensor, "actuatorfrc", name=f"jtorque_{i}", actuator=f"actuator_{i}")

        # Option/visual
        ET.SubElement(mujoco, "option", timestep="0.001", gravity="0 0 -9.81")
        visual = ET.SubElement(mujoco, "visual")
        ET.SubElement(visual, "quality", shadowsize="4096")
        ET.SubElement(visual, "global", offwidth="1280", offheight="720")

        tree = ET.ElementTree(mujoco)
        tree.write(str(path), xml_declaration=True)
        logger.info("Generated DH-parameterized MJCF with actuators: %s", path)

    def _import_from_source(
        self, source_code: str, module_name: str, class_name: str,
    ) -> Tuple[Optional[object], Optional[callable]]:
        """Write source to temp dir, import, instantiate class.

        Returns (instance, cleanup_fn) on success, (None, None) on failure.
        cleanup_fn restores sys.path and removes temp directory.
        """
        tmpdir = tempfile.TemporaryDirectory()
        try:
            src_path = Path(tmpdir.name) / f"{module_name}.py"
            src_path.write_text(source_code, encoding="utf-8")
            old_path = sys.path.copy()
            sys.path.insert(0, tmpdir.name)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            instance = cls()

            def cleanup():
                if tmpdir.name in sys.path:
                    sys.path.remove(tmpdir.name)
                if module_name in sys.modules:
                    del sys.modules[module_name]
                tmpdir.cleanup()
                sys.path[:] = old_path

            return instance, cleanup
        except Exception:
            tmpdir.cleanup()
            logger.warning(
                "Failed to import %s.%s: %s",
                module_name, class_name, traceback.format_exc()[:200],
            )
            return None, None

    def _execute_trajectory_planner(
        self, code: GeneratedCode, waypoints: list,
    ) -> Optional[TrajectoryCommand]:
        """Import generated TrajectoryPlanner and call plan_ptp between waypoints.

        Validates the generated code actually runs and produces a valid trajectory.
        Returns None on failure (caller falls back to S-curve interpolation).
        """
        tp_key = next(
            (k for k in code.package_structure if "trajectory_planner" in k.lower()),
            None,
        )
        if tp_key is None:
            return None

        planner, cleanup = self._import_from_source(
            code.package_structure[tp_key], "trajectory_planner", "TrajectoryPlanner",
        )
        if planner is None:
            return None

        try:
            all_positions = []
            all_times = []
            time_offset = 0.0
            dt = 0.01

            for i in range(len(waypoints) - 1):
                start = waypoints[i].tolist()
                goal = waypoints[i + 1].tolist()
                segment = planner.plan_ptp(start, goal, dt=dt)

                if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                    continue

                for pt in segment:
                    pos = pt.get("positions", pt.get("position"))
                    t = pt.get("time", 0.0)
                    if pos is not None:
                        all_positions.append(np.array(pos[:6], dtype=float))
                        all_times.append(t + time_offset)

                time_offset = all_times[-1] if all_times else 0.0

            if len(all_positions) < 2:
                return None

            pos_array = np.array(all_positions)
            time_array = np.array(all_times)
            logger.info(
                "Generated trajectory from TrajectoryPlanner: %d steps, %d waypoints",
                len(time_array), len(waypoints),
            )
            return TrajectoryCommand(
                joint_positions=pos_array,
                joint_velocities=np.zeros_like(pos_array),
                time_steps=time_array,
            )
        except Exception:
            logger.warning("TrajectoryPlanner execution failed: %s", traceback.format_exc()[:200])
            return None
        finally:
            if cleanup:
                cleanup()

    def _extract_trajectory(self, code: GeneratedCode) -> TrajectoryCommand:
        """Extract trajectory data from generated code or use DH-based default."""
        import ast
        import re

        all_content = "\n".join(code.package_structure.values())
        n_dof = 6
        positions: list = []

        # Phase 1: AST-based WAYPOINTS extraction (safe, no code execution)
        # Looks for top-level WAYPOINTS = [[...], [...], ...] in trajectory files
        tp_keys = [k for k in code.package_structure if "trajectory" in k.lower()]
        for key in tp_keys:
            try:
                tree = ast.parse(code.package_structure[key])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id in ("WAYPOINTS", "waypoints"):
                                raw = ast.literal_eval(node.value)
                                if isinstance(raw, list) and len(raw) >= 2:
                                    pts = [
                                        np.array(p[:6], dtype=float) for p in raw
                                        if isinstance(p, (list, tuple)) and len(p) >= 6
                                    ]
                                    if len(pts) >= 2:
                                        positions = pts
                                        break
                    if positions:
                        break
            except (SyntaxError, ValueError, TypeError, AttributeError):
                continue
            if positions:
                break

        # Phase 1.5: Try executing generated TrajectoryPlanner code with found waypoints
        if positions:
            traj_cmd = self._execute_trajectory_planner(code, positions)
            if traj_cmd is not None:
                return traj_cmd
            logger.warning("TrajectoryPlanner execution failed, falling back to S-curve interpolation")

        # Phase 2: Fall back to generic regex scan across all files
        if not positions:
            for match in re.finditer(r'\[([\d.,\s-]+)\]', all_content):
                try:
                    vals = [float(x) for x in match.group(1).split(",") if x.strip()]
                    if len(vals) >= 6:
                        positions.append(np.array(vals[:6]))
                except ValueError:
                    continue

        # Phase 3: Default S-curve if no waypoints found
        if not positions:
            # Generate default trajectory: smooth S-curve from 0 to 0.05 rad
            # Conservative amplitude (~3 deg) ensures torque limits are respected
            # on any arm model without overloading distal joints
            duration = 5.0
            n_pts = 100
            time_steps = np.linspace(0, duration, n_pts)
            pos_array = np.zeros((n_pts, n_dof))

            # Quintic S-curve: s(t) = 10(t/T)³ - 15(t/T)⁴ + 6(t/T)⁵
            # Max velocity at t=T/2: ds/dt = 1.875/T, acceleration limited
            for i, t in enumerate(time_steps):
                s = t / duration
                if s <= 0:
                    s_curve = 0.0
                elif s >= 1:
                    s_curve = 1.0
                else:
                    s_curve = (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
                pos_array[i] = 0.05 * s_curve

            return TrajectoryCommand(
                joint_positions=pos_array,
                joint_velocities=np.zeros_like(pos_array),
                time_steps=time_steps,
            )

        n_pts = len(positions)
        duration = max(5.0, n_pts * 2.0)
        time_steps = np.linspace(0, duration, max(n_pts * 20, 100))
        pos_array = np.zeros((len(time_steps), n_dof))

        # Quintic S-curve interpolation between waypoints
        for i, t in enumerate(time_steps):
            frac = t / duration if duration > 0 else 0
            raw_idx = frac * (n_pts - 1)
            idx = min(int(raw_idx), n_pts - 2)
            local_frac = raw_idx - idx
            if idx < n_pts - 1:
                # Apply S-curve to local_frac for smooth acceleration
                s = local_frac
                s_curve = (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
                pos_array[i] = positions[idx] + s_curve * (positions[idx + 1] - positions[idx])
            else:
                pos_array[i] = positions[-1]

        return TrajectoryCommand(
            joint_positions=pos_array,
            joint_velocities=np.zeros_like(pos_array),
            time_steps=time_steps,
        )

    def _validate_code_structure(self, code: GeneratedCode) -> Dict[str, MetricResult]:
        """Check generated code structure: syntax validity and required symbols.

        Phase 3: AST-based structural validation — no code execution.
        Returns dict of MetricResult for each structural check.
        """
        import ast

        syntax_errors = 0
        has_planner = False
        has_kinematics = False
        has_plan_ptp = False
        has_forward = False
        has_inverse = False

        for path, content in code.package_structure.items():
            if not content.strip():
                continue
            # Only validate .py files with ast.parse — YAML/XML configs aren't Python
            if not path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if node.name == "TrajectoryPlanner":
                            has_planner = True
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef) and item.name == "plan_ptp":
                                    has_plan_ptp = True
                        if node.name == "ArmKinematics":
                            has_kinematics = True
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    if item.name == "forward":
                                        has_forward = True
                                    if item.name == "inverse":
                                        has_inverse = True
            except SyntaxError:
                syntax_errors += 1

        missing = []
        if not has_planner:
            missing.append("TrajectoryPlanner")
        if not has_kinematics:
            missing.append("ArmKinematics")
        if not has_plan_ptp:
            missing.append("plan_ptp")
        if not has_forward:
            missing.append("forward_kinematics")
        if not has_inverse:
            missing.append("inverse_kinematics")

        results = {}
        results["code_syntax_valid"] = MetricResult(
            name="code_syntax_valid",
            passed=syntax_errors == 0,
            value=float(syntax_errors),
            threshold=0.5,
            unit="errors",
            explanation=(
                f"{syntax_errors} file(s) with syntax errors"
                if syntax_errors > 0 else
                "All files parse without syntax errors"
            ),
        )
        results["code_required_symbols"] = MetricResult(
            name="code_required_symbols",
            passed=len(missing) == 0,
            value=float(len(missing)),
            threshold=0.5,
            unit="missing",
            explanation=(
                f"Missing: {', '.join(missing)}"
                if missing else
                "All required symbols present"
            ),
        )
        return results

    def _validate_kinematics(self, code: GeneratedCode) -> Dict[str, MetricResult]:
        """Import generated ArmKinematics and test FK/IK round-trip.

        Phase 2: Code execution validation — verifies forward and inverse
        kinematics produce valid transformations.
        """
        kin_key = next(
            (k for k in code.package_structure if "kinematics" in k.lower()),
            None,
        )

        if kin_key is None:
            return {
                "kinematics_fk": MetricResult(
                    name="kinematics_fk", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="No kinematics file found",
                ),
                "kinematics_ik_roundtrip": MetricResult(
                    name="kinematics_ik_roundtrip", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="No kinematics file found",
                ),
                "kinematics_ik_orientation": MetricResult(
                    name="kinematics_ik_orientation", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="No kinematics file found",
                ),
            }

        kin, cleanup = self._import_from_source(
            code.package_structure[kin_key], "arm_kinematics", "ArmKinematics",
        )
        if kin is None:
            return {
                "kinematics_fk": MetricResult(
                    name="kinematics_fk", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="Failed to import ArmKinematics",
                ),
                "kinematics_ik_roundtrip": MetricResult(
                    name="kinematics_ik_roundtrip", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="Failed to import ArmKinematics",
                ),
            }

        try:
            n_dof = len(kin.dh_params) if hasattr(kin, 'dh_params') else 6
            test_angles = [
                [0.0] * n_dof,
            ]
            if n_dof >= 1:
                test_angles.append(
                    [0.5 if j == 0 else 0.0 for j in range(n_dof)]
                )
            if n_dof >= 3:
                test_angles.append(
                    [0.5 if j == 0 else (-0.2 if j == 1 else (0.3 if j == 2 else 0.0))
                     for j in range(n_dof)]
                )

            fk_ok = True
            last_pose = None
            for angles in test_angles:
                T = kin.forward(angles)
                if not isinstance(T, np.ndarray) or T.shape != (4, 4):
                    fk_ok = False
                    break
                last_pose = T

            results = {}
            results["kinematics_fk"] = MetricResult(
                name="kinematics_fk", passed=fk_ok,
                value=0.0 if fk_ok else 1.0,
                threshold=0.5, unit="bool",
                explanation=(
                    "Forward kinematics returns 4x4 matrix"
                    if fk_ok else
                    "Forward kinematics did not return valid 4x4 matrix"
                ),
            )

            if fk_ok and last_pose is not None:
                ik_result = kin.inverse(last_pose)
                if isinstance(ik_result, tuple) and len(ik_result) == 2:
                    q_ik, _ = ik_result
                else:
                    q_ik = ik_result

                q_ik_arr = np.asarray(q_ik, dtype=float).ravel()
                T_round = kin.forward(q_ik_arr.tolist())

                # Position round-trip error
                pos_error = float(np.linalg.norm(T_round[:3, 3] - last_pose[:3, 3]))
                ik_ok = pos_error < 1e-3
                results["kinematics_ik_roundtrip"] = MetricResult(
                    name="kinematics_ik_roundtrip", passed=ik_ok,
                    value=pos_error,
                    threshold=0.001, unit="m",
                    explanation=(
                        f"IK round-trip position error: {pos_error:.6f}m"
                        if ik_ok else
                        f"IK round-trip error too large: {pos_error:.6f}m"
                    ),
                )

                # Orientation round-trip error
                R_round = T_round[:3, :3]
                R_target = last_pose[:3, :3]
                R_diff = R_round.T @ R_target
                cos_angle = max(-1.0, min(1.0, (np.trace(R_diff) - 1.0) / 2.0))
                orient_error = float(math.acos(cos_angle))
                orient_ok = orient_error < 0.017  # < 1 degree
                results["kinematics_ik_orientation"] = MetricResult(
                    name="kinematics_ik_orientation", passed=orient_ok,
                    value=orient_error,
                    threshold=0.017, unit="rad",
                    explanation=(
                        f"IK round-trip orientation error: {orient_error:.6f} rad ({math.degrees(orient_error):.3f} deg)"
                        if orient_ok else
                        f"IK round-trip orientation error too large: {orient_error:.6f} rad"
                    ),
                )
            else:
                results["kinematics_ik_roundtrip"] = MetricResult(
                    name="kinematics_ik_roundtrip", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="IK round-trip skipped — FK failed",
                )
                results["kinematics_ik_orientation"] = MetricResult(
                    name="kinematics_ik_orientation", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="IK orientation skipped — FK failed",
                )

            return results
        except Exception:
            logger.warning("Kinematics validation failed: %s", traceback.format_exc()[:200])
            return {
                "kinematics_fk": MetricResult(
                    name="kinematics_fk", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="Kinematics execution error",
                ),
                "kinematics_ik_roundtrip": MetricResult(
                    name="kinematics_ik_roundtrip", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="Kinematics execution error",
                ),
                "kinematics_ik_orientation": MetricResult(
                    name="kinematics_ik_orientation", passed=False,
                    value=1.0, threshold=0.5, unit="bool",
                    explanation="Kinematics execution error",
                ),
            }
        finally:
            if cleanup:
                cleanup()

    def _validate_code_execution(self, code: GeneratedCode) -> Dict[str, MetricResult]:
        """Orchestrate all code execution validation checks.

        Runs structural analysis (AST), kinematics execution (FK/IK),
        and returns merged results dict.
        """
        metrics = {}
        metrics.update(self._validate_code_structure(code))
        metrics.update(self._validate_kinematics(code))
        return metrics

    def _merge_code_metrics(
        self, report: ValidationReport, code_metrics: Dict[str, MetricResult],
    ) -> ValidationReport:
        """Merge code validation metrics into the simulation validation report."""
        merged = dict(report.metrics)
        merged.update(code_metrics)
        return ValidationReport(
            passed=report.passed,
            metrics=merged,
            summary=report.summary,
            recommendations=report.recommendations,
        )

    def _autotune_actuators(
        self, report: ValidationReport, n_dof: int, attempt: int,
    ) -> Optional[dict]:
        """Compute automatic actuator overrides from failed metrics.

        Independent of LLM — directly adjusts kp/kv/forcerange_scale based on
        which metric categories failed. Called when LLM repair doesn't resolve
        torque/velocity-related failures.
        """
        failed_names = {m.name for m in report.metrics.values() if not m.passed}
        if not failed_names:
            return None

        torque_fail = "joint_torque_peak" in failed_names
        velocity_fail = "joint_velocity_overshoot" in failed_names
        accel_fail = "joint_acceleration_peak" in failed_names

        if not (torque_fail or velocity_fail or accel_fail):
            return None

        # Use per-joint torque data if available to target specific joints
        torque_metrics = {}
        for m in report.metrics.values():
            if m.name == "joint_torque_peak":
                torque_metrics = getattr(m, "data", None)
                break

        overrides = {}
        for j in range(n_dof):
            entry = {}
            if torque_fail:
                # Progressive kp reduction: 20 → 15 → 10 (smoother than old 20→12→7)
                kp = max(8, int(20 - (attempt - 1) * 5))
                entry["kp"] = kp
            if accel_fail and not torque_fail:
                # Only acceleration issue: gentle kp reduction
                kp = max(10, int(20 - (attempt - 1) * 3))
                entry["kp"] = kp
            if velocity_fail:
                # Moderate kv increase for velocity damping: 0.5 → 1.0 → 2.0
                kv = 0.5 * (2 ** (attempt - 1))
                entry["kv"] = min(kv, 3.0)
            if torque_fail and attempt >= 3:
                # After multiple failed attempts, allow slight torque headroom
                entry["forcerange_scale"] = 1.1
            if entry:
                overrides[str(j)] = entry

        return overrides if overrides else None

    async def _attempt_repair(
        self, ctx: StageContext, report: ValidationReport,
        sim_data: SimulationData, attempt: int,
    ) -> ValidationReport:
        """Auto-repair: LLM modifies generated code to fix simulation failures."""
        failed = [m for m in report.metrics.values() if not m.passed]
        if not failed:
            return report

        code = ctx.generated_code
        if not code or not code.package_structure:
            logger.warning("No generated code to repair")
            return report

        # 1. LLM generates code fixes based on failed metrics
        print(f"  Analyzing failures and generating code fix (attempt {attempt})...")
        try:
            repair_prompt, repair_msg = self.skill.prompt_mgr.render(
                "code_repair",
                failed_metrics=[{
                    "name": m.name, "value": m.value,
                    "threshold": m.threshold, "unit": m.unit,
                } for m in failed],
                code_files=code.package_structure,
            )

            from pydantic import BaseModel, Field

            class RepairSchema(BaseModel):
                files: dict
                explanation: str
                actuator_overrides: Optional[dict] = Field(
                    default=None,
                    description=(
                        "Per-joint actuator settings for MJCF simulation model. "
                        'Key is joint index as string "0", "1", etc. '
                        "Each value has optional: kp (position gain, default 20), "
                        "kv (velocity gain, default 0.5), "
                        "forcerange_scale (torque limit multiplier, default 1.0). "
                        "Only include joints that need changes."
                    ),
                )

            repair = await self.llm.complete_structured(
                system=repair_prompt,
                messages=[{"role": "user", "content": repair_msg}],
                output_model=RepairSchema,
                max_tokens=8192,
                temperature=0.2,
            )

            if not repair.files:
                print("  LLM did not suggest any changes.")
                return report

            # 2. Apply changes to generated code
            modified = 0
            for file_path, new_content in repair.files.items():
                if file_path in code.package_structure:
                    old_content = code.package_structure[file_path]
                    if new_content != old_content:
                        code.package_structure[file_path] = new_content
                        modified += 1

            if modified == 0:
                print("  LLM suggested changes, but none differ from current code.")
                return report

            changed = list(repair.files.keys())
            print(f"  Modified {modified} file(s): {', '.join(changed[:3])}{'...' if len(changed) > 3 else ''}")
            print(f"  Reason: {repair.explanation[:120]}")

            # Validate repaired code before re-running simulation
            code_metrics = self._validate_code_execution(code)
            if code_metrics:
                code_failures = [m for m in code_metrics.values() if not m.passed]
                if code_failures:
                    print(f"  [yellow]Repair introduced {len(code_failures)} code issue(s):[/yellow]")
                    for m in code_failures:
                        print(f"    - {m.name}: {m.explanation}")
                    # Only abort if syntax or required symbols are broken
                    critical = {m.name for m in code_failures
                                if m.name in ("code_syntax_valid", "code_required_symbols")}
                    if critical and attempt < self.MAX_REPAIR_ATTEMPTS:
                        print("  [red]Critical code issues detected. Reverting repair.[/red]")
                        return report

        except Exception as e:
            logger.warning("LLM code repair failed: %s", e)
            print(f"  Code repair unavailable: {e}")
            return report

        # 3. Check for MJCF actuator overrides from LLM repair
        actuator_overrides = getattr(repair, 'actuator_overrides', None)

        # 4. Re-generate models (using actuator overrides if provided)
        print(f"  Re-running simulation (attempt {attempt})...")
        simulator = MuJoCoSimulator()
        arm = ctx.technical_approach.arm_parameters if ctx.technical_approach else None
        urdf_path, mjcf_path = await self._prepare_models(
            ctx, code, arm, actuator_overrides,
        )
        traj_cmd = self._extract_trajectory(code)
        sim_data = await simulator.run_trajectory(mjcf_path, traj_cmd)
        sim_data.arm_model = arm or {}
        if sim_data.kinematic_only:
            print("  [yellow](estimated kinematics — no physics)[/yellow]")

        # 5. Re-validate
        engine = ValidationEngine()
        new_report = await engine.validate(sim_data)

        # 5.5 Re-validate code execution after repair
        code_metrics = self._validate_code_execution(code)
        if code_metrics:
            new_report = self._merge_code_metrics(new_report, code_metrics)

        return new_report

    def _display_results(self, report: ValidationReport):
        """Display validation results in console."""
        print(f"\n  Validation: {'PASS' if report.passed else 'FAIL'}")
        print(f"  Pass rate: {report.pass_rate:.0%} ({len(report.passed_metrics)}/{len(report.metrics)})")

        for name, result in sorted(report.metrics.items()):
            icon = "OK" if result.passed else "FAIL"
            print(f"    [{icon:4s}] {name}: {result.value:.4f} (threshold: {result.threshold})")

    def _to_report(self, report: ValidationReport, sim_data: Optional[SimulationData] = None) -> SimulationReport:
        """Convert ValidationReport to SimulationReport."""
        from articulate_core.pipeline.models import MetricResult as ModelMetricResult
        from articulate_core.pipeline.models import RepairAttempt as ModelRepairAttempt

        metrics = {}
        for name, result in report.metrics.items():
            metrics[name] = ModelMetricResult(
                name=result.name, passed=result.passed,
                value=result.value, threshold=result.threshold,
                unit=result.unit, explanation=result.explanation,
            )

        return SimulationReport(
            passed=report.passed,
            metrics=metrics,
            summary=report.summary,
            recommendations=report.recommendations,
            kinematic_only=sim_data.kinematic_only if sim_data else False,
        )

    def _save_report(self, ctx: StageContext, report: ValidationReport):
        """Save validation report to project directory."""
        report_dir = ctx.project_dir / "deploy"
        report_dir.mkdir(parents=True, exist_ok=True)

        engine = ValidationEngine()

        md = engine.generate_report_markdown(report)
        (report_dir / "validation_report.md").write_text(md, encoding="utf-8")

        html = engine.generate_report_html(report)
        (report_dir / "validation_report.html").write_text(html, encoding="utf-8")

        print(f"\n  Report saved to: {report_dir}/validation_report.{'md'}, .html")

    async def _confirm(self, msg: str) -> bool:
        import click
        return click.confirm(f"\n{msg}", default=True)

    async def rollback(self, ctx: StageContext) -> StageContext:
        ctx.simulation_report = None
        ctx.should_continue = False
        return ctx
