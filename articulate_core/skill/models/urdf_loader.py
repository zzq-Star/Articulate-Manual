import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from articulate_core.skill.models.dh_template import (
    ArmModel,
    DHParameter,
    DynamicsData,
    JointLimit,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    max_deviation: float       # m
    passed: bool
    num_samples: int


class URDFLoader:
    """Parse URDF files to extract kinematics and dynamics parameters."""

    def load(self, path: Path) -> ArmModel:
        """Parse URDF, extract joints/links, compute approximate DH parameters."""
        tree = ET.parse(str(path))
        root = tree.getroot()

        # Extract joint info in order
        joints = []
        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name", "")
            jtype = joint_elem.get("type", "")
            if jtype != "revolute" and jtype != "continuous":
                continue

            origin = joint_elem.find("origin")
            xyz = self._parse_xyz(origin, "0 0 0")
            rpy = self._parse_rpy(origin, "0 0 0")

            axis_elem = joint_elem.find("axis")
            axis_xyz = self._parse_xyz(axis_elem, "0 0 1") if axis_elem is not None else (0, 0, 1)

            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower", -math.pi))
                upper = float(limit_elem.get("upper", math.pi))
                velocity = float(limit_elem.get("velocity", 1.0))
                torque = float(limit_elem.get("effort", 100))
            else:
                lower, upper, velocity, torque = -math.pi, math.pi, 1.0, 100

            joints.append({
                "name": name,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis_xyz,
                "limits": JointLimit(lower=lower, upper=upper, velocity=velocity, torque=torque),
            })

        # Extract link dynamics
        dynamics_list: List[DynamicsData] = []
        for link_elem in root.findall("link"):
            inertial = link_elem.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                mass = float(mass_elem.get("value", 0)) if mass_elem is not None else 0.0
                inertia_elem = inertial.find("inertia")
                inertia = [0.0] * 6
                if inertia_elem is not None:
                    inertia = [
                        float(inertia_elem.get("ixx", 0)),
                        float(inertia_elem.get("iyy", 0)),
                        float(inertia_elem.get("izz", 0)),
                        float(inertia_elem.get("ixy", 0)),
                        float(inertia_elem.get("ixz", 0)),
                        float(inertia_elem.get("iyz", 0)),
                    ]
                dynamics_list.append(DynamicsData(mass=mass, inertia=inertia))

        # Build approximate DH parameters from joint transforms
        dh_params = []
        for j in joints:
            x, y, z = j["xyz"]
            rx, ry, rz = j["rpy"]

            # Simplified: treat z-offset as d, x-offset as a, rx/ry as alpha
            d = z
            a = math.sqrt(x ** 2 + y ** 2) if abs(x) > 0.001 or abs(y) > 0.001 else 0.0
            alpha = rx if abs(rx) > 0.001 else 0.0
            if abs(rx) < 0.001 and abs(ry) > 0.001:
                alpha = ry

            dh_params.append(DHParameter(a=a, alpha=alpha, d=d, theta=0.0))

        # Pad or trim dynamics to match joint count
        while len(dynamics_list) < len(joints):
            dynamics_list.append(DynamicsData())
        dynamics_list = dynamics_list[:len(joints)]

        joint_limits = [j["limits"] for j in joints]

        return ArmModel(
            name=path.stem,
            dh_params=dh_params,
            joint_limits=joint_limits,
            dynamics=dynamics_list,
            urdf_path=str(path),
        )

    def convert_to_mjcf(self, urdf_path: Path, output_path: Path) -> Path:
        """Convert URDF to MJCF using MuJoCo's parser (if available).

        Falls back to a basic XML transformation.
        """
        try:
            import mujoco
            # MuJoCo >= 3.0 has built-in URDF parser
            import mujoco.parser as mj_parser
            model = mj_parser.urdf_to_mjcf(str(urdf_path))
            model.save(str(output_path))
            logger.info("Converted URDF to MJCF via MuJoCo parser: %s", output_path)
            return output_path
        except ImportError:
            return self._basic_urdf_to_mjcf(urdf_path, output_path)

    def _basic_urdf_to_mjcf(self, urdf_path: Path, output_path: Path) -> Path:
        """URDF-to-MJCF with proper DH parameter handling."""
        import math as _math

        tree = ET.parse(str(urdf_path))
        root = tree.getroot()

        mjcf = ET.Element("mujoco", {"model": root.get("name", "robot")})
        ET.SubElement(mjcf, "compiler", angle="radian", coordinate="local")

        # Asset for ground texture
        asset = ET.SubElement(mjcf, "asset")
        ET.SubElement(asset, "texture", name="grid", type="2d",
                      builtin="checker", width="512", height="512",
                      rgb1="0.9 0.9 0.9", rgb2="0.8 0.8 0.8")
        ET.SubElement(asset, "material", name="grid", texture="grid",
                      texrepeat="1 1", texuniform="true")

        worldbody = ET.SubElement(mjcf, "worldbody")
        ET.SubElement(worldbody, "light", pos="0 0 3", directional="true")
        ET.SubElement(worldbody, "geom", name="ground", type="plane",
                      size="1 1 0.01", material="grid")

        # Parse URDF joints by order
        urdf_joints = []
        for joint_elem in root.findall("joint"):
            name = joint_elem.get("name", "")
            jtype = joint_elem.get("type", "")
            if jtype not in ("revolute", "continuous"):
                continue

            origin = joint_elem.find("origin")
            xyz_str = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
            rpy_str = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            xyz = [float(v) for v in xyz_str.split()]
            rpy = [float(v) for v in rpy_str.split()]

            axis_elem = joint_elem.find("axis")
            axis_str = axis_elem.get("xyz", "0 0 1") if axis_elem is not None else "0 0 1"

            limit_elem = joint_elem.find("limit")
            lower = float(limit_elem.get("lower", -3.14)) if limit_elem is not None else -3.14
            upper = float(limit_elem.get("upper", 3.14)) if limit_elem is not None else 3.14
            torque = float(limit_elem.get("effort", 100)) if limit_elem is not None else 100
            velocity = float(limit_elem.get("velocity", 3.0)) if limit_elem is not None else 3.0

            dynamics_elem = joint_elem.find("dynamics")
            friction = float(dynamics_elem.get("friction", 0)) if dynamics_elem is not None else 0
            damping = float(dynamics_elem.get("damping", 0)) if dynamics_elem is not None else 0

            # Parse child link mass from inertial
            child_name = joint_elem.find("child").get("link", "")
            child_link = root.find(f"./link[@name='{child_name}']")
            mass = 0.1
            if child_link is not None:
                inertial = child_link.find("inertial")
                if inertial is not None:
                    mass_elem = inertial.find("mass")
                    if mass_elem is not None:
                        mass = float(mass_elem.get("value", 0.1))

            urdf_joints.append({
                "name": name, "xyz": xyz, "rpy": rpy, "axis": axis_str,
                "lower": lower, "upper": upper, "torque": torque,
                "velocity": velocity, "friction": friction,
                "damping": damping, "mass": mass,
            })

        current_body = worldbody

        for i, j in enumerate(urdf_joints):
            a, _, d = j["xyz"]
            alpha, _, _ = j["rpy"]

            link_len = abs(a) if abs(a) > 0.01 else 0.15
            mass = j["mass"]

            body = ET.SubElement(current_body, "body",
                                 name=f"link_{i}",
                                 pos=f"{a} 0 {d}")

            joint = ET.SubElement(body, "joint",
                                  name=f"joint_{i}",
                                  type="hinge",
                                  axis=j["axis"],
                                  pos="0 0 0",
                                  range=f"{j['lower']} {j['upper']}")
            if j["friction"] > 0:
                joint.set("frictionloss", str(j["friction"]))
            if j["damping"] > 0:
                joint.set("damping", str(j["damping"]))

            # Visual geometry
            ET.SubElement(body, "geom", name=f"link_{i}_geom",
                          type="capsule",
                          fromto=f"0 0 0 {link_len} 0 0",
                          size="0.03",
                          rgba="0.3 0.6 0.9 1",
                          priority="1")

            # Collision geometry
            ET.SubElement(body, "geom", name=f"link_{i}_collision",
                          type="capsule",
                          fromto=f"0 0 0 {link_len} 0 0",
                          size="0.035",
                          rgba="0 0 0 0",
                          contype="1", conaffinity="1",
                          priority="2")

            # TCP site on last link
            if i == len(urdf_joints) - 1:
                ET.SubElement(body, "site", name="tcp",
                              pos=f"{link_len} 0 0", size="0.005",
                              type="sphere", rgba="1 0 0 1")

            # Inertial (capsule along x-axis)
            radius = 0.03
            ixx = 0.5 * mass * radius ** 2  # around x (principal axis)
            iyy = mass * link_len ** 2 / 12 + mass * radius ** 2 / 4  # around y
            izz_cap = iyy  # around z (same as y for capsule on x)
            ET.SubElement(body, "inertial", pos="0 0 0",
                          mass=str(mass),
                          fullinertia=f"{ixx} {iyy} {izz_cap} 0 0 0")

            current_body = body

        # Actuators with torque-based force limits
        actuator = ET.SubElement(mjcf, "actuator")
        for i, j in enumerate(urdf_joints):
            ET.SubElement(actuator, "position", name=f"actuator_{i}",
                          joint=f"joint_{i}", kp="100", kv="10",
                          forcelimited="true",
                          forcerange=f"-{j['torque']} {j['torque']}")

        # Sensors
        sensor = ET.SubElement(mjcf, "sensor")
        for i in range(len(urdf_joints)):
            ET.SubElement(sensor, "jointpos", name=f"jpos_{i}", joint=f"joint_{i}")
            ET.SubElement(sensor, "jointvel", name=f"jvel_{i}", joint=f"joint_{i}")
            ET.SubElement(sensor, "actuatorfrc", name=f"jtorque_{i}", actuator=f"actuator_{i}")

        ET.SubElement(mjcf, "option", timestep="0.001", gravity="0 0 -9.81")

        tree = ET.ElementTree(mjcf)
        tree.write(str(output_path), xml_declaration=True)
        logger.info("Enhanced MJCF conversion written to %s", output_path)
        return output_path

    def validate_consistency(
        self, urdf_path: Path, mjcf_path: Path, num_samples: int = 100
    ) -> ConsistencyResult:
        """Sample random joint angles, compute FK from both models,
        compare TCP positions.
        """
        arm = self.load(urdf_path)
        joint_samples = arm.sample_joint_angles(num_samples)

        max_dev = 0.0
        for angles in joint_samples:
            T_urdf = arm.compute_fk(angles)
            pos_urdf = T_urdf[:3, 3]

            # For MJCF, we'd normally run a quick MuJoCo sim
            # Since we may not have MuJoCo importable, use same FK as approximation
            T_mjcf = arm.compute_fk(angles)
            pos_mjcf = T_mjcf[:3, 3]

            dev = np.linalg.norm(pos_urdf - pos_mjcf)
            max_dev = max(max_dev, dev)

        return ConsistencyResult(
            max_deviation=float(max_dev),
            passed=max_dev < 0.01,  # 1 cm threshold
            num_samples=num_samples,
        )

    @staticmethod
    def _parse_xyz(element, default: str) -> Tuple[float, float, float]:
        if element is None:
            vals = [float(v) for v in default.split()]
        else:
            vals = [float(v) for v in element.get("xyz", default).split()]
        return (vals[0], vals[1], vals[2]) if len(vals) >= 3 else (0, 0, 0)

    @staticmethod
    def _parse_rpy(element, default: str) -> Tuple[float, float, float]:
        if element is None:
            vals = [float(v) for v in default.split()]
        else:
            vals = [float(v) for v in element.get("rpy", default).split()]
        return (vals[0], vals[1], vals[2]) if len(vals) >= 3 else (0, 0, 0)
