"""Preset arm definitions with DH parameters."""

from articulate_core.skill.models.dh_template import ArmModel, DHParameter, JointLimit, DynamicsData

PRESET_ARMS = {}

# ─── Six-DOF Standard Industrial Arm ─────────────────────────────────

six_dof_standard = ArmModel(
    name="six_dof_standard",
    dh_params=[
        DHParameter(a=0,     alpha=-1.5708, d=0.3,   theta=0),
        DHParameter(a=0.4,   alpha=0,       d=0,     theta=0),
        DHParameter(a=0.35,  alpha=0,       d=0,     theta=0),
        DHParameter(a=0,     alpha=-1.5708, d=0.3,   theta=0),
        DHParameter(a=0,     alpha=1.5708,  d=0,     theta=0),
        DHParameter(a=0,     alpha=0,       d=0.1,   theta=0),
    ],
    joint_limits=[
        JointLimit(lower=-2.967, upper=2.967, velocity=2.0, torque=150),
        JointLimit(lower=-2.094, upper=2.094, velocity=2.0, torque=150),
        JointLimit(lower=-2.967, upper=2.967, velocity=2.5, torque=100),
        JointLimit(lower=-2.094, upper=2.094, velocity=3.0, torque=80),
        JointLimit(lower=-2.967, upper=2.967, velocity=3.0, torque=80),
        JointLimit(lower=-2.094, upper=2.094, velocity=3.0, torque=50),
    ],
    dynamics=[
        DynamicsData(mass=4.0, friction=0.1, damping=0.05),
        DynamicsData(mass=8.0, friction=0.1, damping=0.05),
        DynamicsData(mass=5.0, friction=0.1, damping=0.05),
        DynamicsData(mass=2.0, friction=0.05, damping=0.03),
        DynamicsData(mass=1.5, friction=0.05, damping=0.03),
        DynamicsData(mass=0.5, friction=0.02, damping=0.02),
    ],
)
PRESET_ARMS["six_dof_standard"] = six_dof_standard

# ─── Six-DOF Collaborative Arm ───────────────────────────────────────

six_dof_collaborative = ArmModel(
    name="six_dof_collaborative",
    dh_params=[
        DHParameter(a=0,     alpha=-1.5708, d=0.25,  theta=0),
        DHParameter(a=0.3,   alpha=0,       d=0,     theta=0),
        DHParameter(a=0.25,  alpha=0,       d=0,     theta=0),
        DHParameter(a=0,     alpha=-1.5708, d=0.25,  theta=0),
        DHParameter(a=0,     alpha=1.5708,  d=0,     theta=0),
        DHParameter(a=0,     alpha=0,       d=0.08,  theta=0),
    ],
    joint_limits=[
        JointLimit(lower=-2.618, upper=2.618, velocity=1.5, torque=80),
        JointLimit(lower=-2.094, upper=2.094, velocity=1.5, torque=80),
        JointLimit(lower=-2.618, upper=2.618, velocity=2.0, torque=50),
        JointLimit(lower=-2.094, upper=2.094, velocity=2.5, torque=40),
        JointLimit(lower=-2.618, upper=2.618, velocity=2.5, torque=40),
        JointLimit(lower=-2.094, upper=2.094, velocity=2.5, torque=25),
    ],
)
PRESET_ARMS["six_dof_collaborative"] = six_dof_collaborative

# ─── Seven-DOF Redundant Arm ─────────────────────────────────────────

seven_dof_standard = ArmModel(
    name="seven_dof_standard",
    dh_params=[
        DHParameter(a=0,     alpha=-1.5708, d=0.25,  theta=0),
        DHParameter(a=0.3,   alpha=1.5708,  d=0,     theta=0),
        DHParameter(a=0,     alpha=-1.5708, d=0.3,   theta=0),
        DHParameter(a=0,     alpha=1.5708,  d=0.3,   theta=0),
        DHParameter(a=0.2,   alpha=-1.5708, d=0,     theta=0),
        DHParameter(a=0,     alpha=1.5708,  d=0.2,   theta=0),
        DHParameter(a=0,     alpha=0,       d=0.1,   theta=0),
    ],
    joint_limits=[
        JointLimit(lower=-2.618, upper=2.618, velocity=1.5, torque=100),
        JointLimit(lower=-2.094, upper=2.094, velocity=1.5, torque=100),
        JointLimit(lower=-2.618, upper=2.618, velocity=2.0, torque=80),
        JointLimit(lower=-2.094, upper=2.094, velocity=2.0, torque=80),
        JointLimit(lower=-2.618, upper=2.618, velocity=2.5, torque=60),
        JointLimit(lower=-2.094, upper=2.094, velocity=2.5, torque=50),
        JointLimit(lower=-2.618, upper=2.618, velocity=2.5, torque=40),
    ],
)
PRESET_ARMS["seven_dof_standard"] = seven_dof_standard
