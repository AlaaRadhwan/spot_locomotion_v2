"""Spot V3 robot configuration for MjLab 1.5.3.

Rigid-body model:
    src/spot_locomotion_v2/robot/mjcf/v3/spot.xml

Actuator baseline:
    MjLab BuiltinPositionActuatorCfg
    Kp = 300
    Kd = 4
    HX/HY effort limit = 45 N*m
    knee effort limit = 115 N*m

This file intentionally keeps the existing V2 home pose and 500/50 Hz
timing so that the rigid-body + actuator change can be evaluated cleanly.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_HERE = Path(__file__).resolve().parent

SPOT_V3_XML = _HERE / "mjcf" / "v3" / "spot.xml"

if not SPOT_V3_XML.is_file():
    raise FileNotFoundError(
        f"Spot V3 MJCF model was not found: {SPOT_V3_XML}\n"
        "Run tools/build_spot_v3_model.py first."
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

PHYSICS_FREQUENCY_HZ = 500
POLICY_FREQUENCY_HZ = 50

PHYSICS_TIMESTEP = 1.0 / PHYSICS_FREQUENCY_HZ
POLICY_TIMESTEP = 1.0 / POLICY_FREQUENCY_HZ

DECIMATION = PHYSICS_FREQUENCY_HZ // POLICY_FREQUENCY_HZ

if PHYSICS_FREQUENCY_HZ % POLICY_FREQUENCY_HZ != 0:
    raise ValueError("Physics frequency must be divisible by policy frequency.")


# ---------------------------------------------------------------------------
# Joint ordering and initial pose
# ---------------------------------------------------------------------------

JOINT_NAMES = (
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
)

HOME_JOINT_POSITIONS = (
    0.0, 1.04, -1.8,
    0.0, 1.04, -1.8,
    0.0, 1.04, -1.8,
    0.0, 1.04, -1.8,
)

HOME_JOINT_POSITION_MAP = dict(
    zip(JOINT_NAMES, HOME_JOINT_POSITIONS, strict=True)
)


# ---------------------------------------------------------------------------
# V3 actuator baseline
# ---------------------------------------------------------------------------

V3_KP = 300.0
V3_KD = 4

V3_HIP_EFFORT_LIMIT = 45.0
V3_KNEE_EFFORT_LIMIT = 115.0

SPOT_V3_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hx",
        ".*_hy",
    ),
    stiffness=V3_KP,
    damping=V3_KD,
    effort_limit=V3_HIP_EFFORT_LIMIT,
)

SPOT_V3_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_kn",),
    stiffness=V3_KP,
    damping=V3_KD,
    effort_limit=V3_KNEE_EFFORT_LIMIT,
)


# ---------------------------------------------------------------------------
# MjLab entity
# ---------------------------------------------------------------------------

def get_v3_spec() -> mujoco.MjSpec:
    """Load a fresh actuator-free V3 Spot specification.

    MjLab will create the BuiltinPosition actuators programmatically from
    SPOT_V3_ARTICULATION_CFG.

    Existing MJCF keyframes are removed because EntityCfg creates its own
    initialization keyframe from SPOT_V3_INIT_STATE.
    """
    spec = mujoco.MjSpec.from_file(str(SPOT_V3_XML))

    # The V3 rigid-body MJCF is intended to be actuator-free, but keep this
    # defensive cleanup so this configuration remains unambiguous.
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    for keyframe in list(spec.keys):
        spec.delete(keyframe)

    return spec


SPOT_V3_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.46),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos=HOME_JOINT_POSITION_MAP,
    joint_vel={".*": 0.0},
)

SPOT_V3_ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        SPOT_V3_HIP_ACTUATOR_CFG,
        SPOT_V3_KNEE_ACTUATOR_CFG,
    ),
)


def get_spot_v3_robot_cfg() -> EntityCfg:
    """Return a fresh MjLab 1.5.3 Spot V3 EntityCfg."""
    return EntityCfg(
        init_state=SPOT_V3_INIT_STATE,
        spec_fn=get_v3_spec,
        articulation=SPOT_V3_ARTICULATION_CFG,
        sort_actuators=True,
    )


def build_v3_model_with_ground() -> mujoco.MjModel:
    """Build the V3 MjLab entity, add a flat ground plane, and compile.

    This function is for deterministic local tests.  MjLab is responsible
    for inserting the BuiltinPosition actuators into the MjSpec; MuJoCo's
    passive viewer is then used only to step and visualize the compiled model.
    """
    robot = get_spot_v3_robot_cfg().build()
    spec = robot.spec

    spec.option.timestep = PHYSICS_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0

    terrain = spec.worldbody.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.01],
        friction=[0.8, 0.02, 0.01],
    )
    # Slight neutral grey for visual clarity.
    terrain.rgba[:] = [0.35, 0.35, 0.35, 1.0]

    model = spec.compile()

    actual_joint_names = tuple(robot.joint_names)
    actual_actuator_names = tuple(robot.actuator_names)

    if actual_joint_names != JOINT_NAMES:
        raise RuntimeError(
            "Compiled joint ordering does not match JOINT_NAMES.\n"
            f"Expected: {JOINT_NAMES}\n"
            f"Actual:   {actual_joint_names}"
        )

    if actual_actuator_names != JOINT_NAMES:
        raise RuntimeError(
            "Compiled actuator ordering does not match JOINT_NAMES.\n"
            f"Expected: {JOINT_NAMES}\n"
            f"Actual:   {actual_actuator_names}"
        )

    if model.nu != len(JOINT_NAMES):
        raise RuntimeError(
            f"Expected 12 actuators, but compiled model has {model.nu}."
        )

    return model


if __name__ == "__main__":
    model = build_v3_model_with_ground()
    print("PASS: Spot V3 MjLab model compiled.")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"physics timestep={model.opt.timestep:.6f} s")
