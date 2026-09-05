"""Spot robot and actuator configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


##
# MJCF model.
##

_HERE = Path(__file__).resolve().parent

SPOT_XML = (
    _HERE
    / "mjcf"
    / "experiments"
    / "spot_floating_base_explicit_pd.xml"
)

if not SPOT_XML.is_file():
    raise FileNotFoundError(
        f"Spot MJCF model was not found: {SPOT_XML}"
    )


def get_spec() -> mujoco.MjSpec:
    """Load a clean Spot specification for MjLab.

    The experimental MJCF contains motors for the standalone MuJoCo
    tests. MjLab's IdealPdActuatorCfg creates its own motor actuators,
    so the XML actuators must be removed first.

    The XML home keyframe is also removed because MjLab will create its
    initialization keyframe from SPOT_INIT_STATE below.
    """

    spec = mujoco.MjSpec.from_file(str(SPOT_XML))

    for actuator in list(spec.actuators):
        spec.delete(actuator)

    for keyframe in list(spec.keys):
        spec.delete(keyframe)

    return spec


##
# Timing.
##

PHYSICS_FREQUENCY_HZ = 500
POLICY_FREQUENCY_HZ = 50

PHYSICS_TIMESTEP = 1.0 / PHYSICS_FREQUENCY_HZ
POLICY_TIMESTEP = 1.0 / POLICY_FREQUENCY_HZ

DECIMATION = PHYSICS_FREQUENCY_HZ // POLICY_FREQUENCY_HZ

if PHYSICS_FREQUENCY_HZ % POLICY_FREQUENCY_HZ != 0:
    raise ValueError(
        "Physics frequency must be divisible by policy frequency."
    )


##
# Joint ordering and home pose.
##

JOINT_NAMES = (
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
)

HOME_JOINT_POSITIONS = (
    0.0,
    1.04,
    -1.8,
    0.0,
    1.04,
    -1.8,
    0.0,
    1.04,
    -1.8,
    0.0,
    1.04,
    -1.8,
)

HOME_JOINT_POSITION_MAP = dict(
    zip(
        JOINT_NAMES,
        HOME_JOINT_POSITIONS,
        strict=True,
    )
)


##
# Provisional explicit PD actuator model.
##

HIP_KP = 500.0
HIP_KD = 10.0

KNEE_KP = 500.0
KNEE_KD = 10.0

HIP_TORQUE_LIMIT = 69.52

# Local approximation around the home knee angle q = -1.8 rad.
# This will eventually be replaced by an angle-dependent limit.
KNEE_TORQUE_LIMIT = 164.15

SPOT_HIP_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=(
        ".*_hx",
        ".*_hy",
    ),
    stiffness=HIP_KP,
    damping=HIP_KD,
    effort_limit=HIP_TORQUE_LIMIT,
)

SPOT_KNEE_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=(".*_kn",),
    stiffness=KNEE_KP,
    damping=KNEE_KD,
    effort_limit=KNEE_TORQUE_LIMIT,
)


##
# Initial state.
##

SPOT_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.46),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos=HOME_JOINT_POSITION_MAP,
    joint_vel={".*": 0.0},
)


##
# Complete articulation and entity configurations.
##

SPOT_ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        SPOT_HIP_ACTUATOR_CFG,
        SPOT_KNEE_ACTUATOR_CFG,
    ),
)


def get_spot_robot_cfg() -> EntityCfg:
    """Return a fresh MjLab Spot entity configuration."""

    return EntityCfg(
        init_state=SPOT_INIT_STATE,
        spec_fn=get_spec,
        articulation=SPOT_ARTICULATION_CFG,

        # The two actuator groups would otherwise be arranged as
        # eight hip actuators followed by four knee actuators.
        # Sorting makes model.ctrl follow the MJCF joint order.
        sort_actuators=True,
    )


def main() -> None:
    """Compile the MjLab entity and validate its basic structure."""

    robot = get_spot_robot_cfg().build()
    model = robot.spec.compile()

    actual_joint_names = tuple(robot.joint_names)
    actual_actuator_names = tuple(robot.actuator_names)

    print(f"Model: {SPOT_XML}")
    print(f"Physics frequency: {PHYSICS_FREQUENCY_HZ} Hz")
    print(f"Policy frequency: {POLICY_FREQUENCY_HZ} Hz")
    print(f"Decimation: {DECIMATION}")
    print(f"Joints: {actual_joint_names}")
    print(f"Actuators: {actual_actuator_names}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(
        "Actuator force ranges: "
        f"{model.actuator_forcerange.tolist()}"
    )

    if actual_joint_names != JOINT_NAMES:
        raise RuntimeError(
            "Compiled joint ordering does not match JOINT_NAMES."
        )

    if actual_actuator_names != JOINT_NAMES:
        raise RuntimeError(
            "Compiled actuator ordering does not match JOINT_NAMES."
        )

    if model.nu != len(JOINT_NAMES):
        raise RuntimeError(
            f"Expected 12 actuators, but the model has {model.nu}."
        )

    print(
        "\nPASS: Spot EntityCfg compiled with the expected "
        "joint and actuator order."
    )


if __name__ == "__main__":
    main()