from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "reference"
    / "scene.xml"
)


# -------------------------------------------------------------------
# Comparison configuration
# -------------------------------------------------------------------

# Available modes:
#
# "original":
#     Keep the original joint torque limits of ±1000 N·m.
#
# "matched":
#     Use the same hip and knee limits as our explicit controller.
#
# "scaled":
#     Scale the physical Spot torque limits according to the mass ratio
#     between our 50.34 kg model and the approximately 32.5 kg robot.
LIMIT_MODE = "scaled"


# Available modes:
#
# "original":
#     Keep the original MuJoCo actuator gains:
#     Kp = 500, Kd = 40.
#
# "explicit":
#     Use the family-specific gains tested with our explicit controller:
#     HX/HY Kp = 100, knee Kp = 300, all Kd = 8.
GAIN_MODE = "explicit"


JOINT_NAMES = [
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
]


# Physical-reference limits used in the explicit controller.
CURRENT_TORQUE_LIMITS = np.asarray(
    [
        44.88,
        44.88,
        105.98,
        44.88,
        44.88,
        105.98,
        44.88,
        44.88,
        105.98,
        44.88,
        44.88,
        105.98,
    ],
    dtype=np.float64,
)


# This scaling is only an engineering comparison, not a claim about
# the real actuator's continuous capability.
MASS_SCALE = 50.34 / 32.5

SCALED_TORQUE_LIMITS = (
    CURRENT_TORQUE_LIMITS * MASS_SCALE
)


EXPLICIT_KP_VALUES = np.asarray(
    [
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
        500.0,
    ],
    dtype=np.float64,
)


EXPLICIT_KD_VALUES = np.asarray(
    [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ],
    dtype=np.float64,
)

COMPARISON_KP_VALUES = np.full(
    12,
    500.0,
    dtype=np.float64,
)

COMPARISON_KD_VALUES = np.full(
    12,
    10.0,
    dtype=np.float64,
)


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """Return the MuJoCo ID of a named object."""

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(
            f"MuJoCo object not found: {name}"
        )

    return object_id


def configure_torque_limits(
    model: mujoco.MjModel,
    joint_ids: np.ndarray,
    actuator_ids: np.ndarray,
) -> np.ndarray:
    """
    Configure the joint-level actuator-force limits.

    The original XML limits actuator contributions at the joint using
    jnt_actfrclimited and jnt_actfrcrange. We preserve that same
    location in the actuator pipeline for a fair comparison.
    """

    if LIMIT_MODE == "original":
        # Keep the compiled ±1000 N·m limits from the original XML.
        return model.jnt_actfrcrange[
            joint_ids
        ].copy()

    if LIMIT_MODE == "matched":
        selected_limits = (
            CURRENT_TORQUE_LIMITS.copy()
        )

    elif LIMIT_MODE == "scaled":
        selected_limits = (
            SCALED_TORQUE_LIMITS.copy()
        )

    else:
        raise ValueError(
            f"Unknown LIMIT_MODE: {LIMIT_MODE}"
        )

    # Ensure there is no second actuator-level force clamp.
    model.actuator_forcelimited[
        actuator_ids
    ] = 0

    model.actuator_forcerange[
        actuator_ids,
        :,
    ] = 0.0

    # Enable and set joint-level actuator-force limits.
    model.jnt_actfrclimited[
        joint_ids
    ] = 1

    model.jnt_actfrcrange[
        joint_ids,
        0,
    ] = -selected_limits

    model.jnt_actfrcrange[
        joint_ids,
        1,
    ] = selected_limits

    return model.jnt_actfrcrange[
        joint_ids
    ].copy()


def configure_actuator_gains(
    model: mujoco.MjModel,
    actuator_ids: np.ndarray,
) -> None:
    """
    Configure the built-in MuJoCo position actuators.

    For the affine position actuator:

        actuator output =
            Kp * ctrl
            - Kp * q
            - Kd * qdot

    which is equivalent to:

        Kp * (q_des - q) - Kd * qdot
    """

    if GAIN_MODE == "original":
        # Keep the original compiled Kp=500 and Kd=40.
        return

    if GAIN_MODE != "explicit":
        raise ValueError(
            f"Unknown GAIN_MODE: {GAIN_MODE}"
        )

    # if GAIN_MODE == "comparison":
    #     model.actuator_gainprm[
    #         actuator_ids,
    #         0,
    #     ] = COMPARISON_KP_VALUES

    #     model.actuator_biasprm[
    #         actuator_ids,
    #         0,
    #     ] = 0.0

    #     model.actuator_biasprm[
    #         actuator_ids,
    #         1,
    #     ] = -COMPARISON_KP_VALUES

    #     model.actuator_biasprm[
    #         actuator_ids,
    #         2,
    #     ] = -COMPARISON_KD_VALUES

    #     return

    # Control gain multiplying q_des.
    model.actuator_gainprm[
        actuator_ids,
        0,
    ] = EXPLICIT_KP_VALUES

    # Constant bias coefficient remains zero.
    model.actuator_biasprm[
        actuator_ids,
        0,
    ] = 0.0

    # Position bias coefficient multiplying current q.
    model.actuator_biasprm[
        actuator_ids,
        1,
    ] = -EXPLICIT_KP_VALUES

    # Velocity bias coefficient multiplying qdot.
    model.actuator_biasprm[
        actuator_ids,
        2,
    ] = -EXPLICIT_KD_VALUES


def print_configuration(
    model: mujoco.MjModel,
    joint_ids: np.ndarray,
    actuator_ids: np.ndarray,
    selected_force_ranges: np.ndarray,
) -> None:
    """Print the actual compiled configuration used by the test."""

    kp_values = model.actuator_gainprm[
        actuator_ids,
        0,
    ]

    position_bias = model.actuator_biasprm[
        actuator_ids,
        1,
    ]

    kd_values = -model.actuator_biasprm[
        actuator_ids,
        2,
    ]

    print(f"Model: {MODEL_PATH}")
    print(
        f"Physics timestep: "
        f"{model.opt.timestep:.6f} s"
    )

    print(f"\nLimit mode: {LIMIT_MODE}")
    print(f"Gain mode:  {GAIN_MODE}")

    print("\nJoint order:")
    for index, joint_name in enumerate(
        JOINT_NAMES
    ):
        print(
            f"  {index:2d}: {joint_name}"
        )

    print("\nCompiled actuator Kp:")
    print(kp_values)

    print("\nCompiled actuator Kd:")
    print(kd_values)

    print("\nCompiled position-bias coefficients:")
    print(position_bias)

    print("\nJoint-level actuator-force limited:")
    print(
        model.jnt_actfrclimited[
            joint_ids
        ]
    )

    print("\nJoint-level actuator-force ranges:")
    print(selected_force_ranges)

    print("\nActuator-level force limited:")
    print(
        model.actuator_forcelimited[
            actuator_ids
        ]
    )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    np.set_printoptions(
        precision=3,
        suppress=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    home_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    joint_ids = np.asarray(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            for joint_name in JOINT_NAMES
        ],
        dtype=int,
    )

    actuator_ids = np.asarray(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )
            for joint_name in JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = model.jnt_qposadr[
        joint_ids
    ].astype(int)

    dof_addresses = model.jnt_dofadr[
        joint_ids
    ].astype(int)

    selected_force_ranges = (
        configure_torque_limits(
            model,
            joint_ids,
            actuator_ids,
        )
    )

    configure_actuator_gains(
        model,
        actuator_ids,
    )

    # Create MjData after configuring the model.
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    mujoco.mj_forward(
        model,
        data,
    )

    home_control = model.key_ctrl[
        home_id
    ].copy()

    home_joint_positions = data.qpos[
        qpos_addresses
    ].copy()

    print_configuration(
        model,
        joint_ids,
        actuator_ids,
        selected_force_ranges,
    )

    print("\nHome joint positions:")
    print(home_joint_positions)

    print("\nHome control targets:")
    print(home_control)

    print(
        "\nViewer controls:\n"
        "  Ctrl + right mouse drag: apply force/torque\n"
        "  Double-click a body first if the force is not "
        "applied to the torso\n"
        "  Close the viewer window to stop\n"
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        while viewer.is_running():
            step_start = time.monotonic()

            # For the reference position actuators, ctrl contains
            # desired joint angles—not torques.
            data.ctrl[:] = home_control

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            elapsed = (
                time.monotonic()
                - step_start
            )

            remaining = (
                model.opt.timestep
                - elapsed
            )

            if remaining > 0.0:
                time.sleep(remaining)

    final_joint_positions = data.qpos[
        qpos_addresses
    ].copy()

    final_joint_velocities = data.qvel[
        dof_addresses
    ].copy()

    final_joint_errors = (
        home_joint_positions
        - final_joint_positions
    )

    print("\nFinal state after closing viewer:")
    print(
        f"Final base height: "
        f"{data.qpos[2]:.6f} m"
    )

    print(
        "\njoint  | target  | actual  | error   | velocity"
    )

    for (
        joint_name,
        target,
        actual,
        error,
        velocity,
    ) in zip(
        JOINT_NAMES,
        home_joint_positions,
        final_joint_positions,
        final_joint_errors,
        final_joint_velocities,
        strict=True,
    ):
        print(
            f"{joint_name:6s} | "
            f"{target:+7.3f} | "
            f"{actual:+7.3f} | "
            f"{error:+7.3f} | "
            f"{velocity:+8.4f}"
        )


if __name__ == "__main__":
    main()