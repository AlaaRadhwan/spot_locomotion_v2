#!/usr/bin/env python3
"""Spot V2 planar inverse-kinematics test.

Current goal
------------
Test front-left leg planar IK before adding the hip-roll joint fl_hx.

The desired foot position is specified relative to the main body frame:

    (x_body, z_body)

For this planar test:
    qhx = 0

The IK internally converts the desired body-frame foot position into
coordinates relative to the HY joint, solves the two-link triangle, and
returns:

    qhy, qkn

The script verifies the result in two ways:

1. IK -> our analytical FK
2. IK -> MuJoCo geometry directly

Only after those tests are passed does the explicit-PD controller move
the simulated leg toward the calculated joint angles.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from spot_locomotion_v2.robot.spot_constants import (
    HIP_KD,
    HIP_KP,
    HIP_TORQUE_LIMIT,
    KNEE_KD,
    KNEE_KP,
    KNEE_TORQUE_LIMIT,
    PHYSICS_FREQUENCY_HZ,
)


DT = 1.0 / PHYSICS_FREQUENCY_HZ
TOTAL_TIME_S = 180.0


# ---------------------------------------------------------------------------
# Desired planar foot target, relative to the MAIN BODY frame.
#
# +X = forward
# +Z = upward
#
# Because this is still planar IK, qhx = 0 and we do not command Y yet.
# ---------------------------------------------------------------------------

TARGET_X_BODY = 0.125
TARGET_Z_BODY = -0.400


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


HOME = {
    "fl_hx": 0.0,
    "fl_hy": 1.04,
    "fl_kn": -1.80,
    "fr_hx": 0.0,
    "fr_hy": 1.04,
    "fr_kn": -1.80,
    "hl_hx": 0.0,
    "hl_hy": 1.04,
    "hl_kn": -1.80,
    "hr_hx": 0.0,
    "hr_hy": 1.04,
    "hr_kn": -1.80,
}


def repo_root() -> Path:
    """Return repository root."""
    return Path(__file__).resolve().parents[1]


def model_path() -> Path:
    path = (
        repo_root()
        / "src"
        / "spot_locomotion_v2"
        / "robot"
        / "mjcf"
        / "experiments"
        / "spot_fixed_base_explicit_pd.xml"
    )

    if not path.is_file():
        raise FileNotFoundError(f"Spot MJCF not found: {path}")

    return path


def get_joint_addresses(
    model: mujoco.MjModel,
    joint_name: str,
) -> tuple[int, int]:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint {joint_name!r} not found"
        )

    qpos_adr = int(
        model.jnt_qposadr[joint_id]
    )

    dof_adr = int(
        model.jnt_dofadr[joint_id]
    )

    return qpos_adr, dof_adr


def get_actuator_id(
    model: mujoco.MjModel,
    actuator_name: str,
) -> int:

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator {actuator_name!r} not found"
        )

    return int(actuator_id)


def torque_limit_for_joint(
    joint_name: str,
) -> float:

    if joint_name.endswith(("_hx", "_hy")):
        return HIP_TORQUE_LIMIT

    if joint_name.endswith("_kn"):
        return KNEE_TORQUE_LIMIT

    raise ValueError(
        f"No torque limit defined for "
        f"joint {joint_name!r}"
    )


def gains_for_joint(
    joint_name: str,
) -> tuple[float, float]:

    if joint_name.endswith(("_hx", "_hy")):
        return float(HIP_KP), float(HIP_KD)

    if joint_name.endswith("_kn"):
        return float(KNEE_KP), float(KNEE_KD)

    raise ValueError(
        f"No gains defined for {joint_name}"
    )


def initialize_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: dict[str, int],
) -> None:
    """Initialize exactly at the home joint pose."""

    mujoco.mj_resetData(
        model,
        data,
    )

    home_key = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(
            model,
            data,
            int(home_key),
        )

    else:
        for name in JOINT_NAMES:
            data.qpos[
                qpos_addr[name]
            ] = HOME[name]

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def apply_explicit_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_des: dict[str, float],
    qpos_addr: dict[str, int],
    dof_addr: dict[str, int],
    actuator_id: dict[str, int],
) -> tuple[
    dict[str, float],
    dict[str, float],
]:

    raw_torque: dict[str, float] = {}
    applied_torque: dict[str, float] = {}

    for name in JOINT_NAMES:

        q = float(
            data.qpos[
                qpos_addr[name]
            ]
        )

        qd = float(
            data.qvel[
                dof_addr[name]
            ]
        )

        kp, kd = gains_for_joint(name)

        tau_raw = (
            kp * (q_des[name] - q)
            - kd * qd
        )

        tau_limit = torque_limit_for_joint(
            name
        )

        tau_applied = float(
            np.clip(
                tau_raw,
                -tau_limit,
                tau_limit,
            )
        )

        data.ctrl[
            actuator_id[name]
        ] = tau_applied

        raw_torque[name] = float(
            tau_raw
        )

        applied_torque[name] = (
            tau_applied
        )

    return raw_torque, applied_torque


# ===========================================================================
# FORWARD KINEMATICS
# ===========================================================================


def forward_kinematics_fl(
    model: mujoco.MjModel,
    qhx: float,
    qhy: float,
    qkn: float,
) -> np.ndarray:
    """Return FL foot position relative to main body frame."""

    fl_hip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_hip",
    )

    fl_uleg_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_uleg",
    )

    fl_lleg_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_lleg",
    )

    fl_foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "FL_site",
    )

    # Fixed geometry
    r_body_hx = model.body_pos[
        fl_hip_id
    ]

    r_hx_hy = model.body_pos[
        fl_uleg_id
    ]

    r_hy_kn = model.body_pos[
        fl_lleg_id
    ]

    r_kn_foot = model.site_pos[
        fl_foot_id
    ]

    # ------------------------------------------------------------------
    # Rotate HY -> KN about Y
    # ------------------------------------------------------------------

    x_hy_kn = (
        r_hy_kn[0] * np.cos(qhy)
        + r_hy_kn[2] * np.sin(qhy)
    )

    z_hy_kn = (
        -r_hy_kn[0] * np.sin(qhy)
        + r_hy_kn[2] * np.cos(qhy)
    )

    rot_r_hy_kn = np.array([
        x_hy_kn,
        0.0,
        z_hy_kn,
    ])

    # ------------------------------------------------------------------
    # Rotate KN -> foot
    #
    # Knee rotation is relative to the already-rotated upper-leg frame.
    # Therefore the total Y rotation is:
    #
    #     qhy + qkn
    # ------------------------------------------------------------------

    lower_angle = qhy + qkn

    x_kn_foot = (
        r_kn_foot[2]
        * np.sin(lower_angle)
    )

    z_kn_foot = (
        r_kn_foot[2]
        * np.cos(lower_angle)
    )

    rot_r_kn_foot = np.array([
        x_kn_foot,
        0.0,
        z_kn_foot,
    ])

    # ------------------------------------------------------------------
    # Vector HX -> foot before HX rotation
    # ------------------------------------------------------------------

    r_hx_foot = (
        r_hx_hy
        + rot_r_hy_kn
        + rot_r_kn_foot
    )

    # ------------------------------------------------------------------
    # Rotate entire leg about X by qhx
    # ------------------------------------------------------------------

    x_hx_foot = r_hx_foot[0]

    y_hx_foot = (
        r_hx_foot[1] * np.cos(qhx)
        - r_hx_foot[2] * np.sin(qhx)
    )

    z_hx_foot = (
        r_hx_foot[1] * np.sin(qhx)
        + r_hx_foot[2] * np.cos(qhx)
    )

    rot_r_hx_foot = np.array([
        x_hx_foot,
        y_hx_foot,
        z_hx_foot,
    ])

    # Body -> foot
    p_foot = (
        r_body_hx
        + rot_r_hx_foot
    )

    return p_foot


# ===========================================================================
# PLANAR INVERSE KINEMATICS
# ===========================================================================


def inverse_kinematics_fl_planar(
    model: mujoco.MjModel,
    x_body: float,
    z_body: float,
) -> tuple[float, float]:
    """Solve FL planar IK from a BODY-frame foot target.

    Assumptions:
        qhx = 0

    Inputs:
        x_body:
            desired FL foot X coordinate relative to main body

        z_body:
            desired FL foot Z coordinate relative to main body

    Returns:
        qhy, qkn
    """

    fl_hip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_hip",
    )

    fl_uleg_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_uleg",
    )

    fl_lleg_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_lleg",
    )

    fl_foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "FL_site",
    )

    # ------------------------------------------------------------------
    # Fixed geometry
    # ------------------------------------------------------------------

    r_body_hx = model.body_pos[
        fl_hip_id
    ]

    r_hx_hy = model.body_pos[
        fl_uleg_id
    ]

    r_hy_kn = model.body_pos[
        fl_lleg_id
    ]

    r_kn_foot = model.site_pos[
        fl_foot_id
    ]

    # ------------------------------------------------------------------
    # HY position relative to the body.
    #
    # Since qhx = 0 in this planar version:
    #
    # body -> HY
    # =
    # body -> HX + HX -> HY
    # ------------------------------------------------------------------

    p_hy_body = (
        r_body_hx
        + r_hx_hy
    )

    # ------------------------------------------------------------------
    # Convert desired BODY-frame target into HY-relative coordinates.
    #
    # Example:
    #
    # x_body = 0.125
    # x_HY   = 0.29785
    #
    # x relative to HY:
    #
    #     0.125 - 0.29785
    #     = -0.17285
    # ------------------------------------------------------------------

    x_hy = (
        x_body
        - p_hy_body[0]
    )

    z_hy = (
        z_body
        - p_hy_body[2]
    )

    # ------------------------------------------------------------------
    # Link lengths
    # ------------------------------------------------------------------

    L1 = float(
        np.linalg.norm(r_hy_kn)
    )

    L2 = float(
        np.linalg.norm(r_kn_foot)
    )

    # Built-in upper-leg geometry offset
    alpha = np.arctan2(
        r_hy_kn[0],
        -r_hy_kn[2],
    )

    # Distance HY -> desired foot
    d = np.sqrt(
        x_hy**2
        + z_hy**2
    )

    # ------------------------------------------------------------------
    # Reachability
    # ------------------------------------------------------------------

    if (
        d > L1 + L2
        or d < abs(L1 - L2)
    ):
        raise ValueError(
            "Desired foot position is unreachable.\n"
            f"HY-relative target: "
            f"x={x_hy:.6f}, "
            f"z={z_hy:.6f}\n"
            f"d={d:.6f} m\n"
            f"reachable range = "
            f"[{abs(L1 - L2):.6f}, "
            f"{L1 + L2:.6f}] m"
        )

    # ------------------------------------------------------------------
    # Knee triangle angle gamma
    #
    #        L1² + L2² - d²
    # cos γ = ----------------
    #             2 L1 L2
    # ------------------------------------------------------------------

    cos_gamma = (
        L1**2
        + L2**2
        - d**2
    ) / (
        2.0 * L1 * L2
    )

    cos_gamma = np.clip(
        cos_gamma,
        -1.0,
        1.0,
    )

    gamma = np.arccos(
        cos_gamma
    )

    # Convert triangle knee angle -> MuJoCo qkn
    qkn = (
        gamma
        - np.pi
        - alpha
    )

    # ------------------------------------------------------------------
    # Direction HY -> desired foot.
    #
    # phi is measured from downward vertical.
    # ------------------------------------------------------------------

    phi = np.arctan2(
        -x_hy,
        -z_hy,
    )

    # ------------------------------------------------------------------
    # Angle beta between HY->foot and HY->knee
    #
    #        L1² + d² - L2²
    # cos β = ----------------
    #             2 L1 d
    # ------------------------------------------------------------------

    cos_beta = (
        L1**2
        + d**2
        - L2**2
    ) / (
        2.0 * L1 * d
    )

    cos_beta = np.clip(
        cos_beta,
        -1.0,
        1.0,
    )

    beta = np.arccos(
        cos_beta
    )

    # For Spot's normal knee configuration:
    #
    # theta_upper = phi + beta
    #
    # qhy also includes the small built-in alpha offset.

    qhy = (
        phi
        + beta
        + alpha
    )

    return float(qhy), float(qkn)


# ===========================================================================
# MUJOCO GEOMETRY CHECK
# ===========================================================================


def mujoco_fl_foot_position_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """Return MuJoCo FL foot position relative to the fixed main body."""

    fl_foot_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "FL_site",
    )

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "body",
    )

    # The current model is fixed and body orientation is identity,
    # so subtracting world positions gives body-relative coordinates.
    return (
        data.site_xpos[fl_foot_id]
        - data.xpos[body_id]
    ).copy()


def verify_planar_ik(
    model: mujoco.MjModel,
    qpos_addr: dict[str, int],
    x_body: float,
    z_body: float,
) -> tuple[float, float]:
    """Verify IK using both analytical FK and MuJoCo."""

    qhy, qkn = inverse_kinematics_fl_planar(
        model,
        x_body,
        z_body,
    )

    # ------------------------------------------------------------------
    # Since qhx = 0, desired Y is fixed by the body's FL hip geometry.
    # ------------------------------------------------------------------

    fl_hip_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_hip",
    )

    fl_uleg_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "fl_uleg",
    )

    p_hy_body = (
        model.body_pos[fl_hip_id]
        + model.body_pos[fl_uleg_id]
    )

    target_body = np.array([
        x_body,
        p_hy_body[1],
        z_body,
    ])

    # ------------------------------------------------------------------
    # Test 1:
    #
    # desired foot
    # -> IK
    # -> qhy/qkn
    # -> our FK
    # -> foot
    # ------------------------------------------------------------------

    p_fk = forward_kinematics_fl(
        model,
        qhx=0.0,
        qhy=qhy,
        qkn=qkn,
    )

    fk_error = (
        p_fk
        - target_body
    )

    # ------------------------------------------------------------------
    # Test 2:
    #
    # Put those exact joint angles into a separate MuJoCo data object.
    # This avoids involving the PD controller.
    # ------------------------------------------------------------------

    test_data = mujoco.MjData(
        model
    )

    initialize_home(
        model,
        test_data,
        qpos_addr,
    )

    test_data.qpos[
        qpos_addr["fl_hx"]
    ] = 0.0

    test_data.qpos[
        qpos_addr["fl_hy"]
    ] = qhy

    test_data.qpos[
        qpos_addr["fl_kn"]
    ] = qkn

    test_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        test_data,
    )

    p_mujoco = (
        mujoco_fl_foot_position_body(
            model,
            test_data,
        )
    )

    mujoco_error = (
        p_mujoco
        - target_body
    )

    print()
    print("========================================")
    print("PLANAR IK VERIFICATION")
    print("========================================")

    print()
    print("Desired foot position, body frame:")
    print(target_body)

    print()
    print("IK joint solution:")
    print(
        f"qhy = {qhy:.9f} rad "
        f"({np.degrees(qhy):.3f} deg)"
    )
    print(
        f"qkn = {qkn:.9f} rad "
        f"({np.degrees(qkn):.3f} deg)"
    )

    print()
    print("Our FK(IK(target)):")
    print(p_fk)

    print("FK error:")
    print(fk_error)

    print()
    print("MuJoCo position using IK angles:")
    print(p_mujoco)

    print("MuJoCo error:")
    print(mujoco_error)

    print()

    if np.allclose(
        p_fk,
        target_body,
        atol=1e-9,
    ):
        print(
            "PASS: analytical FK(IK(target)) "
            "matches target."
        )
    else:
        print(
            "FAIL: analytical FK(IK(target)) "
            "does not match target."
        )

    if np.allclose(
        p_mujoco,
        target_body,
        atol=1e-9,
    ):
        print(
            "PASS: MuJoCo geometry also "
            "matches target."
        )
    else:
        print(
            "FAIL: MuJoCo geometry does not "
            "match target."
        )

    print("========================================")
    print()

    return qhy, qkn


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:

    xml_path = model_path()

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )

    data = mujoco.MjData(
        model
    )

    model.opt.timestep = DT

    qpos_addr: dict[str, int] = {}
    dof_addr: dict[str, int] = {}
    actuator_id: dict[str, int] = {}

    for name in JOINT_NAMES:

        (
            qpos_addr[name],
            dof_addr[name],
        ) = get_joint_addresses(
            model,
            name,
        )

        actuator_id[name] = (
            get_actuator_id(
                model,
                name,
            )
        )

    if model.nu != 12:
        raise RuntimeError(
            f"Expected 12 torque motors, "
            f"model has nu={model.nu}"
        )

    initialize_home(
        model,
        data,
        qpos_addr,
    )

    print()
    print("Spot V2 planar IK test")
    print("----------------------")
    print(f"Model: {xml_path}")
    print(
        f"Physics: "
        f"{PHYSICS_FREQUENCY_HZ:.0f} Hz"
    )

    print(
        f"HX/HY gains: "
        f"Kp={HIP_KP}, "
        f"Kd={HIP_KD}"
    )

    print(
        f"Knee gains: "
        f"Kp={KNEE_KP}, "
        f"Kd={KNEE_KD}"
    )

    print(
        f"Limits: "
        f"hip={HIP_TORQUE_LIMIT:.2f} Nm, "
        f"knee={KNEE_TORQUE_LIMIT:.2f} Nm"
    )

    # ------------------------------------------------------------------
    # FIRST: prove the geometry is correct.
    # ------------------------------------------------------------------

    qhy_target, qkn_target = verify_planar_ik(
        model,
        qpos_addr,
        TARGET_X_BODY,
        TARGET_Z_BODY,
    )

    # ------------------------------------------------------------------
    # THEN: command those joint angles through the real explicit-PD
    # controller and watch the leg move.
    # ------------------------------------------------------------------

    q_des = dict(HOME)

    q_des["fl_hx"] = 0.0
    q_des["fl_hy"] = qhy_target
    q_des["fl_kn"] = qkn_target

    next_status_print = 0.0

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while (
            data.time < TOTAL_TIME_S
            and viewer.is_running()
        ):

            step_wall_start = (
                time.perf_counter()
            )

            apply_explicit_pd(
                model,
                data,
                q_des,
                qpos_addr,
                dof_addr,
                actuator_id,
            )

            mujoco.mj_step(
                model,
                data,
            )

            # Print approximately once per second,
            # rather than 500 times per second.
            if data.time >= next_status_print:

                qhx_actual = float(
                    data.qpos[
                        qpos_addr["fl_hx"]
                    ]
                )

                qhy_actual = float(
                    data.qpos[
                        qpos_addr["fl_hy"]
                    ]
                )

                qkn_actual = float(
                    data.qpos[
                        qpos_addr["fl_kn"]
                    ]
                )

                p_fk_actual = (
                    forward_kinematics_fl(
                        model,
                        qhx_actual,
                        qhy_actual,
                        qkn_actual,
                    )
                )

                p_mujoco_actual = (
                    mujoco_fl_foot_position_body(
                        model,
                        data,
                    )
                )

                print(
                    f"t = {data.time:6.2f} s"
                )

                print(
                    "  actual joints: "
                    f"qhx={qhx_actual:+.5f}, "
                    f"qhy={qhy_actual:+.5f}, "
                    f"qkn={qkn_actual:+.5f}"
                )

                print(
                    "  our FK: ",
                    p_fk_actual,
                )

                print(
                    "  MuJoCo: ",
                    p_mujoco_actual,
                )

                print()

                next_status_print += 1.0

            viewer.sync()

            elapsed = (
                time.perf_counter()
                - step_wall_start
            )

            remaining = (
                DT - elapsed
            )

            if remaining > 0.0:
                time.sleep(
                    remaining
                )


if __name__ == "__main__":
    main()