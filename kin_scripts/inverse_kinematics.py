#!/usr/bin/env python3
"""Spot V2 full 3D inverse-kinematics test for the front-left leg.

Goal
----
Given a desired FL foot position relative to the main BODY frame:

    [x_body, y_body, z_body]

calculate:

    qhx, qhy, qkn

The IK works in two stages:

1. Use the Y-Z geometry to solve qhx.
2. Undo that sideways rotation and solve qhy/qkn using the
   planar triangle IK already verified previously.

The solution is checked using:

    desired foot
        -> IK
        -> qhx, qhy, qkn
        -> our analytical FK

and independently using MuJoCo geometry.
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


# ===========================================================================
# DESIRED FOOT POSITION
#
# Relative to the MAIN BODY frame:
#
# +X = forward
# +Y = left
# +Z = up
#
# We deliberately use Y = 0.20 instead of the old 0.1658 so that
# qhx must actually move.
# ===========================================================================

TARGET_X_BODY = 0.125
TARGET_Y_BODY = 0.200
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


# ===========================================================================
# MODEL / ACTUATOR HELPERS
# ===========================================================================


def repo_root() -> Path:
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
        raise FileNotFoundError(
            f"Spot MJCF not found: {path}"
        )

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
        f"No torque limit defined for {joint_name!r}"
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

        kp, kd = gains_for_joint(
            name
        )

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
# FL LEG GEOMETRY
# ===========================================================================


def get_fl_geometry(
    model: mujoco.MjModel,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

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

    r_body_hx = model.body_pos[
        fl_hip_id
    ].copy()

    r_hx_hy = model.body_pos[
        fl_uleg_id
    ].copy()

    r_hy_kn = model.body_pos[
        fl_lleg_id
    ].copy()

    r_kn_foot = model.site_pos[
        fl_foot_id
    ].copy()

    return (
        r_body_hx,
        r_hx_hy,
        r_hy_kn,
        r_kn_foot,
    )


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

    (
        r_body_hx,
        r_hx_hy,
        r_hy_kn,
        r_kn_foot,
    ) = get_fl_geometry(model)

    # ------------------------------------------------------------------
    # HY -> KN rotated around Y by qhy
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
    # KN -> foot
    #
    # The knee rotates relative to the already-rotated upper leg.
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
    # HX -> foot before qhx rotation
    # ------------------------------------------------------------------

    r_hx_foot_before_hx = (
        r_hx_hy
        + rot_r_hy_kn
        + rot_r_kn_foot
    )

    x_before = r_hx_foot_before_hx[0]
    y_before = r_hx_foot_before_hx[1]
    z_before = r_hx_foot_before_hx[2]

    # ------------------------------------------------------------------
    # Rotate entire HX -> foot vector around X by qhx
    # ------------------------------------------------------------------

    x_after = x_before

    y_after = (
        y_before * np.cos(qhx)
        - z_before * np.sin(qhx)
    )

    z_after = (
        y_before * np.sin(qhx)
        + z_before * np.cos(qhx)
    )

    r_hx_foot_after_hx = np.array([
        x_after,
        y_after,
        z_after,
    ])

    # Body -> foot
    p_foot_body = (
        r_body_hx
        + r_hx_foot_after_hx
    )

    return p_foot_body


# ===========================================================================
# FULL 3D INVERSE KINEMATICS
# ===========================================================================


def inverse_kinematics_fl(
    model: mujoco.MjModel,
    x_body: float,
    y_body: float,
    z_body: float,
) -> tuple[float, float, float]:
    """Solve full FL leg IK.

    Input
    -----
    Desired foot position relative to the main body:

        [x_body, y_body, z_body]

    Output
    ------
        qhx, qhy, qkn

    Normal downward-leg IK branch is used.
    """

    (
        r_body_hx,
        r_hx_hy,
        r_hy_kn,
        r_kn_foot,
    ) = get_fl_geometry(model)

    # ==================================================================
    # STEP 1
    # Desired BODY-frame foot target -> HX-relative target
    #
    # p_HX_to_foot =
    #     p_body_to_foot - p_body_to_HX
    # ==================================================================

    X = (
        x_body
        - r_body_hx[0]
    )

    Y = (
        y_body
        - r_body_hx[1]
    )

    Z = (
        z_body
        - r_body_hx[2]
    )

    # HX -> HY lateral offset.
    #
    # FL:
    #     s = +0.1108 m
    #
    # Before qhx rotates:
    #
    # HX -> foot =
    #
    #     [X,
    #      s,
    #      z_planar]
    #
    s = float(
        r_hx_hy[1]
    )

    # ==================================================================
    # STEP 2
    # Recover z_planar.
    #
    # Rotation about X preserves the Y-Z distance:
    #
    #     Y² + Z² = s² + z_planar²
    #
    # therefore:
    #
    #     z_planar² = Y² + Z² - s²
    # ==================================================================

    z_planar_squared = (
        Y**2
        + Z**2
        - s**2
    )

    # Small negative numbers can occur from floating-point roundoff.
    tolerance = 1e-12

    if z_planar_squared < -tolerance:

        raise ValueError(
            "Desired Y/Z position cannot be reached "
            "by the FL hip-roll geometry.\n"
            f"Y relative HX = {Y:.6f}\n"
            f"Z relative HX = {Z:.6f}\n"
            f"HX->HY offset = {s:.6f}\n"
            f"Y^2 + Z^2 = {Y**2 + Z**2:.6f}\n"
            f"s^2 = {s**2:.6f}"
        )

    z_planar_squared = max(
        z_planar_squared,
        0.0,
    )

    # We choose the normal branch where the foot lies below the hip.
    z_planar = -np.sqrt(
        z_planar_squared
    )

    # ==================================================================
    # STEP 3
    # Solve qhx.
    #
    # Before qhx:
    #
    #     [Y, Z] = [s, z_planar]
    #
    # After qhx:
    #
    #     [Y, Z] = desired [Y, Z]
    #
    # So:
    #
    # qhx =
    #     desired Y-Z vector angle
    #     -
    #     original Y-Z vector angle
    # ==================================================================

    theta_desired = np.arctan2(
        Z,
        Y,
    )

    theta_before = np.arctan2(
        z_planar,
        s,
    )

    qhx = (
        theta_desired
        - theta_before
    )

    # Normalize qhx to [-pi, pi]
    qhx = (
        qhx + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    # ==================================================================
    # STEP 4
    # Now the remaining problem is exactly the planar IK we already
    # solved.
    #
    # HY -> foot before qhx rotation:
    #
    #     [X,
    #      0,
    #      z_planar]
    #
    # So the two-link triangle uses:
    #
    #     x_planar = X
    #     z_planar = z_planar
    # ==================================================================

    x_planar = X

    # Link lengths
    L1 = float(
        np.linalg.norm(
            r_hy_kn
        )
    )

    L2 = float(
        np.linalg.norm(
            r_kn_foot
        )
    )

    # Small built-in tilt of HY -> KN relative to downward vertical
    alpha = np.arctan2(
        r_hy_kn[0],
        -r_hy_kn[2],
    )

    # HY -> foot distance
    d = np.sqrt(
        x_planar**2
        + z_planar**2
    )

    # ==================================================================
    # STEP 5
    # Check planar reachability
    # ==================================================================

    d_min = abs(
        L1 - L2
    )

    d_max = (
        L1 + L2
    )

    if (
        d > d_max
        or d < d_min
    ):

        raise ValueError(
            "Desired foot position is outside the "
            "two-link leg workspace.\n"
            f"x_planar = {x_planar:.6f}\n"
            f"z_planar = {z_planar:.6f}\n"
            f"d = {d:.6f}\n"
            f"reachable d = "
            f"[{d_min:.6f}, {d_max:.6f}]"
        )

    # ==================================================================
    # STEP 6
    # Solve knee
    # ==================================================================

    cos_gamma = (
        L1**2
        + L2**2
        - d**2
    ) / (
        2.0
        * L1
        * L2
    )

    gamma = np.arccos(
        np.clip(
            cos_gamma,
            -1.0,
            1.0,
        )
    )

    qkn = (
        gamma
        - np.pi
        - alpha
    )

    # ==================================================================
    # STEP 7
    # Solve hip pitch
    # ==================================================================

    phi = np.arctan2(
        -x_planar,
        -z_planar,
    )

    cos_beta = (
        L1**2
        + d**2
        - L2**2
    ) / (
        2.0
        * L1
        * d
    )

    beta = np.arccos(
        np.clip(
            cos_beta,
            -1.0,
            1.0,
        )
    )

    qhy = (
        phi
        + beta
        + alpha
    )

    return (
        float(qhx),
        float(qhy),
        float(qkn),
    )


# ===========================================================================
# MUJOCO POSITION CHECK
# ===========================================================================


def mujoco_fl_foot_position_body(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    """Return FL foot position relative to main body.

    This subtraction is valid for the CURRENT fixed-base model because
    the body frame has the same orientation as the world frame.
    """

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

    return (
        data.site_xpos[fl_foot_id]
        - data.xpos[body_id]
    ).copy()


# ===========================================================================
# IK VERIFICATION
# ===========================================================================


def verify_3d_ik(
    model: mujoco.MjModel,
    qpos_addr: dict[str, int],
    x_body: float,
    y_body: float,
    z_body: float,
) -> tuple[float, float, float]:

    target_body = np.array([
        x_body,
        y_body,
        z_body,
    ])

    # ------------------------------------------------------------------
    # Solve IK
    # ------------------------------------------------------------------

    qhx, qhy, qkn = (
        inverse_kinematics_fl(
            model,
            x_body,
            y_body,
            z_body,
        )
    )

    # ------------------------------------------------------------------
    # Verification 1:
    #
    # target -> IK -> our FK
    # ------------------------------------------------------------------

    p_fk = forward_kinematics_fl(
        model,
        qhx,
        qhy,
        qkn,
    )

    fk_error = (
        p_fk - target_body
    )

    # ------------------------------------------------------------------
    # Verification 2:
    #
    # Put exact IK angles directly into MuJoCo.
    # No PD controller is involved.
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
    ] = qhx

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
    print(
        "========================================"
    )
    print(
        "FULL 3D IK VERIFICATION"
    )
    print(
        "========================================"
    )

    print()
    print(
        "Desired foot position, body frame:"
    )
    print(
        target_body
    )

    print()
    print(
        "IK joint solution:"
    )

    print(
        f"qhx = {qhx:.9f} rad "
        f"({np.degrees(qhx):.3f} deg)"
    )

    print(
        f"qhy = {qhy:.9f} rad "
        f"({np.degrees(qhy):.3f} deg)"
    )

    print(
        f"qkn = {qkn:.9f} rad "
        f"({np.degrees(qkn):.3f} deg)"
    )

    print()
    print(
        "Our FK(IK(target)):"
    )
    print(
        p_fk
    )

    print(
        "FK error:"
    )
    print(
        fk_error
    )

    print()
    print(
        "MuJoCo position using IK angles:"
    )
    print(
        p_mujoco
    )

    print(
        "MuJoCo error:"
    )
    print(
        mujoco_error
    )

    print()

    fk_pass = np.allclose(
        p_fk,
        target_body,
        atol=1e-9,
    )

    mujoco_pass = np.allclose(
        p_mujoco,
        target_body,
        atol=1e-9,
    )

    if fk_pass:

        print(
            "PASS: analytical FK(IK(target)) "
            "matches target."
        )

    else:

        print(
            "FAIL: analytical FK(IK(target)) "
            "does not match target."
        )

    if mujoco_pass:

        print(
            "PASS: MuJoCo geometry also "
            "matches target."
        )

    else:

        print(
            "FAIL: MuJoCo geometry does not "
            "match target."
        )

    print(
        "========================================"
    )
    print()

    return (
        qhx,
        qhy,
        qkn,
    )


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
    print(
        "Spot V2 full 3D IK test"
    )
    print(
        "------------------------"
    )

    print(
        f"Model: {xml_path}"
    )

    print(
        f"Foot target in body frame: "
        f"[{TARGET_X_BODY:.4f}, "
        f"{TARGET_Y_BODY:.4f}, "
        f"{TARGET_Z_BODY:.4f}]"
    )

    # ==================================================================
    # FIRST:
    # verify the geometry without involving the PD controller.
    # ==================================================================

    (
        qhx_target,
        qhy_target,
        qkn_target,
    ) = verify_3d_ik(
        model,
        qpos_addr,
        TARGET_X_BODY,
        TARGET_Y_BODY,
        TARGET_Z_BODY,
    )

    # ==================================================================
    # THEN:
    # command the IK result through the actual explicit-PD controller.
    # ==================================================================

    q_des = dict(
        HOME
    )

    q_des["fl_hx"] = qhx_target
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

            # Recompute all derived kinematics for the
            # newly-integrated qpos before comparing site_xpos.
            mujoco.mj_forward(
                model,
                data,
            )

            if (
                data.time
                >= next_status_print
            ):

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

                target = np.array([
                    TARGET_X_BODY,
                    TARGET_Y_BODY,
                    TARGET_Z_BODY,
                ])

                foot_error = (
                    p_mujoco_actual
                    - target
                )

                print(
                    f"t = {data.time:6.2f} s"
                )

                print(
                    "  target joints: "
                    f"qhx={qhx_target:+.5f}, "
                    f"qhy={qhy_target:+.5f}, "
                    f"qkn={qkn_target:+.5f}"
                )

                print(
                    "  actual joints: "
                    f"qhx={qhx_actual:+.5f}, "
                    f"qhy={qhy_actual:+.5f}, "
                    f"qkn={qkn_actual:+.5f}"
                )

                print(
                    "  our FK:     ",
                    p_fk_actual,
                )

                print(
                    "  MuJoCo:     ",
                    p_mujoco_actual,
                )

                print(
                    "  foot error: ",
                    foot_error,
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