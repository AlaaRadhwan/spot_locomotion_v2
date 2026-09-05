#!/usr/bin/env python3
"""Visual free-standing push-recovery test for Spot V2.

Purpose
-------
Evaluate whether the CURRENT explicit-PD gains provide sensible load-bearing
and disturbance-rejection behavior with the current physical torque limits.

The robot:
    1. settles in the home pose,
    2. receives one downward external force on the torso,
    3. recovers while continuing to command the home pose.

A MuJoCo viewer stays open during the test.

The script records all 12 joints but produces only the plots needed to answer:
    - how far the body is pushed down,
    - how quickly it recovers,
    - how much HY/knee motion occurs,
    - how fast those joints move,
    - whether the actuator limits are reached.

Outputs
-------
reports/actuator_characterization/<gain-specific-directory>/
    push_recovery.csv
    base_height.png
    base_vertical_velocity.png
    hy_positions.png
    knee_positions.png
    joint_velocity_envelope.png
    actuator_utilization.png
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
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


# ---------------------------------------------------------------------------
# Test definition.
# ---------------------------------------------------------------------------

DT = 1.0 / PHYSICS_FREQUENCY_HZ

# One standardized vertical disturbance.
#
# Spot V2 mass is about 50.34 kg, so body weight is about 494 N.
# 250 N is roughly one-half additional body weight. This is large enough to
# challenge the stance controller without intentionally trying to slam the
# robot into the ground.
DOWNWARD_FORCE_N = 250.0

SETTLE_TIME_S = 2.0
PUSH_START_S = 2.0
PUSH_DURATION_S = 0.25
PUSH_END_S = PUSH_START_S + PUSH_DURATION_S
TOTAL_TIME_S = 6.0

BASE_BODY_NAME = "body"
FREE_JOINT_NAME = "freejoint"

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

HY_JOINTS = (
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
)

KNEE_JOINTS = (
    "fl_kn",
    "fr_kn",
    "hl_kn",
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
    """Return repository root when this file is placed in scripts/."""
    return Path(__file__).resolve().parents[1]


def scene_path() -> Path:
    path = (
        repo_root()
        / "src"
        / "spot_locomotion_v2"
        / "robot"
        / "mjcf"
        / "experiments"
        / "scene_free_standing.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Free-standing scene not found: {path}")
    return path


def output_dir() -> Path:
    path = (
        repo_root()
        / "reports"
        / "actuator_characterization"
        / (
            f"push_hip_kp{HIP_KP:g}_kd{HIP_KD:g}"
            f"_knee_kp{KNEE_KP:g}_kd{KNEE_KD:g}"
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def name_to_id(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    name: str,
) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"Could not find {name!r}")
    return int(obj_id)


def get_joint_addresses(
    model: mujoco.MjModel,
    joint_name: str,
) -> tuple[int, int]:
    joint_id = name_to_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    return (
        int(model.jnt_qposadr[joint_id]),
        int(model.jnt_dofadr[joint_id]),
    )


def torque_limit_for_joint(joint_name: str) -> float:
    if joint_name.endswith(("_hx", "_hy")):
        return float(HIP_TORQUE_LIMIT)
    if joint_name.endswith("_kn"):
        return float(KNEE_TORQUE_LIMIT)
    raise ValueError(f"No torque limit for joint {joint_name!r}")


def gains_for_joint(joint_name: str) -> tuple[float, float]:
    if joint_name.endswith(("_hx", "_hy")):
        return float(HIP_KP), float(HIP_KD)
    if joint_name.endswith("_kn"):
        return float(KNEE_KP), float(KNEE_KD)
    raise ValueError(f"No gains defined for joint {joint_name!r}")


def initialize_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: dict[str, int],
) -> None:
    """Reset to the floating-base home keyframe."""
    mujoco.mj_resetData(model, data)

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
        # The free-floating XML normally contains the home keyframe.
        # This fallback preserves the current base state and initializes joints.
        for name in JOINT_NAMES:
            data.qpos[qpos_addr[name]] = HOME[name]

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def apply_home_pose_pd(
    data: mujoco.MjData,
    qpos_addr: dict[str, int],
    dof_addr: dict[str, int],
    actuator_id: dict[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply the current V2 explicit PD controller to all 12 joints."""
    raw_torque: dict[str, float] = {}
    applied_torque: dict[str, float] = {}

    for name in JOINT_NAMES:
        q = float(data.qpos[qpos_addr[name]])
        qdot = float(data.qvel[dof_addr[name]])

        kp, kd = gains_for_joint(name)
        tau_raw = float(
            kp * (HOME[name] - q)
            - kd * qdot
        )

        limit = torque_limit_for_joint(name)
        tau_applied = float(
            np.clip(
                tau_raw,
                -limit,
                limit,
            )
        )

        data.ctrl[actuator_id[name]] = tau_applied

        raw_torque[name] = tau_raw
        applied_torque[name] = tau_applied

    return raw_torque, applied_torque


def push_is_active(t: float) -> bool:
    return PUSH_START_S <= t < PUSH_END_S


def shade_push(ax) -> None:
    ax.axvspan(
        PUSH_START_S,
        PUSH_END_S,
        alpha=0.15,
        label="Downward push",
    )


def save_base_height_plot(
    out_dir: Path,
    time_s: np.ndarray,
    base_z_m: np.ndarray,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, base_z_m, label="Base height")
    ax.axhline(
        float(base_z_m[0]),
        linestyle="--",
        label="Initial height",
    )
    shade_push(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Base height [m]")
    ax.set_title("Spot push recovery: base height")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "base_height.png", dpi=160)
    plt.close(fig)


def save_base_velocity_plot(
    out_dir: Path,
    time_s: np.ndarray,
    base_vz_m_s: np.ndarray,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, base_vz_m_s)
    ax.axhline(0.0, linewidth=0.8)
    shade_push(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vertical velocity [m/s]")
    ax.set_title("Spot push recovery: base vertical velocity")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        out_dir / "base_vertical_velocity.png",
        dpi=160,
    )
    plt.close(fig)


def save_group_position_plot(
    out_dir: Path,
    filename: str,
    title: str,
    desired: float,
    joint_names: tuple[str, ...],
    time_s: np.ndarray,
    rows: list[dict[str, float]],
) -> None:
    fig, ax = plt.subplots()

    ax.axhline(
        desired,
        linestyle="--",
        label="Desired position",
    )

    for name in joint_names:
        values = np.asarray(
            [row[f"{name}_q_rad"] for row in rows]
        )
        ax.plot(time_s, values, label=name)

    shade_push(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def save_joint_velocity_envelope(
    out_dir: Path,
    time_s: np.ndarray,
    rows: list[dict[str, float]],
) -> None:
    hy_max_abs = np.asarray(
        [
            max(
                abs(row[f"{name}_qdot_rad_s"])
                for name in HY_JOINTS
            )
            for row in rows
        ]
    )

    knee_max_abs = np.asarray(
        [
            max(
                abs(row[f"{name}_qdot_rad_s"])
                for name in KNEE_JOINTS
            )
            for row in rows
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(
        time_s,
        hy_max_abs,
        label="Max |HY velocity|",
    )
    ax.plot(
        time_s,
        knee_max_abs,
        label="Max |knee velocity|",
    )
    shade_push(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Absolute joint velocity [rad/s]")
    ax.set_title("Push recovery: joint velocity envelope")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "joint_velocity_envelope.png",
        dpi=160,
    )
    plt.close(fig)


def save_actuator_utilization_plot(
    out_dir: Path,
    time_s: np.ndarray,
    rows: list[dict[str, float]],
) -> None:
    hip_names = tuple(
        name
        for name in JOINT_NAMES
        if name.endswith(("_hx", "_hy"))
    )

    knee_names = KNEE_JOINTS

    hip_util_pct = np.asarray(
        [
            100.0
            * max(
                abs(row[f"{name}_tau_applied_nm"])
                / HIP_TORQUE_LIMIT
                for name in hip_names
            )
            for row in rows
        ]
    )

    knee_util_pct = np.asarray(
        [
            100.0
            * max(
                abs(row[f"{name}_tau_applied_nm"])
                / KNEE_TORQUE_LIMIT
                for name in knee_names
            )
            for row in rows
        ]
    )

    fig, ax = plt.subplots()
    ax.plot(
        time_s,
        hip_util_pct,
        label="Max hip utilization",
    )
    ax.plot(
        time_s,
        knee_util_pct,
        label="Max knee utilization",
    )
    ax.axhline(
        100.0,
        linestyle="--",
        label="Torque limit",
    )
    shade_push(ax)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Actuator utilization [%]")
    ax.set_title("Push recovery: actuator torque utilization")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        out_dir / "actuator_utilization.png",
        dpi=160,
    )
    plt.close(fig)


def write_csv(
    out_dir: Path,
    rows: list[dict[str, float]],
) -> None:
    csv_path = out_dir / "push_recovery.csv"

    if not rows:
        return

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, float]],
) -> None:
    base_z = np.asarray(
        [row["base_z_m"] for row in rows]
    )
    base_vz = np.asarray(
        [row["base_vz_m_s"] for row in rows]
    )

    initial_window = [
        row
        for row in rows
        if 1.5 <= row["time_s"] < PUSH_START_S
    ]

    if initial_window:
        standing_z = float(
            np.mean(
                [row["base_z_m"] for row in initial_window]
            )
        )
    else:
        standing_z = float(base_z[0])

    min_z = float(np.min(base_z))
    drop_m = standing_z - min_z

    max_down_v = float(np.min(base_vz))
    max_up_v = float(np.max(base_vz))

    hip_names = tuple(
        name
        for name in JOINT_NAMES
        if name.endswith(("_hx", "_hy"))
    )

    peak_hip_util = max(
        100.0
        * abs(row[f"{name}_tau_applied_nm"])
        / HIP_TORQUE_LIMIT
        for row in rows
        for name in hip_names
    )

    peak_knee_util = max(
        100.0
        * abs(row[f"{name}_tau_applied_nm"])
        / KNEE_TORQUE_LIMIT
        for row in rows
        for name in KNEE_JOINTS
    )

    print()
    print("Quick summary")
    print("-------------")
    print(f"Settled base height: {standing_z:.4f} m")
    print(f"Minimum base height: {min_z:.4f} m")
    print(
        f"Maximum downward displacement: "
        f"{1000.0 * drop_m:.1f} mm"
    )
    print(
        f"Peak downward base velocity: "
        f"{max_down_v:.3f} m/s"
    )
    print(
        f"Peak upward recovery velocity: "
        f"{max_up_v:.3f} m/s"
    )
    print(
        f"Peak hip actuator utilization: "
        f"{peak_hip_util:.1f}%"
    )
    print(
        f"Peak knee actuator utilization: "
        f"{peak_knee_util:.1f}%"
    )


def main() -> None:
    xml_path = scene_path()
    out_dir = output_dir()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Match the V2 physics frequency exactly.
    model.opt.timestep = DT

    base_body_id = name_to_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        BASE_BODY_NAME,
    )

    free_joint_id = name_to_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        FREE_JOINT_NAME,
    )
    free_dof_adr = int(model.jnt_dofadr[free_joint_id])

    qpos_addr: dict[str, int] = {}
    dof_addr: dict[str, int] = {}
    actuator_id: dict[str, int] = {}

    for name in JOINT_NAMES:
        qpos_addr[name], dof_addr[name] = (
            get_joint_addresses(model, name)
        )
        actuator_id[name] = name_to_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

    if model.nu != 12:
        raise RuntimeError(
            f"Expected 12 torque motors, model has nu={model.nu}"
        )

    initialize_home(
        model,
        data,
        qpos_addr,
    )

    rows: list[dict[str, float]] = []

    print()
    print("Spot V2 visual push-recovery test")
    print("---------------------------------")
    print(f"Scene: {xml_path}")
    print(f"Physics: {PHYSICS_FREQUENCY_HZ:.0f} Hz")
    print(f"HX/HY gains: Kp={HIP_KP}, Kd={HIP_KD}")
    print(f"Knee gains:  Kp={KNEE_KP}, Kd={KNEE_KD}")
    print(
        f"Limits: hip={HIP_TORQUE_LIMIT:.2f} Nm, "
        f"knee={KNEE_TORQUE_LIMIT:.2f} Nm"
    )
    print(
        f"Disturbance: {DOWNWARD_FORCE_N:.1f} N downward "
        f"for {PUSH_DURATION_S:.2f} s"
    )
    print()
    print("Timeline:")
    print(f"  0-{SETTLE_TIME_S:.2f} s : settle")
    print(
        f"  {PUSH_START_S:.2f}-{PUSH_END_S:.2f} s : "
        "downward torso force"
    )
    print(
        f"  {PUSH_END_S:.2f}-{TOTAL_TIME_S:.2f} s : recovery"
    )
    print()
    print("Close the MuJoCo viewer to abort.")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()

        while (
            data.time < TOTAL_TIME_S
            and viewer.is_running()
        ):
            step_wall_start = time.perf_counter()

            raw_torque, applied_torque = apply_home_pose_pd(
                data,
                qpos_addr,
                dof_addr,
                actuator_id,
            )

            data.xfrc_applied[base_body_id, :] = 0.0

            force_n = 0.0
            if push_is_active(float(data.time)):
                force_n = -DOWNWARD_FORCE_N
                data.xfrc_applied[
                    base_body_id,
                    2,
                ] = force_n

            row: dict[str, float] = {
                "time_s": float(data.time),
                "external_force_z_n": force_n,
                "base_z_m": float(
                    data.xpos[base_body_id, 2]
                ),
                "base_vz_m_s": float(
                    data.qvel[free_dof_adr + 2]
                ),
            }

            for name in JOINT_NAMES:
                row[f"{name}_q_des_rad"] = HOME[name]
                row[f"{name}_q_rad"] = float(
                    data.qpos[qpos_addr[name]]
                )
                row[f"{name}_qdot_rad_s"] = float(
                    data.qvel[dof_addr[name]]
                )
                row[f"{name}_tau_raw_nm"] = (
                    raw_torque[name]
                )
                row[f"{name}_tau_applied_nm"] = (
                    applied_torque[name]
                )

            rows.append(row)

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = (
                time.perf_counter()
                - step_wall_start
            )
            remaining = DT - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

        wall_elapsed = (
            time.perf_counter()
            - wall_start
        )

    if not rows:
        raise RuntimeError("No test data were recorded.")

    write_csv(out_dir, rows)

    time_s = np.asarray(
        [row["time_s"] for row in rows]
    )
    base_z = np.asarray(
        [row["base_z_m"] for row in rows]
    )
    base_vz = np.asarray(
        [row["base_vz_m_s"] for row in rows]
    )

    save_base_height_plot(
        out_dir,
        time_s,
        base_z,
    )

    save_base_velocity_plot(
        out_dir,
        time_s,
        base_vz,
    )

    save_group_position_plot(
        out_dir=out_dir,
        filename="hy_positions.png",
        title="Push recovery: HY joint positions",
        desired=HOME["fl_hy"],
        joint_names=HY_JOINTS,
        time_s=time_s,
        rows=rows,
    )

    save_group_position_plot(
        out_dir=out_dir,
        filename="knee_positions.png",
        title="Push recovery: knee joint positions",
        desired=HOME["fl_kn"],
        joint_names=KNEE_JOINTS,
        time_s=time_s,
        rows=rows,
    )

    save_joint_velocity_envelope(
        out_dir,
        time_s,
        rows,
    )

    save_actuator_utilization_plot(
        out_dir,
        time_s,
        rows,
    )

    summarize(rows)

    print()
    print("PASS: visual push-recovery test completed.")
    print(f"Wall time: {wall_elapsed:.2f} s")
    print()
    print("Saved:")
    for filename in (
        "push_recovery.csv",
        "base_height.png",
        "base_vertical_velocity.png",
        "hy_positions.png",
        "knee_positions.png",
        "joint_velocity_envelope.png",
        "actuator_utilization.png",
    ):
        print(f"  {out_dir / filename}")


if __name__ == "__main__":
    main()