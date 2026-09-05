#!/usr/bin/env python3
"""Visual actuator step-response test for Spot V2.

Purpose
-------
Characterize the CURRENT explicit-PD actuator in isolation before changing
Kp, Kd, or torque limits.

The test uses the existing fixed-base Spot MJCF and exercises only:
    - front-left hip pitch (fl_hy)
    - front-left knee (fl_kn)

For each joint it records:
    - desired vs actual joint position
    - joint velocity
    - raw PD torque and clipped/applied torque

A MuJoCo viewer is opened so the motion can be inspected visually while the
test runs. Plots and a CSV are saved automatically under:
    reports/actuator_characterization/<gain-specific-directory>/

This is intentionally a small diagnostic test, not a parameter sweep.
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

DT = 1.0 / PHYSICS_FREQUENCY_HZ


# ---------------------------------------------------------------------------
# Test definition.
# ---------------------------------------------------------------------------

HY_JOINT = "fl_hy"
KNEE_JOINT = "fl_kn"

# Representative locomotion-sized command changes.
HY_STEP_RAD = 0.20
KNEE_STEP_RAD = 0.25

# Timeline:
# 0-1 s: settle at home
# 1-2 s: HY step
# 2-3 s: HY return
# 3-4 s: settle
# 4-5 s: knee step
# 5-6 s: knee return
# 6-7 s: final settle
TOTAL_TIME_S = 7.0


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
    """Return repository root when this file is placed in scripts/."""
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
        raise FileNotFoundError(f"Spot fixed-base MJCF not found: {path}")
    return path


def output_dir() -> Path:
    path = (
        repo_root()
        / "reports"
        / "actuator_characterization"
        / (
            f"step_hip_kp{HIP_KP:g}_kd{HIP_KD:g}"
            f"_knee_kp{KNEE_KP:g}_kd{KNEE_KD:g}"
        )
    )
    path.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"Joint not found: {joint_name}")

    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
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
        raise RuntimeError(f"Actuator not found: {actuator_name}")
    return int(actuator_id)


def torque_limit_for_joint(joint_name: str) -> float:
    if joint_name.endswith(("_hx", "_hy")):
        return HIP_TORQUE_LIMIT
    if joint_name.endswith("_kn"):
        return KNEE_TORQUE_LIMIT
    raise ValueError(f"No torque limit defined for {joint_name}")


def gains_for_joint(joint_name: str) -> tuple[float, float]:
    if joint_name.endswith(("_hx", "_hy")):
        return float(HIP_KP), float(HIP_KD)
    if joint_name.endswith("_kn"):
        return float(KNEE_KP), float(KNEE_KD)
    raise ValueError(f"No gains defined for {joint_name}")


def desired_positions(t: float) -> dict[str, float]:
    """Return desired joint positions for the current test time."""
    q_des = dict(HOME)

    if 1.0 <= t < 2.0:
        q_des[HY_JOINT] = HOME[HY_JOINT] + HY_STEP_RAD

    if 4.0 <= t < 5.0:
        # Positive step makes the knee less flexed:
        # -1.80 -> -1.55 rad.
        q_des[KNEE_JOINT] = HOME[KNEE_JOINT] + KNEE_STEP_RAD

    return q_des


def initialize_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: dict[str, int],
) -> None:
    """Initialize exactly at the V2 home joint pose."""
    mujoco.mj_resetData(model, data)

    # Use the existing home keyframe if available.
    home_key = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, int(home_key))
    else:
        for name in JOINT_NAMES:
            data.qpos[qpos_addr[name]] = HOME[name]

    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def apply_explicit_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_des: dict[str, float],
    qpos_addr: dict[str, int],
    dof_addr: dict[str, int],
    actuator_id: dict[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute and apply the current V2 explicit-PD law.

    Returns:
        raw_torque: unclipped Kp/Kd torque command
        applied_torque: torque after physical effort clipping
    """
    raw_torque: dict[str, float] = {}
    applied_torque: dict[str, float] = {}

    for name in JOINT_NAMES:
        q = float(data.qpos[qpos_addr[name]])
        qd = float(data.qvel[dof_addr[name]])

        kp, kd = gains_for_joint(name)
        tau_raw = kp * (q_des[name] - q) - kd * qd
        tau_limit = torque_limit_for_joint(name)
        tau_applied = float(
            np.clip(tau_raw, -tau_limit, tau_limit)
        )

        data.ctrl[actuator_id[name]] = tau_applied

        raw_torque[name] = float(tau_raw)
        applied_torque[name] = tau_applied

    return raw_torque, applied_torque


def make_position_plot(
    out_dir: Path,
    joint_name: str,
    time_s: np.ndarray,
    desired_rad: np.ndarray,
    actual_rad: np.ndarray,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, desired_rad, label="Desired position")
    ax.plot(time_s, actual_rad, label="Actual position")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint position [rad]")
    ax.set_title(f"{joint_name}: position tracking")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{joint_name}_position.png", dpi=160)
    plt.close(fig)


def make_velocity_plot(
    out_dir: Path,
    joint_name: str,
    time_s: np.ndarray,
    velocity_rad_s: np.ndarray,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, velocity_rad_s)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint velocity [rad/s]")
    ax.set_title(f"{joint_name}: velocity response")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{joint_name}_velocity.png", dpi=160)
    plt.close(fig)


def make_torque_plot(
    out_dir: Path,
    joint_name: str,
    time_s: np.ndarray,
    raw_torque_nm: np.ndarray,
    applied_torque_nm: np.ndarray,
    torque_limit_nm: float,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_s, raw_torque_nm, label="Raw PD torque")
    ax.plot(time_s, applied_torque_nm, label="Applied torque")
    ax.axhline(
        torque_limit_nm,
        linestyle="--",
        label=f"+ limit ({torque_limit_nm:.2f} Nm)",
    )
    ax.axhline(
        -torque_limit_nm,
        linestyle="--",
        label=f"- limit ({torque_limit_nm:.2f} Nm)",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title(f"{joint_name}: PD torque and saturation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{joint_name}_torque.png", dpi=160)
    plt.close(fig)


def write_csv(
    out_dir: Path,
    rows: list[dict[str, float]],
) -> None:
    csv_path = out_dir / "visual_step_response.csv"
    if not rows:
        return

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    xml_path = model_path()
    out_dir = output_dir()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Force the exact V2 physics rate even if an experimental XML was saved
    # with a different timestep.
    model.opt.timestep = DT

    qpos_addr: dict[str, int] = {}
    dof_addr: dict[str, int] = {}
    actuator_id: dict[str, int] = {}

    for name in JOINT_NAMES:
        qpos_addr[name], dof_addr[name] = get_joint_addresses(
            model,
            name,
        )
        actuator_id[name] = get_actuator_id(model, name)

    if model.nu != 12:
        raise RuntimeError(
            f"Expected 12 torque motors, model has nu={model.nu}"
        )

    initialize_home(model, data, qpos_addr)

    rows: list[dict[str, float]] = []

    print()
    print("Spot V2 visual actuator step test")
    print("--------------------------------")
    print(f"Model: {xml_path}")
    print(f"Physics: {PHYSICS_FREQUENCY_HZ:.0f} Hz")
    print(f"HX/HY gains: Kp={HIP_KP}, Kd={HIP_KD}")
    print(f"Knee gains:  Kp={KNEE_KP}, Kd={KNEE_KD}")
    print(
        f"Limits: hip={HIP_TORQUE_LIMIT:.2f} Nm, "
        f"knee={KNEE_TORQUE_LIMIT:.2f} Nm"
    )
    print()
    print("Timeline:")
    print("  0-1 s : settle at home")
    print(f"  1-2 s : {HY_JOINT} +{HY_STEP_RAD:.2f} rad")
    print("  2-4 s : return / settle")
    print(f"  4-5 s : {KNEE_JOINT} +{KNEE_STEP_RAD:.2f} rad")
    print("  5-7 s : return / settle")
    print()
    print("Close the MuJoCo viewer to abort the test.")
    print()

    # Run approximately in real time so the motion is easy to inspect.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()

        while data.time < TOTAL_TIME_S and viewer.is_running():
            step_wall_start = time.perf_counter()

            q_des = desired_positions(float(data.time))
            raw_torque, applied_torque = apply_explicit_pd(
                model,
                data,
                q_des,
                qpos_addr,
                dof_addr,
                actuator_id,
            )

            row: dict[str, float] = {
                "time_s": float(data.time),
            }

            for name in (HY_JOINT, KNEE_JOINT):
                row[f"{name}_q_des_rad"] = q_des[name]
                row[f"{name}_q_rad"] = float(
                    data.qpos[qpos_addr[name]]
                )
                row[f"{name}_qdot_rad_s"] = float(
                    data.qvel[dof_addr[name]]
                )
                row[f"{name}_tau_raw_nm"] = raw_torque[name]
                row[f"{name}_tau_applied_nm"] = applied_torque[name]

            rows.append(row)

            mujoco.mj_step(model, data)

            # Sync viewer at a comfortable visual rate.
            viewer.sync()

            # Keep approximately real-time playback.
            elapsed = time.perf_counter() - step_wall_start
            remaining = DT - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

        wall_elapsed = time.perf_counter() - wall_start

    if not rows:
        raise RuntimeError("No test data were recorded.")

    write_csv(out_dir, rows)

    time_s = np.asarray([row["time_s"] for row in rows])

    for name in (HY_JOINT, KNEE_JOINT):
        desired = np.asarray(
            [row[f"{name}_q_des_rad"] for row in rows]
        )
        actual = np.asarray(
            [row[f"{name}_q_rad"] for row in rows]
        )
        velocity = np.asarray(
            [row[f"{name}_qdot_rad_s"] for row in rows]
        )
        raw_tau = np.asarray(
            [row[f"{name}_tau_raw_nm"] for row in rows]
        )
        applied_tau = np.asarray(
            [row[f"{name}_tau_applied_nm"] for row in rows]
        )

        make_position_plot(
            out_dir,
            name,
            time_s,
            desired,
            actual,
        )
        make_velocity_plot(
            out_dir,
            name,
            time_s,
            velocity,
        )
        make_torque_plot(
            out_dir,
            name,
            time_s,
            raw_tau,
            applied_tau,
            torque_limit_for_joint(name),
        )

    print()
    print("PASS: visual step-response test completed.")
    print(f"Wall time: {wall_elapsed:.2f} s")
    print()
    print("Saved:")
    print(f"  {out_dir / 'visual_step_response.csv'}")
    print(f"  {out_dir / 'fl_hy_position.png'}")
    print(f"  {out_dir / 'fl_hy_velocity.png'}")
    print(f"  {out_dir / 'fl_hy_torque.png'}")
    print(f"  {out_dir / 'fl_kn_position.png'}")
    print(f"  {out_dir / 'fl_kn_velocity.png'}")
    print(f"  {out_dir / 'fl_kn_torque.png'}")


if __name__ == "__main__":
    main()