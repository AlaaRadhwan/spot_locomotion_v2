#!/usr/bin/env python3
"""Spot V3 fixed-base step-response test.

Target stack:
- MjLab 1.5.3
- MuJoCo 3.10
- Spot V3 rigid-body model
- current actuator parameters imported from spot_v3_constants.py

This test deliberately removes only the floating-base free joint, sets gravity
to zero, and keeps all 12 native MjLab BuiltinPosition actuators active.

Two policy-sized steps are tested:
1) FL hip-y (fl_hy): +0.05 rad
2) FL knee  (fl_kn): +0.08 rad

For each test it reports:
- first-sample torque after the step
- 10-90% rise time
- overshoot
- 2% settling time
- peak velocity
- peak joint torque
- whether the tested joint hit its effort limit
- maximum disturbance of any other joint

CSV and PNG outputs are written to reports/v3_actuator/.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from spot_locomotion_v2.robot.spot_v3_constants import (
    HOME_JOINT_POSITION_MAP,
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    PHYSICS_TIMESTEP,
    SPOT_V3_ARTICULATION_CFG,
    V3_HIP_EFFORT_LIMIT,
    V3_KD,
    V3_KNEE_EFFORT_LIMIT,
    V3_KP,
    get_v3_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "v3_actuator"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STEP_TIME_S = 0.10
SIMULATION_DURATION_S = 0.80

TESTS = (
    ("fl_hy", 0.05),
    ("fl_kn", 0.08),
)


@dataclass
class StepResult:
    joint_name: str
    step_size: float
    initial_target: float
    final_target: float
    first_step_torque: float
    rise_time_s: float | None
    overshoot_percent: float
    settling_time_s: float | None
    peak_velocity_rad_s: float
    peak_joint_torque_nm: float
    effort_limit_nm: float
    saturated: bool
    max_other_joint_disturbance_rad: float
    worst_other_joint: str
    final_error_rad: float


def gain_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Object not found: {name}")
    return object_id


def fixed_base_spec() -> mujoco.MjSpec:
    """Return the V3 spec with only the floating-base free joint removed."""
    spec = get_v3_spec()

    free_joints = [
        joint
        for joint in spec.joints
        if joint.type == mujoco.mjtJoint.mjJNT_FREE
    ]

    if len(free_joints) != 1:
        raise RuntimeError(
            "Expected exactly one free joint in the V3 floating-base model, "
            f"but found {len(free_joints)}."
        )

    spec.delete(free_joints[0])
    return spec


def build_fixed_base_model() -> mujoco.MjModel:
    """Build the fixed-base model through MjLab's actual actuator pipeline."""
    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            joint_pos=HOME_JOINT_POSITION_MAP,
            joint_vel={".*": 0.0},
        ),
        spec_fn=fixed_base_spec,
        articulation=SPOT_V3_ARTICULATION_CFG,
        sort_actuators=True,
    )

    robot = cfg.build()
    spec = robot.spec
    spec.option.timestep = PHYSICS_TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0

    model = spec.compile()

    actual_joint_names = tuple(robot.joint_names)
    actual_actuator_names = tuple(robot.actuator_names)

    if actual_joint_names != JOINT_NAMES:
        raise RuntimeError(
            "Joint ordering mismatch.\n"
            f"Expected: {JOINT_NAMES}\n"
            f"Actual:   {actual_joint_names}"
        )

    if actual_actuator_names != JOINT_NAMES:
        raise RuntimeError(
            "Actuator ordering mismatch.\n"
            f"Expected: {JOINT_NAMES}\n"
            f"Actual:   {actual_actuator_names}"
        )

    if model.nq != 12 or model.nv != 12 or model.nu != 12:
        raise RuntimeError(
            "Fixed-base model dimensions are wrong: "
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}. "
            "Expected nq=12, nv=12, nu=12."
        )

    # Isolate actuator/joint dynamics from gravity.
    model.opt.gravity[:] = 0.0

    return model


def get_addresses(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            get_object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )

    actuator_ids = np.asarray(
        [
            get_object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = model.jnt_qposadr[joint_ids].astype(int)
    dof_addresses = model.jnt_dofadr[joint_ids].astype(int)

    return qpos_addresses, dof_addresses, actuator_ids


def reset_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    actuator_ids: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)

    home = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)

    data.qpos[qpos_addresses] = home
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = home

    mujoco.mj_forward(model, data)


def first_crossing_time(
    times: np.ndarray,
    response: np.ndarray,
    threshold: float,
) -> float | None:
    indices = np.flatnonzero(response >= threshold)
    if len(indices) == 0:
        return None
    return float(times[indices[0]])


def settling_time(
    times: np.ndarray,
    errors: np.ndarray,
    tolerance: float,
) -> float | None:
    """Earliest time after which all remaining samples stay within tolerance."""
    inside = np.abs(errors) <= tolerance

    suffix_all_inside = np.logical_and.accumulate(inside[::-1])[::-1]
    indices = np.flatnonzero(suffix_all_inside)

    if len(indices) == 0:
        return None

    return float(times[indices[0]])


def run_step(
    model: mujoco.MjModel,
    joint_name: str,
    step_size: float,
) -> tuple[StepResult, dict[str, np.ndarray]]:
    data = mujoco.MjData(model)

    qpos_addresses, dof_addresses, actuator_ids = get_addresses(model)
    reset_home(model, data, qpos_addresses, actuator_ids)

    joint_index = JOINT_NAMES.index(joint_name)
    actuator_id = int(actuator_ids[joint_index])
    qpos_address = int(qpos_addresses[joint_index])
    dof_address = int(dof_addresses[joint_index])

    home = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)
    initial_target = float(home[joint_index])
    final_target = initial_target + step_size

    if joint_name.endswith("_kn"):
        effort_limit = float(V3_KNEE_EFFORT_LIMIT)
    else:
        effort_limit = float(V3_HIP_EFFORT_LIMIT)

    times: list[float] = []
    targets: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    torques: list[float] = []
    all_positions: list[np.ndarray] = []

    n_steps = int(SIMULATION_DURATION_S / model.opt.timestep) + 1

    for _ in range(n_steps):
        t = float(data.time)

        data.ctrl[actuator_ids] = home

        target = initial_target if t < STEP_TIME_S else final_target
        data.ctrl[actuator_id] = target

        # Evaluate actuator force at the current state with the current target
        # before integration, so the first step sample captures the instantaneous
        # PD response to the setpoint jump.
        mujoco.mj_forward(model, data)

        times.append(t)
        targets.append(target)
        positions.append(float(data.qpos[qpos_address]))
        velocities.append(float(data.qvel[dof_address]))
        torques.append(float(data.qfrc_actuator[dof_address]))
        all_positions.append(
            np.asarray(data.qpos[qpos_addresses], dtype=np.float64).copy()
        )

        mujoco.mj_step(model, data)

    times_a = np.asarray(times, dtype=np.float64)
    targets_a = np.asarray(targets, dtype=np.float64)
    positions_a = np.asarray(positions, dtype=np.float64)
    velocities_a = np.asarray(velocities, dtype=np.float64)
    torques_a = np.asarray(torques, dtype=np.float64)
    all_positions_a = np.asarray(all_positions, dtype=np.float64)

    step_index = int(np.searchsorted(times_a, STEP_TIME_S, side="left"))

    post_times = times_a[step_index:] - STEP_TIME_S
    post_positions = positions_a[step_index:]
    post_velocities = velocities_a[step_index:]
    post_torques = torques_a[step_index:]

    direction = 1.0 if step_size > 0.0 else -1.0
    signed_response = direction * (post_positions - initial_target)
    abs_step = abs(step_size)

    t10 = first_crossing_time(post_times, signed_response, 0.10 * abs_step)
    t90 = first_crossing_time(post_times, signed_response, 0.90 * abs_step)

    if t10 is not None and t90 is not None and t90 >= t10:
        rise_time = t90 - t10
    else:
        rise_time = None

    max_signed_response = float(np.max(signed_response))
    overshoot = max(0.0, (max_signed_response - abs_step) / abs_step * 100.0)

    post_error = final_target - post_positions
    settling = settling_time(
        post_times,
        post_error,
        tolerance=0.02 * abs_step,
    )

    other_indices = [i for i in range(len(JOINT_NAMES)) if i != joint_index]
    other_deviation = np.abs(
        all_positions_a[step_index:, other_indices]
        - home[other_indices]
    )

    flat_index = int(np.argmax(other_deviation))
    sample_i, other_local_i = np.unravel_index(
        flat_index,
        other_deviation.shape,
    )
    worst_global_i = other_indices[other_local_i]
    max_other_disturbance = float(other_deviation[sample_i, other_local_i])
    worst_other_joint = JOINT_NAMES[worst_global_i]

    peak_torque = float(np.max(np.abs(post_torques)))

    # A small numerical tolerance avoids calling exact-limit rounding "over".
    saturated = peak_torque >= 0.999 * effort_limit

    result = StepResult(
        joint_name=joint_name,
        step_size=step_size,
        initial_target=initial_target,
        final_target=final_target,
        first_step_torque=float(torques_a[step_index]),
        rise_time_s=rise_time,
        overshoot_percent=overshoot,
        settling_time_s=settling,
        peak_velocity_rad_s=float(np.max(np.abs(post_velocities))),
        peak_joint_torque_nm=peak_torque,
        effort_limit_nm=effort_limit,
        saturated=saturated,
        max_other_joint_disturbance_rad=max_other_disturbance,
        worst_other_joint=worst_other_joint,
        final_error_rad=float(final_target - positions_a[-1]),
    )

    arrays = {
        "time_s": times_a,
        "target_rad": targets_a,
        "position_rad": positions_a,
        "velocity_rad_s": velocities_a,
        "joint_torque_nm": torques_a,
    }

    return result, arrays


def save_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    keys = list(arrays.keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(arrays[key] for key in keys), strict=True))


def save_plot(
    path: Path,
    joint_name: str,
    arrays: dict[str, np.ndarray],
) -> None:
    time_s = arrays["time_s"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(time_s, arrays["target_rad"], "--", label="target")
    axes[0].plot(time_s, arrays["position_rad"], label="position")
    axes[0].axvline(STEP_TIME_S, linestyle=":", label="step")
    axes[0].set_ylabel("Position [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_s, arrays["velocity_rad_s"])
    axes[1].axvline(STEP_TIME_S, linestyle=":")
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].grid(True)

    axes[2].plot(time_s, arrays["joint_torque_nm"])
    axes[2].axvline(STEP_TIME_S, linestyle=":")
    axes[2].set_ylabel("Joint torque [N*m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True)

    fig.suptitle(
        f"Spot V3 fixed-base step: {joint_name} | "
        f"Kp={V3_KP:g}, Kd={V3_KD:g}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def format_optional(value: float | None, unit: str) -> str:
    if value is None:
        return f"not reached within {SIMULATION_DURATION_S - STEP_TIME_S:.2f} s"
    return f"{value:.6f} {unit}"


def print_result(result: StepResult) -> None:
    print(f"\n=== {result.joint_name} step result ===")
    print(
        f"Target:                 "
        f"{result.initial_target:+.6f} -> {result.final_target:+.6f} rad"
    )
    print(f"Step size:              {result.step_size:+.6f} rad")
    print(f"First-step torque:      {result.first_step_torque:+.6f} N*m")
    print(
        "Expected pure P torque: "
        f"{V3_KP * result.step_size:+.6f} N*m"
    )
    print(
        f"10-90% rise time:       "
        f"{format_optional(result.rise_time_s, 's')}"
    )
    print(f"Overshoot:              {result.overshoot_percent:.3f} %")
    print(
        f"2% settling time:       "
        f"{format_optional(result.settling_time_s, 's')}"
    )
    print(f"Peak |velocity|:        {result.peak_velocity_rad_s:.6f} rad/s")
    print(f"Peak |joint torque|:    {result.peak_joint_torque_nm:.6f} N*m")
    print(f"Effort limit:           {result.effort_limit_nm:.3f} N*m")
    print(f"Hit effort limit:       {'YES' if result.saturated else 'no'}")
    print(
        f"Max other-joint motion: {result.max_other_joint_disturbance_rad:.6f} rad "
        f"({result.worst_other_joint})"
    )
    print(f"Final error:            {result.final_error_rad:+.8f} rad")


def main() -> None:
    print("Building fixed-base Spot V3 through MjLab...")
    model = build_fixed_base_model()

    print("\n=== Test configuration ===")
    print("Actuator: BuiltinPositionActuatorCfg")
    print(f"Kp: {V3_KP:g}")
    print(f"Kd: {V3_KD:g}")
    print(f"HX/HY effort limit: {V3_HIP_EFFORT_LIMIT:g} N*m")
    print(f"Knee effort limit:  {V3_KNEE_EFFORT_LIMIT:g} N*m")
    print(f"Physics timestep:   {model.opt.timestep:.6f} s")
    print("Gravity:            disabled for actuator isolation")
    print(f"Fixed-base dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    print(f"Integrator enum:    {int(model.opt.integrator)}")
    print(f"Cone enum:          {int(model.opt.cone)}")
    print(f"Impratio:           {model.opt.impratio}")

    kp_tag = gain_tag(float(V3_KP))
    kd_tag = gain_tag(float(V3_KD))

    all_results: list[StepResult] = []

    for joint_name, step_size in TESTS:
        result, arrays = run_step(model, joint_name, step_size)
        all_results.append(result)
        print_result(result)

        stem = (
            f"fixed_base_{joint_name}_step_"
            f"kp{kp_tag}_kd{kd_tag}"
        )

        csv_path = REPORTS_DIR / f"{stem}.csv"
        png_path = REPORTS_DIR / f"{stem}.png"

        save_csv(csv_path, arrays)
        save_plot(png_path, joint_name, arrays)

        print(f"CSV:  {csv_path}")
        print(f"Plot: {png_path}")

    print("\n=== Overall check ===")

    any_saturation = any(result.saturated for result in all_results)
    max_cross_motion = max(
        result.max_other_joint_disturbance_rad
        for result in all_results
    )

    print(
        "Any tested-joint effort saturation: "
        f"{'YES' if any_saturation else 'no'}"
    )
    print(
        "Largest cross-joint disturbance:    "
        f"{max_cross_motion:.6f} rad"
    )

    print(
        "\nSend me the two result blocks above. "
        "If both are clean, the next test is the 50 Hz ZOH policy-like "
        "reference sequence."
    )


if __name__ == "__main__":
    main()
