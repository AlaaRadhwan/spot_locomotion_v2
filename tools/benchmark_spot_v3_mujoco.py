#!/usr/bin/env python3
"""Automated MuJoCo benchmark for the approved Spot V3 PPO policy.

This intentionally mirrors the ROS 2 / Gazebo `spot_eval` command sequence so
the two simulators can be compared phase-by-phase.

Example:
    RUN=logs/rsl_rl/spot_velocity_v3/2026-09-07_15-57-38_v3_baseline_kp300_kd40_tl45_115

    uv run python tools/benchmark_spot_v3_mujoco.py \
        --checkpoint "$RUN/model_550.pt"

Outputs:
    benchmark_runs/mujoco/spot_v3_mujoco_eval_<timestamp>/
        samples.csv
        summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

# Importing this module registers the Spot tasks with MjLab.
import spot_locomotion_v2.tasks.velocity  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from spot_locomotion_v2.robot.spot_v3_constants import (
    JOINT_NAMES,
    PHYSICS_FREQUENCY_HZ,
    POLICY_FREQUENCY_HZ,
    V3_HIP_EFFORT_LIMIT,
    V3_KD,
    V3_KNEE_EFFORT_LIMIT,
    V3_KP,
)


TASK_ID = "Mjlab-Velocity-Flat-Spot-V3"
COMMAND_NAME = "twist"

# Exactly the same command sequence used by the Gazebo evaluator.
PHASES = (
    ("stand_initial", 5.0, 0.0, 0.0, 0.0),
    ("forward",       5.0, 0.5, 0.0, 0.0),
    ("pause_1",       2.0, 0.0, 0.0, 0.0),
    ("backward",      5.0, -0.5, 0.0, 0.0),
    ("pause_2",       2.0, 0.0, 0.0, 0.0),
    ("left",          5.0, 0.0, 0.3, 0.0),
    ("pause_3",       2.0, 0.0, 0.0, 0.0),
    ("right",         5.0, 0.0, -0.3, 0.0),
    ("pause_4",       2.0, 0.0, 0.0, 0.0),
    ("yaw_left",      5.0, 0.0, 0.0, 0.5),
    ("pause_5",       2.0, 0.0, 0.0, 0.0),
    ("yaw_right",     5.0, 0.0, 0.0, -0.5),
    ("pause_6",       2.0, 0.0, 0.0, 0.0),
    ("combined",      7.0, 0.4, 0.2, 0.3),
    ("stand_final",   5.0, 0.0, 0.0, 0.0),
)

BASE_COLUMNS = (
    "t_s",
    "phase",
    "phase_t_s",
    "cmd_vx",
    "cmd_vy",
    "cmd_wz",
    "actual_vx",
    "actual_vy",
    "actual_vz",
    "actual_wx",
    "actual_wy",
    "actual_wz",
    "base_x",
    "base_y",
    "base_z",
    "roll",
    "pitch",
    "yaw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the approved RSL-RL model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_runs/mujoco"),
        help="Parent directory for timestamped benchmark runs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Simulation/inference device. Default: cuda:0 if available, else cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic environment seed.",
    )
    parser.add_argument(
        "--steady-state-skip-s",
        type=float,
        default=1.0,
        help="Ignore this much time at the start of each phase for summary metrics.",
    )
    parser.add_argument(
        "--saturation-fraction",
        type=float,
        default=0.98,
        help="Fraction of actuator effort limit counted as saturation.",
    )
    return parser.parse_args()


def wxyz_to_rpy(quat: torch.Tensor) -> tuple[float, float, float]:
    """Convert one MuJoCo/MjLab quaternion (w, x, y, z) to roll/pitch/yaw."""
    q = quat.detach().float().cpu()
    w, x, y, z = (float(v) for v in q)

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return math.nan, math.nan, math.nan

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = (
        math.copysign(math.pi / 2.0, sinp)
        if abs(sinp) >= 1.0
        else math.asin(sinp)
    )

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def finite_values(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def mean(values: list[float]) -> float:
    vals = finite_values(values)
    return fmean(vals) if vals else math.nan


def rms(values: list[float]) -> float:
    vals = finite_values(values)
    return math.sqrt(fmean(v * v for v in vals)) if vals else math.nan


def mae(values: list[float], target: float) -> float:
    vals = finite_values(values)
    return fmean(abs(v - target) for v in vals) if vals else math.nan


def rmse(values: list[float], target: float) -> float:
    vals = finite_values(values)
    return math.sqrt(fmean((v - target) ** 2 for v in vals)) if vals else math.nan


def max_abs(values: list[float]) -> float:
    vals = [abs(v) for v in finite_values(values)]
    return max(vals) if vals else math.nan


def phase_steps(duration_s: float, step_dt: float) -> int:
    steps = round(duration_s / step_dt)
    if not math.isclose(steps * step_dt, duration_s, abs_tol=1e-9):
        raise RuntimeError(
            f"Phase duration {duration_s} s is not an integer multiple "
            f"of policy step {step_dt} s."
        )
    return steps


def freeze_command_generator(command_term) -> None:
    """Disable random command resampling/command modes for this deterministic run."""
    command_term.time_left.fill_(1.0e9)

    for name in (
        "is_heading_env",
        "is_standing_env",
        "is_world_env",
        "is_forward_env",
    ):
        value = getattr(command_term, name, None)
        if isinstance(value, torch.Tensor):
            value.fill_(False)


def set_command(
    command_term,
    vx: float,
    vy: float,
    wz: float,
) -> None:
    command = torch.tensor(
        [[vx, vy, wz]],
        dtype=command_term.vel_command_b.dtype,
        device=command_term.vel_command_b.device,
    )
    command_term.vel_command_b[:] = command

    # Not used because world-frame mode is disabled, but keep it synchronized
    # so diagnostics cannot contain a stale randomly sampled command.
    if hasattr(command_term, "vel_command_w"):
        command_term.vel_command_w[:] = command


def record_sample(
    *,
    robot,
    action: torch.Tensor,
    t_s: float,
    phase: str,
    phase_t_s: float,
    vx: float,
    vy: float,
    wz: float,
) -> dict[str, Any]:
    pose = robot.data.root_link_pose_w[0]
    lin_vel = robot.data.root_link_lin_vel_b[0]
    ang_vel = robot.data.root_link_ang_vel_b[0]
    joint_pos = robot.data.joint_pos[0]
    joint_vel = robot.data.joint_vel[0]
    actuator_force = robot.data.actuator_force[0]

    roll, pitch, yaw = wxyz_to_rpy(pose[3:7])

    pose_cpu = pose.detach().float().cpu()
    lin_cpu = lin_vel.detach().float().cpu()
    ang_cpu = ang_vel.detach().float().cpu()
    q_cpu = joint_pos.detach().float().cpu()
    dq_cpu = joint_vel.detach().float().cpu()
    effort_cpu = actuator_force.detach().float().cpu()
    action_cpu = action[0].detach().float().cpu()

    row: dict[str, Any] = {
        "t_s": t_s,
        "phase": phase,
        "phase_t_s": phase_t_s,
        "cmd_vx": vx,
        "cmd_vy": vy,
        "cmd_wz": wz,
        "actual_vx": float(lin_cpu[0]),
        "actual_vy": float(lin_cpu[1]),
        "actual_vz": float(lin_cpu[2]),
        "actual_wx": float(ang_cpu[0]),
        "actual_wy": float(ang_cpu[1]),
        "actual_wz": float(ang_cpu[2]),
        "base_x": float(pose_cpu[0]),
        "base_y": float(pose_cpu[1]),
        "base_z": float(pose_cpu[2]),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }

    for index, name in enumerate(JOINT_NAMES):
        row[f"{name}_q"] = float(q_cpu[index])
        row[f"{name}_dq"] = float(dq_cpu[index])
        row[f"{name}_effort"] = float(effort_cpu[index])
        row[f"{name}_action"] = float(action_cpu[index])

    return row


def summarize_phase(
    rows: list[dict[str, Any]],
    *,
    phase_name: str,
    command: tuple[float, float, float],
    steady_state_skip_s: float,
    saturation_fraction: float,
) -> dict[str, Any]:
    phase_rows = [row for row in rows if row["phase"] == phase_name]
    steady_rows = [
        row
        for row in phase_rows
        if float(row["phase_t_s"]) >= steady_state_skip_s
    ]
    if not steady_rows:
        steady_rows = phase_rows

    actual_vx = [float(r["actual_vx"]) for r in steady_rows]
    actual_vy = [float(r["actual_vy"]) for r in steady_rows]
    actual_wz = [float(r["actual_wz"]) for r in steady_rows]
    base_z = [float(r["base_z"]) for r in steady_rows]
    roll = [float(r["roll"]) for r in steady_rows]
    pitch = [float(r["pitch"]) for r in steady_rows]

    joint_velocities: list[float] = []
    joint_efforts: list[float] = []
    saturated_count = 0
    effort_count = 0
    max_effort_ratio = 0.0

    for row in steady_rows:
        for name in JOINT_NAMES:
            dq = float(row[f"{name}_dq"])
            effort = float(row[f"{name}_effort"])
            joint_velocities.append(dq)
            joint_efforts.append(effort)

            limit = (
                float(V3_KNEE_EFFORT_LIMIT)
                if name.endswith("_kn")
                else float(V3_HIP_EFFORT_LIMIT)
            )
            ratio = abs(effort) / limit
            max_effort_ratio = max(max_effort_ratio, ratio)
            effort_count += 1
            if ratio >= saturation_fraction:
                saturated_count += 1

    cmd_vx, cmd_vy, cmd_wz = command

    return {
        "phase": phase_name,
        "samples_total": len(phase_rows),
        "samples_steady": len(steady_rows),
        "steady_state_skip_s": steady_state_skip_s,
        "command": {
            "vx_mps": cmd_vx,
            "vy_mps": cmd_vy,
            "wz_radps": cmd_wz,
        },
        "tracking": {
            "mean_actual_vx_mps": mean(actual_vx),
            "mean_actual_vy_mps": mean(actual_vy),
            "mean_actual_wz_radps": mean(actual_wz),
            "vx_mae_mps": mae(actual_vx, cmd_vx),
            "vy_mae_mps": mae(actual_vy, cmd_vy),
            "wz_mae_radps": mae(actual_wz, cmd_wz),
            "vx_rmse_mps": rmse(actual_vx, cmd_vx),
            "vy_rmse_mps": rmse(actual_vy, cmd_vy),
            "wz_rmse_radps": rmse(actual_wz, cmd_wz),
        },
        "body": {
            "height_mean_m": mean(base_z),
            "height_min_m": min(base_z) if base_z else math.nan,
            "height_max_m": max(base_z) if base_z else math.nan,
            "roll_rms_rad": rms(roll),
            "roll_max_abs_rad": max_abs(roll),
            "pitch_rms_rad": rms(pitch),
            "pitch_max_abs_rad": max_abs(pitch),
        },
        "joints": {
            "max_abs_velocity_radps": max_abs(joint_velocities),
            "max_abs_effort_nm": max_abs(joint_efforts),
            "max_effort_limit_ratio": max_effort_ratio,
            "saturation_threshold_fraction": saturation_fraction,
            "saturated_joint_samples": saturated_count,
            "effort_joint_samples": effort_count,
            "saturation_percent": (
                100.0 * saturated_count / effort_count
                if effort_count
                else math.nan
            ),
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n===== Spot V3 MuJoCo evaluation summary =====")

    for phase in summary["phases"]:
        cmd = phase["command"]
        tracking = phase["tracking"]
        body = phase["body"]
        joints = phase["joints"]

        print(
            "\n"
            f"{phase['phase']}\n"
            f"  cmd:              "
            f"vx={cmd['vx_mps']:+.2f}  "
            f"vy={cmd['vy_mps']:+.2f}  "
            f"wz={cmd['wz_radps']:+.2f}\n"
            f"  mean actual:      "
            f"vx={tracking['mean_actual_vx_mps']:+.3f}  "
            f"vy={tracking['mean_actual_vy_mps']:+.3f}  "
            f"wz={tracking['mean_actual_wz_radps']:+.3f}\n"
            f"  RMSE:             "
            f"vx={tracking['vx_rmse_mps']:.3f}  "
            f"vy={tracking['vy_rmse_mps']:.3f}  "
            f"wz={tracking['wz_rmse_radps']:.3f}\n"
            f"  body height:      "
            f"mean={body['height_mean_m']:.3f} m  "
            f"min={body['height_min_m']:.3f} m\n"
            f"  attitude max abs: "
            f"roll={math.degrees(body['roll_max_abs_rad']):.2f} deg  "
            f"pitch={math.degrees(body['pitch_max_abs_rad']):.2f} deg\n"
            f"  joints:           "
            f"max|dq|={joints['max_abs_velocity_radps']:.2f} rad/s  "
            f"max|effort|={joints['max_abs_effort_nm']:.1f} Nm  "
            f"sat={joints['saturation_percent']:.3f}%"
        )


def save_outputs(
    *,
    rows: list[dict[str, Any]],
    output_root: Path,
    checkpoint: Path,
    step_dt: float,
    steady_state_skip_s: float,
    saturation_fraction: float,
    terminated_early: bool,
    termination_step: int | None,
    seed: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"spot_v3_mujoco_eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    columns = list(BASE_COLUMNS)
    for name in JOINT_NAMES:
        columns.extend(
            (
                f"{name}_q",
                f"{name}_dq",
                f"{name}_effort",
                f"{name}_action",
            )
        )

    csv_path = run_dir / "samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    phase_summaries = [
        summarize_phase(
            rows,
            phase_name=name,
            command=(vx, vy, wz),
            steady_state_skip_s=steady_state_skip_s,
            saturation_fraction=saturation_fraction,
        )
        for name, _duration, vx, vy, wz in PHASES
        if any(row["phase"] == name for row in rows)
    ]

    summary = {
        "schema_version": 1,
        "simulator": "mujoco",
        "task_id": TASK_ID,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "physics_hz": float(PHYSICS_FREQUENCY_HZ),
        "policy_frequency_hz": float(POLICY_FREQUENCY_HZ),
        "policy_step_s": float(step_dt),
        "low_level_kp": float(V3_KP),
        "low_level_kd": float(V3_KD),
        "hip_effort_limit_nm": float(V3_HIP_EFFORT_LIMIT),
        "knee_effort_limit_nm": float(V3_KNEE_EFFORT_LIMIT),
        "samples_recorded": len(rows),
        "recorded_duration_s": float(rows[-1]["t_s"]) if rows else 0.0,
        "terminated_early": terminated_early,
        "termination_step": termination_step,
        "phases": phase_summaries,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    print_summary(summary)
    print()
    print(f"[PASS] CSV:     {csv_path}")
    print(f"[PASS] Summary: {summary_path}")

    return run_dir


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.steady_state_skip_s < 0.0:
        raise ValueError("--steady-state-skip-s cannot be negative.")
    if not 0.0 < args.saturation_fraction <= 1.0:
        raise ValueError("--saturation-fraction must be in (0, 1].")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Task:       {TASK_ID}")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Device:     {device}")
    print(f"[INFO] Seed:       {args.seed}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed

    # The normal play config already disables observation corruption, curriculum,
    # pushes, friction randomization, COM randomization and encoder bias.
    #
    # For this benchmark we additionally remove the randomized root reset so the
    # robot starts from the exact V3 nominal pose instead of a random x/y/z/yaw
    # perturbation. Joint reset remains at zero offset from the home pose.
    env_cfg.events.pop("reset_base", None)

    # We want a single uninterrupted benchmark. Play mode already uses an
    # extremely long episode, and auto_reset=False additionally ensures that a
    # real fall/termination is reported instead of silently resetting the robot.
    env_cfg.auto_reset = False

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode=None,
    )
    wrapped_env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    rows: list[dict[str, Any]] = []
    terminated_early = False
    termination_step: int | None = None

    try:
        step_dt = float(env.step_dt)

        expected_step_dt = 1.0 / float(POLICY_FREQUENCY_HZ)
        if not math.isclose(step_dt, expected_step_dt, abs_tol=1e-12):
            raise RuntimeError(
                f"Unexpected policy step: {step_dt}; "
                f"expected {expected_step_dt}."
            )

        if not math.isclose(
            float(env.physics_dt),
            1.0 / float(PHYSICS_FREQUENCY_HZ),
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Unexpected physics step: {env.physics_dt}; "
                f"expected {1.0 / float(PHYSICS_FREQUENCY_HZ)}."
            )

        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(
            wrapped_env,
            asdict(agent_cfg),
            log_dir=None,
            device=device,
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        command_term = env.command_manager.get_term(COMMAND_NAME)
        robot = command_term.robot

        actual_joint_names = tuple(robot.joint_names)
        actual_actuator_names = tuple(robot.actuator_names)

        if actual_joint_names != tuple(JOINT_NAMES):
            raise RuntimeError(
                "Joint order mismatch.\n"
                f"Expected: {tuple(JOINT_NAMES)}\n"
                f"Actual:   {actual_joint_names}"
            )
        if actual_actuator_names != tuple(JOINT_NAMES):
            raise RuntimeError(
                "Actuator order mismatch.\n"
                f"Expected: {tuple(JOINT_NAMES)}\n"
                f"Actual:   {actual_actuator_names}"
            )

        freeze_command_generator(command_term)

        total_steps = sum(
            phase_steps(duration, step_dt)
            for _name, duration, _vx, _vy, _wz in PHASES
        )
        print(
            f"[INFO] Physics:    {1.0 / env.physics_dt:.0f} Hz\n"
            f"[INFO] Policy:     {1.0 / step_dt:.0f} Hz\n"
            f"[INFO] Steps:      {total_steps}\n"
            f"[INFO] Duration:   {total_steps * step_dt:.2f} s"
        )

        global_step = 0

        for phase_name, duration_s, vx, vy, wz in PHASES:
            count = phase_steps(duration_s, step_dt)
            print(
                f"[PHASE] {phase_name:14s} "
                f"cmd=[{vx:+.2f}, {vy:+.2f}, {wz:+.2f}] "
                f"duration={duration_s:.1f}s"
            )

            set_command(command_term, vx, vy, wz)

            for phase_step in range(count):
                # Re-assert the deterministic command before observation
                # construction. This guarantees the actor sees the exact
                # benchmark command at every 50 Hz policy step.
                set_command(command_term, vx, vy, wz)

                obs_td = wrapped_env.get_observations().to(device)

                with torch.inference_mode():
                    action = policy(obs_td)

                _obs, _reward, dones, _extras = wrapped_env.step(action)

                global_step += 1
                phase_t_s = (phase_step + 1) * step_dt
                t_s = global_step * step_dt

                rows.append(
                    record_sample(
                        robot=robot,
                        action=action,
                        t_s=t_s,
                        phase=phase_name,
                        phase_t_s=phase_t_s,
                        vx=vx,
                        vy=vy,
                        wz=wz,
                    )
                )

                if int(dones[0].item()) != 0:
                    terminated_early = True
                    termination_step = global_step
                    print(
                        "[FAIL] Environment terminated during "
                        f"{phase_name} at t={t_s:.3f} s."
                    )
                    break

            if terminated_early:
                break

        run_dir = save_outputs(
            rows=rows,
            output_root=output_root,
            checkpoint=checkpoint,
            step_dt=step_dt,
            steady_state_skip_s=args.steady_state_skip_s,
            saturation_fraction=args.saturation_fraction,
            terminated_early=terminated_early,
            termination_step=termination_step,
            seed=args.seed,
        )

        if terminated_early:
            raise RuntimeError(
                "Benchmark terminated early. Partial results were saved to "
                f"{run_dir}"
            )

        print(
            "\n[PASS] MuJoCo benchmark complete. "
            "Upload samples.csv and summary.json for the Gazebo comparison."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
