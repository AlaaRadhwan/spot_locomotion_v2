#!/usr/bin/env python3
"""Export the approved Spot V3 PPO checkpoint for ROS 2 / Gazebo deployment.

This script:
1. Loads the registered Mjlab-Velocity-Flat-Spot-V3 task.
2. Loads the requested RSL-RL checkpoint.
3. Exports the actor to TorchScript as policy_jit.pt.
4. Attempts an ONNX export as policy.onnx.
5. Writes policy_metadata.json for the ROS spot_policy package.
6. Verifies the TorchScript output against the loaded inference policy.

Example:
    uv run python tools/export_spot_v3_policy.py \
        --checkpoint logs/rsl_rl/spot_velocity_v3/<RUN>/model_550.pt \
        --output-dir exported/spot_v3_model_550
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

# Importing the task module registers the Spot tasks with MjLab.
import spot_locomotion_v2.tasks.velocity  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from spot_locomotion_v2.robot.spot_v3_constants import (
    DECIMATION,
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    PHYSICS_TIMESTEP,
    POLICY_FREQUENCY_HZ,
)

TASK_ID = "Mjlab-Velocity-Flat-Spot-V3-Robust"

ACTION_SCALES = (
    0.30, 0.40, 0.50,
    0.30, 0.40, 0.50,
    0.30, 0.40, 0.50,
    0.30, 0.40, 0.50,
)

OBSERVATION_LAYOUT = [
    {"name": "base_lin_vel", "shape": [3], "start": 0, "end": 3},
    {"name": "base_ang_vel", "shape": [3], "start": 3, "end": 6},
    {"name": "projected_gravity", "shape": [3], "start": 6, "end": 9},
    {"name": "joint_pos", "shape": [12], "start": 9, "end": 21},
    {"name": "joint_vel", "shape": [12], "start": 21, "end": 33},
    {"name": "actions", "shape": [12], "start": 33, "end": 45},
    {"name": "command", "shape": [3], "start": 45, "end": 48},
]

OBSERVATION_SIZE = 48
ACTION_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the RSL-RL model_*.pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exported/spot_v3"),
        help="Directory for policy_jit.pt, policy.onnx, and metadata.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference/export device. Default: cuda:0 if available, else cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Task:       {TASK_ID}")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Device:     {device}")
    print(f"[INFO] Output dir: {output_dir}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode=None,
    )
    wrapped_env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    try:
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

        inference_policy = runner.get_inference_policy(device=device)

        # Export TorchScript. This is what the existing ROS policy node loads.
        runner.export_policy_to_jit(
            str(output_dir),
            filename="policy_jit.pt",
        )

        jit_path = output_dir / "policy_jit.pt"
        if not jit_path.is_file():
            raise RuntimeError(f"TorchScript export was not created: {jit_path}")

        # ONNX is useful as a secondary portable artifact, but ROS currently
        # uses the TorchScript model.
        onnx_path = output_dir / "policy.onnx"
        try:
            runner.export_policy_to_onnx(
                str(output_dir),
                filename="policy.onnx",
            )
            print(f"[PASS] ONNX:       {onnx_path}")
        except Exception as exc:
            print(f"[WARN] ONNX export skipped/failed: {exc}")

        # ------------------------------------------------------------------
        # Verify the TorchScript model against the loaded inference actor.
        #
        # RSL-RL 5.x uses a TensorDict for the live MLPModel, while the
        # exported JIT wrapper intentionally accepts a single pre-concatenated
        # tensor. Use a real environment observation so both paths receive
        # exactly the same values.
        # ------------------------------------------------------------------
        scripted = torch.jit.load(str(jit_path), map_location=device)
        scripted.eval()

        obs_td = wrapped_env.get_observations().to(device)

        actor_groups = list(inference_policy.obs_groups)
        if not actor_groups:
            raise RuntimeError("Inference policy has no observation groups.")

        missing_groups = [name for name in actor_groups if name not in obs_td.keys()]
        if missing_groups:
            raise RuntimeError(
                "Missing actor observation groups from environment: "
                f"{missing_groups}; available={list(obs_td.keys())}"
            )

        flat_obs = torch.cat(
            [obs_td[name] for name in actor_groups],
            dim=-1,
        )

        if flat_obs.shape[-1] != OBSERVATION_SIZE:
            raise RuntimeError(
                "Unexpected actor observation size: "
                f"{flat_obs.shape[-1]}; expected {OBSERVATION_SIZE}. "
                f"Actor groups: {actor_groups}"
            )

        with torch.inference_mode():
            reference = inference_policy(obs_td)
            exported = scripted(flat_obs)
            max_error = torch.max(torch.abs(reference - exported)).item()

        if max_error > 1e-5:
            raise RuntimeError(
                "TorchScript verification failed: "
                f"max |reference-exported| = {max_error:.8e}"
            )

        # Verify the exported model contract expected by ROS.
        with torch.inference_mode():
            out = scripted(
                torch.zeros(
                    (1, OBSERVATION_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
            )

        if tuple(out.shape) != (1, ACTION_SIZE):
            raise RuntimeError(
                f"Unexpected TorchScript output shape: {tuple(out.shape)}; "
                f"expected (1, {ACTION_SIZE})"
            )

        metadata = {
            "schema_version": 2,
            "task_id": TASK_ID,
            "checkpoint": str(checkpoint),
            "policy_frequency_hz": float(POLICY_FREQUENCY_HZ),
            "physics_timestep_s": float(PHYSICS_TIMESTEP),
            "control_decimation": int(DECIMATION),
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
            "joint_order": list(JOINT_NAMES),
            "default_joint_positions": [float(x) for x in HOME_JOINT_POSITIONS],
            "action_scales": [float(x) for x in ACTION_SCALES],
            "observation_order": OBSERVATION_LAYOUT,
            "command_order": ["vx", "vy", "wz"],
            "command_units": ["m/s", "m/s", "rad/s"],
            "deployment_notes": {
                "gazebo_physics_hz": 2000,
                "ros2_control_hz": 2000,
                "low_level_kp": 300.0,
                "low_level_kd": 40.0,
                "hip_effort_limit_nm": 45.0,
                "knee_effort_limit_nm": 115.0,
            },
        }

        metadata_path = output_dir / "policy_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        print()
        print("[PASS] Spot V3 policy export complete.")
        print(f"[PASS] TorchScript: {jit_path}")
        print(f"[PASS] Metadata:    {metadata_path}")
        print(
            "[PASS] Contract:    "
            f"input=(1,{OBSERVATION_SIZE}) output=(1,{ACTION_SIZE})"
        )
        print(
            "[PASS] JIT parity:  "
            f"max absolute error = {max_error:.8e}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
