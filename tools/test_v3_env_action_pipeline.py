#!/usr/bin/env python3
"""Verify Spot V3's actual MjLab action pipeline before PPO training.

Checks:
1. Joint/action ordering.
2. Resolved per-joint action scales.
3. Default/home position offsets.
4. Exact affine mapping:
       processed_target = raw_action * scale + home
5. A small physical test through env.step():
       all four knees receive raw action +0.10
       -> expected q_ref shift +0.05 rad
   then return to zero action.

This uses the real ManagerBasedRlEnv and JointPositionActionCfg path.
"""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv

from spot_locomotion_v2.robot.spot_v3_constants import (
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
)
from spot_locomotion_v2.tasks.velocity.flat_env_cfg_v3 import (
    make_spot_v3_flat_velocity_env_cfg,
)


EXPECTED_SCALE = torch.tensor(
    [
        0.30, 0.40, 0.50,
        0.30, 0.40, 0.50,
        0.30, 0.40, 0.50,
        0.30, 0.40, 0.50,
    ],
    dtype=torch.float32,
)

KNEE_INDICES = (2, 5, 8, 11)


def as_row_tensor(value, action_dim: int, device: str) -> torch.Tensor:
    """Normalize scalar or tensor scale/offset to shape (1, action_dim)."""
    if isinstance(value, (float, int)):
        return torch.full(
            (1, action_dim),
            float(value),
            device=device,
            dtype=torch.float32,
        )

    tensor = value.detach().to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.shape[0] != 1:
        tensor = tensor[:1]

    return tensor


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = make_spot_v3_flat_velocity_env_cfg(play=True)
    cfg.scene.num_envs = 1

    # We are testing the action path, not reset behavior.
    cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=cfg, device=device)

    try:
        env.reset()

        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")

        print("\n=== V3 environment action-pipeline audit ===")
        print(f"Device:             {device}")
        print(f"Environment dt:     {env.step_dt:.6f} s")
        print(f"Action dimension:   {env.action_manager.total_action_dim}")
        print(f"Robot joints:       {len(robot.joint_names)}")

        target_names = tuple(action_term.target_names)

        print("\nResolved action target order:")
        for i, name in enumerate(target_names):
            print(f"  {i:2d}: {name}")

        if target_names != JOINT_NAMES:
            raise RuntimeError(
                "FAIL: action target ordering does not match JOINT_NAMES.\n"
                f"Expected: {JOINT_NAMES}\n"
                f"Actual:   {target_names}"
            )

        if tuple(robot.joint_names) != JOINT_NAMES:
            raise RuntimeError(
                "FAIL: robot joint ordering does not match JOINT_NAMES.\n"
                f"Expected: {JOINT_NAMES}\n"
                f"Actual:   {tuple(robot.joint_names)}"
            )

        action_dim = env.action_manager.total_action_dim

        scale = as_row_tensor(action_term.scale, action_dim, device)
        offset = as_row_tensor(action_term.offset, action_dim, device)

        expected_scale = EXPECTED_SCALE.to(device).unsqueeze(0)
        expected_offset = torch.tensor(
            HOME_JOINT_POSITIONS,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)

        scale_error = torch.max(torch.abs(scale - expected_scale)).item()
        offset_error = torch.max(torch.abs(offset - expected_offset)).item()

        print("\nResolved affine mapping:")
        print(" idx  joint      scale     offset/home [rad]")
        for i, name in enumerate(target_names):
            print(
                f" {i:2d}   {name:6s}   "
                f"{scale[0, i].item():7.3f}   "
                f"{offset[0, i].item():+10.6f}"
            )

        print(f"\nMax scale error:   {scale_error:.3e}")
        print(f"Max offset error:  {offset_error:.3e}")

        if scale_error > 1e-6:
            raise RuntimeError("FAIL: resolved action scales are wrong.")

        if offset_error > 1e-6:
            raise RuntimeError("FAIL: resolved default/home offsets are wrong.")

        # --------------------------------------------------------------
        # Unique raw-action mapping check.
        # --------------------------------------------------------------
        # Different tiny value for every joint makes ordering mistakes obvious.
        raw_probe = (
            torch.arange(
                1,
                action_dim + 1,
                device=device,
                dtype=torch.float32,
            )
            * 0.01
        ).unsqueeze(0)

        env.action_manager.process_action(raw_probe)

        # Diagnostic access to the concrete BaseAction implementation.
        processed = action_term._processed_actions.detach().clone()
        expected_processed = raw_probe * expected_scale + expected_offset

        mapping_error = torch.max(
            torch.abs(processed - expected_processed)
        ).item()

        print("\nUnique raw-action probe:")
        print(
            " idx  joint      raw       expected q_ref     actual q_ref"
        )
        for i, name in enumerate(target_names):
            print(
                f" {i:2d}   {name:6s}   "
                f"{raw_probe[0, i].item():+7.3f}   "
                f"{expected_processed[0, i].item():+14.6f}   "
                f"{processed[0, i].item():+12.6f}"
            )

        print(f"\nMax mapping error: {mapping_error:.3e}")

        if mapping_error > 1e-6:
            raise RuntimeError(
                "FAIL: raw action -> processed joint target mapping is wrong."
            )

        # Clear the probe before physical stepping.
        zero_action = torch.zeros(
            (1, action_dim),
            device=device,
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Small physical test through env.step().
        # --------------------------------------------------------------
        # Settle for one second with true zero policy action.
        for _ in range(50):
            env.step(zero_action)

        before_height = robot.data.root_link_pos_w[0, 2].item()
        before_knees = (
            robot.data.joint_pos[0, list(KNEE_INDICES)]
            .detach()
            .cpu()
            .clone()
        )

        knee_probe = zero_action.clone()
        knee_probe[0, list(KNEE_INDICES)] = 0.10

        # Knee scale = 0.50, so +0.10 raw means +0.05 rad q_ref.
        expected_knee_target = -1.80 + 0.10 * 0.50

        peak_torque = 0.0

        # Hold for 0.5 s = 25 policy steps.
        for _ in range(25):
            env.step(knee_probe)
            peak_torque = max(
                peak_torque,
                float(torch.max(torch.abs(robot.data.actuator_force)).item()),
            )

        during_height = robot.data.root_link_pos_w[0, 2].item()
        during_knees = (
            robot.data.joint_pos[0, list(KNEE_INDICES)]
            .detach()
            .cpu()
            .clone()
        )

        # Return to home/zero action for 1.0 s.
        for _ in range(50):
            env.step(zero_action)
            peak_torque = max(
                peak_torque,
                float(torch.max(torch.abs(robot.data.actuator_force)).item()),
            )

        after_height = robot.data.root_link_pos_w[0, 2].item()
        after_knees = (
            robot.data.joint_pos[0, list(KNEE_INDICES)]
            .detach()
            .cpu()
            .clone()
        )

        print("\n=== Small physical action test ===")
        print("Raw action on all four knees: +0.10")
        print("Knee action scale:            0.50")
        print(
            f"Expected knee q_ref:          "
            f"{expected_knee_target:+.6f} rad"
        )

        print("\nMean measured knee position:")
        print(f"  before probe: {before_knees.mean().item():+.6f} rad")
        print(f"  during probe: {during_knees.mean().item():+.6f} rad")
        print(f"  after return: {after_knees.mean().item():+.6f} rad")

        print("\nBase height:")
        print(f"  before probe: {before_height:.6f} m")
        print(f"  during probe: {during_height:.6f} m")
        print(f"  after return: {after_height:.6f} m")

        print(f"\nPeak |actuator torque|: {peak_torque:.3f} N*m")

        values = (
            before_height,
            during_height,
            after_height,
            peak_torque,
            *before_knees.tolist(),
            *during_knees.tolist(),
            *after_knees.tolist(),
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise RuntimeError(
                "FAIL: non-finite state encountered during physical action test."
            )

        print("\nPASS: V3 MjLab action pipeline is internally consistent.")
        print(
            "The next step is a fresh PPO characterization run from scratch."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
