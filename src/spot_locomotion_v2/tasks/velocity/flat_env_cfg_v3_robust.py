"""Spot V3 robust flat-terrain velocity environment.

Sibling task for the validated Mjlab-Velocity-Flat-Spot-V3 baseline.

Robust V1 changes only:
1. explicit pure-yaw command coverage;
2. moderate interval velocity pushes;
3. modest reset attitude/velocity perturbations during training.

The validated V3 plant, actuator model, rewards, observations, action space,
solver settings, and PPO-facing semantics remain unchanged.

Playback keeps pushes active so robustness is visible. The inherited V3
play configuration still disables observation corruption and the other
training randomizations (friction, COM, encoder bias).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)

from .flat_env_cfg_v3 import make_spot_v3_flat_velocity_env_cfg


@dataclass(kw_only=True)
class RobustVelocityCommandCfg(UniformVelocityCommandCfg):
    """Flat-ground command sampler with explicit pure-yaw coverage.

    The category probabilities below are conditional on a non-standing
    environment. Standing probability is controlled by rel_standing_envs.

    Moving-command distribution:
      65% longitudinal/diagonal + yaw
      20% pure lateral
      15% pure yaw
    """

    lin_vel_x_abs_range: tuple[float, float] = (0.35, 0.65)
    forward_probability: float = 0.50
    general_lin_vel_y_range: tuple[float, float] = (-0.20, 0.20)

    pure_lateral_probability: float = 0.20
    lin_vel_y_abs_range: tuple[float, float] = (0.20, 0.40)
    left_probability: float = 0.50

    pure_yaw_probability: float = 0.15
    yaw_abs_range: tuple[float, float] = (0.25, 0.60)
    yaw_left_probability: float = 0.50

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.heading_command:
            raise ValueError(
                "RobustVelocityCommandCfg expects heading_command=False."
            )
        if self.init_velocity_prob != 0.0:
            raise ValueError(
                "RobustVelocityCommandCfg requires init_velocity_prob=0.0."
            )

        min_x, max_x = self.lin_vel_x_abs_range
        min_y, max_y = self.lin_vel_y_abs_range
        min_y_general, max_y_general = self.general_lin_vel_y_range
        min_yaw, max_yaw = self.yaw_abs_range

        if min_x <= 0.0 or max_x < min_x:
            raise ValueError("Invalid lin_vel_x_abs_range.")
        if min_y <= 0.0 or max_y < min_y:
            raise ValueError("Invalid lin_vel_y_abs_range.")
        if min_y_general > max_y_general:
            raise ValueError("Invalid general_lin_vel_y_range.")
        if min_yaw <= 0.0 or max_yaw < min_yaw:
            raise ValueError("Invalid yaw_abs_range.")

        for name, probability in (
            ("forward_probability", self.forward_probability),
            ("pure_lateral_probability", self.pure_lateral_probability),
            ("left_probability", self.left_probability),
            ("pure_yaw_probability", self.pure_yaw_probability),
            ("yaw_left_probability", self.yaw_left_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.pure_lateral_probability + self.pure_yaw_probability > 1.0:
            raise ValueError(
                "pure_lateral_probability + pure_yaw_probability "
                "must not exceed 1."
            )

    def build(self, env):
        return RobustVelocityCommand(self, env)


class RobustVelocityCommand(UniformVelocityCommand):
    """Sample mutually exclusive flat-ground motion categories."""

    cfg: RobustVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Let the base class initialize command bookkeeping such as the
        # standing mask. All non-standing commands are overwritten below.
        super()._resample_command(env_ids)

        if len(env_ids) == 0:
            return

        standing_mask = self.is_standing_env[env_ids]
        moving_ids = env_ids[~standing_mask]
        standing_ids = env_ids[standing_mask]

        if len(standing_ids) > 0:
            self.vel_command_b[standing_ids, :] = 0.0

        if len(moving_ids) > 0:
            category = torch.rand(len(moving_ids), device=self.device)

            pure_yaw_mask = category < self.cfg.pure_yaw_probability
            pure_lateral_mask = (
                (category >= self.cfg.pure_yaw_probability)
                & (
                    category
                    < self.cfg.pure_yaw_probability
                    + self.cfg.pure_lateral_probability
                )
            )
            locomotion_mask = ~(pure_yaw_mask | pure_lateral_mask)

            yaw_ids = moving_ids[pure_yaw_mask]
            lateral_ids = moving_ids[pure_lateral_mask]
            locomotion_ids = moving_ids[locomotion_mask]

            # Longitudinal/diagonal locomotion, optionally with yaw.
            if len(locomotion_ids) > 0:
                min_x, max_x = self.cfg.lin_vel_x_abs_range
                x_mag = torch.empty(
                    len(locomotion_ids), device=self.device
                ).uniform_(min_x, max_x)

                forward = (
                    torch.rand(len(locomotion_ids), device=self.device)
                    < self.cfg.forward_probability
                )
                x_sign = torch.where(
                    forward,
                    torch.ones_like(x_mag),
                    -torch.ones_like(x_mag),
                )

                min_y, max_y = self.cfg.general_lin_vel_y_range
                lateral = torch.empty(
                    len(locomotion_ids), device=self.device
                ).uniform_(min_y, max_y)

                min_wz, max_wz = self.cfg.ranges.ang_vel_z
                yaw = torch.empty(
                    len(locomotion_ids), device=self.device
                ).uniform_(min_wz, max_wz)

                self.vel_command_b[locomotion_ids, 0] = x_sign * x_mag
                self.vel_command_b[locomotion_ids, 1] = lateral
                self.vel_command_b[locomotion_ids, 2] = yaw

            # Pure lateral: x = 0, wz = 0.
            if len(lateral_ids) > 0:
                min_y, max_y = self.cfg.lin_vel_y_abs_range
                y_mag = torch.empty(
                    len(lateral_ids), device=self.device
                ).uniform_(min_y, max_y)

                left = (
                    torch.rand(len(lateral_ids), device=self.device)
                    < self.cfg.left_probability
                )
                y_sign = torch.where(
                    left,
                    torch.ones_like(y_mag),
                    -torch.ones_like(y_mag),
                )

                self.vel_command_b[lateral_ids, 0] = 0.0
                self.vel_command_b[lateral_ids, 1] = y_sign * y_mag
                self.vel_command_b[lateral_ids, 2] = 0.0

            # Pure yaw: vx = vy = 0.
            if len(yaw_ids) > 0:
                min_wz, max_wz = self.cfg.yaw_abs_range
                yaw_mag = torch.empty(
                    len(yaw_ids), device=self.device
                ).uniform_(min_wz, max_wz)

                left_turn = (
                    torch.rand(len(yaw_ids), device=self.device)
                    < self.cfg.yaw_left_probability
                )
                yaw_sign = torch.where(
                    left_turn,
                    torch.ones_like(yaw_mag),
                    -torch.ones_like(yaw_mag),
                )

                self.vel_command_b[yaw_ids, 0] = 0.0
                self.vel_command_b[yaw_ids, 1] = 0.0
                self.vel_command_b[yaw_ids, 2] = yaw_sign * yaw_mag

        # World-frame mode is disabled, but synchronize this tensor so no
        # stale command remains from the base sampler.
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


def make_spot_v3_robust_flat_velocity_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the robust V1 Spot V3 flat-terrain task."""

    # Start from the validated V3 task. For play=True, its existing play
    # configuration disables observation corruption, foot-friction DR,
    # COM DR and encoder bias. We add the push event after this call so
    # pushes remain active during playback/video evaluation.
    cfg = make_spot_v3_flat_velocity_env_cfg(play=play)

    old_command = cfg.commands["twist"]
    if not isinstance(old_command, UniformVelocityCommandCfg):
        raise TypeError("Expected a UniformVelocityCommandCfg for 'twist'.")

    cfg.commands["twist"] = RobustVelocityCommandCfg(
        entity_name=old_command.entity_name,
        resampling_time_range=(3.0, 6.0),
        debug_vis=old_command.debug_vis,
        heading_command=False,
        heading_control_stiffness=old_command.heading_control_stiffness,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        rel_standing_envs=0.10,
        init_velocity_prob=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.65, 0.65),
            lin_vel_y=(-0.40, 0.40),
            ang_vel_z=(-0.60, 0.60),
            heading=None,
        ),
        viz=UniformVelocityCommandCfg.VizCfg(
            z_offset=0.5,
            scale=old_command.viz.scale,
        ),
        lin_vel_x_abs_range=(0.35, 0.65),
        forward_probability=0.50,
        general_lin_vel_y_range=(-0.20, 0.20),
        pure_lateral_probability=0.20,
        lin_vel_y_abs_range=(0.20, 0.40),
        left_probability=0.50,
        pure_yaw_probability=0.15,
        yaw_abs_range=(0.25, 0.60),
        yaw_left_probability=0.50,
    )

    # MjLab's push_by_setting_velocity adds a random increment to the
    # current root velocity. This is intentionally noticeable but still
    # below the generic velocity task's more aggressive default regime.
    cfg.events["push_robot"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.35, 0.35),
                "y": (-0.35, 0.35),
                "z": (-0.10, 0.10),
                "roll": (-0.20, 0.20),
                "pitch": (-0.20, 0.20),
                "yaw": (-0.40, 0.40),
            },
        },
    )

    # Training-only initial perturbations. Playback is easier to interpret
    # when it starts normally and robustness is judged from visible pushes.
    if not play:
        reset_base = cfg.events.get("reset_base")
        if reset_base is None:
            raise RuntimeError("Expected inherited 'reset_base' event.")

        reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.01, 0.05),
            "roll": (-math.radians(7.0), math.radians(7.0)),
            "pitch": (-math.radians(7.0), math.radians(7.0)),
            "yaw": (-math.pi, math.pi),
        }

        reset_base.params["velocity_range"] = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.05, 0.05),
            "yaw": (-0.20, 0.20),
        }

    return cfg
