"""Register the V2 Spot flat-velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_env_cfg import make_spot_flat_velocity_env_cfg
from .legacy_actuator_env_cfg import (
    make_spot_legacy_actuator_env_cfg,
)
from .rl_cfg import spot_v2_ppo_runner_cfg


TASK_ID = "Mjlab-Velocity-Flat-Spot-V2"
LEGACY_ACTUATOR_TASK_ID = (
    "Mjlab-Velocity-Flat-Spot-V2-LegacyActuator"
)


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_spot_flat_velocity_env_cfg(),
    play_env_cfg=make_spot_flat_velocity_env_cfg(play=True),
    rl_cfg=spot_v2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


register_mjlab_task(
    task_id=LEGACY_ACTUATOR_TASK_ID,
    env_cfg=make_spot_legacy_actuator_env_cfg(),
    play_env_cfg=make_spot_legacy_actuator_env_cfg(play=True),
    rl_cfg=spot_v2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)