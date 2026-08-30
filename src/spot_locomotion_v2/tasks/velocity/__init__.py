"""Register the V2 Spot flat-velocity task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .flat_env_cfg import make_spot_flat_velocity_env_cfg
from .rl_cfg import spot_v2_ppo_runner_cfg


TASK_ID = "Mjlab-Velocity-Flat-Spot-V2"


register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_spot_flat_velocity_env_cfg(),
    play_env_cfg=make_spot_flat_velocity_env_cfg(play=True),
    rl_cfg=spot_v2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
