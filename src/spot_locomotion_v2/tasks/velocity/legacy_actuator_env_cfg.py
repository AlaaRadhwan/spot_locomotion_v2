"""Legacy-actuator A/B environment configuration.

This environment reuses the current Spot V2 flat-velocity task unchanged
and replaces only the robot actuator configuration with the legacy
high-authority actuator setup.

It is intended strictly as a diagnostic A/B experiment.
"""

from mjlab.envs import ManagerBasedRlEnvCfg

from spot_locomotion_v2.robot.spot_constants_legacy_actuator import (
    get_spot_legacy_actuator_robot_cfg,
)

from .flat_env_cfg import make_spot_flat_velocity_env_cfg


def make_spot_legacy_actuator_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Spot V2 task with the legacy actuator configuration.

    Everything is inherited from the normal V2 flat-velocity task:

    - physics timestep
    - policy frequency
    - observations
    - actions
    - action scales
    - commands
    - rewards
    - terminations
    - events
    - contact sensors
    - diagnostics

    The only intended behavioral difference is the robot actuator model.
    """

    # Start from the exact current V2 flat-locomotion task.
    cfg = make_spot_flat_velocity_env_cfg(
        play=play,
    )

    # Replace only the robot entity with the legacy-actuator version.
    cfg.scene.entities["robot"] = (
        get_spot_legacy_actuator_robot_cfg()
    )

    # These two metrics in the normal V2 environment explicitly assume
    # the realistic V2 torque limits:
    #
    #   hips:  69.52 Nm
    #   knees: 164.15 Nm
    #
    # Those limits are not valid for the legacy ±1000 Nm experiment,
    # so remove the metrics rather than log misleading values.
    cfg.metrics.pop(
        "max_actuator_utilization",
        None,
    )

    cfg.metrics.pop(
        "actuator_saturation_fraction",
        None,
    )

    return cfg