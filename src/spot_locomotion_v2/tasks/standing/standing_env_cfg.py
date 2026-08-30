"""Minimal Spot standing environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from spot_locomotion_v2.robot.spot_constants import (
    DECIMATION,
    PHYSICS_TIMESTEP,
    get_spot_robot_cfg,
)

ACTION_SCALE = 0.25

# Joint-position targets after scaling and offsetting. These bounds
# are equivalent to clamping every raw policy action to [-1, 1]
ACTION_TARGET_LIMITS = {
    ".*_hx": (-0.25, 0.25),
    ".*_hy": (0.79, 1.29),
    ".*_kn": (-2.05, -1.55),
}

def make_standing_env_cfg(
    num_envs: int = 1,
) -> ManagerBasedRlEnvCfg:
    """Create the minimal flat-ground standing environment."""

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms={**actor_terms},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=ACTION_SCALE,
            use_default_offset=True,
            clip=ACTION_TARGET_LIMITS,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="plane",
            ),
            entities={
                "robot": get_spot_robot_cfg(),
            },
            num_envs=num_envs,
            env_spacing=2.0,
        ),
        observations=observations,
        actions=actions,

        rewards={},
        terminations={},
        commands={},
        curriculum={},

        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=PHYSICS_TIMESTEP,
                integrator="implicitfast",
                cone="elliptic",
                impratio=100.0,
            ),
        ),
        decimation=DECIMATION,
        episode_length_s=5.0,
        seed=0,
    )

    