"""PPO configuration for Spot V3 velocity training."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def spot_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Use the current V2 PPO architecture with a separate V3 log namespace."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="spot_velocity_v3",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
