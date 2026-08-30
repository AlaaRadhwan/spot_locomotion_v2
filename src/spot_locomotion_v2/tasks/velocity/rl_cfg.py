"""PPO configuration for V2 Spot velocity training."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def spot_v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Preserve the PPO design used by the successful V1 task."""

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
        
        experiment_name="spot_velocity_v2",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=10_000,
    )
