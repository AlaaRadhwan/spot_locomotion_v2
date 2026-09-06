"""Spot V3 flat-terrain velocity environment.

This deliberately reuses the already-working V2 flat task definition and
changes only the plant-specific pieces:

- Spot V3 EntityCfg / Isaac-derived rigid-body model
- V3 native BuiltinPosition actuator baseline
- V3 effort limits used by actuator-utilization diagnostics

The task semantics, observations, action scaling, rewards, commands, sensors,
500/50 Hz timing, and MuJoCo solver settings remain unchanged for the first
V3 integration test.
"""

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from spot_locomotion_v2.robot.spot_v3_constants import (
    DECIMATION,
    PHYSICS_TIMESTEP,
    V3_HIP_EFFORT_LIMIT,
    V3_KNEE_EFFORT_LIMIT,
    get_spot_v3_robot_cfg,
)

from .flat_env_cfg import make_spot_flat_velocity_env_cfg


class V3ActuatorUtilizationMetric:
    """Measure V3 actuator utilization or saturation.

    This is intentionally separate from the V2 metric because the V2 task
    closes over the V2 effort-limit constants.
    """

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        robot = env.scene[cfg.params["entity_name"]]

        effort_limits: list[float] = []
        for name in robot.actuator_names:
            if name.endswith(("_hx", "_hy")):
                effort_limits.append(float(V3_HIP_EFFORT_LIMIT))
            elif name.endswith("_kn"):
                effort_limits.append(float(V3_KNEE_EFFORT_LIMIT))
            else:
                raise ValueError(
                    "No V3 effort-limit mapping is defined for "
                    f"actuator {name!r}."
                )

        self.effort_limits = torch.tensor(
            effort_limits,
            device=env.device,
            dtype=torch.float32,
        )
        self.statistic = cfg.params["statistic"]
        self.saturation_threshold = float(
            cfg.params.get("saturation_threshold", 0.99)
        )

    def __call__(
        self,
        env,
        entity_name: str,
        statistic: str,
        saturation_threshold: float = 0.99,
    ) -> torch.Tensor:
        del statistic, saturation_threshold

        robot = env.scene[entity_name]
        utilization = (
            torch.abs(robot.data.actuator_force)
            / self.effort_limits.unsqueeze(0)
        )

        if self.statistic == "max":
            return torch.max(utilization, dim=1).values

        if self.statistic == "saturation_fraction":
            return torch.mean(
                (
                    utilization
                    >= self.saturation_threshold
                ).float(),
                dim=1,
            )

        raise ValueError(
            f"Unknown V3 actuator statistic: {self.statistic!r}"
        )


def make_spot_v3_flat_velocity_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the first Spot V3 flat-terrain velocity environment.

    Start from the current V2 task so that the first V3 test changes the plant,
    not the locomotion task. Then replace only V3-specific components.
    """

    cfg = make_spot_flat_velocity_env_cfg(play=play)

    # ------------------------------------------------------------------
    # V3 plant
    # ------------------------------------------------------------------
    cfg.scene.entities = {
        "robot": get_spot_v3_robot_cfg(),
    }

    # Keep the simulation options explicit here rather than relying on the
    # robot MJCF to propagate global options through MjSpec attachment.
    cfg.sim.mujoco.timestep = PHYSICS_TIMESTEP
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 100.0
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.decimation = DECIMATION

    # ------------------------------------------------------------------
    # V3 effort-limit diagnostics
    # ------------------------------------------------------------------
    # The V2 task's metric class uses V2 torque-limit constants. Replace
    # those two metric terms so V3 diagnostics normalize against 45/115 Nm.
    cfg.metrics["max_actuator_utilization"] = MetricsTermCfg(
        func=V3ActuatorUtilizationMetric,
        params={
            "entity_name": "robot",
            "statistic": "max",
        },
        reduce="max",
    )

    cfg.metrics["actuator_saturation_fraction"] = MetricsTermCfg(
        func=V3ActuatorUtilizationMetric,
        params={
            "entity_name": "robot",
            "statistic": "saturation_fraction",
            "saturation_threshold": 0.99,
        },
    )

    return cfg
