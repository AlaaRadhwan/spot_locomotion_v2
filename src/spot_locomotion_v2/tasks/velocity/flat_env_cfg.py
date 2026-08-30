from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from spot_locomotion_v2.robot.spot_constants import (
    DECIMATION,
    HIP_TORQUE_LIMIT,
    KNEE_TORQUE_LIMIT,
    PHYSICS_TIMESTEP,
    get_spot_robot_cfg,
)


# Preserve the per-joint action mapping used by the successful V1 task.
# The policy output is multiplied by these values and added to the home pose.
SPOT_ACTION_SCALE: dict[str, float] = {
    ".*_hx": 0.30,
    ".*_hy": 0.40,
    ".*_kn": 0.50,
}

SPOT_BASE_BODY = "body"
SPOT_FOOT_SITES = (
    "FL_site",
    "FR_site",
    "HL_site",
    "HR_site",
)
SPOT_FOOT_GEOMS = (
    "FL",
    "FR",
    "HL",
    "HR",
)

@dataclass(kw_only=True)
class SplitVelocityCommandCfg(UniformVelocityCommandCfg):
    """Sample longitudinal, diagonal, and pure-lateral commands."""

    lin_vel_x_abs_range: tuple[float, float] = (0.35, 0.65)
    forward_probability: float = 0.5
    pure_lateral_probability: float = 0.20
    lin_vel_y_abs_range: tuple[float, float] = (0.20, 0.40)
    left_probability: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()

        min_x, max_x = self.lin_vel_x_abs_range
        min_y, max_y = self.lin_vel_y_abs_range

        if min_x <= 0.0 or max_x < min_x:
            raise ValueError("Invalid lin_vel_x_abs_range.")
        if min_y <= 0.0 or max_y < min_y:
            raise ValueError("Invalid lin_vel_y_abs_range.")

        for name, probability in (
            ("forward_probability", self.forward_probability),
            ("pure_lateral_probability", self.pure_lateral_probability),
            ("left_probability", self.left_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.init_velocity_prob != 0.0:
            raise ValueError(
                "SplitVelocityCommandCfg requires init_velocity_prob=0.0."
            )

    def build(self, env):
        return SplitVelocityCommand(self, env)


class SplitVelocityCommand(UniformVelocityCommand):
    """Sample longitudinal/diagonal or pure-lateral motion."""

    cfg: SplitVelocityCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        num_envs = len(env_ids)
        if num_envs == 0:
            return

        lateral_mask = (
            torch.rand(num_envs, device=self.device)
            < self.cfg.pure_lateral_probability
        )
        longitudinal_ids = env_ids[~lateral_mask]
        lateral_ids = env_ids[lateral_mask]

        if len(longitudinal_ids) > 0:
            min_x, max_x = self.cfg.lin_vel_x_abs_range
            magnitude = torch.empty(
                len(longitudinal_ids),
                device=self.device,
            ).uniform_(min_x, max_x)

            forward_mask = (
                torch.rand(len(longitudinal_ids), device=self.device)
                < self.cfg.forward_probability
            )
            direction = torch.where(
                forward_mask,
                torch.ones_like(magnitude),
                -torch.ones_like(magnitude),
            )
            self.vel_command_b[longitudinal_ids, 0] = direction * magnitude

        if len(lateral_ids) > 0:
            min_y, max_y = self.cfg.lin_vel_y_abs_range
            magnitude = torch.empty(
                len(lateral_ids),
                device=self.device,
            ).uniform_(min_y, max_y)

            left_mask = (
                torch.rand(len(lateral_ids), device=self.device)
                < self.cfg.left_probability
            )
            direction = torch.where(
                left_mask,
                torch.ones_like(magnitude),
                -torch.ones_like(magnitude),
            )

            self.vel_command_b[lateral_ids, 0] = 0.0
            self.vel_command_b[lateral_ids, 1] = direction * magnitude


def _moving_command_mask(
    env,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    """Return True for environments currently commanded to move."""

    command = env.command_manager.get_command(command_name)
    command_magnitude = (
        torch.linalg.vector_norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
    )
    return command_magnitude > command_threshold



def excessive_foot_air_time(
    env,
    sensor_name: str,
    command_name: str,
    command_threshold: float,
    max_air_time: float,
) -> torch.Tensor:
    """Penalize a foot only if it remains continuously airborne too long.

    This is the same simple safeguard used in the earlier successful Spot
    project. It does not reward a particular gait, cadence, duty factor, or
    diagonal pairing.
    """

    sensor = env.scene[sensor_name]

    if sensor.data.current_air_time is None:
        raise RuntimeError(
            f"Sensor '{sensor_name}' must use track_air_time=True."
        )

    air_time = sensor.data.current_air_time

    # Zero below the permitted air time. Above the threshold, scale the
    # longest airborne foot into a bounded [0, 1] penalty.
    excess = torch.clamp(
        (air_time - max_air_time) / max_air_time,
        min=0.0,
        max=1.0,
    )
    penalty = torch.max(excess, dim=1).values

    return penalty * _moving_command_mask(
        env,
        command_name,
        command_threshold,
    ).float()

class PerFootCompletedPhaseDuration:
    """Running mean completed air/stance duration for one specific foot."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        self.phase = cfg.params["phase"]
        if self.phase not in ("air", "stance"):
            raise ValueError(
                f"Unknown phase {self.phase!r}; expected 'air' or 'stance'."
            )

        sensor = env.scene[cfg.params["sensor_name"]]
        foot_name = cfg.params["foot_name"]

        try:
            self.foot_idx = list(sensor.primary_names).index(foot_name)
        except ValueError as exc:
            raise ValueError(
                f"Foot {foot_name!r} not found in "
                f"{list(sensor.primary_names)!r}."
            ) from exc

        self.duration_sum = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.event_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        command_threshold: float,
        phase: str,
        foot_name: str,
    ) -> torch.Tensor:
        del phase, foot_name

        sensor = env.scene[sensor_name]
        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )

        if self.phase == "air":
            durations = sensor.data.last_air_time
            completed = sensor.compute_first_contact(dt=env.step_dt)
        else:
            durations = sensor.data.last_contact_time
            completed = sensor.compute_first_air(dt=env.step_dt)

        if durations is None:
            raise RuntimeError(
                f"Sensor '{sensor_name}' must use track_air_time=True."
            )

        completed = completed[:, self.foot_idx] & active
        completed_float = completed.float()

        self.duration_sum += (
            durations[:, self.foot_idx] * completed_float
        )
        self.event_count += completed_float

        return self.duration_sum / torch.clamp(
            self.event_count,
            min=1.0,
        )

    def reset(self, env_ids) -> None:
        self.duration_sum[env_ids] = 0.0
        self.event_count[env_ids] = 0.0


class ActiveContactPatternFraction:
    """Fraction of commanded-moving time spent in a selected contact pattern."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        sensor = env.scene[cfg.params["sensor_name"]]
        primary_names = list(sensor.primary_names)

        self.indices = {
            name: primary_names.index(name)
            for name in ("FL", "FR", "HL", "HR")
        }
        self.pattern = cfg.params["pattern"]

        self.match_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.active_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        command_threshold: float,
        pattern: str,
    ) -> torch.Tensor:
        del pattern

        sensor = env.scene[sensor_name]
        current_contact_time = sensor.data.current_contact_time
        if current_contact_time is None:
            raise RuntimeError(
                f"Sensor '{sensor_name}' must use track_air_time=True."
            )

        contact = current_contact_time > 0.0

        fl = contact[:, self.indices["FL"]]
        fr = contact[:, self.indices["FR"]]
        hl = contact[:, self.indices["HL"]]
        hr = contact[:, self.indices["HR"]]

        diag_fl_hr = fl & hr & ~fr & ~hl
        diag_fr_hl = fr & hl & ~fl & ~hr
        all_four = fl & fr & hl & hr
        flight = ~(fl | fr | hl | hr)

        if self.pattern == "diag_fl_hr":
            match = diag_fl_hr
        elif self.pattern == "diag_fr_hl":
            match = diag_fr_hl
        elif self.pattern == "all_four":
            match = all_four
        elif self.pattern == "flight":
            match = flight
        elif self.pattern == "other":
            match = ~(
                diag_fl_hr
                | diag_fr_hl
                | all_four
                | flight
            )
        else:
            raise ValueError(
                f"Unknown contact pattern {self.pattern!r}."
            )

        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )

        self.match_count += (match & active).float()
        self.active_count += active.float()

        return self.match_count / torch.clamp(
            self.active_count,
            min=1.0,
        )

    def reset(self, env_ids) -> None:
        self.match_count[env_ids] = 0.0
        self.active_count[env_ids] = 0.0


class CompletedFootPhaseDuration:
    """Measure mean completed air or stance duration while commanded to move."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        self.phase = cfg.params["phase"]
        if self.phase not in ("air", "stance"):
            raise ValueError(
                f"Unknown foot phase {self.phase!r}; expected 'air' or 'stance'."
            )

        self.duration_sum = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.event_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        command_threshold: float,
        phase: str,
    ) -> torch.Tensor:
        del phase

        sensor = env.scene[sensor_name]
        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )

        if self.phase == "air":
            durations = sensor.data.last_air_time
            if durations is None:
                raise RuntimeError(
                    f"Sensor '{sensor_name}' must use track_air_time=True."
                )
            completed = sensor.compute_first_contact(dt=env.step_dt)
        else:
            durations = sensor.data.last_contact_time
            if durations is None:
                raise RuntimeError(
                    f"Sensor '{sensor_name}' must use track_air_time=True."
                )
            completed = sensor.compute_first_air(dt=env.step_dt)

        completed = completed & active.unsqueeze(1)
        completed_float = completed.float()

        self.duration_sum += torch.sum(
            durations * completed_float,
            dim=1,
        )
        self.event_count += torch.sum(completed_float, dim=1)

        return self.duration_sum / torch.clamp(
            self.event_count,
            min=1.0,
        )

    def reset(self, env_ids) -> None:
        self.duration_sum[env_ids] = 0.0
        self.event_count[env_ids] = 0.0



class MovingMechanicalPower:
    """Measure mean absolute joint mechanical power while commanded to move."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        del cfg
        self.power_sum = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.sample_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    def __call__(
        self,
        env,
        entity_name: str,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        robot = env.scene[entity_name]
        power_w = torch.sum(
            torch.abs(
                robot.data.actuator_force
                * robot.data.joint_vel
            ),
            dim=1,
        )

        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )
        active_float = active.float()

        self.power_sum += power_w * active_float
        self.sample_count += active_float

        return self.power_sum / torch.clamp(
            self.sample_count,
            min=1.0,
        )

    def reset(self, env_ids) -> None:
        self.power_sum[env_ids] = 0.0
        self.sample_count[env_ids] = 0.0


class ActuatorUtilizationMetric:
    """Measure normalized actuator utilization or saturation."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        robot = env.scene[cfg.params["entity_name"]]

        effort_limits = []
        for name in robot.actuator_names:
            if name.endswith(("_hx", "_hy")):
                effort_limits.append(float(HIP_TORQUE_LIMIT))
            elif name.endswith("_kn"):
                effort_limits.append(float(KNEE_TORQUE_LIMIT))
            else:
                raise ValueError(
                    "No torque limit mapping is defined for "
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
            f"Unknown actuator statistic: {self.statistic!r}"
        )


class MovingBaseVerticalVelocityRms:
    """Measure RMS base vertical velocity while commanded to move."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        del cfg
        self.squared_sum = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.sample_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    def __call__(
        self,
        env,
        entity_name: str,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        robot = env.scene[entity_name]
        vertical_velocity = robot.data.root_link_lin_vel_b[:, 2]

        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )
        active_float = active.float()

        self.squared_sum += (
            torch.square(vertical_velocity) * active_float
        )
        self.sample_count += active_float

        mean_square = self.squared_sum / torch.clamp(
            self.sample_count,
            min=1.0,
        )
        return torch.sqrt(mean_square)

    def reset(self, env_ids) -> None:
        self.squared_sum[env_ids] = 0.0
        self.sample_count[env_ids] = 0.0


class FootContactSwitchRate:
    """Measure contact transitions per foot per second while commanded to move."""

    def __init__(self, cfg: MetricsTermCfg, env) -> None:
        sensor = env.scene[cfg.params["sensor_name"]]
        num_feet = len(sensor.primary_names)

        self.num_feet = float(num_feet)
        self.previous_contact = torch.zeros(
            (env.num_envs, num_feet),
            device=env.device,
            dtype=torch.bool,
        )
        self.initialized = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        )
        self.switch_count = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.active_time = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )
        self.step_dt = float(env.step_dt)

    def __call__(
        self,
        env,
        sensor_name: str,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        if sensor.data.found is None:
            raise RuntimeError(
                f"Sensor '{sensor_name}' must include the 'found' field."
            )

        contact = sensor.data.found > 0
        active = _moving_command_mask(
            env,
            command_name,
            command_threshold,
        )

        switches = (
            (contact != self.previous_contact)
            & self.initialized.unsqueeze(1)
            & active.unsqueeze(1)
        )

        self.switch_count += torch.sum(switches.float(), dim=1)
        self.active_time += active.float() * self.step_dt

        self.previous_contact.copy_(contact)
        self.initialized.fill_(True)

        denominator = torch.clamp(
            self.active_time * self.num_feet,
            min=self.step_dt * self.num_feet,
        )
        return self.switch_count / denominator

    def reset(self, env_ids) -> None:
        self.previous_contact[env_ids] = False
        self.initialized[env_ids] = False
        self.switch_count[env_ids] = 0.0
        self.active_time[env_ids] = 0.0


def make_spot_flat_velocity_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the V2 Spot flat-terrain velocity environment."""

    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.timestep = PHYSICS_TIMESTEP
    cfg.sim.mujoco.integrator = "implicitfast"
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 100.0
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.decimation = DECIMATION

    cfg.sim.njmax = 300
    cfg.sim.nconmax = 50
    cfg.sim.contact_sensor_maxmatch = 64

    cfg.scene.entities = {
        "robot": get_spot_robot_cfg(),
    }

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            if not isinstance(sensor, TerrainHeightSensorCfg):
                raise TypeError(
                    f"foot_height_scan sensor has an unexpected type: {type(sensor)}"
                )

            sensor.frame = tuple(
                ObjRef(type="site", name=name, entity="robot")
                for name in SPOT_FOOT_SITES
            )
            sensor.pattern = RingPatternCfg.single_ring(
                radius=0.03,
                num_samples=4,
            )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=SPOT_FOOT_GEOMS,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True, 
    )

    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r".*_collision$",
            exclude=SPOT_FOOT_GEOMS,
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=DECIMATION,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
    )

    joint_pos_action = cfg.actions["joint_pos"]
    if not isinstance(joint_pos_action, JointPositionActionCfg):
        raise TypeError("Expected a JointPositionActionCfg.")
    joint_pos_action.scale = SPOT_ACTION_SCALE

    cfg.viewer.body_name = SPOT_BASE_BODY
    cfg.viewer.distance = 2.0
    cfg.viewer.elevation = -10.0

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = SPOT_FOOT_GEOMS

    cfg.events["base_com"].params[
        "asset_cfg"
    ].body_names = (SPOT_BASE_BODY,)

    cfg.rewards["pose"].params["std_standing"] = {
        ".*_hx": 0.05,
        ".*_hy": 0.05,
        ".*_kn": 0.10,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        ".*_hx": 0.30,
        ".*_hy": 0.30,
        ".*_kn": 0.60,
    }
    cfg.rewards["pose"].params["std_running"] = {
        ".*_hx": 0.30,
        ".*_hy": 0.30,
        ".*_kn": 0.60,
    }

    cfg.rewards["upright"].params[
        "asset_cfg"
    ].body_names = (SPOT_BASE_BODY,)

    cfg.rewards["body_ang_vel"].params[
        "asset_cfg"
    ].body_names = (SPOT_BASE_BODY,)

    for reward_name in (
        "foot_clearance",
        "foot_slip",
    ):
        cfg.rewards[reward_name].params[
            "asset_cfg"
        ].site_names = SPOT_FOOT_SITES


    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.5
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0

    # Keep gait discovery simple: no explicit air-time reward, no gait-phase
    # reward, no diagonal-pair reward, and no support-balance reward.
    cfg.rewards["air_time"].weight = 0.0

    # Only prevent a foot from remaining continuously airborne for too long.
    # This matches the simple safeguard used by the earlier successful Spot
    # locomotion project.
    # cfg.rewards["excessive_air_time"] = RewardTermCfg(
    #     func=excessive_foot_air_time,
    #     weight=-0.35,
    #     params={
    #         "sensor_name": feet_ground_cfg.name,
    #         "command_name": "twist",
    #         "command_threshold": 0.2,
    #         "max_air_time": 0.35,
    #     },
    # )

    cfg.rewards["foot_clearance"].weight = -2.0
    cfg.rewards["foot_clearance"].params["target_height"] = 0.08
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.2

    cfg.rewards["foot_swing_height"].weight = -0.25
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.08
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.2

    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.2
    
    cfg.rewards["soft_landing"].weight = -1e-5
    cfg.rewards["soft_landing"].params["command_threshold"] = 0.2

    # Locomotion diagnostics only. These do not affect rewards or policy.
    # Gait timing is measured only while commanded to move. Mechanical power and
    # vertical RMS are also movement-gated. Actuator metrics remain global so
    # saturation during both locomotion and support can be detected.
    cfg.metrics.update(
        {
            "mean_air_time_at_landing_s": MetricsTermCfg(
                func=CompletedFootPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "air",
                },
                reduce="last",
            ),
            "mean_stance_time_at_liftoff_s": MetricsTermCfg(
                func=CompletedFootPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "stance",
                },
                reduce="last",
            ),
            "foot_contact_switch_rate_hz": MetricsTermCfg(
                func=FootContactSwitchRate,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                },
                reduce="last",
            ),
            "mean_mechanical_power_w": MetricsTermCfg(
                func=MovingMechanicalPower,
                params={
                    "entity_name": "robot",
                    "command_name": "twist",
                    "command_threshold": 0.2,
                },
                reduce="last",
            ),
            "max_actuator_utilization": MetricsTermCfg(
                func=ActuatorUtilizationMetric,
                params={
                    "entity_name": "robot",
                    "statistic": "max",
                },
                reduce="max",
            ),
            "actuator_saturation_fraction": MetricsTermCfg(
                func=ActuatorUtilizationMetric,
                params={
                    "entity_name": "robot",
                    "statistic": "saturation_fraction",
                    "saturation_threshold": 0.99,
                },
            ),
            "base_vertical_velocity_rms": MetricsTermCfg(
                func=MovingBaseVerticalVelocityRms,
                params={
                    "entity_name": "robot",
                    "command_name": "twist",
                    "command_threshold": 0.2,
                },
                reduce="last",
            ),
            "air_time_FL_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "air",
                    "foot_name": "FL",
                },
                reduce="last",
            ),
            "air_time_FR_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "air",
                    "foot_name": "FR",
                },
                reduce="last",
            ),
            "air_time_HL_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "air",
                    "foot_name": "HL",
                },
                reduce="last",
            ),
            "air_time_HR_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "air",
                    "foot_name": "HR",
                },
                reduce="last",
            ),
            "stance_time_FL_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "stance",
                    "foot_name": "FL",
                },
                reduce="last",
            ),
            "stance_time_FR_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "stance",
                    "foot_name": "FR",
                },
                reduce="last",
            ),
            "stance_time_HL_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "stance",
                    "foot_name": "HL",
                },
                reduce="last",
            ),
            "stance_time_HR_s": MetricsTermCfg(
                func=PerFootCompletedPhaseDuration,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "phase": "stance",
                    "foot_name": "HR",
                },
                reduce="last",
            ),
            "diag_FL_HR_fraction": MetricsTermCfg(
                func=ActiveContactPatternFraction,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "pattern": "diag_fl_hr",
                },
                reduce="last",
            ),
            "diag_FR_HL_fraction": MetricsTermCfg(
                func=ActiveContactPatternFraction,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "pattern": "diag_fr_hl",
                },
                reduce="last",
            ),
            "all_four_contact_fraction": MetricsTermCfg(
                func=ActiveContactPatternFraction,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "pattern": "all_four",
                },
                reduce="last",
            ),
            "flight_fraction": MetricsTermCfg(
                func=ActiveContactPatternFraction,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "pattern": "flight",
                },
                reduce="last",
            ),
            "other_contact_pattern_fraction": MetricsTermCfg(
                func=ActiveContactPatternFraction,
                params={
                    "sensor_name": feet_ground_cfg.name,
                    "command_name": "twist",
                    "command_threshold": 0.2,
                    "pattern": "other",
                },
                reduce="last",
            ),
        }
    )

    # History catches illegal contacts that occur between policy updates.
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={
            "sensor_name": nonfoot_ground_cfg.name,
            "force_threshold": 10.0,
        },
    )

    # Flat terrain and flat-task observations.
    if cfg.scene.terrain is None:
        raise RuntimeError("The velocity base task has no terrain.")
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("command_vel", None)
    cfg.events.pop("push_robot", None)

    base_command = cfg.commands["twist"]
    if not isinstance(base_command, UniformVelocityCommandCfg):
        raise TypeError("Expected a UniformVelocityCommandCfg.")

    cfg.commands["twist"] = SplitVelocityCommandCfg(
        entity_name=base_command.entity_name,
        resampling_time_range=(3.0, 6.0),
        debug_vis=base_command.debug_vis,
        heading_command=False,
        heading_control_stiffness=base_command.heading_control_stiffness,
        rel_heading_envs=0.0,
        rel_world_envs=0.0,
        rel_forward_envs=0.0,
        rel_standing_envs=0.10,
        init_velocity_prob=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.65, 0.65),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.50, 0.50),
            heading=None,
        ),
        viz=UniformVelocityCommandCfg.VizCfg(
            z_offset=0.5,
            scale=base_command.viz.scale,
        ),
        lin_vel_x_abs_range=(0.35, 0.65),
        forward_probability=0.50,
        pure_lateral_probability=0.20,
        lin_vel_y_abs_range=(0.20, 0.40),
        left_probability=0.50,
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

        for event_name in (
            "push_robot",
            "foot_friction",
            "base_com",
            "encoder_bias",
        ):
            cfg.events.pop(event_name, None)

    return cfg