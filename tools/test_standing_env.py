"""Validate the minimal Spot standing environment."""

import torch

from mjlab.envs import ManagerBasedRlEnv

from spot_locomotion_v2.robot.spot_constants import (
    HIP_TORQUE_LIMIT,
    JOINT_NAMES,
    KNEE_TORQUE_LIMIT,
)
from spot_locomotion_v2.tasks.standing.standing_env_cfg import (
    ACTION_SCALE,
    make_standing_env_cfg,
)


DEVICE = "cuda:0"
ROLLOUT_DURATION_S = 2.0
EXPECTED_OBSERVATION_DIM = 45


def require_finite(
    name: str,
    value: torch.Tensor,
    step: int,
) -> None:
    """Raise if a simulation tensor contains NaN or infinity."""

    if not torch.all(torch.isfinite(value)):
        raise RuntimeError(
            f"{name} became non-finite at policy step {step}."
        )


def main() -> None:
    cfg = make_standing_env_cfg(num_envs=1)

    env = ManagerBasedRlEnv(
        cfg=cfg,
        device=DEVICE,
    )

    try:
        observations, _ = env.reset(seed=0)

        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")

        actual_action_order = tuple(action_term.target_names)

        if actual_action_order != JOINT_NAMES:
            raise RuntimeError(
                "Action ordering does not match JOINT_NAMES.\n"
                f"Expected: {JOINT_NAMES}\n"
                f"Actual:   {actual_action_order}"
            )

        action_dimension = env.action_manager.total_action_dim

        if action_dimension != len(JOINT_NAMES):
            raise RuntimeError(
                f"Expected 12 actions, got {action_dimension}."
            )

        actor_observation = observations["actor"]

        if actor_observation.shape != (
            env.num_envs,
            EXPECTED_OBSERVATION_DIM,
        ):
            raise RuntimeError(
                "Unexpected actor observation shape: "
                f"{tuple(actor_observation.shape)}"
            )

        zero_action = torch.zeros(
            (
                env.num_envs,
                action_dimension,
            ),
            device=env.device,
        )

        number_of_steps = round(
            ROLLOUT_DURATION_S / env.step_dt
        )

        torque_limits = torch.tensor(
            [
                KNEE_TORQUE_LIMIT
                if name.endswith("_kn")
                else HIP_TORQUE_LIMIT
                for name in JOINT_NAMES
            ],
            device=env.device,
        )

        initial_base_height = float(
            robot.data.root_link_pose_w[0, 2].item()
        )

        minimum_base_height = initial_base_height
        maximum_base_height = initial_base_height

        peak_base_linear_speed = 0.0
        peak_base_angular_speed = 0.0
        peak_joint_speed = 0.0
        maximum_joint_error = 0.0
        peak_actuator_torque = 0.0
        maximum_target_error = 0.0
        maximum_torque_limit_excess = 0.0

        for step in range(number_of_steps):
            (
                observations,
                _reward,
                _terminated,
                _truncated,
                _extras,
            ) = env.step(zero_action)

            root_pose = robot.data.root_link_pose_w
            root_velocity = robot.data.root_link_vel_w
            joint_positions = robot.data.joint_pos
            joint_velocities = robot.data.joint_vel
            joint_targets = robot.data.joint_pos_target
            actuator_forces = robot.data.actuator_force
            default_positions = robot.data.default_joint_pos

            require_finite("root pose", root_pose, step)
            require_finite("root velocity", root_velocity, step)
            require_finite(
                "joint positions",
                joint_positions,
                step,
            )
            require_finite(
                "joint velocities",
                joint_velocities,
                step,
            )
            require_finite(
                "joint targets",
                joint_targets,
                step,
            )
            require_finite(
                "actuator forces",
                actuator_forces,
                step,
            )
            require_finite(
                "actor observations",
                observations["actor"],
                step,
            )

            base_height = float(
                root_pose[0, 2].item()
            )

            base_linear_speed = float(
                torch.linalg.vector_norm(
                    root_velocity[0, :3]
                ).item()
            )

            base_angular_speed = float(
                torch.linalg.vector_norm(
                    root_velocity[0, 3:]
                ).item()
            )

            joint_speed = float(
                torch.max(
                    torch.abs(joint_velocities)
                ).item()
            )

            joint_error = float(
                torch.max(
                    torch.abs(
                        joint_positions
                        - default_positions
                    )
                ).item()
            )

            actuator_torque = float(
                torch.max(
                    torch.abs(actuator_forces)
                ).item()
            )

            target_error = float(
                torch.max(
                    torch.abs(
                        joint_targets
                        - default_positions
                    )
                ).item()
            )

            torque_limit_excess = float(
                torch.max(
                    torch.abs(actuator_forces)
                    - torque_limits
                ).item()
            )

            minimum_base_height = min(
                minimum_base_height,
                base_height,
            )
            maximum_base_height = max(
                maximum_base_height,
                base_height,
            )
            peak_base_linear_speed = max(
                peak_base_linear_speed,
                base_linear_speed,
            )
            peak_base_angular_speed = max(
                peak_base_angular_speed,
                base_angular_speed,
            )
            peak_joint_speed = max(
                peak_joint_speed,
                joint_speed,
            )
            maximum_joint_error = max(
                maximum_joint_error,
                joint_error,
            )
            peak_actuator_torque = max(
                peak_actuator_torque,
                actuator_torque,
            )
            maximum_target_error = max(
                maximum_target_error,
                target_error,
            )
            maximum_torque_limit_excess = max(
                maximum_torque_limit_excess,
                torque_limit_excess,
            )

        final_base_height = float(
            robot.data.root_link_pose_w[0, 2].item()
        )

        final_projected_gravity = (
            robot.data.projected_gravity_b[0]
            .detach()
            .cpu()
            .tolist()
        )

        print("\nStanding-environment summary:")
        print(f"Device: {env.device}")
        print(f"Physics timestep: {env.physics_dt:.6f} s")
        print(f"Policy timestep: {env.step_dt:.6f} s")
        print(f"Decimation: {cfg.decimation}")
        print(f"Action dimension: {action_dimension}")
        print(f"Action scale: {ACTION_SCALE:.3f} rad")
        print(
            "Actor observation shape: "
            f"{tuple(observations['actor'].shape)}"
        )
        print(f"Initial base height: {initial_base_height:.6f} m")
        print(f"Minimum base height: {minimum_base_height:.6f} m")
        print(f"Maximum base height: {maximum_base_height:.6f} m")
        print(f"Final base height: {final_base_height:.6f} m")
        print(
            "Peak base linear speed: "
            f"{peak_base_linear_speed:.6f} m/s"
        )
        print(
            "Peak base angular speed: "
            f"{peak_base_angular_speed:.6f} rad/s"
        )
        print(
            "Peak absolute joint speed: "
            f"{peak_joint_speed:.6f} rad/s"
        )
        print(
            "Maximum joint error: "
            f"{maximum_joint_error:.6f} rad"
        )
        print(
            "Peak actuator torque: "
            f"{peak_actuator_torque:.6f} N·m"
        )
        print(
            "Maximum zero-action target error: "
            f"{maximum_target_error:.9f} rad"
        )
        print(
            "Maximum torque-limit excess: "
            f"{maximum_torque_limit_excess:.9f} N·m"
        )
        print(
            "Final projected gravity: "
            f"{final_projected_gravity}"
        )

        failures = []

        if maximum_target_error > 1e-6:
            failures.append(
                "zero action did not produce the default joint targets"
            )

        if maximum_torque_limit_excess > 1e-4:
            failures.append(
                "an actuator exceeded its configured torque limit"
            )

        if minimum_base_height < 0.35:
            failures.append(
                "the robot appears to have fallen or collapsed"
            )

        if final_projected_gravity[2] > -0.9:
            failures.append(
                "the robot did not finish upright"
            )

        if failures:
            print("\nFAIL:")
            for failure in failures:
                print(f"  - {failure}")

            raise RuntimeError(
                "Standing-environment validation failed."
            )

        print(
            "\nPASS: the zero-action 50 Hz to 500 Hz "
            "standing pipeline remained stable."
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()