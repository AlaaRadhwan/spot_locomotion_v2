from pathlib import Path
import csv

import matplotlib.pyplot as plt
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "experiments"
    / "spot_fixed_base_v1.xml"
)


REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "fixed_base_step_v1.csv"
PLOT_PATH = REPORTS_DIR / "fixed_base_step_v1.png"

TEST_JOINT = "fl_hy"

STEP_TIME = 0.10
STEP_SIZE = 0.10
SIMULATION_DURATION = 0.60


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(f"Object not found: {name}")

    return object_id


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Isolate the actuator from gravitational loading.
    model.opt.gravity[:] = 0.0

    home_keyframe_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_keyframe_id,
    )

    joint_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        TEST_JOINT,
    )

    actuator_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        TEST_JOINT,
    )

    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])

    initial_target = float(data.ctrl[actuator_id])

    # Your first important calculation:
    final_target = initial_target + STEP_SIZE

    expected_initial_step_torque = 500.0 * STEP_SIZE

    print(f"Model: {MODEL_PATH}")
    print(f"Test joint: {TEST_JOINT}")
    print(f"Physics timestep: {model.opt.timestep} s")
    print(f"Initial target: {initial_target:.6f} rad")
    print(f"Final target: {final_target:.6f} rad")
    print(f"Step size: {STEP_SIZE:.6f} rad")
    print(
        "Expected instantaneous step torque: "
        f"{expected_initial_step_torque:.3f} N·m"
    )

    times: list[float] = []
    targets: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    actuator_forces: list[float] = []
    joint_torques: list[float] = []

    number_of_steps = int(
        SIMULATION_DURATION / model.opt.timestep
    )

    for _ in range(number_of_steps + 1):
        current_time = float(data.time)

        # Keep all actuators at their home targets.
        data.ctrl[:] = model.key_ctrl[home_keyframe_id]

        # Apply the target step only after STEP_TIME.
        if current_time < STEP_TIME:
            target = initial_target
        else:
            target = final_target

        data.ctrl[actuator_id] = target

        # Calculate forces at the current state before integrating
        # to the next timestep.
        mujoco.mj_forward(model, data)

        times.append(current_time)
        targets.append(target)
        positions.append(float(data.qpos[qpos_address]))
        velocities.append(float(data.qvel[dof_address]))
        actuator_forces.append(
            float(data.actuator_force[actuator_id])
        )
        joint_torques.append(
            float(data.qfrc_actuator[dof_address])
        )

        mujoco.mj_step(model, data)

    times_array = np.asarray(times)
    targets_array = np.asarray(targets)
    positions_array = np.asarray(positions)
    velocities_array = np.asarray(velocities)
    actuator_forces_array = np.asarray(actuator_forces)
    joint_torques_array = np.asarray(joint_torques)

    step_index = int(
        np.searchsorted(times_array, STEP_TIME)
    )

    print("\nAt the first recorded step sample:")
    print(f"time: {times_array[step_index]:.6f} s")
    print(
        f"position: "
        f"{positions_array[step_index]:.6f} rad"
    )
    print(
        f"velocity: "
        f"{velocities_array[step_index]:.6f} rad/s"
    )
    print(
        f"actuator force: "
        f"{actuator_forces_array[step_index]:.6f} N"
    )
    print(
        f"joint torque: "
        f"{joint_torques_array[step_index]:.6f} N·m"
    )

    print("\nResponse summary:")
    print(
        f"Final position: "
        f"{positions_array[-1]:.6f} rad"
    )
    print(
        f"Final error: "
        f"{final_target - positions_array[-1]:.6f} rad"
    )
    print(
        f"Peak velocity: "
        f"{np.max(np.abs(velocities_array)):.6f} rad/s"
    )
    print(
        f"Peak actuator force: "
        f"{np.max(np.abs(actuator_forces_array)):.6f}"
    )
    print(
        f"Peak joint torque: "
        f"{np.max(np.abs(joint_torques_array)):.6f} N·m"
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "time_s",
                "target_rad",
                "position_rad",
                "velocity_rad_s",
                "actuator_force",
                "joint_torque_nm",
            ]
        )

        writer.writerows(
            zip(
                times_array,
                targets_array,
                positions_array,
                velocities_array,
                actuator_forces_array,
                joint_torques_array,
                strict=True,
            )
        )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        times_array,
        targets_array,
        label="Target",
        linestyle="--",
    )
    axes[0].plot(
        times_array,
        positions_array,
        label="Position",
    )
    axes[0].set_ylabel("Position [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        times_array,
        velocities_array,
    )
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].grid(True)

    axes[2].plot(
        times_array,
        actuator_forces_array,
        label="Actuator force",
    )
    axes[2].plot(
        times_array,
        joint_torques_array,
        label="Joint torque",
        linestyle="--",
    )
    axes[2].set_ylabel("Torque [N·m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True)

    figure.suptitle(
        f"V1 fixed-base step response: {TEST_JOINT}"
    )
    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=160,
    )

    print(f"\nCSV written to: {CSV_PATH}")
    print(f"Plot written to: {PLOT_PATH}")


if __name__ == "__main__":
    main()