from pathlib import Path
import csv
import math
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "experiments"
    / "scene_free_standing.xml"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "free_standing_kp100_300_kd8.csv"
PLOT_PATH = REPORTS_DIR / "free_standing_kp100_300_kd8.png"

SIMULATION_DURATION = 60.0

KP_h = 300.0
KP_kn = 500.0

KP_VALUES = np.asarray(
    [
        KP_h, KP_h, KP_kn,
        KP_h, KP_h, KP_kn,
        KP_h, KP_h, KP_kn,
        KP_h, KP_h, KP_kn,
    ],
    dtype=np.float64,
)

KD_h = 10.0
KD_kn = 10.0

KD_VALUES = np.asarray(
    [
        KD_h, KD_h, KD_kn,
        KD_h, KD_h, KD_kn,
        KD_h, KD_h, KD_kn,
        KD_h, KD_h, KD_kn,
    ],
    dtype=np.float64,
)

HIP_TORQUE_LIMIT = 44.88

# Local approximation around the home knee angle q = -1.8 rad.
# This will later be replaced by an angle-dependent knee limit.
KNEE_TORQUE_LIMIT = 105.98

JOINT_NAMES = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]

FOOT_NAMES = ["FL", "FR", "HL", "HR"]


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """Return the MuJoCo ID of a named object."""

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(f"Object not found: {name}")

    return object_id


def quaternion_to_roll_pitch(
    quaternion: np.ndarray,
) -> tuple[float, float]:
    """Convert a MuJoCo [w, x, y, z] quaternion to roll and pitch."""

    w, x, y, z = quaternion

    sin_roll_cos_pitch = 2.0 * (w * x + y * z)
    cos_roll_cos_pitch = 1.0 - 2.0 * (x * x + y * y)

    roll = math.atan2(
        sin_roll_cos_pitch,
        cos_roll_cos_pitch,
    )

    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))

    pitch = math.asin(sin_pitch)

    return roll, pitch


def count_foot_contacts(
    data: mujoco.MjData,
    terrain_id: int,
    foot_geom_ids: set[int],
) -> int:
    """Count how many distinct feet currently contact the terrain."""

    touching_feet: set[int] = set()

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]

        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 == terrain_id and geom2 in foot_geom_ids:
            touching_feet.add(geom2)

        if geom2 == terrain_id and geom1 in foot_geom_ids:
            touching_feet.add(geom1)

    return len(touching_feet)


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )
    data = mujoco.MjData(model)

    home_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    terrain_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "terrain",
    )

    foot_geom_ids = {
        get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            foot_name,
        )
        for foot_name in FOOT_NAMES
    }

    joint_ids = np.asarray(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            for joint_name in JOINT_NAMES
        ],
        dtype=int,
    )

    actuator_ids = np.asarray(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                joint_name,
            )
            for joint_name in JOINT_NAMES
        ],
        dtype=int,
    )

    # For the floating-base model, joint angles begin after the
    # seven floating-base qpos values. These arrays retrieve the
    # correct address of every joint instead of hard-coding slices.
    qpos_addresses = model.jnt_qposadr[
        joint_ids
    ].astype(int)

    dof_addresses = model.jnt_dofadr[
        joint_ids
    ].astype(int)

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )

    mujoco.mj_forward(model, data)

    # Hold the twelve joint angles stored in the home keyframe.
    desired_positions = data.qpos[
        qpos_addresses
    ].copy()

    torque_limits = np.asarray(
        [
            (
                KNEE_TORQUE_LIMIT
                if joint_name.endswith("_kn")
                else HIP_TORQUE_LIMIT
            )
            for joint_name in JOINT_NAMES
        ],
        dtype=np.float64,
    )

    print(f"Model: {MODEL_PATH}")
    print(
        f"Physics timestep: "
        f"{model.opt.timestep:.6f} s"
    )
    print(
        f"Simulation duration: "
        f"{SIMULATION_DURATION:.2f} s"
    )
    print(f"Kp: {KP_VALUES}")
    print(f"Kd: {KD_VALUES}")
    print(
        f"Initial base height: "
        f"{data.qpos[2]:.6f} m"
    )
    print(
        f"Desired joint positions: "
        f"{desired_positions}"
    )
    print(
        f"Torque limits: "
        f"{torque_limits}"
    )

    times: list[float] = []
    base_heights: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    maximum_joint_errors: list[float] = []
    maximum_torques: list[float] = []
    foot_contact_counts: list[int] = []

    saturation_counts = np.zeros(
        len(JOINT_NAMES),
        dtype=int,
    )

    peak_abs_torque_per_joint = np.zeros(
        len(JOINT_NAMES),
        dtype=np.float64,
    )

    number_of_steps = int(
        SIMULATION_DURATION / model.opt.timestep
    )

    executed_steps = 0

    print(
        "\nOpening MuJoCo viewer. "
        "Close the viewer to stop the test early."
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        for _ in range(number_of_steps):
            if not viewer.is_running():
                break

            wall_step_start = time.monotonic()

            joint_positions = data.qpos[
                qpos_addresses
            ]
            joint_velocities = data.qvel[
                dof_addresses
            ]

            position_error = (
                desired_positions
                - joint_positions
            )

            raw_torque = (
                KP_VALUES * position_error
                - KD_VALUES * joint_velocities
            )

            applied_torque = np.clip(
                raw_torque,
                -torque_limits,
                torque_limits,
            )

            # Clear all motor commands and then assign the
            # twelve explicit joint torque commands.
            data.ctrl[:] = 0.0
            data.ctrl[actuator_ids] = applied_torque

            saturation_counts += (
                np.abs(raw_torque)
                > torque_limits
            )

            peak_abs_torque_per_joint = np.maximum(
                peak_abs_torque_per_joint,
                np.abs(applied_torque),
            )

            mujoco.mj_step(
                model,
                data,
            )

            executed_steps += 1

            # Recalculate errors after the physics step so the
            # recorded state is synchronized with base position.
            current_joint_positions = data.qpos[
                qpos_addresses
            ]

            current_position_error = (
                desired_positions
                - current_joint_positions
            )

            roll, pitch = quaternion_to_roll_pitch(
                data.qpos[3:7]
            )

            foot_contacts = count_foot_contacts(
                data,
                terrain_id,
                foot_geom_ids,
            )

            times.append(float(data.time))
            base_heights.append(
                float(data.qpos[2])
            )
            rolls.append(roll)
            pitches.append(pitch)
            maximum_joint_errors.append(
                float(
                    np.max(
                        np.abs(
                            current_position_error
                        )
                    )
                )
            )
            maximum_torques.append(
                float(
                    np.max(
                        np.abs(applied_torque)
                    )
                )
            )
            foot_contact_counts.append(
                foot_contacts
            )

            # Update the visible robot state.
            viewer.sync()

            # Pace the simulation close to real time so that
            # five simulated seconds take about five real seconds.
            elapsed_wall_time = (
                time.monotonic()
                - wall_step_start
            )

            remaining_time = (
                model.opt.timestep
                - elapsed_wall_time
            )

            if remaining_time > 0.0:
                time.sleep(remaining_time)

    if executed_steps == 0:
        raise RuntimeError(
            "No simulation steps were executed."
        )

    times_array = np.asarray(
        times,
        dtype=np.float64,
    )
    base_heights_array = np.asarray(
        base_heights,
        dtype=np.float64,
    )
    rolls_array = np.asarray(
        rolls,
        dtype=np.float64,
    )
    pitches_array = np.asarray(
        pitches,
        dtype=np.float64,
    )
    joint_errors_array = np.asarray(
        maximum_joint_errors,
        dtype=np.float64,
    )
    torques_array = np.asarray(
        maximum_torques,
        dtype=np.float64,
    )
    contacts_array = np.asarray(
        foot_contact_counts,
        dtype=int,
    )

    saturation_percentages = (
        100.0
        * saturation_counts
        / executed_steps
    )

    final_joint_positions = data.qpos[
        qpos_addresses
    ].copy()

    final_joint_velocities = data.qvel[
        dof_addresses
    ].copy()

    final_joint_errors = (
        desired_positions
        - final_joint_positions
    )

    final_raw_torques = (
        KP_VALUES * final_joint_errors
        - KD_VALUES * final_joint_velocities
    )

    final_applied_torques = np.clip(
        final_raw_torques,
        -torque_limits,
        torque_limits,
    )

    final_roll_deg = math.degrees(
        rolls_array[-1]
    )
    final_pitch_deg = math.degrees(
        pitches_array[-1]
    )

    print("\nStanding summary:")
    print(
        f"Executed simulation time: "
        f"{times_array[-1]:.3f} s"
    )
    print(
        f"Final base height: "
        f"{base_heights_array[-1]:.6f} m"
    )
    print(
        f"Minimum base height: "
        f"{np.min(base_heights_array):.6f} m"
    )
    print(
        f"Final roll: "
        f"{final_roll_deg:.3f} deg"
    )
    print(
        f"Final pitch: "
        f"{final_pitch_deg:.3f} deg"
    )
    print(
        f"Maximum absolute roll: "
        f"{np.max(np.abs(np.degrees(rolls_array))):.3f} deg"
    )
    print(
        f"Maximum absolute pitch: "
        f"{np.max(np.abs(np.degrees(pitches_array))):.3f} deg"
    )
    print(
        f"Maximum joint error: "
        f"{np.max(joint_errors_array):.6f} rad"
    )
    print(
        f"Peak applied torque: "
        f"{np.max(torques_array):.6f} N·m"
    )
    print(
        f"Final number of feet contacting terrain: "
        f"{contacts_array[-1]}"
    )

    print("\nPer-joint saturation:")
    for joint_name, saturation_percentage in zip(
        JOINT_NAMES,
        saturation_percentages,
        strict=True,
    ):
        print(
            f"  {joint_name:6s}: "
            f"{saturation_percentage:7.3f}%"
        )

    print("\nFinal per-joint state:")
    print(
        "joint  | desired | actual  | error   | "
        "velocity | torque  | peak torque"
    )

    for (
        joint_name,
        desired,
        actual,
        error,
        velocity,
        torque,
        peak_torque,
    ) in zip(
        JOINT_NAMES,
        desired_positions,
        final_joint_positions,
        final_joint_errors,
        final_joint_velocities,
        final_applied_torques,
        peak_abs_torque_per_joint,
        strict=True,
    ):
        print(
            f"{joint_name:6s} | "
            f"{desired:+7.3f} | "
            f"{actual:+7.3f} | "
            f"{error:+7.3f} | "
            f"{velocity:+8.4f} | "
            f"{torque:+7.2f} | "
            f"{peak_torque:11.2f}"
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
                "base_height_m",
                "roll_rad",
                "pitch_rad",
                "max_joint_error_rad",
                "max_applied_torque_nm",
                "foot_contact_count",
            ]
        )

        writer.writerows(
            zip(
                times_array,
                base_heights_array,
                rolls_array,
                pitches_array,
                joint_errors_array,
                torques_array,
                contacts_array,
                strict=True,
            )
        )

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    axes[0].plot(
        times_array,
        base_heights_array,
    )
    axes[0].axhline(
        y=float(model.key_qpos[home_id, 2]),
        linestyle="--",
        label="Initial target height",
    )
    axes[0].set_ylabel("Base height [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        times_array,
        np.degrees(rolls_array),
        label="Roll",
    )
    axes[1].plot(
        times_array,
        np.degrees(pitches_array),
        label="Pitch",
    )
    axes[1].set_ylabel("Angle [deg]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        times_array,
        joint_errors_array,
    )
    axes[2].set_ylabel(
        "Max joint error [rad]"
    )
    axes[2].grid(True)

    axes[3].plot(
        times_array,
        torques_array,
        label="Maximum applied torque",
    )
    axes[3].set_ylabel("Torque [N·m]")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True)

    contact_axis = axes[3].twinx()

    contact_axis.step(
        times_array,
        contacts_array,
        label="Feet in contact",
        where="post",
    )
    contact_axis.set_ylabel(
        "Feet in contact"
    )
    contact_axis.set_ylim(
        -0.2,
        4.2,
    )

    torque_lines, torque_labels = (
        axes[3].get_legend_handles_labels()
    )
    contact_lines, contact_labels = (
        contact_axis.get_legend_handles_labels()
    )

    axes[3].legend(
        torque_lines + contact_lines,
        torque_labels + contact_labels,
        loc="best",
    )

    figure.suptitle(
        "Free-standing pose hold: family-specific PD gains"
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