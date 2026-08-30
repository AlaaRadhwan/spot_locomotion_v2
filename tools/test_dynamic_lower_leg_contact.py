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
    / "scene_free_standing.xml"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
CSV_PATH = REPORTS_DIR / "dynamic_lower_leg_contacts.csv"
PLOT_PATH = REPORTS_DIR / "dynamic_lower_leg_contacts.png"

SIMULATION_DURATION = 2.0
EXPECTED_TIMESTEP = 0.002
BASE_LOWERING = 0.030

KP_HIP = 500.0
KP_KNEE = 500.0
KD_HIP = 10.0
KD_KNEE = 10.0

HIP_TORQUE_LIMIT = 69.52

# Local approximation around the home knee angle q = -1.8 rad.
# This will later be replaced by an angle-dependent knee limit.
KNEE_TORQUE_LIMIT = 164.15

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

LOWER_LEG_GEOM_NAMES = [
    "fl_lleg_collision",
    "fr_lleg_collision",
    "hl_lleg_collision",
    "hr_lleg_collision",
]

FOOT_GEOM_NAMES = [
    "FL",
    "FR",
    "HL",
    "HR",
]

KP_VALUES = np.asarray(
    [
        KP_HIP, KP_HIP, KP_KNEE,
        KP_HIP, KP_HIP, KP_KNEE,
        KP_HIP, KP_HIP, KP_KNEE,
        KP_HIP, KP_HIP, KP_KNEE,
    ],
    dtype=np.float64,
)

KD_VALUES = np.asarray(
    [
        KD_HIP, KD_HIP, KD_KNEE,
        KD_HIP, KD_HIP, KD_KNEE,
        KD_HIP, KD_HIP, KD_KNEE,
        KD_HIP, KD_HIP, KD_KNEE,
    ],
    dtype=np.float64,
)


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


def get_touching_geoms(
    data: mujoco.MjData,
    terrain_id: int,
    candidate_geom_ids: set[int],
) -> set[int]:
    """Return candidate geoms currently contacting the terrain."""

    touching_geom_ids: set[int] = set()

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 == terrain_id and geom2 in candidate_geom_ids:
            touching_geom_ids.add(geom2)
        elif geom2 == terrain_id and geom1 in candidate_geom_ids:
            touching_geom_ids.add(geom1)

    return touching_geom_ids


def get_lower_leg_contact_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_id: int,
    lower_leg_ids: set[int],
) -> float:
    """Return summed normal force from lower-leg/terrain contacts."""

    total_normal_force = 0.0
    contact_force = np.zeros(6, dtype=np.float64)

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        is_lower_leg_terrain_contact = (
            geom1 == terrain_id
            and geom2 in lower_leg_ids
        ) or (
            geom2 == terrain_id
            and geom1 in lower_leg_ids
        )

        if not is_lower_leg_terrain_contact:
            continue

        contact_force.fill(0.0)
        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            contact_force,
        )
        total_normal_force += float(contact_force[0])

    return total_normal_force


def require_finite_state(
    data: mujoco.MjData,
) -> None:
    """Raise if any important simulation array contains NaN or infinity."""

    arrays = {
        "qpos": data.qpos,
        "qvel": data.qvel,
        "qacc": data.qacc,
        "ctrl": data.ctrl,
    }

    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Non-finite {name} at t={data.time:.6f} s"
            )


def print_terrain_contact_details(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_id: int,
) -> None:
    """Print solver information for every terrain contact."""

    contact_force = np.zeros(6, dtype=np.float64)

    print("\nInitial terrain-contact details:")

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)

        if geom1 == terrain_id:
            robot_geom_id = geom2
        elif geom2 == terrain_id:
            robot_geom_id = geom1
        else:
            continue

        robot_geom_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            robot_geom_id,
        )

        contact_force.fill(0.0)

        mujoco.mj_contactForce(
            model,
            data,
            contact_index,
            contact_force,
        )

        print(
            f"  {robot_geom_name:20s} "
            f"distance={float(contact.dist):+.6f} m  "
            f"efc_address={int(contact.efc_address):4d}  "
            f"exclude={int(contact.exclude)}  "
            f"normal_force={float(contact_force[0]):10.3f} N"
        )


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )
    data = mujoco.MjData(model)

    if not np.isclose(
        model.opt.timestep,
        EXPECTED_TIMESTEP,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Expected a 500 Hz physics timestep of "
            f"{EXPECTED_TIMESTEP:.6f} s, but model uses "
            f"{model.opt.timestep:.6f} s."
        )

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

    lower_leg_ids = {
        get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        for geom_name in LOWER_LEG_GEOM_NAMES
    }

    foot_ids = {
        get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )
        for geom_name in FOOT_GEOM_NAMES
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

    qpos_addresses = model.jnt_qposadr[
        joint_ids
    ].astype(int)

    dof_addresses = model.jnt_dofadr[
        joint_ids
    ].astype(int)

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

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )
    mujoco.mj_forward(model, data)

    desired_positions = data.qpos[
        qpos_addresses
    ].copy()
    home_base_height = float(data.qpos[2])

    # Lower the whole floating base by 30 mm. The previous kinematic
    # test found first lower-leg contact at about 25 mm, so this creates
    # roughly 5 mm of deliberate lower-leg penetration.
    data.qpos[2] -= BASE_LOWERING
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    require_finite_state(data)

    initial_lower_legs = get_touching_geoms(
        data,
        terrain_id,
        lower_leg_ids,
    )
    initial_feet = get_touching_geoms(
        data,
        terrain_id,
        foot_ids,
    )
    initial_lower_leg_force = (
        get_lower_leg_contact_force(
            model,
            data,
            terrain_id,
            lower_leg_ids,
        )
    )

    print_terrain_contact_details(
        model,
        data,
        terrain_id,
    )

    print(f"Model: {MODEL_PATH}")
    print(
        f"Physics timestep: {model.opt.timestep:.6f} s "
        "(500 Hz)"
    )
    print(
        f"Simulation duration: {SIMULATION_DURATION:.3f} s"
    )
    print(
        f"Home base height: {home_base_height:.6f} m"
    )
    print(
        f"Lowered base height: {data.qpos[2]:.6f} m"
    )
    print(
        f"Base lowering: {BASE_LOWERING:.3f} m"
    )
    print(
        "Initial feet contacting terrain: "
        f"{len(initial_feet)}/4"
    )
    print(
        "Initial lower legs contacting terrain: "
        f"{len(initial_lower_legs)}/4"
    )
    print(
        "Initial summed lower-leg normal force: "
        f"{initial_lower_leg_force:.3f} N"
    )

    times: list[float] = []
    base_heights: list[float] = []
    base_linear_speeds: list[float] = []
    base_angular_speeds: list[float] = []
    peak_joint_speeds_at_step: list[float] = []
    maximum_joint_errors: list[float] = []
    maximum_applied_torques: list[float] = []
    lower_leg_forces: list[float] = []
    lower_leg_contact_counts: list[int] = []
    foot_contact_counts: list[int] = []

    saturation_counts = np.zeros(
        len(JOINT_NAMES),
        dtype=int,
    )

    number_of_steps = int(
        round(
            SIMULATION_DURATION
            / model.opt.timestep
        )
    )

    first_clear_time: float | None = None
    lower_leg_contact_steps = 0

    for _ in range(number_of_steps):
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

        data.ctrl[:] = 0.0
        data.ctrl[actuator_ids] = applied_torque

        saturation_counts += (
            np.abs(raw_torque)
            > torque_limits
        )

        mujoco.mj_step(model, data)
        require_finite_state(data)

        touching_lower_legs = get_touching_geoms(
            data,
            terrain_id,
            lower_leg_ids,
        )
        touching_feet = get_touching_geoms(
            data,
            terrain_id,
            foot_ids,
        )
        lower_leg_force = (
            get_lower_leg_contact_force(
                model,
                data,
                terrain_id,
                lower_leg_ids,
            )
        )

        lower_leg_count = len(
            touching_lower_legs
        )
        foot_count = len(touching_feet)

        if lower_leg_count > 0:
            lower_leg_contact_steps += 1
        elif first_clear_time is None:
            first_clear_time = float(data.time)

        current_position_error = (
            desired_positions
            - data.qpos[qpos_addresses]
        )

        times.append(float(data.time))
        base_heights.append(float(data.qpos[2]))
        base_linear_speeds.append(
            float(
                np.linalg.norm(
                    data.qvel[0:3]
                )
            )
        )
        base_angular_speeds.append(
            float(
                np.linalg.norm(
                    data.qvel[3:6]
                )
            )
        )
        peak_joint_speeds_at_step.append(
            float(
                np.max(
                    np.abs(
                        data.qvel[
                            dof_addresses
                        ]
                    )
                )
            )
        )
        maximum_joint_errors.append(
            float(
                np.max(
                    np.abs(
                        current_position_error
                    )
                )
            )
        )
        maximum_applied_torques.append(
            float(
                np.max(
                    np.abs(applied_torque)
                )
            )
        )
        lower_leg_forces.append(
            lower_leg_force
        )
        lower_leg_contact_counts.append(
            lower_leg_count
        )
        foot_contact_counts.append(
            foot_count
        )

    times_array = np.asarray(
        times,
        dtype=np.float64,
    )
    base_heights_array = np.asarray(
        base_heights,
        dtype=np.float64,
    )
    base_linear_speeds_array = np.asarray(
        base_linear_speeds,
        dtype=np.float64,
    )
    base_angular_speeds_array = np.asarray(
        base_angular_speeds,
        dtype=np.float64,
    )
    joint_speeds_array = np.asarray(
        peak_joint_speeds_at_step,
        dtype=np.float64,
    )
    joint_errors_array = np.asarray(
        maximum_joint_errors,
        dtype=np.float64,
    )
    torques_array = np.asarray(
        maximum_applied_torques,
        dtype=np.float64,
    )
    lower_leg_forces_array = np.asarray(
        lower_leg_forces,
        dtype=np.float64,
    )
    lower_leg_counts_array = np.asarray(
        lower_leg_contact_counts,
        dtype=int,
    )
    foot_counts_array = np.asarray(
        foot_contact_counts,
        dtype=int,
    )

    lower_leg_contact_duration = (
        lower_leg_contact_steps
        * model.opt.timestep
    )

    peak_lower_leg_force = max(
        initial_lower_leg_force,
        float(
            np.max(
                lower_leg_forces_array
            )
        ),
    )

    saturation_percentages = (
        100.0
        * saturation_counts
        / number_of_steps
    )

    print("\nDynamic contact summary:")
    print(
        f"Peak summed lower-leg normal force: "
        f"{peak_lower_leg_force:.3f} N"
    )
    print(
        f"Lower-leg contact duration: "
        f"{lower_leg_contact_duration:.6f} s"
    )
    print(
        "First time all lower legs cleared: "
        + (
            f"{first_clear_time:.6f} s"
            if first_clear_time is not None
            else "never"
        )
    )
    print(
        f"Minimum base height: "
        f"{np.min(base_heights_array):.6f} m"
    )
    print(
        f"Maximum base height: "
        f"{np.max(base_heights_array):.6f} m"
    )
    print(
        f"Final base height: "
        f"{base_heights_array[-1]:.6f} m"
    )
    print(
        f"Peak base linear speed: "
        f"{np.max(base_linear_speeds_array):.6f} m/s"
    )
    print(
        f"Peak base angular speed: "
        f"{np.max(base_angular_speeds_array):.6f} rad/s"
    )
    print(
        f"Peak absolute joint speed: "
        f"{np.max(joint_speeds_array):.6f} rad/s"
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
        "Final feet contacting terrain: "
        f"{foot_counts_array[-1]}/4"
    )
    print(
        "Final lower legs contacting terrain: "
        f"{lower_leg_counts_array[-1]}/4"
    )

    print("\nPer-joint torque saturation:")
    for joint_name, percentage in zip(
        JOINT_NAMES,
        saturation_percentages,
        strict=True,
    ):
        print(
            f"  {joint_name:6s}: "
            f"{percentage:7.3f}%"
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
                "base_linear_speed_m_s",
                "base_angular_speed_rad_s",
                "max_abs_joint_speed_rad_s",
                "max_joint_error_rad",
                "max_applied_torque_nm",
                "lower_leg_normal_force_n",
                "lower_leg_contact_count",
                "foot_contact_count",
            ]
        )
        writer.writerows(
            zip(
                times_array,
                base_heights_array,
                base_linear_speeds_array,
                base_angular_speeds_array,
                joint_speeds_array,
                joint_errors_array,
                torques_array,
                lower_leg_forces_array,
                lower_leg_counts_array,
                foot_counts_array,
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
        label="Base height",
    )
    axes[0].axhline(
        home_base_height,
        linestyle="--",
        label="Home height",
    )
    axes[0].set_ylabel("Height [m]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        times_array,
        lower_leg_forces_array,
    )
    axes[1].set_ylabel(
        "Lower-leg force [N]"
    )
    axes[1].grid(True)

    axes[2].plot(
        times_array,
        base_linear_speeds_array,
        label="Base linear",
    )
    axes[2].plot(
        times_array,
        base_angular_speeds_array,
        label="Base angular",
    )
    axes[2].plot(
        times_array,
        joint_speeds_array,
        label="Max joint",
    )
    axes[2].set_ylabel("Speed")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].step(
        times_array,
        lower_leg_counts_array,
        where="post",
        label="Lower legs",
    )
    axes[3].step(
        times_array,
        foot_counts_array,
        where="post",
        label="Feet",
    )
    axes[3].set_ylabel("Contact count")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylim(-0.2, 4.2)
    axes[3].legend()
    axes[3].grid(True)

    figure.suptitle(
        "Dynamic lower-leg penetration recovery"
    )
    figure.tight_layout()
    figure.savefig(
        PLOT_PATH,
        dpi=160,
    )
    plt.close(figure)

    print(f"\nCSV written to: {CSV_PATH}")
    print(f"Plot written to: {PLOT_PATH}")

    failures: list[str] = []

    if initial_lower_legs != lower_leg_ids:
        failures.append(
            "the initial pose did not place all four "
            "lower legs in terrain contact"
        )

    if initial_feet != foot_ids:
        failures.append(
            "the initial pose did not place all four "
            "feet in terrain contact"
        )

    if peak_lower_leg_force <= 0.0:
        failures.append(
            "no positive lower-leg normal force was measured"
        )

    if first_clear_time is None:
        failures.append(
            "the lower-leg contacts never cleared"
        )

    if lower_leg_counts_array[-1] != 0:
        failures.append(
            "one or more lower legs still contacted "
            "the terrain at the end"
        )

    if foot_counts_array[-1] != len(foot_ids):
        failures.append(
            "not all four feet contacted the terrain "
            "at the end"
        )

    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"  - {failure}")

        raise RuntimeError(
            "Dynamic lower-leg contact validation failed."
        )

    print(
        "\nPASS: the simulation remained finite, generated "
        "measurable lower-leg contact force, cleared all "
        "lower-leg contacts, and finished on all four feet."
    )


if __name__ == "__main__":
    main()