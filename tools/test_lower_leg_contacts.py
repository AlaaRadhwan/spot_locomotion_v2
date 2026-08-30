from pathlib import Path

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


def get_terrain_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    terrain_id: int,
) -> set[int]:
    """Return geom IDs currently contacting the terrain."""

    contacting_geom_ids: set[int] = set()

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]

        if contact.geom1 == terrain_id:
            contacting_geom_ids.add(contact.geom2)
        elif contact.geom2 == terrain_id:
            contacting_geom_ids.add(contact.geom1)

    return contacting_geom_ids


def geom_names_from_ids(
    model: mujoco.MjModel,
    geom_ids: set[int],
) -> list[str]:
    """Convert geom IDs into readable MuJoCo names."""

    names = []

    for geom_id in sorted(geom_ids):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        names.append(
            name if name is not None else f"<unnamed:{geom_id}>"
        )

    return names


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
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

    lower_leg_ids = {
        get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in LOWER_LEG_GEOM_NAMES
    }

    foot_ids = {
        get_object_id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in FOOT_GEOM_NAMES
    }

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_id,
    )
    mujoco.mj_forward(model, data)

    initial_base_height = float(data.qpos[2])

    print(f"Model: {MODEL_PATH}")
    print(f"Initial base height: {initial_base_height:.6f} m")
    print(f"Lower-leg geom IDs: {sorted(lower_leg_ids)}")
    print(f"Foot geom IDs: {sorted(foot_ids)}")

    detected_lower_legs: set[int] = set()
    detected_feet: set[int] = set()

    # Move downward by at most 10 cm, in 1 mm increments.
    probe_depths = np.arange(
        0.0,
        0.101,
        0.001,
    )

    for depth in probe_depths:
        data.qpos[2] = initial_base_height - depth

        # Recalculate kinematics and contacts without advancing time.
        mujoco.mj_forward(model, data)

        terrain_contacts = get_terrain_contacts(
            model,
            data,
            terrain_id,
        )

        current_feet = terrain_contacts & foot_ids
        current_lower_legs = (
            terrain_contacts & lower_leg_ids
        )

        new_feet = current_feet - detected_feet
        new_lower_legs = (
            current_lower_legs - detected_lower_legs
        )

        if new_feet:
            print(
                f"Depth {depth:.3f} m: new foot contacts: "
                f"{geom_names_from_ids(model, new_feet)}"
            )

        if new_lower_legs:
            print(
                f"Depth {depth:.3f} m: new lower-leg contacts: "
                f"{geom_names_from_ids(model, new_lower_legs)}"
            )

        detected_feet.update(current_feet)
        detected_lower_legs.update(current_lower_legs)

        if detected_lower_legs == lower_leg_ids:
            break

    print("\nContact-test summary:")
    print(
        "Detected feet: "
        f"{geom_names_from_ids(model, detected_feet)}"
    )
    print(
        "Detected lower legs: "
        f"{geom_names_from_ids(model, detected_lower_legs)}"
    )

    missing_lower_legs = (
        lower_leg_ids - detected_lower_legs
    )

    # if missing_lower_legs:
    #     missing_names = geom_names_from_ids(
    #         model,
    #         missing_lower_legs,
    #     )
    #     raise RuntimeError(
    #         "Lower-leg contact validation failed. "
    #         f"Never detected: {missing_names}"
    #     )

    # if detected_feet != foot_ids:
    #     missing_feet = geom_names_from_ids(
    #         model,
    #         foot_ids - detected_feet,
    #     )
    #     raise RuntimeError(
    #         "Foot contact validation failed. "
    #         f"Never detected: {missing_feet}"
    #     )

    # if lower_leg_ids & foot_ids:
    #     raise RuntimeError(
    #         "A lower-leg geom ID was incorrectly classified "
    #         "as a foot geom ID."
    #     )

    # print(
    #     "\nPASS: all four lower-leg capsules and all four "
    #     "foot spheres generated distinct terrain contacts."
    # )


if __name__ == "__main__":
    main()