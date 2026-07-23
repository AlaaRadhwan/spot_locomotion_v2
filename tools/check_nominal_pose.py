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
    / "reference"
    / "spot.xml"
)

FOOT_GEOM_NAMES = ["FL", "FR", "HL", "HR"]


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_keyframe_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_keyframe_id < 0:
        raise RuntimeError("Home keyframe not found.")

    # Copy the home keyframe's qpos, qvel, and ctrl into MjData.
    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_keyframe_id,
    )

    # Calculate body and geometry world positions from qpos.
    mujoco.mj_forward(model, data)

    print(f"Model: {MODEL_PATH}")
    print(f"Base position: {data.qpos[0:3]}")
    print(f"Base quaternion: {data.qpos[3:7]}")

    print("\nNominal foot positions:")

    bottom_heights: list[float] = []

    for foot_name in FOOT_GEOM_NAMES:
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            foot_name,
        )

        if geom_id < 0:
            raise RuntimeError(
                f"Foot geometry not found: {foot_name}"
            )

        centre_position = data.geom_xpos[geom_id].copy()
        radius = float(model.geom_size[geom_id, 0])

        # TODO:
        # Calculate the lowest point of the foot sphere.
        bottom_z = buttom_z = centre_position[2] - radius

        # TODO:
        # Convert metres to millimetres.
        bottom_z_mm = bottom_z * 1000

        bottom_heights.append(bottom_z)

        print(
            f"{foot_name} | "
            f"centre={centre_position} | "
            f"radius={radius:.6f} m | "
            f"bottom_z={bottom_z:.6f} m "
            f"({bottom_z_mm:.3f} mm)"
        )

    print("\nClearance summary:")
    print(
        f"minimum={min(bottom_heights) * 1000.0:.3f} mm"
    )
    print(
        f"maximum={max(bottom_heights) * 1000.0:.3f} mm"
    )
    print(
        f"difference="
        f"{(max(bottom_heights) - min(bottom_heights)) * 1000.0:.3f} mm"
    )

    print("\nInitial actuator forces:")
    print(data.actuator_force)


if __name__ == "__main__":
    main()