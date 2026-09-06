"""Shared utilities for visual Spot V3 deterministic tests."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from spot_locomotion_v2.robot.spot_v3_constants import (
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    V3_HIP_EFFORT_LIMIT,
    V3_KD,
    V3_KNEE_EFFORT_LIMIT,
    V3_KP,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports" / "v3_actuator"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FOOT_NAMES = ("FL", "FR", "HL", "HR")


def object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    idx = mujoco.mj_name2id(model, object_type, name)
    if idx < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return idx


def get_joint_arrays(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = np.asarray(
        [
            object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )

    qpos_addresses = model.jnt_qposadr[joint_ids].astype(int)
    dof_addresses = model.jnt_dofadr[joint_ids].astype(int)

    actuator_ids = np.asarray(
        [
            object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )

    return qpos_addresses, dof_addresses, actuator_ids


def reset_to_init_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Reset to the MjLab-generated init-state keyframe when available."""
    key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "init_state",
    )

    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        # Defensive fallback.
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0.0, 0.0, 0.46]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        qpos_addresses, _, _ = get_joint_arrays(model)
        data.qpos[qpos_addresses] = np.asarray(
            HOME_JOINT_POSITIONS,
            dtype=np.float64,
        )

    # The built-in position actuator ctrl signal IS the position target.
    data.ctrl[:] = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)
    mujoco.mj_forward(model, data)


def roll_pitch_deg(q: np.ndarray) -> tuple[float, float]:
    """Return roll and pitch in degrees from MuJoCo wxyz quaternion."""
    w, x, y, z = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))

    return math.degrees(roll), math.degrees(pitch)


def count_feet_on_ground(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> int:
    terrain_id = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    foot_ids = {
        object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_NAMES
    }

    touching: set[int] = set()

    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)

        if g1 == terrain_id and g2 in foot_ids:
            touching.add(g2)
        elif g2 == terrain_id and g1 in foot_ids:
            touching.add(g1)

    return len(touching)


def actual_joint_torques(
    data: mujoco.MjData,
    dof_addresses: np.ndarray,
) -> np.ndarray:
    """Return generalized actuator torque actually applied at each leg joint."""
    return np.asarray(data.qfrc_actuator[dof_addresses], dtype=np.float64).copy()


def print_model_actuator_summary(model: mujoco.MjModel) -> None:
    print("\n=== V3 actuator model ===")
    print("MjLab actuator: BuiltinPositionActuatorCfg")
    print(f"Kp: {V3_KP}")
    print(f"Kd: {V3_KD}")
    print(f"HX/HY effort limit: {V3_HIP_EFFORT_LIMIT:.1f} N*m")
    print(f"knee effort limit:  {V3_KNEE_EFFORT_LIMIT:.1f} N*m")
    print(f"physics timestep:   {model.opt.timestep:.6f} s")
    print(f"compiled nu:        {model.nu}")

    print("\nCompiled actuator order:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i:2d}: {name}")


def save_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
