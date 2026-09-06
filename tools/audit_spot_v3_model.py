#!/usr/bin/env python3
"""Audit the generated Spot V3 rigid-body MJCF.

Checks:
- compiled model dimensions and joint ordering
- total and per-link mass
- principal inertia validity
- V2 -> V3 mass comparison
- HOME apparent joint inertia with floating base
- HOME apparent joint inertia with base constrained

This script does not train anything and does not require MjLab.
It only requires MuJoCo + NumPy, both already present in the project env.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

try:
    import mujoco
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "MuJoCo/NumPy could not be imported.\n"
        "Run with the project environment, for example:\n"
        "  uv run python tools/audit_spot_v3_model.py"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]

V2_XML = (
    REPO_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "reference"
    / "spot.xml"
)

V3_XML = (
    REPO_ROOT
    / "src"
    / "spot_locomotion_v2"
    / "robot"
    / "mjcf"
    / "v3"
    / "spot.xml"
)

EXPECTED_TOTAL_MASS = 31.5999994

EXPECTED_BODY_MASS = {
    "body": 16.707651,
    "fl_hip": 1.1368834,
    "fr_hip": 1.1368834,
    "hl_hip": 1.1368834,
    "hr_hip": 1.1368834,
    "fl_uleg": 2.2562037,
    "fr_uleg": 2.2562037,
    "hl_uleg": 2.2562037,
    "hr_uleg": 2.2562037,
    "fl_lleg": 0.33,
    "fr_lleg": 0.33,
    "hl_lleg": 0.33,
    "hr_lleg": 0.33,
}

JOINT_NAMES = (
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
)


def body_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise RuntimeError(f"Body {name!r} not found")
    return idx


def joint_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if idx < 0:
        raise RuntimeError(f"Joint {name!r} not found")
    return idx


def joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and name != "freejoint":
            names.append(name)
    return tuple(names)


def check_dimensions(model: mujoco.MjModel) -> None:
    expected = {
        "nbody": 14,  # world + 13 physical links
        "njnt": 13,   # free joint + 12 leg joints
        "nq": 19,
        "nv": 18,
        "nu": 0,      # V3 rigid-body model intentionally has no actuators yet
    }

    for attr, value in expected.items():
        actual = int(getattr(model, attr))
        if actual != value:
            raise RuntimeError(
                f"{attr} mismatch: expected {value}, got {actual}"
            )

    actual_joint_names = joint_names(model)
    if actual_joint_names != JOINT_NAMES:
        raise RuntimeError(
            "Joint ordering mismatch.\n"
            f"Expected: {JOINT_NAMES}\n"
            f"Actual:   {actual_joint_names}"
        )


def check_masses(model: mujoco.MjModel) -> None:
    total = float(np.sum(model.body_mass))

    print("\n=== V3 mass audit ===")
    print(f"Total compiled mass: {total:.9f} kg")
    print(f"Expected mass:       {EXPECTED_TOTAL_MASS:.9f} kg")
    print(f"Error:               {total - EXPECTED_TOTAL_MASS:+.3e} kg")

    if not math.isclose(total, EXPECTED_TOTAL_MASS, abs_tol=2e-6):
        raise RuntimeError("V3 total mass does not match the expected Isaac-derived mass")

    print()
    print(f"{'body':<12} {'compiled kg':>13} {'expected kg':>13} {'error':>12}")
    print("-" * 54)

    for name, expected in EXPECTED_BODY_MASS.items():
        bid = body_id(model, name)
        actual = float(model.body_mass[bid])
        err = actual - expected
        print(f"{name:<12} {actual:13.7f} {expected:13.7f} {err:12.3e}")

        if not math.isclose(actual, expected, abs_tol=1e-7):
            raise RuntimeError(
                f"Mass mismatch for {name}: expected {expected}, got {actual}"
            )


def check_inertias(model: mujoco.MjModel) -> None:
    print("\n=== Principal inertia audit ===")
    print(f"{'body':<12} {'Ixx':>12} {'Iyy':>12} {'Izz':>12}  status")
    print("-" * 61)

    for name in EXPECTED_BODY_MASS:
        bid = body_id(model, name)
        I = np.asarray(model.body_inertia[bid], dtype=float)

        positive = bool(np.all(I > 0.0))
        triangle = bool(
            I[0] <= I[1] + I[2] + 1e-12
            and I[1] <= I[0] + I[2] + 1e-12
            and I[2] <= I[0] + I[1] + 1e-12
        )

        status = "PASS" if positive and triangle else "FAIL"
        print(
            f"{name:<12} "
            f"{I[0]:12.8f} {I[1]:12.8f} {I[2]:12.8f}  {status}"
        )

        if status == "FAIL":
            raise RuntimeError(f"Invalid principal inertia for {name}")


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)

        # freejoint qpos starts at 0..6, then 12 hinge joint qpos entries
        data.qpos[2] = 0.46
        data.qpos[3] = 1.0

        home = {
            "fl_hx": 0.0, "fl_hy": 1.04, "fl_kn": -1.8,
            "fr_hx": 0.0, "fr_hy": 1.04, "fr_kn": -1.8,
            "hl_hx": 0.0, "hl_hy": 1.04, "hl_kn": -1.8,
            "hr_hx": 0.0, "hr_hy": 1.04, "hr_kn": -1.8,
        }
        for name, q in home.items():
            jid = joint_id(model, name)
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = q

    mujoco.mj_forward(model, data)


def full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, M)
    return M


def apparent_inertias(model: mujoco.MjModel) -> dict[str, tuple[float, float]]:
    data = mujoco.MjData(model)
    set_home(model, data)
    M = full_mass_matrix(model, data)

    # Floating-base apparent inertia: apply unit torque at the joint while
    # allowing all generalized coordinates, including the base, to respond.
    M_inv = np.linalg.inv(M)

    dofs = [int(model.jnt_dofadr[joint_id(model, n)]) for n in JOINT_NAMES]

    # Base-constrained equivalent: remove the 6 free-base DOFs and invert the
    # remaining joint-space mass block. Other joints remain dynamically free.
    M_joint = M[np.ix_(dofs, dofs)]
    M_joint_inv = np.linalg.inv(M_joint)

    result = {}
    for k, (name, dof) in enumerate(zip(JOINT_NAMES, dofs, strict=True)):
        j_float = 1.0 / M_inv[dof, dof]
        j_base_constrained = 1.0 / M_joint_inv[k, k]
        result[name] = (float(j_float), float(j_base_constrained))

    return result


def compare_v2_v3_masses(v2: mujoco.MjModel, v3: mujoco.MjModel) -> None:
    print("\n=== V2 -> V3 body-mass comparison ===")
    print(f"{'body':<12} {'V2 kg':>10} {'V3 kg':>10} {'change %':>10}")
    print("-" * 46)

    for name in EXPECTED_BODY_MASS:
        m2 = float(v2.body_mass[body_id(v2, name)])
        m3 = float(v3.body_mass[body_id(v3, name)])
        pct = 100.0 * (m3 - m2) / m2
        print(f"{name:<12} {m2:10.5f} {m3:10.5f} {pct:10.2f}")

    total2 = float(np.sum(v2.body_mass))
    total3 = float(np.sum(v3.body_mass))
    pct = 100.0 * (total3 - total2) / total2

    print("-" * 46)
    print(f"{'TOTAL':<12} {total2:10.5f} {total3:10.5f} {pct:10.2f}")


def compare_apparent_inertia(v2: mujoco.MjModel, v3: mujoco.MjModel) -> None:
    a2 = apparent_inertias(v2)
    a3 = apparent_inertias(v3)

    print("\n=== HOME apparent joint inertia ===")
    print(
        f"{'joint':<8} "
        f"{'V2 float':>11} {'V3 float':>11} "
        f"{'V2 basefix':>12} {'V3 basefix':>12}"
    )
    print("-" * 60)

    for name in JOINT_NAMES:
        v2f, v2b = a2[name]
        v3f, v3b = a3[name]
        print(
            f"{name:<8} "
            f"{v2f:11.6f} {v3f:11.6f} "
            f"{v2b:12.6f} {v3b:12.6f}"
        )


def main() -> None:
    if not V3_XML.is_file():
        raise FileNotFoundError(
            f"V3 model not found: {V3_XML}\n"
            "Run first:\n"
            "  uv run python tools/build_spot_v3_model.py"
        )

    if not V2_XML.is_file():
        raise FileNotFoundError(f"V2/reference model not found: {V2_XML}")

    print(f"Loading V3: {V3_XML}")
    v3 = mujoco.MjModel.from_xml_path(str(V3_XML))

    check_dimensions(v3)
    check_masses(v3)
    check_inertias(v3)

    print(f"\nLoading V2 reference for comparison: {V2_XML}")
    v2 = mujoco.MjModel.from_xml_path(str(V2_XML))

    compare_v2_v3_masses(v2, v3)
    compare_apparent_inertia(v2, v3)

    print("\nPASS: Spot V3 rigid-body model compiled and passed the mechanical audit.")
    print("\nAt this point do NOT tune actuators yet.")
    print("First inspect the V2 -> V3 mass/inertia changes printed above.")


if __name__ == "__main__":
    main()
