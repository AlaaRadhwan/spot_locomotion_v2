#!/usr/bin/env python3
"""Comprehensive Spot V2 joint-dynamics audit.

One execution performs:
- mechanical model audit
- visual confirmation of representative poses
- fixed/floating mass-matrix and apparent-inertia analysis
- joint coupling and current-PD small-signal predictions
- direct-torque pulse validation at HOME
- contact-constrained standing support / perturbation test
- CSV, plots, and Markdown summary generation

Recommended:
    uv run python scripts/spot_joint_dynamics_audit.py --interactive
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace

print("[BOOT] Spot joint-dynamics audit starting...", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("[BOOT] matplotlib loaded", flush=True)

import mujoco
import mujoco.viewer
print("[BOOT] MuJoCo + viewer loaded", flush=True)

import numpy as np
print("[BOOT] NumPy loaded", flush=True)


def _load_project_constants_without_importing_mjlab() -> SimpleNamespace:
    """Read simple project constants from spot_constants.py without executing it."""

    repo_root = Path(__file__).resolve().parents[1]
    constants_path = (
        repo_root
        / "src"
        / "spot_locomotion_v2"
        / "robot"
        / "spot_constants.py"
    )

    if not constants_path.is_file():
        raise FileNotFoundError(
            f"spot_constants.py not found: {constants_path}"
        )

    tree = ast.parse(constants_path.read_text())
    values: dict[str, object] = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue

            try:
                values[target.id] = ast.literal_eval(value_node)
            except Exception:
                pass

    required = (
        "PHYSICS_FREQUENCY_HZ",
        "POLICY_FREQUENCY_HZ",
        "JOINT_NAMES",
        "HOME_JOINT_POSITIONS",
        "HIP_TORQUE_LIMIT",
        "KNEE_TORQUE_LIMIT",
    )

    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(
            "Could not read required literal constants from "
            f"{constants_path}: {missing}"
        )

    if "HIP_KP" in values:
        hip_kp = float(values["HIP_KP"])
        hip_kd = float(values["HIP_KD"])
        knee_kp = float(values["KNEE_KP"])
        knee_kd = float(values["KNEE_KD"])
    elif "KP" in values:
        hip_kp = knee_kp = float(values["KP"])
        hip_kd = knee_kd = float(values["KD"])
    else:
        raise RuntimeError(
            "No supported gain constants were found in spot_constants.py."
        )

    joint_names = tuple(values["JOINT_NAMES"])
    home_positions = tuple(values["HOME_JOINT_POSITIONS"])

    if len(joint_names) != len(home_positions):
        raise RuntimeError(
            "JOINT_NAMES and HOME_JOINT_POSITIONS have different lengths."
        )

    home_map = dict(zip(joint_names, home_positions, strict=True))

    robot_dir = constants_path.parent
    spot_xml = (
        robot_dir
        / "mjcf"
        / "experiments"
        / "spot_floating_base_explicit_pd.xml"
    )

    cfg = SimpleNamespace(
        __file__=str(constants_path),
        SPOT_XML=spot_xml,
        PHYSICS_FREQUENCY_HZ=int(values["PHYSICS_FREQUENCY_HZ"]),
        POLICY_FREQUENCY_HZ=int(values["POLICY_FREQUENCY_HZ"]),
        JOINT_NAMES=joint_names,
        HOME_JOINT_POSITIONS=home_positions,
        HOME_JOINT_POSITION_MAP=home_map,
        HIP_KP=hip_kp,
        HIP_KD=hip_kd,
        KNEE_KP=knee_kp,
        KNEE_KD=knee_kd,
        HIP_TORQUE_LIMIT=float(values["HIP_TORQUE_LIMIT"]),
        KNEE_TORQUE_LIMIT=float(values["KNEE_TORQUE_LIMIT"]),
    )

    print(
        "[BOOT] Project constants loaded without importing MjLab: "
        f"physics={cfg.PHYSICS_FREQUENCY_HZ} Hz, "
        f"policy={cfg.POLICY_FREQUENCY_HZ} Hz, "
        f"hips={cfg.HIP_KP:g}/{cfg.HIP_KD:g}, "
        f"knees={cfg.KNEE_KP:g}/{cfg.KNEE_KD:g}",
        flush=True,
    )

    return cfg


sc = _load_project_constants_without_importing_mjlab()


POSES = {
    "HOME": (0.00, 1.04, -1.80),
    "DEEP_CROUCH": (0.00, 1.35, -2.25),
    "EXTENDED": (0.00, 0.85, -1.45),
    # FOOT_FORWARD and FOOT_REARWARD are added automatically from
    # the actual MuJoCo forward kinematics after model loading.
}
REP_JOINTS = ("fl_hx", "fl_hy", "fl_kn")
TARGET_ACCEL = 30.0
PULSE_LIMIT_FRACTION = 0.10
STANCE_SETTLE_S = 2.5
STANCE_SAMPLE_S = 0.5
STANCE_PULSE_S = 0.040
STANCE_RECOVERY_S = 0.45
STANCE_PULSE_FRACTION = 0.05
DEFAULT_HOLD_S = 2.0
DEFAULT_OUTPUT = Path("reports") / "joint_dynamics_audit"


def project_paths():
    root = Path(sc.__file__).resolve().parent / "mjcf" / "experiments"
    fixed = root / "spot_fixed_base_explicit_pd.xml"
    floating = Path(sc.SPOT_XML)
    standing = root / "scene_free_standing.xml"
    for p in (fixed, floating, standing):
        if not p.is_file():
            raise FileNotFoundError(p)
    return fixed, floating, standing


def current_gains():
    if hasattr(sc, "HIP_KP"):
        return {
            "hip_kp": float(sc.HIP_KP), "hip_kd": float(sc.HIP_KD),
            "knee_kp": float(sc.KNEE_KP), "knee_kd": float(sc.KNEE_KD),
        }
    return {
        "hip_kp": float(sc.KP), "hip_kd": float(sc.KD),
        "knee_kp": float(sc.KP), "knee_kd": float(sc.KD),
    }


def current_limits():
    return {
        "hip": float(sc.HIP_TORQUE_LIMIT),
        "knee": float(sc.KNEE_TORQUE_LIMIT),
    }


def gain(name, g):
    if name.endswith(("_hx", "_hy")):
        return g["hip_kp"], g["hip_kd"]
    return g["knee_kp"], g["knee_kd"]


def limit(name, lim):
    return lim["hip"] if name.endswith(("_hx", "_hy")) else lim["knee"]


def obj_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing {name!r}")
    return int(idx)


def joint_addr(model, name):
    jid = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    aid = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), aid, jid


def full_mass_matrix(model, data):
    M = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, M)
    return M


def reset_home(model, data):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, int(kid))
    else:
        mujoco.mj_resetData(model, data)
        for name, q in sc.HOME_JOINT_POSITION_MAP.items():
            qadr, _, _, _ = joint_addr(model, name)
            data.qpos[qadr] = float(q)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def set_free_base(model, data, z=0.75):
    free = np.where(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]
    if free.size == 0:
        return
    qadr = int(model.jnt_qposadr[int(free[0])])
    data.qpos[qadr:qadr+7] = [0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]


def pose_map(triplet):
    hx, hy, kn = triplet
    out = {}
    for p in ("fl", "fr", "hl", "hr"):
        out[f"{p}_hx"] = hx
        out[f"{p}_hy"] = hy
        out[f"{p}_kn"] = kn
    return out


def set_pose(model, data, triplet, floating=False):
    reset_home(model, data)
    if floating:
        set_free_base(model, data)
    for name, q in pose_map(triplet).items():
        qadr, _, _, _ = joint_addr(model, name)
        data.qpos[qadr] = q
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def validate_poses(model):
    for pose_name, triplet in POSES.items():
        for name, q in pose_map(triplet).items():
            _, _, _, jid = joint_addr(model, name)
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                if not (lo <= q <= hi):
                    raise ValueError(
                        f"{pose_name}: {name}={q} outside [{lo}, {hi}]"
                    )


def foot_pos(model, data):
    sid = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, "FL_site")
    return np.asarray(data.site_xpos[sid]).copy()


def hip_pos(model, data):
    bid = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "fl_hip")
    return np.asarray(data.xpos[bid]).copy()


def foot_rel_hip(model, data):
    return foot_pos(model, data) - hip_pos(model, data)


def auto_select_workspace_poses(model):
    """Select distinct forward/rearward static leg poses from actual kinematics."""
    data = mujoco.MjData(model)
    set_pose(model, data, POSES["HOME"], floating=False)
    home_rel = foot_rel_hip(model, data)
    target_z = float(home_rel[2] + 0.05)

    hy_values = np.linspace(0.55, 1.55, 81)
    kn_values = np.linspace(-2.30, -1.20, 89)
    refs = np.array([POSES["HOME"], POSES["DEEP_CROUCH"], POSES["EXTENDED"]], dtype=float)

    candidates = []
    used_band = None
    for z_band in (0.020, 0.035, 0.055, 0.080):
        candidates = []
        for hy in hy_values:
            for kn in kn_values:
                triplet = (0.0, float(hy), float(kn))
                # All candidate values are intentionally conservative, but
                # still validate against the compiled MJCF limits.
                ok = True
                for jname, q in pose_map(triplet).items():
                    _, _, _, jid = joint_addr(model, jname)
                    if model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        if not (lo <= q <= hi):
                            ok = False
                            break
                if not ok:
                    continue

                qvec = np.array(triplet, dtype=float)
                if np.min(np.linalg.norm(refs - qvec[None, :], axis=1)) < 0.18:
                    continue

                set_pose(model, data, triplet, floating=False)
                rel = foot_rel_hip(model, data)
                if abs(float(rel[2]) - target_z) > z_band:
                    continue

                candidates.append({
                    "triplet": triplet,
                    "x": float(rel[0]),
                    "y": float(rel[1]),
                    "z": float(rel[2]),
                })

        if len(candidates) >= 2:
            used_band = z_band
            break

    if len(candidates) < 2:
        raise RuntimeError("Could not find enough valid forward/rearward workspace poses.")

    forward = max(candidates, key=lambda r: r["x"])
    rearward = min(candidates, key=lambda r: r["x"])
    separation = forward["x"] - rearward["x"]
    if separation < 0.12:
        raise RuntimeError(
            f"Automatically selected forward/rearward poses differ by only {separation:.4f} m in foot x."
        )

    selected = {
        "FOOT_FORWARD": tuple(forward["triplet"]),
        "FOOT_REARWARD": tuple(rearward["triplet"]),
    }
    rows = []
    for label, row in (("FOOT_FORWARD", forward), ("FOOT_REARWARD", rearward)):
        hx, hy, kn = row["triplet"]
        rows.append({
            "pose": label,
            "hx_rad": hx,
            "hy_rad": hy,
            "kn_rad": kn,
            "foot_rel_hip_x_m": row["x"],
            "foot_rel_hip_y_m": row["y"],
            "foot_rel_hip_z_m": row["z"],
            "target_foot_rel_hip_z_m": target_z,
            "selection_z_band_m": used_band,
        })
    return selected, rows


def hold_viewer(viewer, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if not viewer.is_running():
            raise KeyboardInterrupt
        viewer.sync()
        time.sleep(0.02)


def visual_checkpoint(viewer, model, data, title, interactive, hold_s):
    foot = foot_pos(model, data)
    rel = foot_rel_hip(model, data)
    print(f"\n[VIEW] {title}")
    print(
        f"       FL foot world position: x={foot[0]:+.4f}, "
        f"y={foot[1]:+.4f}, z={foot[2]:+.4f} m"
    )
    print(
        f"       FL foot relative to hip: x={rel[0]:+.4f}, "
        f"y={rel[1]:+.4f}, z={rel[2]:+.4f} m"
    )
    viewer.sync()
    if interactive:
        ans = input("       Enter=accept, q=abort: ").strip().lower()
        if ans == "q":
            raise KeyboardInterrupt
    else:
        hold_viewer(viewer, hold_s)


def show_poses(model, label, floating, interactive, hold_s):
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for name, triplet in POSES.items():
            set_pose(model, data, triplet, floating=floating)
            visual_checkpoint(
                viewer, model, data, f"{label}: {name} {triplet}",
                interactive, hold_s,
            )


def audit_bodies(model):
    rows, flags = [], []
    total = 0.0
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
        m = float(model.body_mass[bid])
        I = np.asarray(model.body_inertia[bid], dtype=float)
        com = np.asarray(model.body_ipos[bid], dtype=float)
        total += m
        positive = bool(m > 0 and np.all(I > 0))
        triangle = bool(
            I[0] <= I[1] + I[2] + 1e-12 and
            I[1] <= I[0] + I[2] + 1e-12 and
            I[2] <= I[0] + I[1] + 1e-12
        )
        spread = float(np.ptp(I) / max(np.mean(I), 1e-12))
        rows.append({
            "body": name, "mass_kg": m,
            "com_x_m": float(com[0]), "com_y_m": float(com[1]), "com_z_m": float(com[2]),
            "I1_kg_m2": float(I[0]), "I2_kg_m2": float(I[1]), "I3_kg_m2": float(I[2]),
            "positive_mass_and_inertia": positive,
            "triangle_inequality_ok": triangle,
            "principal_inertia_relative_spread": spread,
        })
        if not positive:
            flags.append(f"{name}: non-positive mass or principal inertia")
        if not triangle:
            flags.append(f"{name}: inertia triangle inequality failed")
        if name == "body" and spread < 1e-3:
            flags.append(
                "Torso principal inertias are nearly identical; source-audit this for the elongated torso."
            )
    rows.append({"body": "__TOTAL__", "mass_kg": total})
    return rows, flags


def audit_joint_passive(model):
    rows = []
    for name in sc.JOINT_NAMES:
        _, d, _, _ = joint_addr(model, name)
        rows.append({
            "joint": name,
            "armature_kg_m2": float(model.dof_armature[d]),
            "damping_nms_rad": float(model.dof_damping[d]),
            "frictionloss_nm": float(model.dof_frictionloss[d]),
        })
    return rows


def normalized_coupling(M):
    C = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            den = math.sqrt(max(M[i, i], 0) * max(M[j, j], 0))
            if den > 0:
                C[i, j] = abs(M[i, j]) / den
    return C


def effective_matrix_with_other_dofs_free(M, dofs):
    H = np.linalg.inv(M)
    Hs = H[np.ix_(dofs, dofs)]
    return np.linalg.inv(Hs)


def modal_analysis(M_eff, kp, kd):
    n = len(kp)
    Minv = np.linalg.inv(M_eff)
    K = np.diag(kp)
    D = np.diag(kd)
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-Minv @ K, -Minv @ D],
    ])
    eig = np.linalg.eigvals(A)
    pos_imag = sorted([e for e in eig if e.imag > 1e-7], key=abs)
    rows = []
    if pos_imag:
        for i, e in enumerate(pos_imag, 1):
            wn = abs(e)
            rows.append({
                "mode_index": i,
                "real_part_1_s": float(e.real),
                "imag_part_rad_s": float(e.imag),
                "natural_frequency_hz": float(wn / (2*math.pi)),
                "damped_frequency_hz": float(abs(e.imag) / (2*math.pi)),
                "damping_ratio": float(-e.real / wn),
            })
    else:
        for i, e in enumerate(sorted(eig, key=lambda x: abs(x.real))[:n], 1):
            rows.append({
                "mode_index": i,
                "real_part_1_s": float(e.real),
                "imag_part_rad_s": float(e.imag),
                "natural_frequency_hz": float(abs(e.real) / (2*math.pi)),
                "damped_frequency_hz": 0.0,
                "damping_ratio": 1.0,
            })
    return rows


def analyze_pose(model, label, pose_name, triplet, floating, gains, limits):
    data = mujoco.MjData(model)
    set_pose(model, data, triplet, floating=floating)
    M = full_mass_matrix(model, data)
    H = np.linalg.inv(M)

    dyn_rows = []
    for name in REP_JOINTS:
        qadr, d, _, _ = joint_addr(model, name)
        J_locked = float(M[d, d])
        J_free = float(1.0 / H[d, d])
        kp, kd = gain(name, gains)
        wn = math.sqrt(kp / J_free)
        zeta = kd / (2.0 * math.sqrt(kp * J_free))
        dyn_rows.append({
            "model": label, "pose": pose_name, "joint": name,
            "q_rad": float(data.qpos[qadr]),
            "J_locked_kg_m2": J_locked,
            "J_free_kg_m2": J_free,
            "free_to_locked_ratio": J_free / J_locked,
            "Kp_nm_rad": kp, "Kd_nms_rad": kd,
            "predicted_natural_frequency_hz": wn / (2*math.pi),
            "predicted_damping_ratio": zeta,
            "bias_generalized_force_at_zero_velocity": float(data.qfrc_bias[d]),
            "torque_limit_nm": limit(name, limits),
            "armature_kg_m2": float(model.dof_armature[d]),
            "joint_damping_nms_rad": float(model.dof_damping[d]),
            "joint_frictionloss_nm": float(model.dof_frictionloss[d]),
        })

    dofs = [joint_addr(model, name)[1] for name in REP_JOINTS]
    M_locked = M[np.ix_(dofs, dofs)]
    M_free = effective_matrix_with_other_dofs_free(M, dofs)
    C_locked = normalized_coupling(M_locked)
    C_free = normalized_coupling(M_free)
    coupling_rows = []
    for i, ni in enumerate(REP_JOINTS):
        for j, nj in enumerate(REP_JOINTS):
            coupling_rows.append({
                "model": label, "pose": pose_name,
                "joint_i": ni, "joint_j": nj,
                "M_locked_ij": float(M_locked[i, j]),
                "M_free_effective_ij": float(M_free[i, j]),
                "normalized_locked_coupling": float(C_locked[i, j]),
                "normalized_free_coupling": float(C_free[i, j]),
            })

    kp = np.array([gain(n, gains)[0] for n in REP_JOINTS])
    kd = np.array([gain(n, gains)[1] for n in REP_JOINTS])
    modes = []
    for row in modal_analysis(M_free, kp, kd):
        modes.append({"model": label, "pose": pose_name, **row})
    return dyn_rows, coupling_rows, modes


def new_data_at_pose(model, triplet, floating):
    data = mujoco.MjData(model)
    set_pose(model, data, triplet, floating=floating)
    return data


def pulse_validation(model, label, floating, limits):
    home = POSES["HOME"]
    ref = new_data_at_pose(model, home, floating)
    M = full_mass_matrix(model, ref)
    H = np.linalg.inv(M)
    rows = []
    for name in REP_JOINTS:
        _, d, aid, _ = joint_addr(model, name)
        J_free = float(1.0 / H[d, d])
        tau_req = TARGET_ACCEL * J_free
        tau_cap = PULSE_LIMIT_FRACTION * limit(name, limits)
        tau = math.copysign(min(abs(tau_req), tau_cap), tau_req)
        predicted_vec = H[:, d] * tau

        base = new_data_at_pose(model, home, floating)
        base.ctrl[:] = 0.0
        mujoco.mj_forward(model, base)

        pulse = new_data_at_pose(model, home, floating)
        pulse.ctrl[:] = 0.0
        pulse.ctrl[aid] = tau
        mujoco.mj_forward(model, pulse)
        forward_delta = pulse.qacc - base.qacc

        base_step = new_data_at_pose(model, home, floating)
        pulse_step = new_data_at_pose(model, home, floating)
        pulse_step.ctrl[aid] = tau
        mujoco.mj_step(model, base_step)
        mujoco.mj_step(model, pulse_step)
        step_delta = (pulse_step.qvel - base_step.qvel) / model.opt.timestep

        pred = float(predicted_vec[d])
        fwd = float(forward_delta[d])
        step = float(step_delta[d])
        rows.append({
            "model": label, "joint": name,
            "J_free_kg_m2": J_free,
            "pulse_torque_nm": tau,
            "pulse_fraction_of_limit": abs(tau) / limit(name, limits),
            "predicted_incremental_qacc_rad_s2": pred,
            "mj_forward_incremental_qacc_rad_s2": fwd,
            "one_step_incremental_qacc_rad_s2": step,
            "mj_forward_error_pct": 100*abs(fwd-pred)/max(abs(pred), 1e-12),
            "one_step_error_pct": 100*abs(step-pred)/max(abs(pred), 1e-12),
            "predicted_reaction_accel_norm_other_dofs": float(
                np.linalg.norm(np.delete(predicted_vec, d))
            ),
        })
    return rows


def pd_torques(model, data, gains, limits):
    tau = np.zeros(model.nu)
    for name in sc.JOINT_NAMES:
        qadr, d, aid, _ = joint_addr(model, name)
        kp, kd = gain(name, gains)
        raw = kp * (sc.HOME_JOINT_POSITION_MAP[name] - data.qpos[qadr]) - kd * data.qvel[d]
        tau[aid] = np.clip(raw, -limit(name, limits), limit(name, limits))
    return tau


def settle_stance(model, gains, limits, viewer=None, realtime=False):
    data = mujoco.MjData(model)
    reset_home(model, data)
    steps = int(round(STANCE_SETTLE_S / model.opt.timestep))
    sample_steps = int(round(STANCE_SAMPLE_S / model.opt.timestep))
    samples = []
    start = time.perf_counter()
    for k in range(steps):
        tau = pd_torques(model, data, gains, limits)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if k >= steps - sample_steps:
            row = {"time_s": float(data.time), "base_z_m": float(data.qpos[2])}
            for name in sc.JOINT_NAMES:
                qadr, d, aid, _ = joint_addr(model, name)
                row[f"{name}_q_rad"] = float(data.qpos[qadr])
                row[f"{name}_qd_rad_s"] = float(data.qvel[d])
                row[f"{name}_tau_nm"] = float(tau[aid])
            samples.append(row)
        if viewer is not None:
            viewer.sync()
            if realtime:
                target = start + (k+1)*model.opt.timestep
                rem = target - time.perf_counter()
                if rem > 0: time.sleep(rem)
    return data.qpos.copy(), data.qvel.copy(), samples, data


def reset_state(model, data, qpos, qvel):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def stance_branch(model, qpos0, qvel0, gains, limits, pulse_joint=None, trace=False):
    data = mujoco.MjData(model)
    reset_state(model, data, qpos0, qvel0)
    psteps = int(round(STANCE_PULSE_S / model.opt.timestep))
    rsteps = int(round(STANCE_RECOVERY_S / model.opt.timestep))
    rows = []
    for k in range(psteps+rsteps):
        tau = pd_torques(model, data, gains, limits)
        if pulse_joint is not None and k < psteps:
            _, _, aid, _ = joint_addr(model, pulse_joint)
            L = limit(pulse_joint, limits)
            tau[aid] = np.clip(tau[aid] + STANCE_PULSE_FRACTION*L, -L, L)
        if trace:
            row = {"time_s": k*model.opt.timestep, "pulse_active": float(k < psteps)}
            for name in REP_JOINTS:
                qadr, d, _, _ = joint_addr(model, name)
                row[f"{name}_q_rad"] = float(data.qpos[qadr])
                row[f"{name}_qd_rad_s"] = float(data.qvel[d])
            free = np.where(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]
            if free.size:
                d0 = int(model.jnt_dofadr[int(free[0])])
                for i, key in enumerate(("vx","vy","vz","wx","wy","wz")):
                    row[f"base_{key}"] = float(data.qvel[d0+i])
            rows.append(row)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
    return data.qvel.copy(), rows


def stance_test(model, gains, limits, show_viewer, interactive, hold_s):
    viewer = None
    settle_data = None
    if show_viewer:
        settle_data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, settle_data)
        print("\n[VIEW] STANCE: settling under current PD")
    try:
        if viewer is not None:
            # use the viewer's data object for visual settling
            reset_home(model, settle_data)
            steps = int(round(STANCE_SETTLE_S / model.opt.timestep))
            sample_steps = int(round(STANCE_SAMPLE_S / model.opt.timestep))
            samples = []
            start = time.perf_counter()
            for k in range(steps):
                tau = pd_torques(model, settle_data, gains, limits)
                settle_data.ctrl[:] = tau
                mujoco.mj_step(model, settle_data)
                if k >= steps - sample_steps:
                    row = {"time_s": float(settle_data.time), "base_z_m": float(settle_data.qpos[2])}
                    for name in sc.JOINT_NAMES:
                        qadr, d, aid, _ = joint_addr(model, name)
                        row[f"{name}_q_rad"] = float(settle_data.qpos[qadr])
                        row[f"{name}_qd_rad_s"] = float(settle_data.qvel[d])
                        row[f"{name}_tau_nm"] = float(tau[aid])
                    samples.append(row)
                viewer.sync()
                target = start + (k+1)*model.opt.timestep
                rem = target - time.perf_counter()
                if rem > 0: time.sleep(rem)
            qpos0, qvel0 = settle_data.qpos.copy(), settle_data.qvel.copy()
            if interactive:
                ans = input("       Inspect settled stance. Enter=accept, q=abort: ").strip().lower()
                if ans == "q": raise KeyboardInterrupt
            else:
                hold_viewer(viewer, hold_s)
        else:
            qpos0, qvel0, samples, _ = settle_stance(model, gains, limits)

        support = []
        for name in sc.JOINT_NAMES:
            tv = np.array([r[f"{name}_tau_nm"] for r in samples])
            qv = np.array([r[f"{name}_q_rad"] for r in samples])
            support.append({
                "joint": name,
                "mean_settled_q_rad": float(np.mean(qv)),
                "mean_support_tau_nm": float(np.mean(tv)),
                "mean_abs_support_tau_nm": float(np.mean(abs(tv))),
                "support_utilization_fraction": float(np.mean(abs(tv))/limit(name, limits)),
            })

        summaries, all_traces = [], []
        for name in REP_JOINTS:
            _, base_trace = stance_branch(
                model, qpos0, qvel0, gains, limits, None, True
            )
            _, pulse_trace = stance_branch(
                model, qpos0, qvel0, gains, limits, name, True
            )

            psteps = int(round(STANCE_PULSE_S / model.opt.timestep))
            pulse_end_idx = max(0, psteps - 1)

            delta_joint = []
            delta_base = []
            trace = []

            for rb, rp in zip(base_trace, pulse_trace, strict=True):
                row = dict(rp)
                row["pulse_joint"] = name

                dq = (
                    rp[f"{name}_qd_rad_s"]
                    - rb[f"{name}_qd_rad_s"]
                )
                row["incremental_joint_qd_rad_s"] = float(dq)
                delta_joint.append(float(dq))

                base_keys = (
                    "base_vx", "base_vy", "base_vz",
                    "base_wx", "base_wy", "base_wz",
                )
                if all(k in rp and k in rb for k in base_keys):
                    dv = np.array([rp[k] - rb[k] for k in base_keys])
                    base_norm = float(np.linalg.norm(dv))
                else:
                    base_norm = 0.0

                row["incremental_base_velocity_norm"] = base_norm
                delta_base.append(base_norm)
                trace.append(row)
                all_traces.append(row)

            summaries.append({
                "pulse_joint": name,
                "added_tau_nm": STANCE_PULSE_FRACTION*limit(name, limits),
                "pulse_duration_s": STANCE_PULSE_S,
                "incremental_joint_qd_at_pulse_end_rad_s": float(
                    delta_joint[pulse_end_idx]
                ),
                "peak_abs_incremental_joint_qd_rad_s": float(
                    np.max(np.abs(delta_joint))
                ),
                "peak_incremental_base_velocity_norm": float(
                    np.max(delta_base)
                ),
            })

            if viewer is not None:
                print(f"\n[VIEW] stance perturbation replay: {name}")
                reset_state(model, settle_data, qpos0, qvel0)
                if interactive:
                    ans = input("       Enter=run, q=abort: ").strip().lower()
                    if ans == "q": raise KeyboardInterrupt
                psteps = int(round(STANCE_PULSE_S / model.opt.timestep))
                total = int(round((STANCE_PULSE_S+STANCE_RECOVERY_S)/model.opt.timestep))
                start = time.perf_counter()
                for k in range(total):
                    tau = pd_torques(model, settle_data, gains, limits)
                    if k < psteps:
                        _, _, aid, _ = joint_addr(model, name)
                        L = limit(name, limits)
                        tau[aid] = np.clip(tau[aid] + STANCE_PULSE_FRACTION*L, -L, L)
                    settle_data.ctrl[:] = tau
                    mujoco.mj_step(model, settle_data)
                    viewer.sync()
                    target = start + (k+1)*model.opt.timestep
                    rem = target - time.perf_counter()
                    if rem > 0: time.sleep(rem)
        return support, summaries, all_traces, samples
    finally:
        if viewer is not None:
            viewer.close()


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def plot_by_pose(rows, model_label, key, ylabel, title, path):
    pose_names = list(POSES)
    x = np.arange(len(pose_names))
    plt.figure(figsize=(10,5))
    for joint in REP_JOINTS:
        y = []
        for pose in pose_names:
            row = next(r for r in rows if r["model"]==model_label and r["pose"]==pose and r["joint"]==joint)
            y.append(float(row[key]))
        plt.plot(x, y, marker="o", label=joint)
    plt.xticks(x, pose_names, rotation=20)
    plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path, dpi=170); plt.close()


def plot_coupling(rows, model_label, out):
    for pose in POSES:
        M = np.zeros((3,3))
        for i, ni in enumerate(REP_JOINTS):
            for j, nj in enumerate(REP_JOINTS):
                r = next(x for x in rows if x["model"]==model_label and x["pose"]==pose and x["joint_i"]==ni and x["joint_j"]==nj)
                M[i,j] = float(r["normalized_free_coupling"])
        plt.figure(figsize=(6,5)); im = plt.imshow(M, vmin=0, vmax=1); plt.colorbar(im, label="Normalized coupling")
        plt.xticks(range(3), REP_JOINTS); plt.yticks(range(3), REP_JOINTS)
        for i in range(3):
            for j in range(3): plt.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center")
        plt.title(f"{model_label} {pose}: free effective coupling"); plt.tight_layout()
        plt.savefig(out/f"coupling_{model_label.lower()}_{pose.lower()}.png", dpi=170); plt.close()


def plot_pulse(rows, path):
    labels = [f"{r['model']}\n{r['joint']}" for r in rows]
    vals = [float(r["one_step_error_pct"]) for r in rows]
    x = np.arange(len(labels)); plt.figure(figsize=(10,5)); plt.bar(x, vals)
    plt.xticks(x, labels, rotation=25); plt.ylabel("One-step prediction error [%]")
    plt.title("Direct-torque pulse validation"); plt.grid(True, axis="y"); plt.tight_layout(); plt.savefig(path,dpi=170); plt.close()


def plot_stance(rows, path):
    plt.figure(figsize=(10,5))
    for joint in REP_JOINTS:
        sub = [r for r in rows if r["pulse_joint"]==joint]
        plt.plot([r["time_s"] for r in sub], [r["incremental_joint_qd_rad_s"] for r in sub], label=joint)
    plt.axvline(STANCE_PULSE_S, linestyle="--", label="pulse end")
    plt.xlabel("Time [s]"); plt.ylabel("Incremental joint velocity [rad/s]"); plt.title("Contact-constrained perturbations vs no-pulse baseline")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(path,dpi=170); plt.close()


def md_table(headers, rows):
    out = ["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]
    out += ["| "+" | ".join(row)+" |" for row in rows]
    return "\n".join(out)


def write_summary(path, flags, model_rows, passive_rows, dyn, modes, pulse, support, stance, settle, gains, limits):
    total = next(float(r["mass_kg"]) for r in model_rows if r["body"]=="__TOTAL__")
    L = ["# Spot V2 Joint Dynamics Audit — Auto Summary", ""]
    L += ["## Active baseline", "", "```text",
          f"HIP:  Kp={gains['hip_kp']:g}, Kd={gains['hip_kd']:g}, limit={limits['hip']:g} Nm",
          f"KNEE: Kp={gains['knee_kp']:g}, Kd={gains['knee_kd']:g}, limit={limits['knee']:g} Nm",
          f"Physics={sc.PHYSICS_FREQUENCY_HZ} Hz, policy={sc.POLICY_FREQUENCY_HZ} Hz", "```", ""]
    L += ["## Mechanical audit", "", f"Total body mass: **{total:.4f} kg**", ""]
    if flags:
        L += ["Flags:", ""] + [f"- {x}" for x in flags] + [""]
    props = []
    for r in passive_rows:
        if r["joint"] in REP_JOINTS:
            props.append([r["joint"], f"{r['armature_kg_m2']:.6g}", f"{r['damping_nms_rad']:.6g}", f"{r['frictionloss_nm']:.6g}"])
    L += ["### Representative compiled passive joint properties", "",
          md_table(["Joint","Armature","Damping","Friction loss"], props), ""]
    home = []
    for mdl in ("FIXED","FLOATING"):
        for j in REP_JOINTS:
            r = next(x for x in dyn if x["model"]==mdl and x["pose"]=="HOME" and x["joint"]==j)
            home.append([mdl,j,f"{r['J_locked_kg_m2']:.6f}",f"{r['J_free_kg_m2']:.6f}",f"{r['predicted_natural_frequency_hz']:.2f}",f"{r['predicted_damping_ratio']:.3f}"])
    L += ["## HOME inertia and current-PD prediction", "",
          md_table(["Model","Joint","J locked","J free","Pred fn Hz","Pred ζ"], home), ""]
    hy = next(x for x in dyn if x["model"]=="FLOATING" and x["pose"]=="HOME" and x["joint"]=="fl_hy")
    kn = next(x for x in dyn if x["model"]=="FLOATING" and x["pose"]=="HOME" and x["joint"]=="fl_kn")
    ratio = float(hy["J_free_kg_m2"])/float(kn["J_free_kg_m2"])
    L += [f"Floating HOME `J_HY/J_KN = {ratio:.3f}`. Under equal Kp, the simplest inertia-only model predicts a knee/HY natural-frequency ratio of **{math.sqrt(ratio):.3f}×**.", ""]
    mt=[]
    for mdl in ("FIXED","FLOATING"):
        for r in [x for x in modes if x["model"]==mdl and x["pose"]=="HOME"]:
            mt.append([mdl,str(r["mode_index"]),f"{r['natural_frequency_hz']:.2f}",f"{r['damped_frequency_hz']:.2f}",f"{r['damping_ratio']:.3f}"])
    L += ["## Coupled 3-DOF leg modes at HOME", "", md_table(["Model","Mode","Natural Hz","Damped Hz","ζ"],mt), ""]
    pt=[]
    for r in pulse:
        pt.append([r["model"],r["joint"],f"{r['pulse_torque_nm']:.4f}",f"{r['predicted_incremental_qacc_rad_s2']:.3f}",f"{r['mj_forward_incremental_qacc_rad_s2']:.3f}",f"{r['one_step_incremental_qacc_rad_s2']:.3f}",f"{r['one_step_error_pct']:.3f}"])
    L += ["## Direct-torque validation", "", md_table(["Model","Joint","Pulse Nm","Pred Δqdd","Forward Δqdd","1-step Δqdd","1-step err %"],pt), ""]
    if settle:
        L += ["## Standing support", "", f"Mean settled base height over final {STANCE_SAMPLE_S:.2f} s: **{np.mean([r['base_z_m'] for r in settle]):.5f} m**", ""]
    st=[]
    for r in stance:
        st.append([r["pulse_joint"],f"{r['added_tau_nm']:.3f}",f"{r['incremental_joint_qd_at_pulse_end_rad_s']:.6f}",f"{r['peak_abs_incremental_joint_qd_rad_s']:.6f}",f"{r['peak_incremental_base_velocity_norm']:.6f}"])
    L += [md_table(["Perturbed joint","Added torque Nm","Δqdot at pulse end","Peak |Δqdot|","Peak base Δvel norm"],st), ""]
    L += ["## Interpretation limits", "",
          "- This audit only identifies dynamics already present in the current MuJoCo model.",
          "- It cannot infer real Spot rotor inertia, gearbox friction, motor electrical constants, torque-loop bandwidth, or latency if those are not modeled.",
          "- Fixed-base zero-velocity bias force is not the same as whole-body standing support torque; standing support is measured separately under contact.", ""]
    path.write_text("\n".join(L))


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_S)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
    args=parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    fixed_xml,floating_xml,standing_xml=project_paths()
    gains=current_gains(); limits=current_limits()
    print("\n=== Spot V2 Comprehensive Joint-Dynamics Audit ===", flush=True)
    print("Fixed:",fixed_xml); print("Floating:",floating_xml); print("Standing:",standing_xml)
    print("Gains:",gains); print("Limits:",limits)
    fixed=mujoco.MjModel.from_xml_path(str(fixed_xml)); floating=mujoco.MjModel.from_xml_path(str(floating_xml)); standing=mujoco.MjModel.from_xml_path(str(standing_xml))
    dt=1.0/float(sc.PHYSICS_FREQUENCY_HZ)
    for m in (fixed,floating,standing): m.opt.timestep=dt

    auto_poses, pose_rows = auto_select_workspace_poses(fixed)
    POSES.update(auto_poses)
    write_csv(out/"pose_selection.csv", pose_rows)
    print("\n[AUTO] Kinematically selected workspace poses:", flush=True)
    for r in pose_rows:
        print(
            f"  {r['pose']:13s}: HX={r['hx_rad']:+.3f}, HY={r['hy_rad']:+.3f}, "
            f"KN={r['kn_rad']:+.3f} rad | foot/hip x={r['foot_rel_hip_x_m']:+.3f} m, "
            f"z={r['foot_rel_hip_z_m']:+.3f} m",
            flush=True,
        )
    print("  Viewer confirmation still comes before accepting the pose data.", flush=True)

    validate_poses(fixed)

    print("\n[A] Mechanical model audit")
    model_rows,flags=audit_bodies(floating); passive=audit_joint_passive(floating)
    write_csv(out/"model_audit.csv",model_rows); write_csv(out/"joint_physical_properties.csv",passive)
    for x in flags: print("[FLAG]",x)

    if not args.no_viewer:
        print("\n[B] Visual pose confirmation")
        show_poses(fixed,"FIXED BASE",False,args.interactive,args.hold_seconds)
        show_poses(floating,"FLOATING / FREE SPACE",True,args.interactive,args.hold_seconds)

    print("\n[C] Dynamics extraction")
    dyn=[]; coupling=[]; modes=[]
    for label,model,isfloat in (("FIXED",fixed,False),("FLOATING",floating,True)):
        for pose,triplet in POSES.items():
            a,b,c=analyze_pose(model,label,pose,triplet,isfloat,gains,limits)
            dyn+=a; coupling+=b; modes+=c
            print(f"  {label:8s} {pose:14s} complete")
    write_csv(out/"effective_inertia.csv",dyn); write_csv(out/"coupling.csv",coupling); write_csv(out/"modal_analysis.csv",modes)

    print("\n[D] Direct-torque HOME validation")
    pulse=pulse_validation(fixed,"FIXED",False,limits)+pulse_validation(floating,"FLOATING",True,limits)
    write_csv(out/"pulse_validation.csv",pulse)
    for r in pulse:
        print(f"  {r['model']:8s} {r['joint']:5s}: pred={r['predicted_incremental_qacc_rad_s2']:.3f}, step={r['one_step_incremental_qacc_rad_s2']:.3f}, err={r['one_step_error_pct']:.3f}%")

    print("\n[E] Standing/contact-constrained test")
    support,stance,trace,settle=stance_test(standing,gains,limits,not args.no_viewer,args.interactive,args.hold_seconds)
    write_csv(out/"standing_support.csv",support); write_csv(out/"stance_perturbation_summary.csv",stance); write_csv(out/"stance_perturbation_trace.csv",trace)

    print("\n[F] Plots and summary")
    for label in ("FIXED","FLOATING"):
        low=label.lower()
        plot_by_pose(dyn,label,"J_free_kg_m2","Free apparent inertia [kg m²]",f"{label}: apparent inertia vs pose",out/f"effective_inertia_{low}.png")
        plot_by_pose(dyn,label,"predicted_natural_frequency_hz","Predicted natural frequency [Hz]",f"{label}: current-PD natural frequency",out/f"predicted_frequency_{low}.png")
        plot_by_pose(dyn,label,"predicted_damping_ratio","Predicted damping ratio",f"{label}: current-PD damping ratio",out/f"predicted_damping_{low}.png")
        plot_coupling(coupling,label,out)
    plot_pulse(pulse,out/"pulse_validation_error.png"); plot_stance(trace,out/"stance_perturbation_velocity.png")
    write_summary(out/"summary.md",flags,model_rows,passive,dyn,modes,pulse,support,stance,settle,gains,limits)
    print("\n=== AUDIT COMPLETE ===")
    print("Results:",out.resolve()); print("Main report:",(out/"summary.md").resolve())


if __name__ == "__main__":
    main()
