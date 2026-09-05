"""Spot V2: JTC linear-vs-cubic policy-rate controller A/B test.

JTC_LINEAR
    q_des is linearly interpolated over each 20 ms policy interval.
    qdot_des is constant over the interval and therefore discontinuous at
    policy waypoints.

JTC_CUBIC
    q_des follows a cubic Hermite segment over each 20 ms policy interval.
    qdot_des is continuous at policy waypoints. Endpoint velocities are
    estimated causally from policy targets:

        v_k = (q_k - q_{k-1}) / T_policy

Everything else is identical: fixed-base model, Kp/Kd, torque limits, target
sequence, physics rate, and controller rate.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from spot_locomotion_v2.robot import spot_constants as sc


EXPECTED_KP = 500.0
EXPECTED_KD = 10.0
EXPECTED_HIP_LIMIT = 69.52
EXPECTED_KNEE_LIMIT = 164.15

PHYSICS_HZ = 500
POLICY_HZ = 50
DT = 1.0 / PHYSICS_HZ
POLICY_DT = 1.0 / POLICY_HZ

SETTLE_S = 1.0
ACTIVE_S = 1.10
FINAL_HOLD_S = 0.40
TOTAL_S = SETTLE_S + ACTIVE_S + FINAL_HOLD_S

HY_OFFSETS = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.10, 0.05, 0.00, -0.05, -0.10, -0.05, 0.00],
    dtype=float,
)
KN_OFFSETS = np.array(
    [0.00, 0.08, 0.16, 0.24, 0.16, 0.08, 0.00, -0.08, -0.16, -0.08, 0.00],
    dtype=float,
)

OUTPUT_DIR = (
    Path("reports")
    / "actuator_characterization"
    / "policy_rate_linear_vs_cubic_kp500_kd10"
)


def read_project_gains():
    if hasattr(sc, "HIP_KP"):
        return (
            float(sc.HIP_KP),
            float(sc.HIP_KD),
            float(sc.KNEE_KP),
            float(sc.KNEE_KD),
        )
    return float(sc.KP), float(sc.KD), float(sc.KP), float(sc.KD)


def validate_constants():
    hip_kp, hip_kd, knee_kp, knee_kd = read_project_gains()
    hip_lim = float(sc.HIP_TORQUE_LIMIT)
    knee_lim = float(sc.KNEE_TORQUE_LIMIT)

    actual = np.array(
        [hip_kp, hip_kd, knee_kp, knee_kd, hip_lim, knee_lim], dtype=float
    )
    expected = np.array(
        [
            EXPECTED_KP,
            EXPECTED_KD,
            EXPECTED_KP,
            EXPECTED_KD,
            EXPECTED_HIP_LIMIT,
            EXPECTED_KNEE_LIMIT,
        ],
        dtype=float,
    )

    print("Project constants:")
    print(f"  hips : Kp={hip_kp:g}, Kd={hip_kd:g}, limit={hip_lim:g} Nm")
    print(f"  knees: Kp={knee_kp:g}, Kd={knee_kd:g}, limit={knee_lim:g} Nm")

    if not np.allclose(actual, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            "This A/B is intentionally defined around the original V2 baseline.\n"
            f"Expected [hip Kp, hip Kd, knee Kp, knee Kd, hip limit, knee limit]:\n"
            f"  {expected.tolist()}\n"
            f"Actual:\n  {actual.tolist()}\n\n"
            "Restore spot_constants.py to 500/10 for hips and knees with "
            "69.52/164.15 Nm limits before running this test."
        )

    if int(sc.PHYSICS_FREQUENCY_HZ) != PHYSICS_HZ:
        raise RuntimeError("Project physics frequency is not 500 Hz.")
    if int(sc.POLICY_FREQUENCY_HZ) != POLICY_HZ:
        raise RuntimeError("Project policy frequency is not 50 Hz.")

    return hip_kp, hip_kd, knee_kp, knee_kd, hip_lim, knee_lim


def fixed_base_xml():
    path = (
        Path(sc.__file__).resolve().parent
        / "mjcf"
        / "experiments"
        / "spot_fixed_base_explicit_pd.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def actuator_name(model, aid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)


def joint_addresses(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint {name!r} not found")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def home_vector(model):
    q_home = np.zeros(model.nu)
    for aid in range(model.nu):
        name = actuator_name(model, aid)
        q_home[aid] = float(sc.HOME_JOINT_POSITION_MAP[name])
    return q_home


def controller_vectors(model, hip_kp, hip_kd, knee_kp, knee_kd, hip_lim, knee_lim):
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    limit = np.zeros(model.nu)
    for aid in range(model.nu):
        name = actuator_name(model, aid)
        if name.endswith(("_hx", "_hy")):
            kp[aid], kd[aid], limit[aid] = hip_kp, hip_kd, hip_lim
        elif name.endswith("_kn"):
            kp[aid], kd[aid], limit[aid] = knee_kp, knee_kd, knee_lim
        else:
            raise RuntimeError(f"Unknown actuator {name!r}")
    return kp, kd, limit


def joint_state(model, data):
    q = np.zeros(model.nu)
    qd = np.zeros(model.nu)
    for aid in range(model.nu):
        name = actuator_name(model, aid)
        qadr, dadr = joint_addresses(model, name)
        q[aid] = data.qpos[qadr]
        qd[aid] = data.qvel[dadr]
    return q, qd


def policy_target(index, q_home, hy_aid, kn_aid):
    target = q_home.copy()
    i = index % len(HY_OFFSETS)
    target[hy_aid] += HY_OFFSETS[i]
    target[kn_aid] += KN_OFFSETS[i]
    return target


def policy_velocity(index, q_home, hy_aid, kn_aid):
    """Causal waypoint velocity estimate v_k = (q_k - q_{k-1}) / T."""
    q_k = policy_target(index, q_home, hy_aid, kn_aid)
    q_prev = q_home.copy() if index == 0 else policy_target(index - 1, q_home, hy_aid, kn_aid)
    return (q_k - q_prev) / POLICY_DT


def target_state_linear(t, q_home, hy_aid, kn_aid):
    if t < SETTLE_S or t >= SETTLE_S + ACTIVE_S:
        return q_home.copy(), q_home.copy(), np.zeros_like(q_home), -1

    active_t = t - SETTLE_S
    k = int(np.floor(active_t / POLICY_DT))
    target = policy_target(k, q_home, hy_aid, kn_aid)
    previous = q_home.copy() if k == 0 else policy_target(k - 1, q_home, hy_aid, kn_aid)

    interval_t = active_t - k * POLICY_DT
    alpha = np.clip(interval_t / POLICY_DT, 0.0, 1.0)

    q_des = previous + alpha * (target - previous)
    qd_des = (target - previous) / POLICY_DT
    return target, q_des, qd_des, k


def cubic_hermite(q0, v0, q1, v1, local_t, duration):
    """Cubic Hermite segment with continuous position and velocity."""
    u = np.clip(local_t / duration, 0.0, 1.0)

    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    q_des = h00 * q0 + h10 * duration * v0 + h01 * q1 + h11 * duration * v1

    dh00 = 6.0 * u**2 - 6.0 * u
    dh10 = 3.0 * u**2 - 4.0 * u + 1.0
    dh01 = -6.0 * u**2 + 6.0 * u
    dh11 = 3.0 * u**2 - 2.0 * u

    qd_des = (
        dh00 * q0
        + dh10 * duration * v0
        + dh01 * q1
        + dh11 * duration * v1
    ) / duration

    return q_des, qd_des


def target_state_cubic(t, q_home, hy_aid, kn_aid):
    if t < SETTLE_S or t >= SETTLE_S + ACTIVE_S:
        return q_home.copy(), q_home.copy(), np.zeros_like(q_home), -1

    active_t = t - SETTLE_S
    k = int(np.floor(active_t / POLICY_DT))
    target = policy_target(k, q_home, hy_aid, kn_aid)
    local_t = active_t - k * POLICY_DT

    if k == 0:
        q0 = q_home.copy()
        v0 = np.zeros_like(q_home)
    else:
        q0 = policy_target(k - 1, q_home, hy_aid, kn_aid)
        v0 = policy_velocity(k - 1, q_home, hy_aid, kn_aid)

    q1 = target
    v1 = policy_velocity(k, q_home, hy_aid, kn_aid)

    q_des, qd_des = cubic_hermite(q0, v0, q1, v1, local_t, POLICY_DT)
    return target, q_des, qd_des, k


def run_mode(mode, model, q_home, kp, kd, torque_limit):
    data = mujoco.MjData(model)

    for aid in range(model.nu):
        name = actuator_name(model, aid)
        qadr, dadr = joint_addresses(model, name)
        data.qpos[qadr] = q_home[aid]
        data.qvel[dadr] = 0.0
    mujoco.mj_forward(model, data)

    hy_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_hy")
    kn_aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fl_kn")

    rows = []
    nsteps = int(round(TOTAL_S / DT))

    print(f"\n=== {mode} ===")
    print("Viewer will run in real time for about 2.5 s.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()

        for step in range(nsteps):
            if not viewer.is_running():
                break

            t = step * DT
            if mode == "JTC_LINEAR":
                policy_tgt, q_des, qd_des, policy_index = target_state_linear(
                    t, q_home, hy_aid, kn_aid
                )
            elif mode == "JTC_CUBIC":
                policy_tgt, q_des, qd_des, policy_index = target_state_cubic(
                    t, q_home, hy_aid, kn_aid
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            q, qd = joint_state(model, data)
            p_tau = kp * (q_des - q)
            d_tau = kd * (qd_des - qd)
            raw_tau = p_tau + d_tau
            applied_tau = np.clip(raw_tau, -torque_limit, torque_limit)
            data.ctrl[:] = applied_tau

            row = {"time_s": t, "policy_index": policy_index}
            for name, aid in (("fl_hy", hy_aid), ("fl_kn", kn_aid)):
                row[f"{name}_policy_target_rad"] = policy_tgt[aid]
                row[f"{name}_q_des_rad"] = q_des[aid]
                row[f"{name}_qd_des_rad_s"] = qd_des[aid]
                row[f"{name}_q_rad"] = q[aid]
                row[f"{name}_qd_rad_s"] = qd[aid]
                row[f"{name}_p_tau_nm"] = p_tau[aid]
                row[f"{name}_d_tau_nm"] = d_tau[aid]
                row[f"{name}_tau_raw_nm"] = raw_tau[aid]
                row[f"{name}_tau_applied_nm"] = applied_tau[aid]
                row[f"{name}_saturated"] = float(
                    abs(raw_tau[aid]) >= torque_limit[aid] - 1e-12
                )
            rows.append(row)

            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_s = wall_start + (step + 1) * DT - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def arrays(rows):
    return {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in rows[0]
    }


def summarize(label, rows):
    d = arrays(rows)
    active = (d["time_s"] >= SETTLE_S) & (d["time_s"] < SETTLE_S + ACTIVE_S)
    print(f"\nSummary: {label}")
    for name in ("fl_hy", "fl_kn"):
        print(f"  {name}")
        print(f"    peak |qdot|       = {np.max(np.abs(d[f'{name}_qd_rad_s'][active])):.3f} rad/s")
        print(f"    peak |qdot_des|   = {np.max(np.abs(d[f'{name}_qd_des_rad_s'][active])):.3f} rad/s")
        print(f"    peak |P torque|   = {np.max(np.abs(d[f'{name}_p_tau_nm'][active])):.3f} Nm")
        print(f"    peak |D torque|   = {np.max(np.abs(d[f'{name}_d_tau_nm'][active])):.3f} Nm")
        print(f"    peak |raw torque| = {np.max(np.abs(d[f'{name}_tau_raw_nm'][active])):.3f} Nm")
        print(f"    saturation        = {100*np.mean(d[f'{name}_saturated'][active]):.2f}%")


def make_plots(linear_rows, cubic_rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    linear = arrays(linear_rows)
    cubic = arrays(cubic_rows)

    for name in ("fl_hy", "fl_kn"):
        plt.figure(figsize=(11, 5))
        plt.plot(linear["time_s"], linear[f"{name}_policy_target_rad"], "--", label="50 Hz policy target")
        plt.plot(linear["time_s"], linear[f"{name}_q_des_rad"], label="JTC_LINEAR q_des")
        plt.plot(linear["time_s"], linear[f"{name}_q_rad"], label="JTC_LINEAR actual q")
        plt.plot(cubic["time_s"], cubic[f"{name}_q_des_rad"], label="JTC_CUBIC q_des")
        plt.plot(cubic["time_s"], cubic[f"{name}_q_rad"], label="JTC_CUBIC actual q")
        plt.xlabel("Time [s]")
        plt.ylabel("Position [rad]")
        plt.title(f"{name}: linear vs cubic position")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{name}_position_linear_vs_cubic.png", dpi=160)
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(linear["time_s"], linear[f"{name}_qd_des_rad_s"], "--", label="JTC_LINEAR qdot_des")
        plt.plot(linear["time_s"], linear[f"{name}_qd_rad_s"], label="JTC_LINEAR actual qdot")
        plt.plot(cubic["time_s"], cubic[f"{name}_qd_des_rad_s"], "--", label="JTC_CUBIC qdot_des")
        plt.plot(cubic["time_s"], cubic[f"{name}_qd_rad_s"], label="JTC_CUBIC actual qdot")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [rad/s]")
        plt.title(f"{name}: linear vs cubic velocity")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{name}_velocity_linear_vs_cubic.png", dpi=160)
        plt.close()

        plt.figure(figsize=(11, 5))
        plt.plot(linear["time_s"], linear[f"{name}_tau_applied_nm"], label="JTC_LINEAR applied torque")
        plt.plot(cubic["time_s"], cubic[f"{name}_tau_applied_nm"], label="JTC_CUBIC applied torque")
        plt.plot(cubic["time_s"], cubic[f"{name}_p_tau_nm"], "--", label="JTC_CUBIC P")
        plt.plot(cubic["time_s"], cubic[f"{name}_d_tau_nm"], ":", label="JTC_CUBIC D")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.title(f"{name}: linear vs cubic torque")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{name}_torque_linear_vs_cubic.png", dpi=160)
        plt.close()


def main():
    hip_kp, hip_kd, knee_kp, knee_kd, hip_lim, knee_lim = validate_constants()

    xml = fixed_base_xml()
    print(f"Fixed-base model: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = DT

    if model.nu != 12:
        raise RuntimeError(f"Expected 12 motors, got {model.nu}")

    q_home = home_vector(model)
    kp, kd, torque_limit = controller_vectors(
        model, hip_kp, hip_kd, knee_kp, knee_kd, hip_lim, knee_lim
    )

    print("\nTwo viewer runs will execute in sequence:")
    print("  1. JTC_LINEAR")
    print("  2. JTC_CUBIC")

    linear_rows = run_mode("JTC_LINEAR", model, q_home, kp, kd, torque_limit)
    cubic_rows = run_mode("JTC_CUBIC", model, q_home, kp, kd, torque_limit)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT_DIR / "jtc_linear.csv", linear_rows)
    save_csv(OUTPUT_DIR / "jtc_cubic.csv", cubic_rows)
    summarize("JTC_LINEAR", linear_rows)
    summarize("JTC_CUBIC", cubic_rows)
    make_plots(linear_rows, cubic_rows)

    print("\nPASS: linear-vs-cubic A/B completed")
    print(f"Results: {OUTPUT_DIR}")
    print("Please send me the two CSVs and six PNGs for interpretation.")


if __name__ == "__main__":
    main()
