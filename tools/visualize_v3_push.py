#!/usr/bin/env python3
"""Visual Test 3: 250 N downward torso push on free-standing Spot V3.

Protocol:
  0.00 - 2.00 s : settle at home reference
  2.00 - 2.25 s : apply 250 N downward force to torso
  2.25 - 6.25 s : recover with the same home reference

The force is applied through MuJoCo data.xfrc_applied to the root body.
The viewer runs approximately in real time.

A CSV log is saved to reports/v3_actuator/.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from spot_locomotion_v2.robot.spot_v3_constants import (
    HOME_JOINT_POSITIONS,
    build_v3_model_with_ground,
)

from v3_visual_test_common import (
    REPORTS_DIR,
    actual_joint_torques,
    count_feet_on_ground,
    get_joint_arrays,
    object_id,
    print_model_actuator_summary,
    reset_to_init_state,
    roll_pitch_deg,
    save_csv,
)


SETTLE_S = 2.0
PUSH_DURATION_S = 0.25
RECOVERY_S = 4.0

PUSH_START_S = SETTLE_S
PUSH_END_S = SETTLE_S + PUSH_DURATION_S
END_S = SETTLE_S + PUSH_DURATION_S + RECOVERY_S

PUSH_FORCE_N = 250.0

CSV_PATH = REPORTS_DIR / "push250_builtin_position_kp60_kd1p5.csv"


def main() -> None:
    model = build_v3_model_with_ground()
    data = mujoco.MjData(model)

    qpos_addresses, dof_addresses, actuator_ids = get_joint_arrays(model)
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "body")

    reset_to_init_state(model, data)

    desired = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)

    print_model_actuator_summary(model)

    print("\n=== Visual 250 N push test ===")
    print(f"Settle:    0.00 -> {PUSH_START_S:.2f} s")
    print(
        f"PUSH:      {PUSH_START_S:.2f} -> {PUSH_END_S:.2f} s "
        f"({PUSH_FORCE_N:.0f} N downward on torso)"
    )
    print(f"Recovery:  {PUSH_END_S:.2f} -> {END_S:.2f} s")
    print("Close the viewer at any time to stop early.")

    rows: list[dict[str, float | int | str]] = []

    pre_push_height: float | None = None
    min_height_during_push = float("inf")
    min_height_after_push = float("inf")
    peak_downward_velocity = 0.0
    peak_upward_velocity = 0.0
    peak_abs_torque = np.zeros(len(dof_addresses), dtype=np.float64)

    last_phase: str | None = None
    next_console_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < END_S:
            wall_start = time.monotonic()

            data.ctrl[actuator_ids] = desired

            # Always clear external wrench first.
            data.xfrc_applied[body_id, :] = 0.0

            if data.time < PUSH_START_S:
                phase = "SETTLE"
            elif data.time < PUSH_END_S:
                phase = "PUSH"
                # Force components are x, y, z in world coordinates.
                data.xfrc_applied[body_id, 2] = -PUSH_FORCE_N
            else:
                phase = "RECOVER"

            if phase != last_phase:
                if phase == "PUSH":
                    pre_push_height = float(data.qpos[2])
                    print(
                        f"\n>>> PUSH ACTIVE: {PUSH_FORCE_N:.0f} N downward "
                        f"at t={data.time:.3f}s <<<"
                    )
                elif phase == "RECOVER":
                    print(
                        f"\n>>> PUSH RELEASED at t={data.time:.3f}s; "
                        "watching recovery <<<"
                    )
                else:
                    print("\n>>> SETTLING <<<")
                last_phase = phase

            mujoco.mj_step(model, data)

            torques = actual_joint_torques(data, dof_addresses)
            peak_abs_torque = np.maximum(peak_abs_torque, np.abs(torques))

            base_height = float(data.qpos[2])
            base_vz = float(data.qvel[2])
            roll_deg, pitch_deg = roll_pitch_deg(data.qpos[3:7])
            feet = count_feet_on_ground(model, data)

            if phase == "PUSH":
                min_height_during_push = min(
                    min_height_during_push,
                    base_height,
                )
            elif phase == "RECOVER":
                min_height_after_push = min(
                    min_height_after_push,
                    base_height,
                )

            peak_downward_velocity = min(peak_downward_velocity, base_vz)
            peak_upward_velocity = max(peak_upward_velocity, base_vz)

            current_q = data.qpos[qpos_addresses]
            max_joint_error = float(np.max(np.abs(desired - current_q)))

            rows.append(
                {
                    "time_s": float(data.time),
                    "phase": phase,
                    "base_height_m": base_height,
                    "base_vz_mps": base_vz,
                    "roll_deg": roll_deg,
                    "pitch_deg": pitch_deg,
                    "feet_on_ground": feet,
                    "max_joint_error_rad": max_joint_error,
                    "max_abs_joint_torque_nm": float(np.max(np.abs(torques))),
                    "external_force_z_n": (
                        -PUSH_FORCE_N if phase == "PUSH" else 0.0
                    ),
                }
            )

            if data.time >= next_console_print:
                print(
                    f"t={data.time:5.2f}s | "
                    f"{phase:7s} | "
                    f"z={base_height:.4f}m | "
                    f"vz={base_vz:+.3f}m/s | "
                    f"feet={feet} | "
                    f"max|tau|={np.max(np.abs(torques)):6.2f}Nm"
                )
                next_console_print += 0.25

            viewer.sync()

            elapsed = time.monotonic() - wall_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    save_csv(CSV_PATH, rows)

    final_roll_deg, final_pitch_deg = roll_pitch_deg(data.qpos[3:7])

    print("\n=== Push-test result ===")
    print(f"Executed time:            {data.time:.3f} s")

    if pre_push_height is not None:
        print(f"Height before push:       {pre_push_height:.6f} m")

    if min_height_during_push != float("inf"):
        print(f"Minimum height in push:   {min_height_during_push:.6f} m")
        if pre_push_height is not None:
            compression_mm = 1000.0 * (
                pre_push_height - min_height_during_push
            )
            print(f"Push compression:         {compression_mm:.2f} mm")

    if min_height_after_push != float("inf"):
        print(f"Min height after release: {min_height_after_push:.6f} m")

    print(f"Final base height:        {data.qpos[2]:.6f} m")
    print(f"Peak downward vz:         {peak_downward_velocity:+.4f} m/s")
    print(f"Peak upward vz:           {peak_upward_velocity:+.4f} m/s")
    print(f"Final roll:               {final_roll_deg:+.3f} deg")
    print(f"Final pitch:              {final_pitch_deg:+.3f} deg")
    print(f"Final feet on ground:     {count_feet_on_ground(model, data)}")
    print(f"Peak joint torque:        {np.max(peak_abs_torque):.3f} N*m")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
