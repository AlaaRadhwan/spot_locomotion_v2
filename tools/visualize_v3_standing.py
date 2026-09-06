#!/usr/bin/env python3
"""Visual Test 2: free-standing Spot V3 with MjLab BuiltinPosition actuator.

The robot is commanded to the unchanged V2 home pose for 10 simulated seconds.
The MuJoCo passive viewer runs close to real time so you can directly watch:
- settling / sag
- oscillation
- foot contacts
- roll / pitch drift
- any collapse

A compact CSV log is saved to reports/v3_actuator/.
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
    print_model_actuator_summary,
    reset_to_init_state,
    roll_pitch_deg,
    save_csv,
)


DURATION_S = 10.0
CSV_PATH = REPORTS_DIR / "standing_builtin_position_kp60_kd1p5.csv"


def main() -> None:
    model = build_v3_model_with_ground()
    data = mujoco.MjData(model)

    qpos_addresses, dof_addresses, actuator_ids = get_joint_arrays(model)

    reset_to_init_state(model, data)

    desired = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)

    print_model_actuator_summary(model)

    print("\n=== Visual standing test ===")
    print(f"Duration: {DURATION_S:.1f} s")
    print(f"Initial base height: {data.qpos[2]:.4f} m")
    print("Command: hold the unchanged home pose.")
    print("Close the viewer at any time to stop early.")
    print("\nWatch for:")
    print("  - how far the body settles")
    print("  - whether the feet remain planted")
    print("  - persistent bouncing/oscillation")
    print("  - roll/pitch drift or collapse")

    rows: list[dict[str, float | int | str]] = []

    min_height = float(data.qpos[2])
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    peak_abs_torque = np.zeros(len(dof_addresses), dtype=np.float64)

    next_console_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < DURATION_S:
            wall_start = time.monotonic()

            # Native MuJoCo position actuator target.  We deliberately update
            # at every physics step here; the target itself remains constant.
            data.ctrl[actuator_ids] = desired

            mujoco.mj_step(model, data)

            torques = actual_joint_torques(data, dof_addresses)
            peak_abs_torque = np.maximum(peak_abs_torque, np.abs(torques))

            roll_deg, pitch_deg = roll_pitch_deg(data.qpos[3:7])
            feet = count_feet_on_ground(model, data)

            current_q = data.qpos[qpos_addresses]
            max_joint_error = float(np.max(np.abs(desired - current_q)))

            min_height = min(min_height, float(data.qpos[2]))
            max_abs_roll = max(max_abs_roll, abs(roll_deg))
            max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))

            rows.append(
                {
                    "time_s": float(data.time),
                    "base_height_m": float(data.qpos[2]),
                    "roll_deg": roll_deg,
                    "pitch_deg": pitch_deg,
                    "feet_on_ground": feet,
                    "max_joint_error_rad": max_joint_error,
                    "max_abs_joint_torque_nm": float(np.max(np.abs(torques))),
                }
            )

            if data.time >= next_console_print:
                print(
                    f"t={data.time:5.2f}s | "
                    f"z={data.qpos[2]:.4f}m | "
                    f"roll={roll_deg:+6.2f}deg | "
                    f"pitch={pitch_deg:+6.2f}deg | "
                    f"feet={feet} | "
                    f"max|tau|={np.max(np.abs(torques)):6.2f}Nm"
                )
                next_console_print += 0.5

            viewer.sync()

            elapsed = time.monotonic() - wall_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    save_csv(CSV_PATH, rows)

    final_roll_deg, final_pitch_deg = roll_pitch_deg(data.qpos[3:7])
    final_q = data.qpos[qpos_addresses]
    final_error = desired - final_q

    print("\n=== Standing result ===")
    print(f"Executed time:          {data.time:.3f} s")
    print(f"Final base height:      {data.qpos[2]:.6f} m")
    print(f"Minimum base height:    {min_height:.6f} m")
    print(f"Final roll:             {final_roll_deg:+.3f} deg")
    print(f"Final pitch:            {final_pitch_deg:+.3f} deg")
    print(f"Maximum |roll|:         {max_abs_roll:.3f} deg")
    print(f"Maximum |pitch|:        {max_abs_pitch:.3f} deg")
    print(f"Final feet on ground:   {count_feet_on_ground(model, data)}")
    print(f"Max final joint error:  {np.max(np.abs(final_error)):.6f} rad")
    print(f"Peak joint torque:      {np.max(peak_abs_torque):.3f} N*m")
    print(f"CSV: {CSV_PATH}")

    print("\nFinal per-joint error [rad]:")
    for name, error in zip(
        (
            "fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",
        ),
        final_error,
        strict=True,
    ):
        print(f"  {name:6s}: {error:+.6f}")


if __name__ == "__main__":
    main()
