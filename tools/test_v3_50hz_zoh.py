#!/usr/bin/env python3
"""Visualize the Spot V3 50 Hz ZOH actuator test in real time.

Independent visual version:
- does NOT import test_v3_50hz_zoh.py
- uses the same fixed-base V3 MjLab model
- uses current Kp/Kd from spot_v3_constants.py
- 500 Hz physics
- 50 Hz zero-order-held joint references
- 2 Hz sampled reference
- FL HY +/-0.10 rad
- FL knee +/-0.16 rad
"""

from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from spot_locomotion_v2.robot.spot_v3_constants import (
    HOME_JOINT_POSITIONS,
    JOINT_NAMES,
    POLICY_TIMESTEP,
    V3_KD,
    V3_KP,
)

from test_v3_fixed_base_step import (
    build_fixed_base_model,
    get_addresses,
    reset_home,
)


# Same parameters as the validated non-visual 50 Hz ZOH test.
REFERENCE_FREQUENCY_HZ = 2.0
HY_AMPLITUDE_RAD = 0.10
KNEE_AMPLITUDE_RAD = 0.16

HY_NAME = "fl_hy"
KNEE_NAME = "fl_kn"

SETTLE_TIME_S = 1.0
ACTIVE_DURATION_S = 8.0
TOTAL_DURATION_S = SETTLE_TIME_S + ACTIVE_DURATION_S


def main() -> None:
    model = build_fixed_base_model()
    data = mujoco.MjData(model)

    qpos_addresses, dof_addresses, actuator_ids = get_addresses(model)
    reset_home(model, data, qpos_addresses, actuator_ids)

    home = np.asarray(HOME_JOINT_POSITIONS, dtype=np.float64)

    hy_index = JOINT_NAMES.index(HY_NAME)
    knee_index = JOINT_NAMES.index(KNEE_NAME)

    physics_dt = float(model.opt.timestep)
    policy_dt = float(POLICY_TIMESTEP)

    ratio = policy_dt / physics_dt
    decimation = int(round(ratio))

    if not math.isclose(ratio, decimation, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Policy dt ({policy_dt}) is not an integer multiple of "
            f"physics dt ({physics_dt})."
        )

    print("=== Visual Spot V3 50 Hz ZOH test ===")
    print(f"Kp / Kd:             {V3_KP:g} / {V3_KD:g}")
    print(f"Integrator enum:     {int(model.opt.integrator)}")
    print(
        f"Physics:             {1.0 / physics_dt:.1f} Hz "
        f"(dt={physics_dt:.6f}s)"
    )
    print(
        f"Policy/ZOH updates:  {1.0 / policy_dt:.1f} Hz "
        f"(every {decimation} physics steps)"
    )
    print(f"Reference frequency: {REFERENCE_FREQUENCY_HZ:.2f} Hz")
    print(f"FL HY amplitude:      +/-{HY_AMPLITUDE_RAD:.3f} rad")
    print(f"FL knee amplitude:    +/-{KNEE_AMPLITUDE_RAD:.3f} rad")
    print()
    print(f"0 -> {SETTLE_TIME_S:.1f}s: hold home pose")
    print(
        f"{SETTLE_TIME_S:.1f} -> {TOTAL_DURATION_S:.1f}s: "
        "50 Hz ZOH motion"
    )
    print()
    print("Watch the front-left leg:")
    print("  - smooth motion despite staircase targets")
    print("  - no snapping every 20 ms")
    print("  - no growing oscillation")
    print("  - no significant movement in the other three legs")
    print()
    print("Close the viewer to stop early.")

    current_target = home.copy()
    last_policy_index = -1
    next_console_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < TOTAL_DURATION_S:
            wall_start = time.monotonic()

            t = float(data.time)

            if t < SETTLE_TIME_S:
                current_target[:] = home
            else:
                active_t = t - SETTLE_TIME_S

                policy_index = int(
                    math.floor((active_t + 1e-12) / policy_dt)
                )

                if policy_index != last_policy_index:
                    policy_sample_time = policy_index * policy_dt

                    phase = (
                        2.0
                        * math.pi
                        * REFERENCE_FREQUENCY_HZ
                        * policy_sample_time
                    )

                    current_target[:] = home

                    current_target[hy_index] = (
                        home[hy_index]
                        + HY_AMPLITUDE_RAD * math.sin(phase)
                    )

                    current_target[knee_index] = (
                        home[knee_index]
                        - KNEE_AMPLITUDE_RAD * math.sin(phase)
                    )

                    last_policy_index = policy_index

            # Native position target is held unchanged between 50 Hz ticks.
            data.ctrl[actuator_ids] = current_target

            mujoco.mj_step(model, data)

            if data.time >= next_console_print:
                hy_q = float(data.qpos[qpos_addresses[hy_index]])
                kn_q = float(data.qpos[qpos_addresses[knee_index]])

                hy_qd = float(data.qvel[dof_addresses[hy_index]])
                kn_qd = float(data.qvel[dof_addresses[knee_index]])

                hy_tau = float(
                    data.qfrc_actuator[dof_addresses[hy_index]]
                )
                kn_tau = float(
                    data.qfrc_actuator[dof_addresses[knee_index]]
                )

                print(
                    f"t={data.time:5.2f}s | "
                    f"HY target/q={current_target[hy_index]:+.3f}/{hy_q:+.3f} | "
                    f"KN target/q={current_target[knee_index]:+.3f}/{kn_q:+.3f} | "
                    f"qd={hy_qd:+.2f}/{kn_qd:+.2f} rad/s | "
                    f"tau={hy_tau:+.1f}/{kn_tau:+.1f} Nm"
                )

                next_console_print += 0.5

            viewer.sync()

            elapsed = time.monotonic() - wall_start
            sleep_time = physics_dt - elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)

    print("\nVisual ZOH test finished.")


if __name__ == "__main__":
    main()
