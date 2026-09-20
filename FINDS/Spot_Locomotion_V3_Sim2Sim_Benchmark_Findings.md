# Spot Locomotion V3 — Sim-to-Sim Benchmark Findings

**Date:** 2026-09-10  
**Project:** Spot Locomotion V3  
**Scope:** MuJoCo → ROS 2 / Gazebo sim-to-sim deployment validation

---

## 1. Purpose of this document

This document records the findings obtained after the first successful Spot V3 ROS 2 / Gazebo deployment and the subsequent automated benchmarking work.

The previous engineering update had already established that:

- the V3 rigid-body model is stable and internally consistent,
- the V3 PPO policy walks well in MuJoCo,
- `model_550.pt` is the approved deployment checkpoint,
- the TorchScript export matches the loaded PPO actor,
- the ROS 2 observation pipeline is correct,
- Gazebo is stable with the approved low-level controller when physics and control run at 2000 Hz,
- the live ROS 2 policy controller walks successfully in Gazebo,
- PS5 teleoperation through `/cmd_vel` works cleanly for `vx`, `vy`, and `wz`.

The remaining question was:

> How closely does the deployed Gazebo system reproduce the behavior of the same policy in MuJoCo?

To answer that, equivalent automated benchmarks were created for Gazebo and MuJoCo.

---

# 2. Frozen V3 baseline

## MuJoCo / training

```text
Task:
Mjlab-Velocity-Flat-Spot-V3

Checkpoint:
model_550.pt

Run:
2026-09-07_15-57-38_v3_baseline_kp300_kd40_tl45_115

Robot mass:
31.5999994 kg

Physics:
500 Hz

Policy:
50 Hz

Actuator:
MjLab BuiltinPositionActuator

Kp:
300

Kd:
40

Hip effort limit:
±45 Nm

Knee effort limit:
±115 Nm
```

## ROS 2 / Gazebo deployment

```text
ROS:
Jazzy

Gazebo:
Harmonic

Hardware interface:
effort

Low-level controller:
pid_controller/PidController

Gazebo physics:
2000 Hz

ros2_control:
2000 Hz

Policy:
50 Hz

Kp:
300

Kd:
40

Hip effort limit:
±45 Nm

Knee effort limit:
±115 Nm
```

The deployed policy contract is:

```text
48 observations
      ↓
TorchScript PPO
      ↓
12 actions
      ↓
q_ref = q_home + action_scale × action
```

Action scales:

```text
HX = 0.30 rad
HY = 0.40 rad
KN = 0.50 rad
```

---

# 3. Automated Gazebo benchmark

A dedicated ROS 2 package was created:

```text
spot_eval
```

Its evaluator publishes a deterministic `/cmd_vel` sequence and records:

```text
commanded vx, vy, wz
actual vx, vy, vz
actual angular velocity
base position
roll / pitch / yaw
12 joint positions
12 joint velocities
12 joint efforts
```

The fixed test sequence is:

```text
5 s   stand

5 s   vx = +0.5
2 s   zero

5 s   vx = -0.5
2 s   zero

5 s   vy = +0.3
2 s   zero

5 s   vy = -0.3
2 s   zero

5 s   wz = +0.5
2 s   zero

5 s   wz = -0.5
2 s   zero

7 s   vx = +0.4
      vy = +0.2
      wz = +0.3

5 s   stand
```

The evaluator logs data at 50 Hz and calculates steady-state metrics after excluding the first 1 second of each phase.

Metrics include:

```text
mean actual velocity
MAE
RMSE
body height
roll / pitch RMS
maximum roll / pitch
maximum joint velocity
maximum effort
effort saturation percentage
```

---

# 4. Automated MuJoCo benchmark

A matching MuJoCo benchmark was then created:

```text
tools/benchmark_spot_v3_mujoco.py
```

It runs the same:

```text
59 seconds
2950 policy steps
50 Hz policy loop
```

using the same:

```text
model_550.pt
```

The MuJoCo benchmark:

- disables random command resampling,
- disables randomized base reset for this deterministic comparison,
- uses the exact V3 nominal starting pose,
- uses body-frame root velocity directly from MjLab,
- records actuator forces directly from the MuJoCo plant,
- prevents silent auto-reset during a fall,
- checks the exact 12-joint and 12-actuator ordering.

The benchmark completed successfully:

```text
samples recorded: 2950
duration:         59.0 s
terminated early: false
```

No benchmark phase caused a fall or termination.

---

# 5. Direct MuJoCo vs Gazebo tracking comparison

The central comparison is:

| Phase | Command | MuJoCo | Gazebo | Gazebo - MuJoCo |
|---|---:|---:|---:|---:|
| Forward `vx` | +0.50 | +0.472 | +0.488 | +0.016 m/s |
| Backward `vx` | -0.50 | -0.476 | -0.507 | -0.031 m/s |
| Left `vy` | +0.30 | +0.248 | +0.278 | +0.031 m/s |
| Right `vy` | -0.30 | -0.266 | -0.295 | -0.029 m/s |
| Yaw left `wz` | +0.50 | +0.294 | +0.295 | +0.001 rad/s |
| Yaw right `wz` | -0.50 | -0.288 | -0.325 | -0.037 rad/s |
| Combined `vx` | +0.40 | +0.369 | +0.384 | +0.015 m/s |
| Combined `vy` | +0.20 | +0.180 | +0.192 | +0.012 m/s |
| Combined `wz` | +0.30 | +0.303 | +0.349 | +0.046 rad/s |

The overall result is strong.

Gazebo reproduces the same qualitative locomotion behavior as MuJoCo across:

```text
forward
backward
left lateral
right lateral
pure yaw
combined translation + yaw
standing recovery
```

There is no evidence of a major sim-to-sim policy failure.

---

# 6. Major discovery: pure-yaw weakness belongs to the policy

The Gazebo test initially raised a concern because the robot did not reach the requested pure yaw rate.

For:

```text
wz command = +0.50 rad/s
```

Gazebo achieved approximately:

```text
+0.295 rad/s
```

The equivalent MuJoCo benchmark produced:

```text
+0.294 rad/s
```

The difference is only:

```text
~0.001 rad/s
```

This is extremely small.

The same behavior therefore exists in the training simulator itself.

The causal chain is:

```text
pure yaw command = +0.50
          ↓
same PPO policy
          ↓
MuJoCo ≈ +0.294
Gazebo ≈ +0.295
          ↓
pure-yaw undertracking is not a Gazebo transfer error
```

For negative yaw:

```text
command:
-0.50 rad/s

MuJoCo:
-0.288 rad/s

Gazebo:
-0.325 rad/s
```

The agreement is not as exact as positive yaw, but both systems still show the same fundamental behavior: pure ±0.5 rad/s yaw commands are undertracked.

## Conclusion

The pure-yaw limitation should **not** be addressed by changing:

```text
Gazebo physics rate
ros2_control rate
Kp
Kd
torque limits
robot inertials
ROS observation code
policy export
```

If stronger turn-in-place behavior is desired later, it should be treated as a PPO training / command-distribution problem.

---

# 7. Combined translation + yaw behaves better than pure yaw

An interesting result appears in the combined command:

```text
vx = +0.4
vy = +0.2
wz = +0.3
```

MuJoCo:

```text
vx = +0.369
vy = +0.180
wz = +0.303
```

Gazebo:

```text
vx = +0.384
vy = +0.192
wz = +0.349
```

MuJoCo follows the requested `wz = 0.3` almost exactly.

This supports the earlier suspicion that the policy is more comfortable turning while translating than performing aggressive rotation in place.

The V3 training command distribution emphasizes longitudinal, diagonal, and pure-lateral locomotion and does not explicitly create a dedicated pure-yaw-only training mode.

Therefore the weak turn-in-place response is consistent with the training distribution.

---

# 8. Linear velocity transfer is excellent

Forward and backward transfer are very close.

## Forward

```text
command:
+0.500 m/s

MuJoCo:
+0.472 m/s

Gazebo:
+0.488 m/s
```

## Backward

```text
command:
-0.500 m/s

MuJoCo:
-0.476 m/s

Gazebo:
-0.507 m/s
```

Gazebo is actually slightly closer to the requested command in both longitudinal tests.

Lateral tracking also remains close between the simulators.

## Left

```text
command:
+0.300 m/s

MuJoCo:
+0.248 m/s

Gazebo:
+0.278 m/s
```

## Right

```text
command:
-0.300 m/s

MuJoCo:
-0.266 m/s

Gazebo:
-0.295 m/s
```

These differences are small enough that there is no current justification for changing the policy or plant.

---

# 9. Body-height difference between simulators

A consistent difference was observed in body height.

Approximate steady-state means:

| Phase | MuJoCo | Gazebo | Difference |
|---|---:|---:|---:|
| Standing | 0.454 m | 0.466 m | +12 mm |
| Forward | 0.432 m | 0.447 m | +15 mm |
| Backward | 0.430 m | 0.442 m | +12 mm |
| Left | 0.453 m | 0.467 m | +14 mm |
| Right | 0.457 m | 0.472 m | +15 mm |
| Yaw left | 0.451 m | 0.464 m | +13 mm |
| Yaw right | 0.444 m | 0.459 m | +15 mm |
| Combined | 0.439 m | 0.454 m | +15 mm |

The offset is approximately:

```text
12–15 mm
```

through most of the test.

This difference is currently **not considered a fault**.

Possible explanations include:

```text
different base/root frame origin definitions
contact geometry differences
contact solver differences
implicit MuJoCo actuator vs external Gazebo effort PD
small equilibrium posture differences
```

The offset is quite consistent, which suggests it may partly be a frame/reference issue rather than a major dynamics error.

This should only be investigated further if it becomes relevant to a later hardware or state-estimation requirement.

---

# 10. Gazebo requires higher instantaneous actuator effort

A larger difference appears in actuator effort.

Representative maximum steady-state effort:

| Phase | MuJoCo | Gazebo |
|---|---:|---:|
| Forward | 60.5 Nm | 80.7 Nm |
| Backward | 62.4 Nm | 107.0 Nm |
| Left | 51.5 Nm | 81.9 Nm |
| Right | 51.3 Nm | 72.9 Nm |
| Yaw left | 46.7 Nm | 58.4 Nm |
| Yaw right | 45.6 Nm | 57.9 Nm |
| Combined | 68.5 Nm | 97.2 Nm |

Gazebo generally needs larger instantaneous effort than MuJoCo.

This is not surprising because the two low-level implementations are numerically different.

## MuJoCo

```text
500 Hz physics
BuiltinPositionActuator
implicitfast integration
```

## Gazebo

```text
2000 Hz physics
2000 Hz ros2_control
external effort PID
Kp = 300
Kd = 40
```

The nominal control relationship is similar, but the numerical implementation is not identical.

The important result is that neither system shows sustained actuator saturation during the benchmark.

The Gazebo benchmark showed zero steady-state samples above the chosen:

```text
98% effort-limit threshold
```

and only a very small number of brief saturation-level samples occurred during command transitions.

Therefore the current effort limits remain acceptable:

```text
hips:
±45 Nm

knees:
±115 Nm
```

---

# 11. Body attitude remains stable

Both simulators remain upright throughout the full sequence.

No test phase caused:

```text
collapse
violent roll/pitch growth
limit-cycle behavior
persistent oscillation
failure to recover at zero command
```

The Gazebo robot shows somewhat larger attitude motion in some cases, especially during yaw-right, but the excursions remain small.

The largest notable Gazebo steady-state roll excursion observed during the benchmark was approximately:

```text
2.9 degrees
```

This is not considered problematic.

After each test command, the inserted zero-command pause allows the robot to return close to stationary behavior.

---

# 12. The original 500 Hz Gazebo instability is fully separated from the PPO behavior

An important previous discovery remains confirmed by the benchmark.

At:

```text
Gazebo physics = 500 Hz
ros2_control   = 500 Hz
Kp = 300
Kd = 40
```

the robot was unstable even under a fixed home-position reference.

The behavior included:

```text
rapid joint-velocity sign changes
rail-to-rail effort chatter
repeated torque-limit contact
eventual fall
```

This occurred without PPO.

Changing:

```text
500 Hz → 2000 Hz
```

for both Gazebo physics and low-level control solved the problem while keeping:

```text
Kp = 300
Kd = 40
±45 / ±115 Nm limits
```

unchanged.

The successful PPO benchmark therefore confirms that the resulting 2000 Hz Gazebo controller is not merely stable while standing; it also works during full locomotion.

---

# 13. Sim-to-sim milestone assessment

Based on the automated comparison, the V3 transfer can now be classified as:

```text
V3 rigid-body transfer                 PASS
TorchScript policy export              PASS
observation contract                   PASS
action mapping                         PASS
50 Hz policy execution                 PASS
forward locomotion transfer            PASS
backward locomotion transfer           PASS
lateral locomotion transfer            PASS
pure-yaw behavioral transfer           PASS
combined-command transfer              PASS
standing recovery                      PASS
effort-limit behavior                  PASS
full 59 s uninterrupted benchmark      PASS
```

Perfect numerical equality is not expected because MuJoCo and Gazebo use different contact and low-level-control implementations.

The important engineering question is whether the deployed robot reproduces the useful behavior of the training simulator.

It does.

---

# 14. What is now frozen

The following baseline should remain unchanged unless new evidence specifically points to a problem:

```text
V3 31.5999994 kg rigid-body model

model_550.pt

48-observation policy contract

12-action joint ordering

home pose

action scaling:
HX 0.30
HY 0.40
KN 0.50

MuJoCo:
500 Hz physics
50 Hz policy

Gazebo:
2000 Hz physics
2000 Hz ros2_control
50 Hz policy

Kp = 300
Kd = 40

hip effort limit = ±45 Nm
knee effort limit = ±115 Nm
```

There is currently no justification for reopening:

```text
V3 inertias
low-level gain tuning
torque-limit tuning
reward tuning
air-time tuning
actuator redesign
target interpolation
artificial delays
friction additions
armature additions
```

solely to improve the current MuJoCo → Gazebo transfer.

---

# 15. Remaining known imperfections

The current system is successful but not perfect.

Known differences / limitations are:

## 15.1 Pure-yaw command tracking

The policy substantially undertracks:

```text
wz = ±0.5 rad/s
```

during pure rotation.

This exists in both MuJoCo and Gazebo.

If improved later, it should be addressed through PPO training rather than Gazebo plant tuning.

## 15.2 Gazebo actuator effort is generally higher

Gazebo often needs larger instantaneous joint effort than MuJoCo.

This is currently acceptable because:

```text
locomotion remains stable
tracking remains good
sustained saturation does not occur
```

## 15.3 Gazebo body height is approximately 12–15 mm higher

This is consistent across many phases.

It is not yet known how much comes from:

```text
frame origin
contact model
controller implementation
equilibrium posture
```

It is not important enough to justify changing the baseline now.

## 15.4 Combined yaw differs more than pure positive yaw

For:

```text
vx = 0.4
vy = 0.2
wz = 0.3
```

MuJoCo gives approximately:

```text
wz = 0.303
```

while Gazebo gives:

```text
wz = 0.349
```

This is most likely a coupled locomotion/contact-dynamics difference.

The robot remains stable and the behavior is still useful.

---

# 16. Overall engineering conclusion

The main conclusion from this stage is:

> **The Spot V3 PPO policy has successfully transferred from MuJoCo to ROS 2 / Gazebo.**

The remaining differences are second-order rather than evidence of a broken deployment.

The most important earlier uncertainty — poor pure-yaw tracking — has now been resolved causally:

```text
it already exists in MuJoCo
```

and therefore is not a Gazebo implementation problem.

Similarly, the previously observed 500 Hz Gazebo instability was a low-level numerical/control-rate issue and has been resolved independently of PPO by running the Gazebo plant and controller at 2000 Hz.

The current system therefore provides a strong, experimentally measured sim-to-sim baseline.

---

# 17. Recommended next stage

The project should now move forward rather than continue tuning the already validated flat-ground baseline.

Reasonable next directions are:

```text
1. Preserve the current benchmark results and code in Git.

2. Keep model_550 as the reference flat-ground deployment checkpoint.

3. Decide whether accurate turn-in-place tracking is actually important.
   If yes, train a later policy with better pure-yaw command coverage.

4. Begin controlled robustness tests:
   - slopes
   - uneven ground
   - friction changes
   - pushes/disturbances
   - small obstacles

5. Determine how much robustness the current flat policy already has
   before committing to a new rough-terrain training pipeline.

6. Begin defining the real-hardware low-level-control interface only
   after the simulation baseline remains reproducible.
```

The key rule going forward should remain:

```text
Do not change a validated component without evidence that it is the source
of the next observed limitation.
```

---

# 18. Current milestone

The project has now progressed through:

```text
physically coherent V3 plant
        ↓
stable actuator/controller design
        ↓
successful PPO locomotion
        ↓
exact policy export
        ↓
ROS 2 observation/action deployment
        ↓
stable Gazebo low-level control
        ↓
live Gazebo locomotion
        ↓
PS5 teleoperation
        ↓
automated Gazebo benchmark
        ↓
matching MuJoCo benchmark
        ↓
quantitative sim-to-sim validation
```

**Status: V3 flat-ground MuJoCo → ROS 2 / Gazebo sim-to-sim deployment validated.**
