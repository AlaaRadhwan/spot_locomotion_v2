# Spot Locomotion V2 — Actuator Investigation Update and Actuator Model V3 Rethink

## Status

This document is a continuation of:

1. `Spot_Locomotion_V2_Engineering_Retrospective.md`
2. `Spot_V2_Actuator_Characterization_Report.md`

It records the work performed **after** the previous actuator-characterization report reached the conclusion that:

- the current physical torque limits were not obviously the bottleneck,
- the uniform `Kp=500, Kd=10` controller produced very different closed-loop behavior on HY and knee,
- joint-specific gains were the next logical experiment,
- and further reinforcement-learning reward tuning should be paused until the actuator/controller was better understood.

This continuation covers:

- the explicit high-torque authority experiments,
- the doubled-torque-limit experiment,
- the joint-specific `300/8` knee controller,
- the corrected fixed-base step test,
- the corrected free-standing push test,
- the RL result with `300/8`,
- the stale diagnostic-script issue that initially invalidated one test,
- the stale legacy-actuator import issue that temporarily prevented task registration,
- the decision to stop treating the problem as a gain-tuning exercise,
- new research into Boston Dynamics Spot's public joint-control interface and transmission data,
- MuJoCo/MjLab actuator implementation details,
- relevant sim-to-real actuator-modeling literature,
- and the resulting proposal for a new actuator architecture.

The purpose is to preserve not just the parameter changes, but the **causal reasoning** behind them.

---

# 1. Starting Point After the Previous Report

The previous actuator-characterization report established two apparently conflicting facts.

## 1.1 Standing behavior was good with the current controller

With:

```text
Kp = 500
Kd = 10
```

for all 12 joints, and:

```text
hip torque limit  = 69.52 Nm
knee torque limit = 164.15 Nm
```

the robot could:

- stand under gravity,
- reject a 250 N downward body disturbance,
- recover without sustained oscillation,
- remain well below 100% actuator utilization.

The approximately 50.34 kg robot settled near:

```text
base height ≈ 0.434 m
```

and a 250 N downward push caused only about:

```text
12 mm
```

of additional body compression.

Peak utilization during the push was roughly:

```text
hips  ≈ 16%
knees ≈ 34%
```

This was strong evidence that ordinary standing support was **not torque-authority limited**.

---

## 1.2 Commanded knee motion was extremely aggressive

The fixed-base step test gave a different picture.

For the same:

```text
Kp = 500
Kd = 10
```

the representative HY joint reached roughly:

```text
6.25 rad/s
```

while the knee reached approximately:

```text
19–21 rad/s
```

for a similar-sized position step.

The knee did this without saturating.

Its torque changed sign rapidly because the derivative term became very large once the joint accelerated.

This suggested:

> A controller that is acceptable for load-bearing stance can still have inappropriate closed-loop dynamics for swing-leg motion.

The previous report therefore recommended one joint-specific gain experiment rather than a broad gain sweep.

---

# 2. Before Changing Gains: Torque-Authority A/B Experiments

Before committing fully to the joint-specific gain hypothesis, we wanted stronger evidence that the realistic torque ceilings were not the reason PPO preferred the tapping/support-asymmetry solution.

The old successful project had used much higher actuator authority.

Therefore two direct torque-authority experiments were run using the current explicit `IdealPdActuator` architecture.

The key idea was:

> Keep the explicit controller and the reward/task structure, but raise available torque authority and see whether natural locomotion returns.

If it did, the physical torque limits would remain a plausible root cause.

If it did not, torque authority would become much less likely as the primary cause.

---

# 3. Explicit `500/10` With ±1000 Nm Authority

## 3.1 Configuration

The first high-authority experiment kept:

```text
IdealPdActuator
Kp = 500
Kd = 10
```

but replaced the realistic limits with approximately:

```text
±1000 Nm
```

for the joints.

This was deliberately different from the earlier failed test that had combined:

```text
Kp = 500
Kd = 40
±1000 Nm
```

inside the explicit actuator.

The question here was narrower:

> If the current `500/10` explicit controller is simply starved of torque, does restoring old-project-scale torque authority solve the gait problem?

---

## 3.2 Smoke-test result

The simulation did not immediately produce NaNs, but the dynamics became extremely energetic.

Representative late smoke-test values were approximately:

```text
episode length        ~12–13 steps
mechanical power      ~8.6–9.5 kW
foot slip             ~3.2–3.4 m/s
landing force         ~830–865 N
base vertical RMS     ~0.49–0.53 m/s
flight fraction       ~44%
```

Falls and illegal contacts remained common.

The robot was no longer constrained by realistic torque authority, but this did not create a well-behaved locomotion plant.

Instead, the explicit controller became capable of injecting very large amounts of energy.

---

## 3.3 Interpretation

This experiment strongly weakened the hypothesis:

> "The realistic torque limits are simply too low."

Giving the explicit actuator old-project-scale authority did not restore the old project's natural behavior.

It made the plant violently energetic.

This also reinforced an earlier conclusion:

> The old native MuJoCo position servo cannot be reproduced by copying nominal gains and torque authority into `IdealPdActuator`.

The actuator implementation matters.

---

# 4. Intermediate Experiment: 2× Torque Limits

The ±1000 Nm experiment was intentionally extreme.

A more moderate test was then performed.

## 4.1 Configuration

The physical limits were doubled:

```text
hip limit:
69.52 -> 139.04 Nm

knee limit:
164.15 -> 328.30 Nm
```

while retaining:

```text
IdealPdActuator
Kp = 500
Kd = 10
```

The purpose was to test whether the nominal physical limits were restrictive during RL exploration without giving the controller essentially unlimited authority.

---

## 4.2 Smoke-test behavior

Representative late smoke-test values were approximately:

```text
episode length        ~14.7–17.4 steps
mechanical power      ~4.6–4.7 kW
foot slip             ~2.6–2.9 m/s
landing force         ~747–772 N
base vertical RMS     ~0.39–0.47 m/s
flight fraction       ~0.34–0.38
```

This was less extreme than the ±1000 Nm test, but still much more energetic than the physically limited baseline.

---

## 4.3 Full RL result

A full RL run was then performed with the doubled limits.

The important visual result was:

> The tapping/support-asymmetry problem still remained.

The robot still tended to:

- keep one or two feet airborne for disproportionately long periods,
- use some feet as more persistent support,
- and exhibit the same qualitative family of bad gait solutions.

The larger torque limits changed energy and exploration characteristics, but they did **not** remove the central locomotion pathology.

---

## 4.4 Conclusion of the torque-authority investigation

After:

1. realistic limits,
2. doubled limits,
3. and extremely high limits,

the same explicit actuator architecture still failed to reproduce the good old-project locomotion.

The high-authority cases also became dramatically more energetic.

The correct conclusion is not:

> Torque limits never matter.

They certainly shape exploration and transients.

The stronger conclusion is:

> The realistic torque ceilings are not the primary root cause of the tapping/support asymmetry.

This justified restoring:

```text
hip limit  = 69.52 Nm
knee limit = 164.15 Nm
```

and moving on.

---

# 5. Joint-Specific Gain Experiment

The previous characterization showed the knee was much faster than HY under identical gains.

We therefore designed one intentionally small joint-specific change.

## 5.1 New controller

The hips remained unchanged:

```text
HX/HY:
Kp = 500
Kd = 10
```

The knee was changed to:

```text
Knee:
Kp = 300
Kd = 8
```

Torque limits remained:

```text
hip  = 69.52 Nm
knee = 164.15 Nm
```

---

## 5.2 Why `300/8` was chosen

This was not intended as a final physically identified gain set.

The objective was to answer one experiment:

> If the knee servo is made substantially less aggressive while the hips remain unchanged, does the locomotion failure improve?

The proportional gain was reduced:

```text
500 -> 300
```

The damping was reduced approximately in proportion to the square root of the stiffness ratio:

```text
Kd_new ≈ 10 * sqrt(300/500)
       ≈ 7.75
```

rounded to:

```text
8
```

This approximately preserved the damping-ratio scaling under a simple second-order interpretation.

The important point is that the test changed the knee bandwidth significantly while avoiding a large multi-parameter sweep.

---

# 6. Diagnostic-Script Error Discovered

The first apparent `300/8` step-response test looked almost identical to the old `500/10` result.

Instead of accepting the visual result, the raw CSV was checked.

At knee step onset:

```text
position error ≈ 0.2491 rad
measured raw torque ≈ 124.57 Nm
```

If `Kp=300` had actually been active:

```text
tau ≈ 300 * 0.2491
    ≈ 74.7 Nm
```

But:

```text
124.57 / 0.2491 ≈ 500
```

Therefore the diagnostic was still using:

```text
Kp = 500
```

for the knee.

---

## 6.1 Root cause

The standalone visual step-test script had its own hard-coded:

```python
KP = 500.0
KD = 10.0
```

rather than importing the joint-specific values from the robot configuration.

The push-recovery test also still depended on the old global `KP/KD` names.

This meant changing `spot_constants.py` did not automatically change the diagnostic scripts.

---

## 6.2 Fix

Both diagnostic scripts were changed to import:

```python
HIP_KP
HIP_KD
KNEE_KP
KNEE_KD
HIP_TORQUE_LIMIT
KNEE_TORQUE_LIMIT
```

directly from:

```text
spot_locomotion_v2.robot.spot_constants
```

They also gained explicit per-joint gain selection.

The scripts were:

```text
scripts/actuator_visual_step_test.py
scripts/actuator_visual_push_recovery_test.py
```

Output directories were changed to gain-specific names so previous baselines would not be overwritten.

---

## 6.3 Engineering lesson

This was an important instrumentation failure.

The test itself was valid, but the diagnostic implementation had drifted away from the real robot configuration.

A useful rule was established:

> A characterization script should import the same controller constants used by the real training environment whenever possible.

Another useful lesson was:

> Validate a controller experiment from first principles using measured torque, not only from printed parameter names or graph titles.

The simple calculation:

```text
tau / position_error
```

was enough to expose the problem.

---

# 7. Correct Fixed-Base `300/8` Step Test

After correcting the script, the test was repeated.

The HX/HY controller remained:

```text
500/10
```

The knee now genuinely used:

```text
300/8
```

---

## 7.1 Knee comparison

Approximate results:

| Metric | Knee `500/10` | Knee `300/8` |
|---|---:|---:|
| Initial step torque | ~124.6 Nm | ~74.6 Nm |
| Peak forward velocity | ~19.0 rad/s | ~11.4 rad/s |
| Peak return velocity | ~20.7 rad/s | ~12.4 rad/s |
| 10–90% rise time | ~42 ms | ~54 ms |
| Worst braking torque during forward step | ~−84.5 Nm | ~−23.5 Nm |
| Positive correction after return | ~+101.9 Nm | ~+31.3 Nm |
| Saturation | none | none |

---

## 7.2 Main result

The peak knee velocity dropped by roughly:

```text
40%
```

while rise time increased only moderately:

```text
~42 ms -> ~54 ms
```

The large torque reversal was reduced dramatically.

The old response qualitatively resembled:

```text
+125 Nm
-> approximately -85 Nm
-> large correction
```

The new response was closer to:

```text
+75 Nm
-> approximately -24 Nm
-> smaller correction
```

Position tracking remained fast and stable.

---

## 7.3 Cross-joint coupling also reduced

The HY joint experienced smaller disturbances when the knee was commanded.

Approximate peak HY disturbance velocity during knee motion fell from values around:

```text
2.7–3.6 rad/s
```

to roughly:

```text
1.6–2.2 rad/s
```

depending on motion direction.

This suggested that the original aggressive knee motion was injecting significant disturbance into the rest of the leg.

---

## 7.4 Interpretation

For the exact problem targeted by the fixed-base test, `300/8` was successful.

It produced:

- lower peak velocity,
- much gentler torque reversals,
- less neighboring-joint disturbance,
- no saturation,
- still-fast position response.

At this stage it would have been easy to conclude that the knee-gain hypothesis was correct.

The loaded test was therefore essential.

---

# 8. Correct Free-Standing `300/8` Push Test

The next question was:

> Did reducing the knee's dynamic aggressiveness make the stance too soft?

The same 250 N downward torso disturbance used in the original baseline characterization was repeated.

---

## 8.1 Quantitative comparison

Approximate results:

| Metric | Knee `500/10` | Knee `300/8` |
|---|---:|---:|
| Settled base height | ~0.434 m | ~0.4231 m |
| Extra downward displacement under push | ~12 mm | ~19.1 mm |
| Peak downward base velocity | ~−0.15 m/s | ~−0.182 m/s |
| Peak upward recovery velocity | ~+0.12 m/s | ~+0.157 m/s |
| Peak hip utilization | ~16% | ~30.9% |
| Peak knee utilization | ~34% | ~33.6% |
| Peak HY velocity during push | ~0.23 rad/s | ~0.346 rad/s |
| Peak knee velocity during push | ~0.65 rad/s | ~0.834 rad/s |

---

## 8.2 Static sag matched theory

With the new knee controller:

```text
desired knee ≈ -1.80 rad
actual loaded knee ≈ -1.90 to -1.907 rad
```

The required static error was therefore around:

```text
0.10 rad
```

For:

```text
Kp = 300
```

the generated support torque is approximately:

```text
300 * 0.10 ≈ 30 Nm
```

This matches the approximate knee torque required for static support.

The result was a very clean demonstration of the pure-PD support mechanism.

With no gravity/load feed-forward:

```text
tau_static = Kp * position_error
```

Lowering `Kp` means the joint must sag farther to generate the same support torque.

---

## 8.3 Interpretation

The robot remained stable.

It did not collapse.

It recovered from the disturbance.

However, the stance became clearly softer:

- lower body height,
- more compression under the same load,
- larger vertical velocity,
- more hip involvement in recovery.

The experiment exposed a real control tradeoff:

```text
higher knee Kp
-> better stance stiffness
-> more aggressive swing dynamics

lower knee Kp
-> gentler swing dynamics
-> more stance sag / softer support
```

This was no longer a simple "find the right Kp" problem.

It suggested that one pure PD term was being asked to solve two different jobs:

1. generate static gravity-support torque,
2. define desired swing-joint dynamics.

---

# 9. RL Test With Knee `300/8`

Because the robot still stood and recovered, one fresh RL run was justified.

The purpose was not to optimize reward.

The experiment was:

```text
same robot
same task
same rewards
same observations
same action scales
same command distribution
same torque limits

only meaningful actuator change:
knee 500/10 -> 300/8
```

---

## 9.1 Correct training environment

The normal V2 environment was used:

```text
Mjlab-Velocity-Flat-Spot-V2
```

through:

```python
get_spot_robot_cfg()
```

from the normal:

```text
spot_constants.py
```

The legacy actuator environment was **not** the desired environment for this experiment.

---

## 9.2 Visual RL result

The result was substantially worse than the `500/10` baseline.

Observed behavior:

- the robot struggled more to walk,
- feet appeared to drag or scrape,
- foot lift was less decisive,
- the robot looked mechanically softer,
- the locomotion did not improve in the way predicted by the fixed-base test.

This was the critical negative result.

---

## 9.3 Interpretation

The `300/8` change successfully improved the **unloaded step-response metric** we were watching.

But it made the actual locomotion plant worse.

This means:

> High knee velocity in the fixed-base test was not, by itself, the root cause of the tapping gait.

The lower Kp reduced the available corrective torque for a given policy position error.

For a 0.25 rad target error:

```text
Kp=500 -> 125 Nm proportional demand
Kp=300 ->  75 Nm proportional demand
```

Under locomotion load, this made swing-leg tracking less decisive and increased posture sag.

The robot therefore paid for the gentler unloaded servo with weaker practical leg motion.

---

# 10. What the `300/8` Experiment Ruled Out

The correct conclusion was not that `300/8` is universally "bad."

The experiment ruled out a more specific hypothesis:

> The tapping is mainly caused by the knee servo being too fast/aggressive under `500/10`.

If that had been the dominant cause, substantially lowering knee bandwidth should have improved the learned gait.

It did not.

The result instead showed:

> A gain change that improves one standalone actuator metric can degrade the full locomotion system.

This was strong evidence that continuing to search gain numbers manually would have diminishing value.

---

# 11. Legacy-Actuator Import Failure During the New Run

Changing the normal constants from:

```text
KP / KD
```

to:

```text
HIP_KP / HIP_KD
KNEE_KP / KNEE_KD
```

caused a separate software issue.

The legacy actuator file still imported:

```python
KP
KD
```

from the normal `spot_constants.py`.

The task package imported the legacy environment during package initialization.

Therefore the stale import produced:

```text
ImportError:
cannot import name 'KP' from spot_locomotion_v2.robot.spot_constants
```

Because package initialization failed, the normal Spot task was not registered.

The CLI then reported:

```text
invalid choice 'Mjlab-Velocity-Flat-Spot-V2'
```

The second error was only a consequence of the first.

---

## 11.1 Resolution

The legacy experimental task was temporarily disabled or its stale imports needed to be updated.

The key point was that the normal V2 environment itself was not broken.

The old diagnostic/experimental code had become coupled to a now-changed API.

---

## 11.2 Engineering lesson

Experimental code can become infrastructure debt.

A stale A/B file should not be allowed to prevent the production/current task from importing.

Going forward, experimental actuator configurations should either:

- use explicit, stable parameter names,
- be isolated from package-wide task registration,
- or be removed/archived once the experiment is complete.

---

# 12. Decision: Stop Treating the Problem as Gain Tuning

At this point several plausible actuator hypotheses had been tested.

## 12.1 Torque authority

Tested through:

- realistic limits,
- 2× limits,
- approximately ±1000 Nm.

Result:

> More torque did not remove the central gait pathology.

---

## 12.2 Knee bandwidth

Tested through:

```text
500/10 -> 300/8
```

Result:

> Unloaded knee dynamics improved, but RL locomotion became worse.

---

## 12.3 Reward explanation

Earlier work had already shown that:

- simple rewards reproduced the pathology,
- complicated gait rewards changed or exploited the pathology,
- and the native actuator could learn much better locomotion under essentially the same task.

Result:

> Reward design alone is not the primary explanation.

---

## 12.4 New conclusion

The problem was now better framed as:

> What actuator/control architecture should this robot actually have?

rather than:

> What Kp/Kd values should we try next?

This prompted a clean-sheet actuator-model investigation.

---

# 13. Boston Dynamics Spot Public Joint-Control Interface

Boston Dynamics' current public Joint Control API provides a useful reference because it exposes a joint-level interface very close to what a learned controller could command.

The documented servo law is:

```text
load_servo =
    k_q_p  * (position_desired - position_measured)
  + k_qd_p * (velocity_desired - velocity_measured)
  + load_desired
```

In control notation:

```text
tau =
Kp (q_des - q)
+
Kd (qdot_des - qdot)
+
tau_ff
```

The interface exposes, per joint:

- desired position,
- desired velocity,
- desired load/feed-forward torque,
- position gain,
- velocity gain.

This is materially richer than our current controller:

```text
tau =
Kp (q_des - q)
-
Kd qdot
```

Our current implementation assumes:

```text
qdot_des = 0
tau_ff   = 0
```

at all times.

---

# 14. Command Streaming Rate and Interpolation

Boston Dynamics recommends streaming new joint commands at approximately:

```text
100–333 Hz
```

for high-fidelity trajectories or trained machine-learning policies.

Their public squat example:

- interpolates between joint positions,
- computes a corresponding desired velocity,
- streams commands at approximately 333 Hz,
- and includes measured joint load as feed-forward.

This does **not** prove the proprietary walking controller uses exactly this implementation.

However, it is enough to show that Boston Dynamics considers the following architecture reasonable:

```text
high-rate q_des
+
high-rate qdot_des
+
feed-forward load
+
joint-specific PD gains
```

Our current policy/control path is much more discontinuous:

```text
PPO position target @ 50 Hz
-> zero-order held for 20 ms
-> explicit PD @ 500 Hz
-> qdot_des = 0
-> tau_ff = 0
```

This difference became important enough to investigate explicitly.

---

# 15. Boston Dynamics Public Example Gains

Boston Dynamics publishes example leg gains described as reasonable for position control in kinematics space:

```text
HX:
Kp = 624
Kd = 5.20

HY:
Kp = 936
Kd = 5.20

Knee:
Kp = 286
Kd = 2.04
```

These values should **not** be treated as proprietary production-walking gains.

They are still useful for three reasons.

## 15.1 They are joint-specific

Boston Dynamics does not assume:

```text
HX = HY = knee
```

in controller gains.

This agrees with our measured difference in closed-loop behavior.

---

## 15.2 Knee proportional stiffness is lower

The public reference knee value:

```text
Kp = 286
```

is numerically close to our experimental:

```text
Kp = 300
```

However this does **not** validate our `300/8` experiment, because their published knee damping is:

```text
Kd = 2.04
```

while ours was:

```text
Kd = 8
```

and the rest of the robot/control architecture is also different.

The correct lesson is:

> Our earlier `300/8` was still a heuristic controller, not a reconstruction of Spot's low-level dynamics.

---

## 15.3 Controller gains cannot be copied without the plant

A gain number only has meaning together with:

- reflected inertia,
- transmission ratio,
- joint friction,
- torque/current limits,
- servo rate,
- command interpolation,
- feed-forward,
- robot mass distribution.

Therefore the Boston Dynamics gains are **reference points**, not parameters to copy directly.

---

# 16. Transmission Data and Torque-Limit Reassessment

The Boston Dynamics supplemental joint-control data provides:

```text
HX gear ratio = 51
HY gear ratio = 51
HX/HY max motor torque = 0.88 Nm

knee max motor torque = 1.50 Nm
knee transmission ratio = configuration-dependent
```

The knee uses a ball-screw / push-rod linkage, so joint torque capability varies with knee angle.

---

## 16.1 HX/HY real-Spot-like joint torque

Ignoring efficiency for a first approximation:

```text
tau_joint =
gear_ratio * tau_motor
```

so:

```text
51 * 0.88
= 44.88 Nm
```

---

## 16.2 Why our 69.52 Nm hip limit is not arbitrary

The public Spot reference mass is approximately:

```text
32.5 kg
```

Our model is approximately:

```text
50.34 kg
```

Mass scale:

```text
50.34 / 32.5
≈ 1.5489
```

Scale the Spot-like hip torque by the same factor:

```text
44.88 * 1.5489
≈ 69.51 Nm
```

Our current limit is:

```text
69.52 Nm
```

which is essentially exact.

---

## 16.3 Knee scaling

At approximately:

```text
q_knee ≈ -1.8 rad
```

the published Spot knee torque capability is roughly:

```text
~105.98 Nm
```

Scaling by the same mass ratio:

```text
105.98 * 1.5489
≈ 164.15 Nm
```

which is exactly our current knee limit.

---

## 16.4 New interpretation of the current torque limits

The limits can now be described more rigorously:

> They are Spot-derived transmission limits scaled by the mass ratio of our heavier model to approximately preserve torque-to-weight capability.

This is a much stronger justification than calling them simply "realistic limits."

They are not literal Boston Dynamics Spot limits.

They are a **scaled Spot-like assumption**.

---

# 17. The Current Constant Knee Limit Is Still Incomplete

Boston Dynamics explicitly documents that the knee torque capability changes with joint configuration.

Therefore:

```text
knee_limit = 164.15 Nm
```

as one constant value is only locally representative around the selected reference configuration.

A better future model should implement:

```text
tau_knee_max = f(q_knee)
```

using the published transmission relationship, then apply the same explicit mass/torque scaling assumption if we choose to preserve it.

This would make the model more physically interpretable without inventing a full unknown motor model.

---

# 18. MuJoCo/MjLab Actuator Implementation Difference

MjLab's actuator classes clarify why the old native actuator and the new explicit actuator behave differently even when their equations look similar.

## 18.1 `IdealPdActuator`

`IdealPdActuator` computes:

```text
tau =
Kp (q_des - q)
+
Kd (qdot_des - qdot)
```

outside MuJoCo and sends the result through a `<motor>` actuator.

From MuJoCo's point of view, this is an externally computed force.

The integrator does not have the PD gain derivatives as part of the native actuator model.

---

## 18.2 `BuiltinPdActuator`

MjLab's newer `BuiltinPdActuator` represents the PD controller using native MuJoCo:

```text
<position>
+
<velocity>
```

actuators.

MjLab explicitly describes it as the implicit-integration version of `IdealPdActuator`.

With MuJoCo:

```text
implicit
```

or:

```text
implicitfast
```

the gain derivatives are included in the velocity solve.

This generally makes stiff servo dynamics more numerically stable.

---

## 18.3 Why this matters to our project

Our empirical history already showed:

```text
native position actuator
-> much better locomotion

explicit IdealPd
-> tapping/asymmetry
```

The MjLab implementation details show that there is a real numerical reason these are not equivalent.

This is no longer merely an empirical curiosity.

---

# 19. Why `BuiltinPdActuator` Is Not Automatically the Final Answer

It would be tempting to conclude:

> Use `BuiltinPdActuator` and the problem is solved.

That would be premature.

A native implicit PD implementation addresses **how the servo is numerically integrated**.

It does not by itself model:

- motor torque-speed limits,
- reflected rotor inertia,
- gearbox/friction losses,
- joint friction,
- knee transmission geometry,
- actuator response delay,
- load/feed-forward compensation,
- command interpolation,
- electrical dynamics.

Therefore native PD is best viewed as:

> a better numerical implementation of the joint servo layer,

not a complete electromechanical actuator model.

---

# 20. Reflected Motor / Gearbox Inertia

MuJoCo supports joint `armature` specifically for inertia associated with actuator/drivetrain motion.

For a geared rotational drivetrain:

```text
J_reflected =
J_rotor * N^2
```

For the Spot-like HX/HY transmission:

```text
N = 51
```

therefore:

```text
N^2 = 2601
```

Even modest rotor inertia can therefore become significant when reflected to the joint.

---

## 20.1 Why this matters

Our current model effectively treats the drivetrain as much closer to:

```text
torque source
-> ideal massless transmission
-> joint
```

than a real geared actuator.

Missing reflected inertia can allow unrealistically fast acceleration.

This may be one reason the fixed-base knee and other joints can achieve very large velocities under small target changes.

---

## 20.2 Why we should not add an arbitrary armature immediately

We do not currently have a trusted public value for the real Spot motor rotor inertia.

Adding a guessed `armature` could make the simulation look better while making the physical model less defensible.

The correct next step is:

- find a measured/public rotor inertia if possible,
- estimate it from a known motor/datasheet if a defensible motor analogue is chosen,
- or identify an equivalent armature from target frequency-response data.

---

# 21. Friction and Other Drivetrain Effects

A real actuator transmission can contain:

- Coulomb friction,
- viscous friction,
- gearbox drag,
- efficiency losses,
- backlash/compliance,
- rotor inertia,
- motor electrical dynamics.

The current model captures very little of this.

MuJoCo provides mechanisms for several of these effects, including:

- `frictionloss`,
- joint damping,
- armature,
- native DC motor modeling.

The correct principle is:

> Add only effects that we can parameterize or identify meaningfully.

A complicated model with guessed parameters is not automatically more realistic than a simpler model with defensible assumptions.

---

# 22. Full DC Motor Modeling

MuJoCo now provides a detailed DC motor model that can include effects such as:

- motor torque constant,
- back-EMF,
- winding resistance,
- inductance,
- torque-speed behavior,
- rotor inertia,
- friction,
- thermal dynamics,
- cogging.

This is attractive for long-term sim-to-real work.

However, we currently lack reliable values for many Spot motor parameters, including likely:

- resistance,
- torque constant,
- voltage constant,
- no-load speed,
- rotor inertia,
- current-controller behavior,
- gearbox efficiency,
- joint friction.

Therefore moving directly to a full DC motor model would create many guessed degrees of freedom.

The current educated opinion is:

> Do not use a sophisticated motor model until enough parameters are known to make it meaningful.

---

# 23. Sim-to-Real Literature: Why Actuator Modeling Matters

The decision to rethink the actuator is strongly supported by prior legged-robot sim-to-real work.

---

# 24. Hwangbo et al. — ANYmal Actuator Modeling

The ANYmal sim-to-real work by Hwangbo et al. treated actuator behavior as a major transfer problem.

Their approach used a learned actuator model to predict torque from actuator state/history.

The important conceptual lesson is:

> An ideal analytical servo can be too different from the real actuator for a locomotion policy trained in simulation to transfer reliably.

Their method used actuator-state history specifically because actuator dynamics are not always captured by a memoryless torque equation.

This supports our current concern that:

```text
tau = clip(Kp*error - Kd*qdot)
```

may be too simple even when its static behavior looks plausible.

---

# 25. Minitaur Sim-to-Real Work

Sim-to-real work on Minitaur explicitly modeled effects including:

- DC motor behavior,
- back-EMF / torque-speed behavior,
- latency.

This reinforced another key point:

> Torque limit alone is not enough to define actuator capability.

A motor that can make a certain torque at zero speed may not make the same torque at high speed.

Our current:

```text
|tau| <= constant tau_max
```

model ignores this.

---

# 26. Modern Quadruped Actuator Constraints

More recent high-performance quadruped work often models an actuator's feasible operating region in torque-speed space rather than using only one constant torque cap.

This matters because locomotion contains:

- high-speed swing motion,
- high-torque stance motion,
- rapid transitions between the two.

The motor/transmission may support these regimes differently.

This is another reason not to interpret the current 69.52 / 164.15 Nm values as a complete actuator model.

They are only one layer of the constraints.

---

# 27. Current Actuator Architecture vs More Physical Architecture

## 27.1 Current architecture

Conceptually:

```text
RL policy @ 50 Hz
    |
    | joint position offsets
    v
q_des
    |
    | zero-order hold for 20 ms
    v
IdealPdActuator @ 500 Hz
    |
    | tau = Kp(q_des-q) - Kd*qdot
    |
    | constant torque clamp
    v
MuJoCo rigid-body dynamics
```

Missing or simplified:

```text
qdot_des
tau_ff
trajectory interpolation
servo delay
reflected inertia
torque-speed behavior
knee variable transmission limit
friction
electrical motor dynamics
```

---

# 28. Proposed Actuator Model V3 Architecture

The new design should separate the problem into physically meaningful layers.

```text
RL policy @ 50 Hz
        |
        | high-level joint command
        v

Command / trajectory layer
        |
        | q_des(t)
        | qdot_des(t)
        v

Low-level joint servo @ 500 Hz
        |
        | tau_cmd =
        | Kp(q_des-q)
        | + Kd(qdot_des-qdot)
        | + tau_ff
        v

Motor + transmission constraint layer
        |
        | torque capability
        | torque-speed capability
        | joint-angle-dependent knee ratio
        | efficiency / response constraints
        v

Drivetrain dynamics
        |
        | reflected inertia / armature
        | friction / damping
        v

MuJoCo multibody dynamics + contact
```

This decomposition is important because it separates:

## Controller parameters

```text
Kp
Kd
q_des
qdot_des
tau_ff
```

from:

## Physical actuator parameters

```text
gear/transmission ratio
torque capability
torque-speed curve
rotor inertia
friction
efficiency
latency
```

Earlier work sometimes asked Kp/Kd to compensate for missing physical effects.

The V3 architecture should avoid that.

---

# 29. A Better Way to Design Kp/Kd

The next controller should not begin from:

```text
"500 feels stiff"
```

or:

```text
"300 looks smoother"
```

Instead, gains should be related to desired closed-loop dynamics.

For a simplified joint model:

```text
J_eff * qdd
+
Kd * qd
+
Kp * (q - q_des)
=
0
```

the approximate natural frequency is:

```text
omega_n =
sqrt(Kp / J_eff)
```

and damping ratio:

```text
zeta =
Kd / (2 * sqrt(Kp * J_eff))
```

Therefore gains should ideally be selected from:

- effective joint inertia,
- target servo bandwidth,
- target damping ratio.

---

# 30. Effective Joint Inertia Should Be Measured From the Model

MuJoCo can provide the configuration-dependent mass matrix.

This means we can evaluate the effective inertia seen by:

- HX,
- HY,
- knee,

at:

- home pose,
- loaded stance configurations,
- representative swing configurations.

The result would explain quantitatively why the same `Kp/Kd` gives such different responses.

Instead of saying:

```text
knee is ~3× faster than HY
```

we could connect that to the actual effective inertia.

This is a much more controls-grounded basis for servo design.

---

# 31. Desired-Velocity Targets and Command Interpolation

The current position-action policy produces new targets every:

```text
20 ms
```

At each policy update, the desired position effectively changes as a step.

The low-level controller then assumes:

```text
qdot_des = 0
```

This can create a peculiar servo transient:

1. position target jumps,
2. large proportional torque accelerates the joint,
3. desired velocity remains zero,
4. derivative term immediately treats the useful motion as velocity error,
5. large braking torque appears,
6. contacts or coupling change qdot,
7. torque can reverse rapidly.

This mechanism was visible directly in the original knee step-response test.

---

## 31.1 Candidate V3 command layer

One candidate is to interpolate from the previous policy target to the new target over the 20 ms policy period.

The interpolation can provide both:

```text
q_des(t)
qdot_des(t)
```

to the 500 Hz servo.

This would make the low-level controller follow a short trajectory rather than repeatedly responding to discontinuous position steps.

This is conceptually closer to the public Boston Dynamics joint-control example.

---

## 31.2 Important caution

Interpolation should not be added because it "looks smoother."

It should be tested as an explicit A/B:

```text
zero-order-held q_des, qdot_des=0
vs.
interpolated q_des + consistent qdot_des
```

using the same actuator characterization suite.

---

# 32. Feed-Forward / Load Compensation

The standing tests exposed a limitation of pure PD.

With no feed-forward:

```text
tau_static =
Kp * position_error
```

The knee requires around:

```text
~30 Nm
```

for normal support.

Therefore:

```text
Kp=500 -> ~0.06 rad static sag
Kp=300 -> ~0.10 rad static sag
```

This was measured experimentally.

A feed-forward/gravity/load term could provide part of the required support torque without requiring the joint to sag.

This could potentially decouple:

```text
static support requirement
```

from:

```text
dynamic servo stiffness
```

which was exactly the conflict exposed by the `300/8` experiment.

---

## 32.1 Feed-forward should still be introduced carefully

Possible sources include:

- gravity compensation from the model,
- desired load/torque from another controller,
- learned residual/feed-forward torque.

The next design should define clearly which one is intended.

We should not simply add arbitrary feed-forward torque because it makes standing easier.

---

# 33. Revised Characterization Program Before More RL

The previous two actuator tests were useful but insufficient.

A V3 actuator should be characterized before PPO is allowed to compensate for it.

Recommended tests:

---

## 33.1 Fixed-base step response

Measure:

- rise time,
- overshoot,
- peak qdot,
- torque history,
- saturation,
- neighboring-joint disturbance.

Run for HX/HY/knee rather than only HY and knee if needed.

---

## 33.2 Loaded step response

Repeat target changes while the leg supports or resists realistic load.

This separates unloaded servo bandwidth from practical loaded performance.

---

## 33.3 Gravity hold

Measure:

- equilibrium position error,
- required torque,
- base/body height,
- sensitivity to Kp and feed-forward.

---

## 33.4 Push recovery

Retain the 250 N downward disturbance as a useful whole-body stiffness/damping test.

---

## 33.5 Sine / chirp response

Command sinusoidal joint trajectories over a controlled frequency range.

Measure:

- amplitude ratio,
- phase lag,
- torque demand,
- velocity,
- onset of saturation.

This gives a much richer measure of servo bandwidth than one step response.

---

## 33.6 Policy-rate target-change test

Generate representative 50 Hz target sequences and inspect what the 500 Hz servo actually does between policy updates.

Compare:

```text
zero-order hold
vs.
interpolated position/velocity target
```

---

## 33.7 Torque-speed operating test

Once a physically justified motor/transmission model exists, test available torque as a function of joint speed.

---

# 34. What We Can Now Say With High Confidence

## Confirmed Finding 1

The realistic torque limits are not the primary explanation for the tapping gait.

Evidence:

- standing uses only a fraction of them,
- 2× limits did not remove the gait failure,
- ±1000 Nm made the explicit plant violently energetic rather than naturally locomoting.

---

## Confirmed Finding 2

Uniform `500/10` gains produce strongly different HY and knee dynamics.

This was directly measured.

---

## Confirmed Finding 3

Reducing knee gains to `300/8` genuinely reduced unloaded knee speed and torque reversals.

---

## Confirmed Finding 4

The same `300/8` change caused greater loaded sag and softer whole-body support.

---

## Confirmed Finding 5

RL locomotion with `300/8` was worse, including weaker foot lift / dragging.

Therefore:

> The tapping failure cannot be explained simply by excessive knee bandwidth.

---

## Confirmed Finding 6

The old native MuJoCo actuator and the current explicit `IdealPdActuator` are not dynamically interchangeable.

Both empirical behavior and MjLab's implementation details support this.

---

## Confirmed Finding 7

The current 69.52 Nm / 164.15 Nm limits have a strong physical interpretation as mass-scaled Spot-like transmission limits.

---

## Confirmed Finding 8

The constant knee limit is still an approximation because the real knee transmission is configuration-dependent.

---

## Confirmed Finding 9

Our current pure PD controller forces gravity-support torque to appear through static position error.

This was quantitatively verified in both `500/10` and `300/8` standing tests.

---

# 35. What Has Been Ruled Out or Strongly Weakened

The following explanations should no longer be treated as leading hypotheses.

## "The reward is simply wrong"

Strongly weakened.

Reward changes altered symptoms but did not remove the underlying actuator-dependent behavior.

---

## "The torque limits are simply too low"

Strongly weakened.

Increasing them did not restore natural locomotion.

---

## "The knee only needs a lower Kp/Kd"

Strongly weakened.

The lower-gain knee became gentler in a fixed-base test but produced worse locomotion.

---

## "The old 500/40/±1000 actuator can be copied into IdealPd"

Ruled out.

That approach produced severe instability / enormous energy injection.

---

# 36. What Remains Open

Important unresolved questions include:

1. How much of the gait difference is due specifically to implicit/native PD integration?
2. What closed-loop bandwidth should HX, HY, and knee actually have?
3. What are the model's configuration-dependent effective joint inertias?
4. Should policy target steps be interpolated at the servo rate?
5. How important is nonzero `qdot_des`?
6. How much standing support should come from feed-forward instead of position error?
7. What armature/reflected inertia values are physically defensible?
8. What joint friction values are defensible?
9. How should the angle-dependent knee torque limit be implemented?
10. Is a torque-speed motor envelope necessary for the intended sim-to-sim / future hardware target?
11. What latency/delay should be modeled?
12. Which parameters can be derived from public Spot data and which must be identified experimentally?

---

# 37. Recommended Actuator Model V3 Development Sequence

The next stage should deliberately avoid PPO until the actuator is coherent.

## Phase 1 — Freeze and document the current baseline

Preserve:

```text
IdealPd
500/10 all joints
69.52 / 164.15 Nm limits
```

and the characterization artifacts.

Do not overwrite them.

---

## Phase 2 — Extract effective inertias

Use MuJoCo's mass matrix to quantify effective inertia at representative configurations.

This gives a rational basis for controller bandwidth calculations.

---

## Phase 3 — Native MuJoCo PD implementation

Build a V3 joint-servo implementation using native MuJoCo PD, likely MjLab:

```text
BuiltinPdActuatorCfg
```

Use:

```text
position target
velocity target
```

rather than assuming zero desired velocity by construction.

At first, keep the physical torque-limit assumptions unchanged.

---

## Phase 4 — Command interpolation A/B

Compare:

```text
50 Hz zero-order-held position target
```

with:

```text
500 Hz interpolated q_des/qdot_des
```

using the characterization suite.

---

## Phase 5 — Knee transmission constraint

Replace:

```text
constant knee torque limit
```

with:

```text
angle-dependent scaled Spot-like knee limit
```

derived from the published knee transmission information.

---

## Phase 6 — Feed-forward investigation

Determine whether gravity/load feed-forward should be used.

If used, document exactly what produces it.

---

## Phase 7 — Reflected inertia / friction

Add:

- armature,
- friction,
- damping,

only when parameters are physically justified or identified.

---

## Phase 8 — Latency / command delay

Characterize and add realistic delay if relevant to the intended deployment stack.

---

## Phase 9 — Motor torque-speed model

Only after enough motor/transmission data is available should a fuller DC motor or torque-speed envelope be introduced.

---

## Phase 10 — Freeze actuator and restart RL

Once the actuator/controller passes characterization:

- freeze the actuator model,
- restore the simple locomotion task,
- restart PPO from scratch,
- do not compensate for actuator problems with gait rewards.

---

# 38. Proposed Acceptance Criteria Before PPO

A V3 actuator should not be approved because it "looks good."

It should pass explicit criteria.

Possible criteria to define quantitatively:

```text
step response:
reasonable rise time
limited overshoot
no pathological torque reversal
no unexplained saturation

loaded response:
maintains pose under realistic support loads
acceptable static sag
stable disturbance recovery

frequency response:
known usable bandwidth
acceptable phase lag
no unexpected resonance

target interpolation:
no extreme torque impulses at 50 Hz policy transitions

transmission:
torque constraints consistent with model assumptions

numerics:
stable under implicitfast at 500 Hz

energy:
no unexplained multi-kW behavior in ordinary motion
```

The exact numerical thresholds should be chosen only after the effective inertia and target operating regime are understood.

---

# 39. Repository / Artifact Notes

Relevant current files include:

```text
src/spot_locomotion_v2/robot/spot_constants.py

src/spot_locomotion_v2/robot/spot_constants_legacy_actuator.py

src/spot_locomotion_v2/tasks/velocity/flat_env_cfg.py

src/spot_locomotion_v2/tasks/velocity/legacy_actuator_env_cfg.py

scripts/actuator_visual_step_test.py

scripts/actuator_visual_push_recovery_test.py
```

Main explicit MJCF:

```text
src/spot_locomotion_v2/robot/mjcf/experiments/
spot_floating_base_explicit_pd.xml
```

Fixed-base test MJCF:

```text
src/spot_locomotion_v2/robot/mjcf/experiments/
spot_fixed_base_explicit_pd.xml
```

Free-standing scene:

```text
src/spot_locomotion_v2/robot/mjcf/experiments/
scene_free_standing.xml
```

The original untouched reference MJCF should remain unchanged.

---

# 40. Experiment Matrix So Far

| Experiment | Actuator | Gains | Limits | Main result |
|---|---|---|---|---|
| V2 baseline | IdealPd | all 500/10 | 69.52 / 164.15 | tapping + support asymmetry |
| old successful style | native position | 500/40 | ~±1000 | natural locomotion returned |
| explicit old-gain attempt | IdealPd | 500/40 | ±1000 | catastrophic instability / NaNs |
| explicit high authority | IdealPd | 500/10 | ±1000 | extreme energy, slip, impacts |
| explicit 2× limits | IdealPd | 500/10 | 139.04 / 328.30 | tapping still present |
| joint-specific test | IdealPd | hips 500/10, knee 300/8 | 69.52 / 164.15 | better knee step response, softer stance |
| joint-specific RL | IdealPd | hips 500/10, knee 300/8 | 69.52 / 164.15 | locomotion worse; foot dragging |

This table summarizes why the problem can no longer be reduced to one gain or one torque limit.

---

# 41. Current Best Technical Interpretation

The strongest current interpretation is:

> The Spot V2 locomotion pathology is associated with the overall actuator/control architecture and its interaction with the plant, rather than one isolated reward weight, torque ceiling, or knee gain.

The current explicit controller is too simple in some places and overburdened in others.

It is being asked to provide:

- static gravity support,
- dynamic target tracking,
- damping,
- and actuator realism,

through only:

```text
Kp
Kd
constant torque limit
```

while omitting several physically important layers.

The successful native-actuator A/B demonstrates that the RL task and robot geometry are capable of good locomotion.

The failed high-authority explicit tests demonstrate that actuator authority alone is not the answer.

The failed `300/8` RL test demonstrates that making one joint gentler in isolation is not the answer.

Therefore the correct engineering task is now:

> Design a coherent actuator/control stack first, then train PPO on that plant.

---

# 42. Broader Controls Engineering Lessons

## 42.1 Closed-loop behavior matters more than gain values

A gain pair is not a physical description.

The meaningful quantities are:

- bandwidth,
- damping,
- peak velocity,
- torque history,
- load sensitivity,
- disturbance rejection.

---

## 42.2 Static stiffness and dynamic tracking are different requirements

The `300/8` experiment demonstrated this directly.

A lower Kp improved unloaded motion but worsened stance and locomotion.

This is a classic reason to use:

- feed-forward,
- trajectory targets,
- or more complete actuator dynamics,

rather than forcing one virtual spring to do everything.

---

## 42.3 Numerical integration is part of the controller implementation

A PD law evaluated outside the simulator and a native implicit PD servo are not necessarily equivalent discrete-time systems.

This is particularly important at high stiffness and in contact-rich locomotion.

---

## 42.4 More "realistic" constraints can create a less realistic system if the rest of the actuator is incomplete

Replacing ±1000 Nm with a physical torque cap was directionally correct.

But if:

- drivetrain inertia,
- motor speed limits,
- command shaping,
- feed-forward,
- friction,

are omitted, the resulting system may still not resemble a real actuator.

Realism must be systemic, not parameter-by-parameter.

---

## 42.5 Do not let PPO become the system-identification tool

PPO is very good at compensating for plant quirks.

That can hide bad modeling.

The actuator should first be characterized using deterministic tests.

Only then should PPO be used to discover locomotion.

---

# 43. Broader Reinforcement-Learning Lesson

The project has now produced a clearer hierarchy:

```text
1. Robot mechanics
2. Actuator / low-level controller
3. Action interface
4. Observations
5. Commands
6. Reward
7. PPO optimization
```

Reward tuning was initially receiving too much attention relative to layer 2.

The current evidence indicates that:

> A bad or incomplete actuator plant can create an RL local optimum that no reasonable reward tweak fixes cleanly.

The correct response is not infinite reward shaping.

It is to fix the plant.

---

# 44. Current Decision

At the end of this phase:

```text
STOP:
manual gain guessing
torque-limit sweeps
new gait rewards
more PPO runs on the unresolved actuator

KEEP:
the physically motivated torque-limit scaling assumption
the current robot mass/inertia model
the simple locomotion task
the rich diagnostics
the old/native actuator A/B as a behavioral reference

START:
Actuator Model V3 design
effective-inertia analysis
native PD servo investigation
command interpolation investigation
angle-dependent knee transmission constraint
physically justified drivetrain modeling
```

---

# 45. Source References for the V3 Rethink

These sources were used as external engineering references.

They do not prove that our Spot-like robot should copy Boston Dynamics' proprietary internal controller.

They provide public data and established modeling approaches that help define a defensible design.

## Boston Dynamics

### Joint Control API

Boston Dynamics Spot documentation:

`https://dev.bostondynamics.com/docs/concepts/joint_control/readme`

Relevant points:

- joint-level PD + feed-forward control,
- desired position,
- desired velocity,
- desired load,
- per-joint gains,
- recommended high-rate command streaming,
- robot-state streaming at high frequency.

### Supplemental Joint / Transmission Data

`https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html`

Relevant points:

- HX/HY gear ratio = 51,
- HX/HY max motor torque = 0.88 Nm,
- knee max motor torque = 1.50 Nm,
- knee transmission ratio varies with configuration,
- knee uses ball-screw / push-rod geometry.

### Spot SDK Joint-Control Example Gains

Boston Dynamics Spot SDK:

`python/examples/joint_control/constants.py`

Public example values:

```text
HX: 624 / 5.20
HY: 936 / 5.20
KN: 286 / 2.04
```

These are described as reasonable example gains for kinematics-space position control, not production gait gains.

### Joint-Control Squat / Helper Example

Boston Dynamics Spot SDK:

```text
python/examples/joint_control/noarm_squat.py
python/examples/joint_control/joint_api_helper.py
```

Relevant points:

- interpolated desired position,
- explicit desired velocity,
- load/feed-forward command,
- command streaming around 333 Hz.

---

## MuJoCo / MjLab

### MuJoCo XML Reference — Armature

`https://mujoco.readthedocs.io/en/latest/XMLreference.html`

Relevant point:

```text
J_reflected = J_rotor * gear_ratio^2
```

for geared transmission reflected inertia.

### MjLab BuiltinPdActuator

MjLab release/source documentation:

`https://github.com/mujocolab/mjlab/releases`

`https://github.com/mujocolab/mjlab/blob/main/src/mjlab/actuator/builtin_actuator.py`

Relevant distinction:

```text
IdealPdActuator:
PD force computed externally and sent through motor

BuiltinPdActuator:
native MuJoCo position + velocity actuator
implicit integration can include gain derivatives
```

### MuJoCo DC Motor Model

`https://mujoco.readthedocs.io/en/latest/_static/dcmotor.pdf`

Relevant possible future effects:

- torque constant,
- back-EMF,
- resistance,
- inductance,
- rotor inertia,
- friction,
- torque-speed behavior,
- thermal effects.

---

## Sim-to-Real Legged Robotics

### Hwangbo et al.

"Learning agile and dynamic motor skills for legged robots"

`https://arxiv.org/abs/1901.08652`

Relevant lesson:

- actuator dynamics are a central sim-to-real problem,
- learned actuator models can be used when analytical models are insufficient.

### Tan et al. / Minitaur Sim-to-Real

"Sim-to-Real: Learning Agile Locomotion For Quadruped Robots"

`https://arxiv.org/abs/1804.10332`

Relevant lesson:

- motor dynamics,
- torque-speed behavior,
- and latency

can be important for successful transfer.

---

# 46. Final Summary

Since the previous actuator-characterization report, the project moved through three important stages.

## Stage 1 — Torque authority was tested directly

Increasing torque authority:

```text
1× -> 2× -> approximately ±1000 Nm
```

did not remove the core locomotion failure.

Very high authority instead made the explicit actuator extremely energetic.

Therefore:

> insufficient torque ceiling is no longer the leading explanation.

---

## Stage 2 — Knee gains were isolated

The knee was changed:

```text
500/10 -> 300/8
```

Corrected deterministic tests showed:

- ~40% lower peak knee velocity,
- much smaller torque reversal,
- less cross-joint disturbance,
- but more stance sag and softer push response.

RL then showed:

> the robot walked worse and dragged its feet more.

Therefore:

> excessive unloaded knee bandwidth is not the root cause by itself.

---

## Stage 3 — The actuator architecture itself became the main problem

External research and our own A/B data now point toward a broader issue.

A realistic joint actuator is not just:

```text
PD + constant torque clip
```

It includes, depending on required fidelity:

- position and velocity target trajectories,
- feed-forward/load terms,
- numerically appropriate servo integration,
- transmission geometry,
- torque capability,
- torque-speed behavior,
- drivetrain inertia,
- friction,
- delay.

The current best plan is therefore to stop PPO development temporarily and design **Actuator Model V3** from a controls and physical-modeling perspective.

The central engineering question is now:

> What low-level actuator/controller dynamics should the 50 Hz learned policy actually be commanding?

That question is more fundamental than:

> Which reward or Kp/Kd value makes the robot walk?

Answering it correctly is the next step toward a Spot-like locomotion system that is not only trainable, but physically defensible and transferable.
