# Spot Locomotion V2 — Explicit-PD Actuator Characterization Report

## Status

This document records the actuator-focused investigation performed after the Spot V2 locomotion experiments showed that the current explicit-PD model learned a noticeably worse gait than the earlier MjLab 1.1 / native-position-actuator setup.

The purpose of these tests was **not** to tune every parameter exhaustively.

The purpose was to answer a much narrower engineering question:

> Are the locomotion problems more likely to come from the current PD gains, from the physical torque limits, or from the interaction between them?

Two targeted visual tests were performed:

1. **Fixed-base joint step-response test**
2. **Free-standing downward push-recovery test**

The tests used the current explicit-PD architecture and current V2 limits.

No RL policy was involved in these actuator tests.

---

# 1. Current Explicit-PD Baseline

The current Spot V2 actuator is modeled conceptually as:

```text
tau = clip(
    Kp * (q_des - q)
    - Kd * qdot,
    -tau_max,
    +tau_max
)
```

with:

```text
Kp = 500 Nm/rad
Kd = 10 Nm·s/rad
```

for all 12 joints.

Current torque limits:

```text
HX/HY joint limit = 69.52 Nm
Knee joint limit  = 164.15 Nm
```

The policy/action layer is not part of the tests in this report.

The purpose here is to characterize the low-level actuator/controller itself.

---

# 2. Why These Tests Were Needed

Earlier V2 work showed a major discrepancy:

## Earlier successful actuator setup

The old locomotion project used a native MuJoCo position servo with approximately:

```text
Kp = 500
Kd = 40
very high actuator authority
```

and learned natural-looking locomotion relatively easily.

## Current V2 actuator setup

The new project uses:

```text
explicit PD
Kp = 500
Kd = 10
finite physically motivated torque limits
```

and frequently learned:

- rapid tapping,
- asymmetric diagonal support,
- one diagonal pair acting as anchors,
- less natural stepping.

A legacy native-position-actuator A/B experiment restored much better locomotion.

This made the actuator model the leading suspect.

However, that experiment changed several things together:

- actuator implementation,
- damping,
- torque authority.

Therefore the actuator needed to be characterized directly.

---

# 3. External Spot Reference Information

Before running the tests, publicly available Boston Dynamics information was reviewed for guidance.

This information is **reference context**, not direct proof of what our model should use.

Boston Dynamics' public Joint Control API supports joint-level position/velocity feedback gains and load/feed-forward commands.

Their public joint-control example uses different gains for different leg joints:

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

This is useful because it reinforces one important idea:

> There is no physical reason to assume HX, HY, and knee should all use identical PD gains.

Boston Dynamics also publishes transmission data indicating:

```text
HX gear ratio = 51
HY gear ratio = 51
Knee transmission = configuration-dependent
```

and the knee uses a ball-screw / push-rod mechanism.

The external reference therefore suggests:

- joint-specific dynamics matter,
- the knee is mechanically different,
- a single `Kp/Kd` pair for all joints is a simplification.

The tests below independently support that conclusion.

---

# 4. Test 1 — Fixed-Base Visual Joint Step Response

## 4.1 Purpose

The first test asked:

1. Does `Kp=500` produce sensible commanded motion?
2. Is `Kd=10` weak, excessive, or reasonable during motion?
3. Does the actuator spend significant time at the torque limit?
4. Do HY and knee behave similarly when given the same controller gains?

The robot base was fixed so whole-body motion and foot contact would not obscure the joint response.

Only two representative joints were deliberately excited:

```text
FL_HY
FL_KN
```

The other joints continued holding the home pose.

---

# 5. Test 1 Setup

Physics:

```text
500 Hz
dt = 0.002 s
```

Controller:

```text
Kp = 500
Kd = 10
```

Torque limits:

```text
HY   = 69.52 Nm
Knee = 164.15 Nm
```

Test sequence:

```text
0–1 s
hold home pose

1–2 s
FL_HY:
1.04 -> 1.24 rad

2–4 s
return HY to home and settle

4–5 s
FL_KN:
-1.80 -> -1.55 rad

5–7 s
return knee to home and settle
```

Command sizes:

```text
HY step   = 0.20 rad ≈ 11.5 deg
Knee step = 0.25 rad ≈ 14.3 deg
```

Recorded quantities:

- desired position,
- actual position,
- joint velocity,
- raw PD torque,
- clipped/applied torque.

The test was run with the MuJoCo viewer visible.

---

# 6. Test 1 — HY Position Response

The HY joint tracked:

```text
1.04 -> 1.24 rad
```

very rapidly.

Observed behavior:

- fast rise,
- small overshoot,
- no sustained oscillation,
- very small final error.

Approximate measured behavior:

```text
10–90% rise time ≈ 34 ms
peak overshoot ≈ 1.1%
```

The steady position while loaded by the hanging leg settled slightly below the desired value.

Approximate values:

```text
desired ≈ 1.2400 rad
actual  ≈ 1.2335 rad
```

Steady error:

```text
≈ 0.0065 rad
≈ 0.37 deg
```

This error is physically understandable.

There is no gravity/feed-forward torque in the controller.

At steady state:

```text
qdot = 0
```

so the PD controller must generate the static holding torque through position error:

```text
tau = Kp * error
```

For:

```text
error ≈ 0.0065 rad
```

the controller generates approximately:

```text
500 * 0.0065 ≈ 3.25 Nm
```

which is consistent with the observed holding torque.

### Test 1 HY interpretation

The HY position response is not obviously bad.

It is:

- fast,
- lightly overshooting,
- stable.

However, the position plot alone hides important torque and velocity behavior.

---

# 7. Test 1 — HY Torque Response

At the start of the 0.20 rad step, the raw proportional demand is approximately:

```text
500 * 0.20 = 100 Nm
```

The measured raw PD demand was approximately:

```text
~103 Nm
```

while the physical actuator limit is:

```text
69.52 Nm
```

Therefore the controller clips briefly.

Approximate result:

```text
raw torque  ≈ 103 Nm
applied max = 69.52 Nm
```

The important detail is that the saturation is short.

Approximate saturation duration:

```text
~6 ms
```

Therefore this step is **not dominated by saturation for its entire duration**.

### HY torque conclusion

The hip torque limit affects the initial transient, but this particular 0.20 rad movement is not simply a torque-limited motion from start to finish.

This weakens the hypothesis that the hip limit alone is obviously too restrictive.

---

# 8. Test 1 — HY Velocity and Derivative Action

Peak HY velocity was approximately:

```text
6.25 rad/s
```

At this speed, the derivative term alone becomes:

```text
Kd * qdot
=
10 * 6.25
=
62.5 Nm
```

This is almost the entire hip torque limit:

```text
62.5 / 69.52 ≈ 90%
```

At one point in the motion, the proportional and derivative contributions approximately oppose each other:

```text
P contribution ≈ +65 Nm
D contribution ≈ -62.5 Nm
```

leaving only a small net torque.

### HY derivative conclusion

`Kd=10` is **not a weak correction** during dynamic motion.

At realistic joint speeds it can become a very large component of the commanded torque.

This was an important finding.

---

# 9. Test 1 — Knee Position Response

The knee received:

```text
-1.80 -> -1.55 rad
```

a 0.25 rad step.

The position trace looked nearly ideal:

- extremely fast,
- little visible overshoot,
- rapid settling.

If only the position graph were inspected, the controller could easily be judged excellent.

The velocity and torque plots showed a more complicated picture.

---

# 10. Test 1 — Knee Velocity Response

Peak knee velocity was approximately:

```text
+19 rad/s
```

during the forward step.

On return it reached approximately:

```text
-20.7 rad/s
```

This is dramatically faster than HY:

```text
HY peak ≈ 6.25 rad/s
knee    ≈ 19–21 rad/s
```

Yet both joints use exactly the same:

```text
Kp = 500
Kd = 10
```

### Knee velocity conclusion

The same gains produce very different closed-loop dynamics on different joints.

This is a direct experimental reason to question uniform gains.

---

# 11. Test 1 — Knee Torque Response

Initial proportional demand for the 0.25 rad step is:

```text
500 * 0.25 = 125 Nm
```

The knee limit is:

```text
164.15 Nm
```

Therefore:

```text
125 < 164.15
```

The knee did **not** need to saturate for the commanded step.

This is extremely useful.

The high knee speed is therefore not caused by insufficient torque authority.

It comes from the closed-loop dynamics of:

- effective knee inertia,
- Kp,
- Kd,
- explicit PD implementation,
- 500 Hz update rate.

### Knee torque conclusion

For this fixed-base test:

> The knee torque limit is not the limiting factor.

---

# 12. Test 1 — Aggressive Knee Torque Reversal

The knee response showed a particularly important explicit-PD transient.

Immediately after the command:

```text
tau ≈ +125 Nm
```

A few milliseconds later, after joint velocity rose sharply, the derivative term became very large.

At:

```text
qdot ≈ 19 rad/s
```

the derivative term is approximately:

```text
-Kd*qdot
=
-10*19
=
-190 Nm
```

which can exceed the proportional contribution.

The net controller demand therefore changes sign rapidly.

The observed pattern is qualitatively:

```text
large positive torque
-> large braking torque
-> positive correction
-> negative correction
-> settle
```

The position still looks smooth because these torque reversals happen quickly.

### Important lesson

A good-looking position response does not necessarily mean the actuator/controller is dynamically gentle.

For locomotion and contact, the torque/velocity behavior matters.

---

# 13. Test 1 — Dynamic Coupling

Small responses were visible in neighboring joints when the tested joint moved.

For example:

- HY motion caused a small knee disturbance,
- knee motion caused a small HY disturbance.

This is expected.

The leg is a coupled multibody system.

Joint controllers do not operate on isolated inertias.

This coupling is another reason why gains cannot be interpreted independently of the robot configuration.

---

# 14. Test 1 — Main Findings

The fixed-base step test established:

### Confirmed

1. HY tracks a 0.20 rad step quickly and stably.
2. HY briefly reaches its torque limit.
3. HY does not remain saturated for the whole motion.
4. `Kd=10` produces very large dynamic torque at higher joint speeds.
5. Knee reaches approximately 19–21 rad/s for a 0.25 rad command.
6. Knee does not hit its current torque limit in this test.
7. Same `500/10` gains produce substantially different dynamics on HY and knee.

### Strong implication

The uniform-gain assumption is questionable.

---

# 15. Test 2 — Free-Standing Downward Push Recovery

## 15.1 Purpose

The fixed-base step test studies commanded motion.

It does not answer whether the same controller is capable of supporting the full robot under gravity and rejecting disturbances.

The second test therefore asked:

1. Can the robot support its full body with the current gains?
2. Does a meaningful downward disturbance cause collapse?
3. Do the actuators approach their torque limits?
4. Is `Kd=10` sufficient to damp the recovery?
5. Are the high gains observed in the unloaded test actually necessary under load?

This test used the floating-base Spot on the ground.

---

# 16. Test 2 Setup

Controller:

```text
Kp = 500
Kd = 10
```

Limits:

```text
hips  = 69.52 Nm
knees = 164.15 Nm
```

Disturbance:

```text
250 N downward force
applied to torso
duration = 0.25 s
```

Timeline:

```text
0–2.00 s
settle under gravity

2.00–2.25 s
apply 250 N downward force

2.25–6.00 s
recover
```

The 250 N force is a substantial additional vertical disturbance relative to the weight of the approximately 50 kg robot model.

Recorded:

- base height,
- base vertical velocity,
- all HY joint positions,
- all knee joint positions,
- maximum HY/knee joint velocity,
- actuator torque utilization.

The test was observed in the MuJoCo viewer.

---

# 17. Test 2 — Initial Settling

The simulation starts with the body around:

```text
0.460 m
```

but this is not the loaded equilibrium.

Under gravity, Spot settles to approximately:

```text
0.434 m
```

Therefore the large initial movement immediately after simulation start should not be confused with the 250 N disturbance.

The correct pre-push standing reference is approximately:

```text
z_stand ≈ 0.434 m
```

not the original spawn height.

This is important when interpreting the base-height graph.

---

# 18. Test 2 — Actual Push Displacement

Immediately before the push:

```text
base height ≈ 0.434 m
```

During maximum compression:

```text
base height ≈ 0.422 m
```

Approximate extra displacement:

```text
0.434 - 0.422
≈ 0.012 m
≈ 12 mm
```

So a 250 N downward disturbance produced only approximately 12 mm of additional body compression.

### Interpretation

The current controller is not weak in static/load-bearing terms.

The virtual joint stiffness is substantial.

---

# 19. Test 2 — Base Vertical Velocity

During the actual push, the base reached approximately:

```text
peak downward velocity ≈ -0.15 m/s
```

During rebound:

```text
peak upward velocity ≈ +0.12 m/s
```

After the force was removed, several small oscillations occurred.

They decayed quickly.

There was no sustained bouncing.

### Interpretation

The loaded whole-body system is reasonably well damped under this disturbance.

There is no clear evidence from this test that `Kd=10` is too small for standing recovery.

---

# 20. Test 2 — Hip Actuator Utilization

During the actual disturbance, peak hip utilization was only approximately:

```text
~16%
```

of the available:

```text
69.52 Nm
```

Equivalent torque:

```text
0.16 * 69.52 ≈ 11 Nm
```

The hip joints therefore retained substantial unused torque authority.

### Hip-limit conclusion

This test gives no evidence that the current hip torque limit prevents ordinary push recovery.

---

# 21. Test 2 — Knee Actuator Utilization

Peak knee utilization during the push was approximately:

```text
~34%
```

of:

```text
164.15 Nm
```

Equivalent torque:

```text
0.34 * 164.15
≈ 56 Nm
```

Again, the knee was far from 100% utilization.

### Knee-limit conclusion

This test gives no evidence that the current knee torque limit is the immediate bottleneck during standing disturbance rejection.

---

# 22. Test 2 — Static Knee Position Error

At steady standing:

```text
desired knee ≈ -1.80 rad
actual knee  ≈ -1.86 rad
```

Approximate static error:

```text
0.06 rad
```

With:

```text
Kp = 500
```

this produces approximately:

```text
500 * 0.06 = 30 Nm
```

of static support torque.

This agrees well with the roughly 18–20% knee utilization visible during normal standing:

```text
0.19 * 164.15
≈ 31 Nm
```

This is an excellent internal consistency check.

---

# 23. What the Static Error Means

The current controller has no explicit gravity/feed-forward term.

At steady state:

```text
qdot = 0
```

so:

```text
tau = Kp * (q_des - q)
```

To generate support torque, a finite position error is required.

The knee must deflect away from its desired position to generate enough torque to hold the robot.

This explains why Spot settles below its nominal unloaded pose.

It also explains why reducing `Kp` earlier made the robot appear too weak.

---

# 24. Why Lower Kp Caused More Collapse

Suppose approximately 30 Nm is required from the knee for normal support.

With:

```text
Kp = 500
```

required error is:

```text
30 / 500
= 0.06 rad
```

With:

```text
Kp = 300
```

required error becomes:

```text
30 / 300
= 0.10 rad
```

With:

```text
Kp = 150
```

required error becomes:

```text
30 / 150
= 0.20 rad
```

Therefore lower Kp means:

> the joint must collapse farther before generating the same supporting torque.

This gives a better interpretation of the earlier push experiments.

The issue was not necessarily that the motor lacked available torque.

The virtual spring was becoming softer.

---

# 25. Test 2 — Loaded Joint Velocities

During the actual 250 N disturbance, approximate maximum velocities were much smaller than during the fixed-base commanded step.

Roughly:

```text
HY   ≈ 0.23 rad/s
knee ≈ 0.65 rad/s
```

Compare this with the fixed-base command test:

```text
HY   ≈ 6.25 rad/s
knee ≈ 19–21 rad/s
```

This is a very important difference.

The same controller behaves:

- calmly under a moderate load disturbance,
- very aggressively during direct position commands.

This is exactly why both tests were necessary.

---

# 26. Combined Interpretation of Both Tests

The tests should not be interpreted independently.

Together they show:

## Loaded standing behavior

`Kp=500, Kd=10` gives:

- strong body support,
- modest displacement under a 250 N disturbance,
- stable recovery,
- moderate actuator utilization,
- no evidence of torque-limit bottleneck.

## Commanded joint-motion behavior

The same gains give:

- fast HY motion,
- extremely fast knee motion,
- large derivative torque,
- rapid torque reversals,
- strongly different dynamics between joint types.

This produces the most important conclusion of the investigation so far:

> `Kp=500, Kd=10` can be a good standing controller without necessarily being a good locomotion servo for every joint.

---

# 27. What These Tests Say About the Torque Limits

The original actuator investigation considered two main possibilities:

```text
A. Kp/Kd are poorly selected
B. torque limits are too restrictive
```

The current evidence shifts the balance toward A.

## Evidence against the torque-limit hypothesis

### HY fixed-base test

The hip saturates, but only briefly.

The full movement is not saturation-dominated.

### Knee fixed-base test

The knee does not saturate.

Yet it still moves extremely aggressively.

### Push test

Peak utilization:

```text
hips  ~16%
knees ~34%
```

during the actual 250 N disturbance.

There is substantial unused authority.

### Current conclusion

There is currently **no strong experimental reason to increase the physical torque limits**.

This does not prove the limits are perfect.

It means:

> we should not change them without stronger evidence.

---

# 28. What These Tests Say About Kp

`Kp=500` produces good standing stiffness.

This is confirmed by:

- only ~12 mm extra compression under the push,
- strong recovery,
- no collapse.

Therefore reducing Kp uniformly across all joints would probably make standing softer.

However, the knee commanded-motion response suggests the knee may not need the same Kp as HY.

This motivates **joint-specific gains**, not simply "lower Kp everywhere."

---

# 29. What These Tests Say About Kd

Before testing, one possibility was:

> Kd=10 might be too small.

The fixed-base response does not support that simple explanation.

At:

```text
HY qdot ≈ 6.25 rad/s
```

the derivative torque is approximately:

```text
62.5 Nm
```

At:

```text
knee qdot ≈ 19 rad/s
```

the derivative demand is approximately:

```text
190 Nm
```

Therefore `Kd=10` is dynamically very significant.

In the standing push test, recovery is also reasonably damped.

### Current conclusion

There is no evidence that the next move should be "increase Kd."

The more likely problem is that the same Kd value is inappropriate across joints with different inertias and transmission characteristics.

---

# 30. Why Uniform Gains Are Now the Leading Suspect

The strongest observation from the step test is:

```text
same controller:
Kp=500
Kd=10
```

produces:

```text
HY peak velocity   ≈ 6.25 rad/s
knee peak velocity ≈ 19–21 rad/s
```

The knee is approximately three times faster for a similar-sized command.

This means the closed-loop bandwidth is very different.

The controller is not defining one consistent "servo behavior."

It is defining a different servo behavior on each joint because each joint has different:

- effective inertia,
- geometry,
- transmission,
- coupling.

This aligns with the external Boston Dynamics reference, where HX, HY and knee do not use the same gains.

---

# 31. Important Distinction: Plant vs Controller

The tests reinforce an important controls-engineering principle.

The controller:

```text
Kp = 500
Kd = 10
```

does not define the response by itself.

Closed-loop response depends on:

```text
controller
+
effective inertia
+
link geometry
+
transmission
+
gravity
+
contact
+
torque limit
+
simulation discretization
```

Therefore:

```text
same Kp/Kd
```

does not imply:

```text
same response
```

across joints.

---

# 32. Important Distinction: Standing vs Locomotion

The standing push test validates:

- static support,
- moderate disturbance rejection,
- loaded damping.

It does **not** validate:

- swing-leg bandwidth,
- high-speed target tracking,
- repeated contact transition behavior,
- foot-impact response,
- RL exploration dynamics.

Locomotion contains much more deliberate target motion than standing.

Therefore a controller selected only from push recovery can be inappropriate for locomotion.

This is likely what happened during the original gain selection.

---

# 33. Feed-Forward / Gravity Compensation Observation

The tests also exposed a limitation of the current pure PD structure.

Real joint control can include feed-forward/load torque.

Our current controller is only:

```text
tau =
Kp * position_error
- Kd * velocity
```

There is no:

```text
tau_ff
```

As a result, static gravity support requires position error.

For example:

```text
knee needs ~30 Nm
Kp = 500

required error ≈ 0.06 rad
```

A controller with gravity/load feed-forward could generate part of this support torque without requiring the joint to sag.

However:

> feed-forward should not be added yet.

It would change another major variable before the gain issue has been isolated.

---

# 34. What Is Confirmed

As of these two actuator tests, the following statements are supported.

## Confirmed Finding 1

The current torque limits are not heavily used during the 250 N standing disturbance.

## Confirmed Finding 2

The knee is not torque-limited during the fixed-base 0.25 rad step.

## Confirmed Finding 3

The same `Kp/Kd` produces substantially different HY and knee dynamics.

## Confirmed Finding 4

`Kp=500` provides useful load-bearing stiffness.

## Confirmed Finding 5

`Kd=10` is not dynamically insignificant; it produces large transient torque.

## Confirmed Finding 6

The standing system is reasonably damped under the tested disturbance.

## Confirmed Finding 7

A pure PD controller requires static joint error to create gravity-support torque.

---

# 35. What Is Not Yet Confirmed

The tests do **not** prove:

- that 69.52 Nm and 164.15 Nm are perfect real-world limits,
- that torque limits never influence PPO locomotion learning,
- that the knee Kp must be exactly a particular lower value,
- that the Boston Dynamics example gains should be copied,
- that feed-forward is required,
- that actuator gains alone explain every remaining gait issue.

The tests only narrow the problem.

---

# 36. Current Best Engineering Interpretation

Before testing, the main uncertainty was:

```text
bad gains
vs.
insufficient torque authority
```

After testing, the evidence favors:

```text
joint-specific closed-loop gain design
```

as the next thing to address.

The torque limits should remain unchanged for the next experiment unless new evidence shows they are restrictive.

---

# 37. Minimal Next Plan

The project requirement is to avoid unnecessary parameter sweeps.

Therefore the next step should be deliberately small.

## Step 1

Design **one** joint-specific alternative gain set.

The purpose is not to find final perfect gains.

The purpose is to reduce the knee's excessively aggressive commanded response while preserving standing robustness.

## Step 2

Run the same two visual tests again:

1. fixed-base HY/knee step response,
2. free-standing 250 N push recovery.

Acceptance questions:

```text
Does knee commanded velocity become more reasonable?

Does knee torque reversal become less aggressive?

Does HY remain well behaved?

Does Spot still support itself?

Does push recovery remain stable?

Do torque limits remain comfortably below 100%?
```

## Step 3

If those two tests remain good, train **one fresh RL policy**.

Do not add new gait rewards.

Keep:

- same task,
- same torque limits,
- same simple reward structure,
- same observations,
- same action scales.

Only change the explicit PD gains.

## Step 4

If locomotion is still poor after a physically sensible gain set, then perform **one** torque-authority A/B experiment.

This preserves causal clarity.

---

# 38. Why We Should Not Run a Large Gain Sweep

A large grid such as:

```text
Kp = 100, 200, 300, 400, 500, 600...
Kd = 1, 2, 5, 10, 15, 20...
```

would generate many simulations but relatively little engineering understanding.

The tests have already told us where the mismatch is:

> knee and HY do not have comparable closed-loop dynamics with identical gains.

The next design should therefore be informed by the observed response, not brute-force search.

---

# 39. Controls Engineering Reflection

These tests demonstrate the difference between:

```text
controller gains
```

and:

```text
closed-loop behavior
```

`Kp=500, Kd=10` are not meaningful in isolation.

The physically important result is:

```text
rise time
overshoot
joint speed
torque
saturation
disturbance rejection
settling
```

A controls engineer should tune toward desired dynamic behavior, not toward familiar-looking gain numbers.

---

# 40. Engineering Reflection on the Original Push-Test Tuning

The original decision to use approximately:

```text
Kp=500
Kd=10
```

was based partly on whether Spot could recover after being pushed down.

That test was valuable.

The new results show exactly what it accomplished:

> It selected a controller with enough virtual stiffness and damping to support the robot.

What it did **not** establish was:

> that the same gains give appropriate high-speed commanded dynamics for each joint.

Therefore the earlier method was not wrong.

It was incomplete.

That distinction is important.

---

# 41. Engineering Reflection on Graphs vs Scalar Numbers

The graphs exposed information that would have been difficult to notice from scalar summaries.

Examples:

- HY saturation lasted only briefly.
- Knee velocity was extremely high.
- Knee did not saturate.
- torque changed sign rapidly even while position tracking looked smooth.
- the body recovered with decaying oscillations.
- static knee error directly corresponded to static support torque.

This supports continuing to use visual time-series diagnostics for actuator work.

---

# 42. Current Decision

At this point the recommended decision is:

```text
KEEP:
69.52 Nm hip limit
164.15 Nm knee limit

DO NOT:
increase torque limits yet

INVESTIGATE NEXT:
joint-specific Kp/Kd
```

The immediate objective is:

> produce a more coherent explicit-PD joint servo before asking PPO to learn around it.

---

# 43. Current Working Hypothesis

The best-supported hypothesis now is:

> The current locomotion difficulty is more likely caused by the closed-loop behavior of the uniform explicit-PD gains than by insufficient static torque authority.

More specifically:

> `Kp=500, Kd=10` produces acceptable loaded standing behavior but mismatched commanded dynamics across HY and knee.

This is now the primary actuator-design problem to solve.

---

# 44. Test Artifacts

## Fixed-base step-response artifacts

Expected directory:

```text
reports/actuator_characterization/current_kp500_kd10/
```

Files:

```text
visual_step_response.csv
fl_hy_position.png
fl_hy_velocity.png
fl_hy_torque.png
fl_kn_position.png
fl_kn_velocity.png
fl_kn_torque.png
```

## Free-standing push-recovery artifacts

Expected directory:

```text
reports/actuator_characterization/push_recovery_kp500_kd10/
```

Files:

```text
push_recovery.csv
base_height.png
base_vertical_velocity.png
hy_positions.png
knee_positions.png
joint_velocity_envelope.png
actuator_utilization.png
```

These should be kept with the project as the baseline actuator characterization before the next gain change.

---

# 45. Final Summary

The two tests gave complementary information.

## Fixed-base step test

Showed:

```text
HY:
fast, stable, brief saturation

Knee:
very fast,
large velocity,
no saturation,
aggressive derivative/torque reversal
```

## Free-standing push test

Showed:

```text
good body support
~12 mm additional compression
stable recovery
hip utilization ~16%
knee utilization ~34%
no evidence of torque-authority bottleneck
```

Together they imply:

```text
Standing controller:
good enough

Torque limits:
not currently the leading suspect

Uniform joint gains:
strongly questionable

Next action:
one joint-specific gain redesign
```

The important engineering shift is that the problem is no longer being treated as:

> "PPO needs a better reward."

It is being treated as:

> "The low-level actuator/controller must first have physically coherent closed-loop dynamics."

That is the correct level to solve before continuing locomotion reward tuning.
