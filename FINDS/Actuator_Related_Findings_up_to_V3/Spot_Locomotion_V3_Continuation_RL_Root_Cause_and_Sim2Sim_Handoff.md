# Spot Locomotion V3 — RL Integration, Actuator Root-Cause Isolation, Final Servo Approval, and Sim-to-Sim Handoff

**Project:** `spot_locomotion_v2`  
**Repository:** `~/Projects/spot_locomotion_v2`  
**Stack:** MjLab 1.5.3 + MuJoCo 3.10.0  
**Date:** 2026-09-07  
**Status:** V3 rigid-body model and low-level servo are now approved for locomotion. The next phase is MuJoCo → ROS 2 Jazzy + Gazebo Harmonic sim-to-sim transfer.

---

# 1. Purpose and Scope of This Continuation

This document continues the earlier:

```text
Spot_Locomotion_V3_Comprehensive_Engineering_Log.md
```

That previous document ended with a **provisional** V3 actuator baseline:

```text
BuiltinPositionActuatorCfg
Kp = 300
Kd = 4
HX/HY effort limit = 45 N·m
knee effort limit  = 115 N·m
```

At that point the rigid-body model and deterministic actuator tests had passed, but PPO had not yet been used to validate whether that servo was actually suitable for learned locomotion.

The purpose of this continuation is to record everything learned after that point:

1. integration of the frozen V3 plant into the real MjLab velocity environment;
2. the zero-action and action-pipeline integration tests;
3. the first fresh V3 PPO run and the return of the old tapping/asymmetric gait;
4. the decision **not** to restart the reward-tuning loop;
5. re-examination of the old successful MjLab 1.1 and later V2 native-actuator A/B evidence;
6. controlled actuator experiments separating:
   - actuator implementation,
   - torque authority,
   - proportional stiffness,
   - derivative damping;
7. quantitative TensorBoard comparisons across those runs;
8. identification of insufficient damping as the main cause of the locomotion pathology;
9. approval of the final V3 servo:
   - `Kp = 300`
   - `Kd = 40`
   - hips `±45 N·m`
   - knees `±115 N·m`;
10. the resulting ROS 2 / Gazebo deployment architecture and validation plan.

The most important correction to the previous V3 document is:

> `Kp=300, Kd=4` was a valid stable deterministic servo, but it was **not** a suitable locomotion servo.  
> The approved V3 locomotion servo is now `Kp=300, Kd=40`.

---

# 2. Starting Point From the Previous V3 Log

The previous phase had successfully frozen the V3 mechanical plant.

## 2.1 V3 rigid-body model

```text
Total articulated mass: 31.5999994 kg
```

The model uses:

- Isaac-derived masses;
- Isaac-derived COM positions;
- Isaac-derived principal inertias;
- principal-axis orientations from Isaac;
- the existing project geometry, kinematics, collisions, sensors, and foot geoms.

The original V2 home pose was retained:

```text
HX = 0.0 rad
HY = 1.04 rad
KN = -1.80 rad
```

Joint order:

```text
fl_hx, fl_hy, fl_kn,
fr_hx, fr_hy, fr_kn,
hl_hx, hl_hy, hl_kn,
hr_hx, hr_hy, hr_kn
```

## 2.2 Timing and action semantics

```text
physics frequency = 500 Hz
physics dt        = 0.002 s

policy frequency  = 50 Hz
policy dt         = 0.020 s

decimation        = 10
```

Policy action mapping:

```text
q_ref = q_home + action_scale * action
```

with:

```text
HX scale   = 0.30 rad
HY scale   = 0.40 rad
knee scale = 0.50 rad
```

The target is zero-order held between policy updates.

## 2.3 MuJoCo settings

The intended environment settings remain:

```text
integrator    = implicitfast
cone          = elliptic
impratio      = 100
solver        = newton
iterations    = 100
ls_iterations = 50
ccd_iterations = 50
```

## 2.4 Provisional actuator at that time

The previous document ended with:

```text
BuiltinPositionActuatorCfg

Kp = 300
Kd = 4

hips  = ±45 N·m
knees = ±115 N·m
```

This choice had passed:

- visual standing;
- 250 N push recovery;
- fixed-base HY step;
- fixed-base knee step;
- 50 Hz ZOH target tracking;
- visual ZOH verification.

The remaining question was whether PPO would discover clean locomotion on that plant.

---

# 3. V3 Integration Into the Actual MjLab RL Environment

A dedicated V3 flat-velocity task was introduced while preserving the existing V2 task.

Conceptually:

```text
flat_env_cfg.py
    existing V2 task

flat_env_cfg_v3.py
    V3 plant integration
```

The first V3 task deliberately reused the current V2 task definition and changed only plant-specific pieces.

The goal was a clean causal experiment.

Preserved:

```text
task structure
observations
policy action type
action scales
commands
rewards
terminations
contact sensors
diagnostic metrics
500/50 Hz timing
MuJoCo solver settings
PPO architecture
```

Changed:

```text
V2 robot entity
    ↓
V3 robot entity

V2 actuator-limit diagnostics
    ↓
V3 actuator-limit diagnostics
```

The new task was registered separately as:

```text
Mjlab-Velocity-Flat-Spot-V3
```

A separate PPO experiment namespace was also used so V2 and V3 logs would not be mixed.

---

# 4. Real MjLab Zero-Action Smoke Test

Before training, the actual MjLab environment was launched with:

```text
1 environment
zero policy action
native viewer
```

This was more important than the previous direct-MuJoCo standing test because it exercised the real runtime path:

```text
ManagerBasedRlEnv
    ↓
JointPositionActionCfg
    ↓
default/home offset
    ↓
V3 BuiltinPositionActuator
    ↓
MuJoCo
```

## Result

The user reported:

> Zero action looks good.

The robot:

- spawned correctly;
- settled into the expected standing posture;
- did not collapse;
- did not jitter violently;
- did not rest on upper-leg collision geometry;
- did not repeatedly reset.

### Conclusion

The V3 plant was mechanically valid inside the actual MjLab task.

---

# 5. Actual MjLab Action-Pipeline Audit

A dedicated deterministic action-pipeline test was then run through the real environment.

The audit checked:

1. the 12-joint ordering;
2. the resolved action scales;
3. the default/home offsets;
4. the exact affine action mapping;
5. a small physical knee command through `env.step()`.

The expected mapping was:

```text
processed target
=
raw policy action * scale
+
home position
```

The test confirmed exactly:

```text
HX   scale = 0.30
HY   scale = 0.40
knee scale = 0.50
```

and the correct home offsets for all 12 joints.

A unique per-joint probe produced:

```text
maximum mapping error = 0
```

to numerical precision.

A physical knee test used:

```text
raw knee action = +0.10
knee scale      = 0.50
```

therefore:

```text
Δq_ref = +0.05 rad
```

and the target became:

```text
-1.80 + 0.05 = -1.75 rad
```

The robot responded correctly and returned to the previous standing equilibrium after the action was removed.

Peak actuator torque was approximately:

```text
33.47 N·m
```

and no non-finite state occurred.

## Conclusion

The real MjLab action path was correct.

This ruled out:

```text
wrong joint order
wrong action scale
wrong home offset
broken JointPositionAction mapping
```

as explanations for later locomotion problems.

---

# 6. First Fresh V3 PPO Run With `300/4, 45/115`

With the integrated environment passing, a fresh PPO policy was trained from scratch.

Plant:

```text
V3 rigid body
BuiltinPositionActuatorCfg
Kp = 300
Kd = 4
hips  = ±45 N·m
knees = ±115 N·m
```

No gait reward was added.

The purpose was to observe what locomotion PPO naturally discovered.

---

# 7. The Old Tapping / Support-Asymmetry Pathology Returned

The result was immediately important.

Visually, the same family of behavior seen in V2 returned:

- rapid foot tapping / stuttering;
- one diagonal pair doing most of the real support;
- the opposite diagonal pair participating poorly;
- one front foot often spending disproportionately long periods airborne;
- motion that was stable and reward-efficient but mechanically undesirable.

This was a major negative result because the entire rigid-body plant had changed from V2 to V3.

The problem had survived:

```text
50.34 kg V2 model
    ↓
31.6 kg Isaac-derived V3 model
```

and survived the move from explicit `IdealPdActuator` to native MuJoCo position actuation.

This meant the diagnosis had to become more specific.

---

# 8. TensorBoard Confirmation of the First `300/4` Failure

The first V3 PPO baseline event file ended around iteration 283.

Representative late values:

```text
mean reward                    90.64
mean episode length            991.36

foot contact switch rate       9.46 Hz
mean mechanical power          138.08 W
base vertical velocity RMS     0.116 m/s
slip velocity mean             0.253 m/s
landing force mean             130.2 N
mean action acceleration       0.271

actuator saturation fraction   0.47%
```

Per-foot timing exposed the asymmetry clearly:

```text
air time:
FL = 0.024 s
FR = 0.119 s
HL = 0.169 s
HR = 0.025 s

stance time:
FL = 0.171 s
FR = 0.064 s
HL = 0.063 s
HR = 0.220 s
```

The exact diagonal support fractions were:

```text
FL + HR only ≈ 29.38%
FR + HL only ≈ 0.008%
```

The second diagonal pair had essentially disappeared as a true support pair.

This was not merely a subjective visual impression.

It was quantitatively the same structured failure mode seen in previous V2 training.

---

# 9. Why We Did Not Return to Reward Tuning

At first glance, the failed V3 PPO run could have been interpreted as another reward exploit.

However, the project had already spent a large amount of time trying:

- air-time rewards;
- diagonal gait rewards;
- faithful Isaac timing rewards;
- support-balance penalties;
- foot-clearance changes;
- excessive-air-time guardrails;
- slip changes.

Those experiments repeatedly changed the **shape** of the exploit without eliminating the underlying actuator-dependent locomotion problem.

The earlier V2 retrospective had already reached the principle:

> Stop solving every visible gait problem by adding another reward.

Therefore the V3 failure was not treated as a reason to restart reward engineering.

Instead, the earlier successful actuator evidence was re-opened.

---

# 10. Re-Examining the Old Successful MjLab 1.1 Project

The old repository:

```text
anymal_c_velocity
```

was inspected directly.

It was pinned to:

```text
MjLab 1.1.0
```

and used native MuJoCo position actuators loaded from XML.

The old Spot MJCF used approximately:

```xml
<joint actuatorfrcrange="-1000 1000"/>
<position kp="500" kv="40"/>
```

with:

```text
implicitfast
elliptic
impratio = 100
```

The successful old servo was therefore:

```text
native MuJoCo position servo
Kp = 500
Kd = 40
very high torque authority
```

The old project learned clean locomotion without the V2/V3 tapping pathology.

---

# 11. The More Important Historical Evidence: MjLab 1.5.3 Native A/B

A stronger piece of evidence already existed in the V2 project.

A clean diagnostic environment had been created in MjLab 1.5.3 that reused the normal V2 task unchanged and swapped only the actuator to:

```text
BuiltinPositionActuatorCfg
Kp = 500
Kd = 40
effort limit = ±1000 N·m
```

That A/B run restored natural locomotion even in MjLab 1.5.3.

This ruled out a simplistic explanation such as:

```text
MjLab 1.1 works
MjLab 1.5.3 is broken
```

It also showed that the actuator did not have to be physically authored in XML.

MjLab 1.5.3 `BuiltinPositionActuatorCfg` creates the same native MuJoCo position-actuator family programmatically.

Therefore the real distinction was not:

```text
XML actuator
vs
Python actuator configuration
```

The important variables were now:

```text
servo stiffness
servo damping
torque authority
interaction with the plant
```

---

# 12. Critical Reinterpretation of the V3 Baseline

The current V3 already used:

```text
BuiltinPositionActuatorCfg
```

so both:

```text
known-good legacy A/B
```

and:

```text
bad V3 300/4 baseline
```

were using native MuJoCo position actuation.

The comparison became:

```text
KNOWN GOOD
Kp = 500
Kd = 40
limit = 1000

versus

BAD V3 BASELINE
Kp = 300
Kd = 4
limits = 45 / 115
```

The key question was therefore:

> Which part of the parameterization made the native servo easy for PPO to control?

A controlled experiment series was started.

---

# 13. Experiment 1 — `Kp=500, Kd=40, ±1000 N·m`

The V3 model was trained with:

```text
Kp = 500
Kd = 40

hip limit  = 1000 N·m
knee limit = 1000 N·m
```

Everything else remained the same V3 task.

## Visual result

The user reported:

> The robot walked very nicely.

The previous tapping and one-diagonal support exploit disappeared.

## Representative late TensorBoard values

```text
mean reward                    109.30
mean episode length            1000

contact switch rate            4.82 Hz
mechanical power               76.15 W
base vertical velocity RMS     0.0655 m/s
slip velocity mean             0.1426 m/s
landing force mean             47.1 N
mean action acceleration       0.172

actuator saturation fraction   0%
max utilization                ~13.4% of the 1000 N·m limit
```

Per-foot timing:

```text
air time:
FL = 0.098 s
FR = 0.165 s
HL = 0.142 s
HR = 0.098 s

stance time:
FL = 0.358 s
FR = 0.262 s
HL = 0.295 s
HR = 0.373 s
```

Diagonal support:

```text
FL + HR only ≈ 27.2%
FR + HL only ≈ 19.4%
```

Both support diagonals now participated meaningfully.

### Immediate conclusion

The V3 rigid-body model was capable of excellent locomotion.

The task and reward were also capable of producing it.

The failure was inside the low-level servo parameterization.

---

# 14. Experiment 2 — `Kp=500, Kd=40, 45/115 N·m`

The next experiment changed only torque authority:

```text
Kp = 500
Kd = 40

hip limit  = 45 N·m
knee limit = 115 N·m
```

These were the V3 physically motivated limits.

## Visual result

Again:

> The robot walked very nicely.

The low limits did not reintroduce the tapping pathology.

## Representative late TensorBoard values

```text
mean reward                    108.21
mean episode length            1000

contact switch rate            4.95 Hz
mechanical power               88.91 W
base vertical velocity RMS     0.0730 m/s
slip velocity mean             0.1426 m/s
landing force mean             57.6 N
mean action acceleration       0.169

actuator saturation fraction   0.97%
```

Per-foot timing:

```text
air time:
FL = 0.157 s
FR = 0.127 s
HL = 0.124 s
HR = 0.139 s

stance time:
FL = 0.277 s
FR = 0.312 s
HL = 0.334 s
HR = 0.301 s
```

Diagonal support:

```text
FL + HR only ≈ 24.7%
FR + HL only ≈ 31.7%
```

Again, both diagonal pairs participated strongly.

---

# 15. Important Early-Saturation Observation in the `500/40, 45/115` Run

The lower-limit `500/40` policy experienced substantial saturation during early PPO exploration:

```text
iteration 0:    ~32.4%
iteration 50:   ~26.6%
iteration 100:  ~20.3%
iteration 150:  ~10.1%
iteration 200:  ~3.25%
iteration 250:  ~1.51%
iteration 300:  ~1.20%
late:           ~0.97%
```

Despite that early saturation, PPO still discovered excellent locomotion.

This was a decisive result.

It showed:

> High torque authority is not required for gait discovery.

The policy can learn a good gait even when early random exploration frequently hits the physical torque limits.

As the policy improves, it naturally leaves saturation.

---

# 16. Torque-Authority Hypothesis Was Effectively Rejected

The comparison:

```text
500/40 + 1000/1000
    → good

500/40 + 45/115
    → good
```

settled an important question.

The V3 physically motivated effort limits:

```text
hips  ±45 N·m
knees ±115 N·m
```

are sufficient for flat-terrain locomotion.

The old `±1000 N·m` authority is therefore not necessary.

This is especially important for deployment because the final model no longer needs unrealistic actuator authority merely to train successfully.

---

# 17. Experiment 3 — `Kp=300, Kd=4, ±1000 N·m`

The next test went in the opposite direction.

Keep the bad low-gain servo:

```text
Kp = 300
Kd = 4
```

but remove torque-authority concerns:

```text
hip limit  = 1000 N·m
knee limit = 1000 N·m
```

## Visual result

The user reported:

- tapping became obvious;
- locomotion looked worse;
- one front leg sometimes appeared to collapse or stop contributing normally.

## Representative late TensorBoard values

```text
mean reward                    83.02
mean episode length            992.6

contact switch rate            7.08 Hz
mechanical power               137.02 W
base vertical velocity RMS     0.118 m/s
slip velocity mean             0.266 m/s
landing force mean             123.6 N
mean action acceleration       0.249

actuator saturation fraction   0%
max actuator utilization       ~15.4%
```

Per-foot timing included an extreme failure:

```text
HL air time    ≈ 1.10 s
HL stance time ≈ 0.026 s
```

Diagonal support:

```text
FL + HR only ≈ 13.5%
FR + HL only ≈ 0.039%
```

The second diagonal support pattern again effectively disappeared.

### Conclusion

The low-gain failure was **not caused by torque saturation**.

The controller had enormous unused authority and still learned the bad gait.

---

# 18. Experiment 4 — `Kp=300, Kd=4, ±500 N·m`

A second high-authority test used:

```text
Kp = 300
Kd = 4

hip limit  = 500 N·m
knee limit = 500 N·m
```

## Visual result

The same bad behavior remained.

## Representative late TensorBoard values

```text
mean reward                    75.96
mean episode length            965.3

contact switch rate            10.28 Hz
mechanical power               153.16 W
base vertical velocity RMS     0.129 m/s
slip velocity mean             0.314 m/s
landing force mean             116.2 N
mean action acceleration       0.279

actuator saturation fraction   0%
max actuator utilization       ~27.5%
```

Diagonal support:

```text
FL + HR only ≈ 24.0%
FR + HL only ≈ 0.116%
```

Again, one diagonal support phase had essentially disappeared.

### Conclusion

Changing:

```text
115 → 500 → 1000 N·m
```

did not rescue `300/4`.

The available torque ceiling was not the governing problem.

---

# 19. What the Four Main A/B Runs Established

At this stage the evidence was:

| Kp | Kd | Effort limits | Result |
|---:|---:|---|---|
| 300 | 4 | 45 / 115 | Bad tapping / support asymmetry |
| 300 | 4 | 500 / 500 | Bad tapping / support asymmetry |
| 300 | 4 | 1000 / 1000 | Bad tapping / leg participation failure |
| 500 | 40 | 45 / 115 | Good natural walking |
| 500 | 40 | 1000 / 1000 | Good natural walking |

This strongly separated:

```text
torque authority
```

from:

```text
servo dynamics
```

The torque-limit question was now essentially solved.

The next remaining question was:

> Is the critical difference Kp, Kd, or the 500/40 combination?

---

# 20. Experiment 5 — `Kp=300, Kd=40, 45/115 N·m`

A final narrow gain-isolation experiment was run:

```text
Kp = 300
Kd = 40

hip limit  = 45 N·m
knee limit = 115 N·m
```

This was especially informative because it changed only:

```text
Kd: 4 → 40
```

relative to the original bad V3 baseline.

`Kp` remained 300.

The physically motivated effort limits remained unchanged.

---

# 21. `300/40` Result — Excellent Natural Locomotion

The user reported:

> `Kp=300, Kd=40` was very good.

The TensorBoard log strongly confirmed that visual assessment.

Representative late metrics:

```text
mean reward                    109.99
mean episode length            1000

contact switch rate            3.78 Hz
mechanical power               67.21 W
base vertical velocity RMS     0.0595 m/s
slip velocity mean             0.1197 m/s
landing force mean             33.9 N
mean action acceleration       0.209

actuator saturation fraction   0.286%
```

Per-foot timing became extremely balanced:

```text
air time:
FL = 0.176 s
FR = 0.168 s
HL = 0.157 s
HR = 0.161 s

stance time:
FL = 0.397 s
FR = 0.408 s
HL = 0.414 s
HR = 0.412 s
```

Diagonal support:

```text
FL + HR only ≈ 25.03%
FR + HL only ≈ 25.72%
```

This is almost perfectly balanced diagonal participation.

The robot also achieved:

```text
flight fraction ≈ 0.20%
all-four contact ≈ 25.6%
```

and maintained full episode survival.

---

# 22. Direct `300/4` vs `300/40` Comparison

This is the cleanest root-cause experiment in the entire V3 investigation.

Everything important stayed the same:

```text
same V3 rigid body
same MjLab version
same MuJoCo version
same native position actuator class
same Kp = 300
same torque limits = 45 / 115
same action scales
same observations
same rewards
same commands
same PPO structure
same 500 / 50 Hz timing
same ZOH target semantics
```

Only:

```text
Kd = 4
```

became:

```text
Kd = 40
```

The behavior changed from:

```text
rapid tapping
one diagonal pair nearly absent
poor foot participation
high slip
hard landings
higher power
```

to:

```text
natural stepping
balanced diagonal support
all four feet participating
lower slip
soft landings
lower power
lower contact-switch rate
```

Therefore the strongest supported conclusion is:

> **Insufficient low-level joint damping was the primary cause of the V3 tapping / asymmetric support pathology.**

---

# 23. What the `500/40` and `300/40` Runs Say About Kp

Both:

```text
500 / 40
```

and:

```text
300 / 40
```

learned good locomotion.

Therefore:

> `Kp=500` is not required to solve the gait problem.

The project now has direct evidence that:

```text
Kp = 300
```

is sufficient when paired with appropriate damping.

This is useful because 300 gives a more compliant position servo than 500 while still producing excellent locomotion.

The final `300/40` run also compared favorably with the `500/40` runs in:

- switch rate;
- mechanical power;
- slip;
- landing force;
- diagonal balance.

Because PPO is stochastic, small numerical differences between successful runs should not be over-interpreted.

The robust conclusion is simply:

```text
300/40 works very well
500/40 works very well
300/4 does not
```

---

# 24. Why the Original `Kd=4` Looked Reasonable in Deterministic Tests

This result forces an important reinterpretation of the earlier actuator-validation methodology.

`Kd=4` had been chosen using approximate second-order critical-damping reasoning based on apparent joint inertia.

For example:

```text
Kd_crit = 2 sqrt(Kp J)
```

gave rough values around:

```text
HX   ~6.5
HY   ~8.6
knee ~4.0
```

The knee fixed-base step at `300/4` looked particularly good.

The servo also:

- stood correctly;
- survived a 250 N push;
- tracked a small 2 Hz ZOH target;
- showed no numerical instability.

All of those tests were valid.

The mistake was the conclusion drawn from them.

They proved:

> `300/4` is a stable servo near the tested deterministic operating conditions.

They did **not** prove:

> `300/4` is a suitable low-level plant for contact-rich learned quadruped locomotion.

---

# 25. Why Locomotion Requires More Damping Than the Isolated-Joint Estimate Suggested

Quadruped locomotion is not a collection of isolated second-order joints.

During walking, each actuator interacts with:

- rapidly changing reflected inertia;
- rigid-body coupling through the torso;
- stance constraints;
- foot impacts;
- swing/stance transitions;
- contact solver impulses;
- body load transfer;
- abrupt 50 Hz policy target changes;
- simultaneous motion of all 12 joints;
- policy exploration noise.

A fixed-base apparent inertia at one pose cannot capture all of this.

The effective locomotion plant is closer to:

```text
rigid-body mechanics
+
servo dynamics
+
contact constraints
+
terrain contact solver
+
policy update semantics
```

The `Kd=40` result demonstrates that a damping level that appears high from an isolated-joint calculation can still produce a much calmer **whole locomotion system**.

---

# 26. Higher Damping Did Not Make Locomotion More Energetic

One might expect a tenfold increase in derivative gain to make the system inefficient or overly resistant to motion.

The experiments showed the opposite.

Approximate late mechanical power:

```text
300/4 high-limit runs:
~137–153 W

first 300/4 V3 baseline:
~138 W

500/40 good runs:
~76–89 W

300/40:
~67 W
```

Approximate slip:

```text
300/4:
~0.25–0.31 m/s

500/40:
~0.143 m/s

300/40:
~0.120 m/s
```

Approximate landing force:

```text
300/4:
~116–130 N

500/40:
~47–58 N

300/40:
~34 N
```

Therefore stronger damping did not simply dissipate more energy.

It changed the optimization landscape so PPO learned:

- smoother leg trajectories;
- fewer rapid contact transitions;
- softer landing behavior;
- less slip;
- more coordinated support;
- lower mechanical power.

This is one of the most important project findings.

---

# 27. The Reward Function Was Not the Root Cause of This Repeated Pathology

The good and bad servo experiments used the same task/reward structure.

With:

```text
300/4
```

the policy found the tapping exploit.

With:

```text
300/40
```

the same optimization problem produced natural locomotion.

Therefore, for this specific recurring pathology:

> The reward function was not the root cause.

This does not mean reward design never matters.

It means that adding more gait-shaping terms would have been compensating for a low-level plant problem rather than solving the underlying cause.

This validates the earlier decision to stop reward stacking.

---

# 28. Torque Saturation Was Also Not the Root Cause

The evidence is now unusually strong.

## Good low-limit run

```text
500/40 + 45/115
```

experienced major saturation early in training and still converged to excellent walking.

## Bad high-limit runs

```text
300/4 + 500
300/4 + 1000
```

had essentially zero saturation and still converged to bad locomotion.

Therefore:

```text
good gait
≠
requires zero saturation

bad gait
≠
caused by saturation
```

The critical variable was the closed-loop servo dynamics, especially damping.

---

# 29. Final Root-Cause Matrix

As of this phase:

| Hypothesis | Status |
|---|---|
| V3 rigid-body model causes tapping | Strongly rejected |
| MjLab 1.5.3 causes tapping | Rejected |
| XML actuator vs programmatic actuator causes tapping | Rejected |
| Joint/action ordering bug | Rejected |
| Action scaling bug | Rejected |
| Reward shaping is the main root cause | Strongly rejected for this pathology |
| 45/115 N·m limits are too low | Rejected |
| PPO needs ±1000 N·m to discover walking | Rejected |
| Native MuJoCo position actuation can learn walking | Confirmed |
| `Kp=500` is required | Rejected |
| `Kd=4` is sufficient for locomotion | Rejected |
| Insufficient joint damping drives the tapping/asymmetry | Strongly confirmed |
| `Kp=300, Kd=40, 45/115` is a viable V3 servo | Confirmed |

---

# 30. Final Approved V3 Low-Level Servo

Gain testing is now considered complete.

The approved baseline is:

```python
V3_KP = 300.0
V3_KD = 40.0

V3_HIP_EFFORT_LIMIT = 45.0
V3_KNEE_EFFORT_LIMIT = 115.0
```

Actuator:

```text
MjLab BuiltinPositionActuatorCfg
```

Reference semantics:

```text
position reference
qdot_ref implicitly zero
zero-order hold between 50 Hz policy updates
```

No further gain sweep is planned.

No `Kd=20` midpoint test is required.

The engineering objective is no longer to find the minimum damping that still works.

The current objective is to move forward with a validated servo that produces clean locomotion.

---

# 31. Final Approved V3 Plant Specification

## Robot

```text
Spot-like quadruped
12 actuated joints
31.5999994 kg total articulated mass
Isaac-derived rigid-body inertials
existing project geometry/collision scaffold
```

## Home pose

```text
HX = 0.0 rad
HY = 1.04 rad
KN = -1.80 rad
```

## Action interface

```text
12 normalized policy outputs

q_ref =
q_home
+
action_scale * action
```

Scales:

```text
HX   = 0.30 rad
HY   = 0.40 rad
knee = 0.50 rad
```

## Low-level actuator

```text
BuiltinPositionActuatorCfg

Kp = 300
Kd = 40

HX/HY effort limit = 45 N·m
knee effort limit  = 115 N·m
```

## Timing

```text
MuJoCo physics = 500 Hz
policy         = 50 Hz
decimation     = 10
ZOH q_ref      = 20 ms
```

## MuJoCo

```text
MuJoCo 3.10.0
integrator = implicitfast
cone       = elliptic
impratio   = 100

solver        = newton
iterations    = 100
ls_iterations = 50
ccd_iterations = 50
```

This specification should now be treated as the frozen V3 MuJoCo locomotion plant unless sim-to-sim evidence exposes a concrete issue.

---

# 32. What We Should Not Reopen Before Sim-to-Sim

The following should now stay fixed:

```text
V3 rigid-body inertials
home pose
joint order
action scales
policy rate
physics rate
ZOH action semantics
BuiltinPosition actuator family
Kp = 300
Kd = 40
hip limit = 45 N·m
knee limit = 115 N·m
simple reward structure
```

Do not restart:

- reward stacking;
- arbitrary gait rewards;
- gain sweeps;
- torque-limit sweeps;
- interpolation experiments;
- arbitrary armature/friction additions;
- DC motor modeling;
- actuator delay modeling;

unless Gazebo transfer or later hardware evidence gives a concrete reason.

---

# 33. Implication for Sim-to-Sim Deployment

The project can now move into ROS 2 Jazzy + Gazebo Harmonic with a well-defined control contract.

The high-level policy remains:

```text
50 Hz
12 joint-position offsets
```

The desired deployment architecture is:

```text
/cmd_vel
    ↓

RL policy @ 50 Hz
    ↓
q_ref[12]
    ↓
hold latest q_ref for 20 ms
    ↓

low-level joint servo @ 500 Hz
    ↓
τ = 300(q_ref - q) - 40 qdot
    ↓
torque clamp
hips  ±45 N·m
knees ±115 N·m
    ↓

ros2_control effort interface
    ↓
gz_ros2_control
    ↓
Gazebo Harmonic
```

This is the natural deployment-side interpretation of the approved MuJoCo servo.

---

# 34. Why Gazebo Should Use an Effort Interface

The policy should not simply send position targets to Gazebo's simulator-native position command interface.

Doing so would make Gazebo's own hidden position-controller behavior part of the deployed plant.

Training would then use:

```text
MuJoCo native position servo
```

while deployment would use:

```text
Gazebo position interface behavior
```

with no guarantee of equivalence.

A cleaner boundary is:

```text
our controller computes torque
    ↓
effort interface
    ↓
Gazebo physics
```

This gives us direct control over:

```text
Kp
Kd
torque limits
controller rate
ZOH semantics
```

and is conceptually portable to later hardware.

---

# 35. Initial ROS 2 Low-Level Controller Candidate

For the first Gazebo implementation, a standard ROS 2 Jazzy controller is preferred over custom C++ unless a standard controller proves insufficient.

A strong initial candidate is:

```text
pid_controller/PidController
```

configured for:

```text
position reference
position + velocity state
effort command
```

with:

```text
P = 300
I = 0
D = 40

qdot_ref = 0
```

and output limits:

```text
HX/HY = ±45 N·m
KN    = ±115 N·m
```

Conceptually:

```text
τ =
300(q_ref - q)
+
40(0 - qdot)
```

This matches the desired deployment law.

---

# 36. Important MuJoCo-vs-Gazebo Numerical Caveat

The same gain numbers do not guarantee identical dynamics.

In MuJoCo, the current `BuiltinPositionActuatorCfg` creates a native position servo whose stiffness and damping participate in MuJoCo's native actuator/integration machinery under:

```text
implicitfast
```

In Gazebo, an effort controller computing:

```text
τ = 300e - 40qdot
```

will apply an explicit torque command through `ros2_control`.

The continuous-time control law is nominally equivalent, but the numerical implementation is not identical.

Therefore the sim-to-sim target is **not**:

> same Kp and Kd numbers.

The target is:

> sufficiently similar closed-loop joint and whole-body behavior.

---

# 37. Deterministic MuJoCo → Gazebo Validation Before Policy Transfer

The first Gazebo milestone should not be PPO deployment.

The exact deterministic tests already used in MuJoCo should be recreated in Gazebo.

Recommended order:

## Test 1 — Zero-action standing

Compare:

```text
settled base height
roll / pitch
joint sag
standing torque
non-foot contact
```

## Test 2 — Fixed-base HY step

Compare:

```text
rise time
overshoot
settling time
peak velocity
peak torque
```

## Test 3 — Fixed-base knee step

Compare the same quantities.

## Test 4 — 250 N downward torso push

Compare:

```text
compression
vertical velocity
rebound
settling
joint torques
```

## Test 5 — 50 Hz ZOH position-reference sequence

Replay the same 50 Hz reference pattern and compare:

```text
tracking RMS
phase lag
peak velocity
peak torque
cross-joint coupling
```

Only after these tests are acceptably close should the learned policy be loaded in Gazebo.

---

# 38. Initial Gazebo Timing Target

To preserve parity initially:

```text
Gazebo physics      = 500 Hz
ros2_control update = 500 Hz
RL policy           = 50 Hz
```

Therefore:

```text
physics dt     = 0.002 s
controller dt  = 0.002 s
policy dt      = 0.020 s
```

A faster Gazebo physics rate can be tested later if contact stability needs it, but the first transfer should minimize timing differences.

---

# 39. V3 Mechanical Model for Gazebo

The Gazebo robot description should reproduce the frozen V3 model as closely as practical:

```text
31.6 kg total mass
same per-link masses
same COM positions
same inertias
same joint axes
same joint limits
same home pose
same relevant collision geometry
same foot radius / contact geometry
```

The goal is not perfect engine identity.

The goal is to avoid creating avoidable mechanical differences before evaluating the controller.

---

# 40. Proposed ROS 2 Package Separation

A clean structure remains:

```text
spot_ros2/
├── spot_description/
│   ├── urdf/
│   ├── meshes/
│   └── config/
│
├── spot_gazebo/
│   ├── launch/
│   ├── worlds/
│   └── config/
│       └── controllers.yaml
│
└── spot_policy/
    ├── models/
    ├── src/
    └── launch/
```

Responsibilities:

```text
spot_description
    V3 robot mechanics

spot_gazebo
    Gazebo + gz_ros2_control + controller config

spot_policy
    observation construction
    TorchScript / exported policy inference
    50 Hz action mapping
    q_ref publishing
```

---

# 41. Policy/Controller Contract to Preserve

The following contract should be frozen across MuJoCo and Gazebo.

## Policy input

Deployment-compatible observations only.

Examples:

```text
base linear velocity
base angular velocity
projected gravity
joint position
joint velocity
previous action
velocity command
```

## Policy output

```text
12 normalized actions
```

## Action mapping

```text
q_ref =
q_home
+
action_scale * action
```

## Servo

```text
500 Hz
Kp = 300
Kd = 40
Ki = 0
qdot_ref = 0
```

## Effort limit

```text
HX/HY = 45 N·m
knee  = 115 N·m
```

## Policy rate

```text
50 Hz
```

This contract is the central sim-to-sim interface.

---

# 42. What the Project Has Learned About Actuator Validation

The actuator-validation process itself needs a revised hierarchy.

Earlier, the following tests were treated as sufficient to freeze a servo:

```text
standing
push
fixed-base step
small 50 Hz ZOH reference
```

The new PPO results show they are necessary but not sufficient.

A better validation ladder is:

```text
1. mechanical model audit
2. zero-action standing
3. disturbance recovery
4. fixed-base step
5. policy-rate ZOH reference
6. actual MjLab action-pipeline validation
7. short fresh PPO characterization
8. only then freeze for locomotion
```

The missing step had been:

```text
Does PPO discover a physically acceptable solution on this servo?
```

That is now part of actuator validation.

---

# 43. Engineering Lesson: Stable Does Not Mean Learnable

The `300/4` servo was stable.

It was not numerically broken.

It could:

- stand;
- reject a push;
- track targets.

But it changed the optimization landscape enough that PPO repeatedly found a bad high-cadence asymmetric solution.

Therefore:

> A low-level controller can be dynamically stable yet still be a poor plant for reinforcement-learning locomotion.

For learned control, actuator validation must include both:

```text
classical closed-loop behavior
```

and:

```text
effect on policy optimization
```

---

# 44. Engineering Lesson: Damping Shapes the RL Optimization Landscape

The project now has direct evidence that damping influences much more than overshoot.

Increasing:

```text
Kd = 4
```

to:

```text
Kd = 40
```

changed:

- contact cadence;
- foot participation;
- diagonal support balance;
- mechanical power;
- slip;
- landing impact;
- action smoothness;
- policy reward;
- episode survival.

This means derivative damping is part of the effective plant that PPO optimizes against.

It should not be chosen only from a local isolated-joint critical-damping estimate.

---

# 45. Engineering Lesson: Torque Limits and Servo Dynamics Must Be Separated Experimentally

The clean A/B sequence prevented a wrong conclusion.

Without the experiments, it would have been easy to say:

> The realistic 45/115 N·m limits are too restrictive, so the robot taps.

But:

```text
500/40 + 45/115
```

walked very well, while:

```text
300/4 + 1000/1000
```

walked poorly.

Therefore high torque authority and good locomotion are not synonymous.

This is exactly why one-variable A/B testing is valuable.

---

# 46. Engineering Lesson: Per-Foot Diagnostics Were Essential

Aggregate metrics could not explain the failure.

The critical diagnostics were:

```text
air_time_FL
air_time_FR
air_time_HL
air_time_HR

stance_time_FL
stance_time_FR
stance_time_HL
stance_time_HR

diag_FL_HR_fraction
diag_FR_HL_fraction
```

These exposed extreme cases where a foot could spend around a second airborne while another diagonal pair carried the robot.

The final `300/40` run showed the opposite:

```text
all four air times ≈ 0.16–0.18 s
all four stance times ≈ 0.40–0.41 s
both diagonals ≈ 25%
```

This gave a quantitative signature for a healthy gait.

---

# 47. Engineering Lesson: Do Not Over-Interpret One Successful Scalar Metric

The earlier fixed-base knee step at `300/4` looked excellent.

That did not make the controller a good locomotion plant.

Similarly, a high PPO mean reward can coexist with an ugly gait.

The project should continue to use multiple levels of evidence:

```text
equations
deterministic tests
time-series diagnostics
per-foot metrics
visual inspection
PPO behavior
cross-simulator validation
```

No single metric is enough.

---

# 48. Current Final Technical Interpretation

The V3 investigation has now progressed through:

```text
QUESTION:
Is the robot model causing the tapping?

ANSWER:
No strong evidence. V3 can walk cleanly.

QUESTION:
Is MjLab 1.5.3 causing it?

ANSWER:
No. Good locomotion exists in 1.5.3.

QUESTION:
Is the native position actuator itself the problem?

ANSWER:
No. Native position actuation produces both good and bad cases depending on gains.

QUESTION:
Are the 45/115 N·m limits too low?

ANSWER:
No. Good locomotion was learned with those exact limits.

QUESTION:
Is Kp=500 required?

ANSWER:
No. Kp=300 works excellently.

QUESTION:
What changed the pathology decisively?

ANSWER:
Kd=4 → Kd=40.
```

The current best-supported root cause is therefore:

> The repeated tapping and diagonal support imbalance were primarily caused by insufficient damping in the low-level position servo.

---

# 49. Final Project State

```text
V3 rigid-body model
    APPROVED

31.6 kg Isaac-derived inertials
    APPROVED

MjLab action pipeline
    VERIFIED

50 Hz policy interface
    VERIFIED

ZOH action semantics
    APPROVED

native position actuator family
    APPROVED

Kp = 300
    APPROVED

Kd = 40
    APPROVED

hip limit = 45 N·m
    APPROVED

knee limit = 115 N·m
    APPROVED

flat locomotion PPO
    SUCCESSFUL

old tapping root cause
    ISOLATED

reward-tuning loop
    CLOSED

gain-tuning loop
    CLOSED

next phase
    ROS 2 / Gazebo sim-to-sim
```

---

# 50. Immediate Next Milestone

The next work should begin with:

```text
V3 MuJoCo model
        ↓
V3 ROS 2 / Gazebo robot description
        ↓
500 Hz effort-based low-level controller
        ↓
deterministic MuJoCo ↔ Gazebo parity tests
        ↓
export trained policy
        ↓
policy inference node @ 50 Hz
        ↓
Gazebo sim-to-sim locomotion
```

The next engineering question is no longer:

> Why does PPO tap its feet?

That question is sufficiently resolved.

The next question is:

> Can Gazebo reproduce the closed-loop behavior of the approved `300/40, 45/115` V3 MuJoCo plant closely enough that the trained policy transfers without retraining?

That is the new phase of the project.
