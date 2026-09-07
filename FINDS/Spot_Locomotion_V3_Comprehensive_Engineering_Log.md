# Spot Locomotion V2 → V3
## Comprehensive Engineering Log: Isaac-Derived Robot Model, MjLab 1.5.3 Actuator Characterization, and Current V3 Baseline

**Project:** `spot_locomotion_v2`  
**Repository:** `~/Projects/spot_locomotion_v2`  
**GitHub:** `https://github.com/AlaaRadhwan/spot_locomotion_v2`  
**Current stack:** MjLab 1.5.3 + MuJoCo 3.10.0  
**Document scope:** This report covers the work done from the beginning of the current chat through the successful visual 50 Hz ZOH actuator test.

---

# 1. Purpose of This Work

The goal of this phase was to stop tuning reinforcement-learning rewards on an uncertain physical plant and instead define a clean, internally consistent **Spot V3 robot model and low-level joint-servo baseline** before returning to PPO training.

The central problem with the previous V2 pipeline was that locomotion failures could not be cleanly attributed to either:

- reward shaping,
- policy optimization,
- torque limits,
- joint gains,
- controller timing,
- actuator numerical implementation,
- or the rigid-body model itself.

Earlier V2 experiments had shown that the same task could behave very differently when using:

- explicit sampled PD control, versus
- MuJoCo's native built-in position actuator.

That made it clear that the actuator/controller and robot dynamics had to be treated as part of the learning plant rather than as secondary implementation details.

This chat therefore focused on:

1. replacing the old V2 rigid-body inertial model with a more plausible Isaac-derived model;
2. keeping the project geometry and kinematic structure simple and unchanged where possible;
3. validating the new model mechanically;
4. selecting and characterizing a new MjLab 1.5.3 actuator baseline;
5. validating that actuator under:
   - standing,
   - external push,
   - fixed-base position steps,
   - and actual 50 Hz zero-order-held policy-style references;
6. freezing a nominal V3 plant before any fresh PPO training.

---

# 2. Environment and Software Versions

A project `uv.lock` file was provided during the discussion to remove ambiguity about the exact software stack.

The relevant environment was confirmed as:

- **MjLab:** `1.5.3`
- **MuJoCo:** `3.10.0`
- **Python:** `>=3.13`  
  The project itself is using Python 3.13.x.
- **PyTorch:** `2.13.0`
- **mujoco-warp:** `3.10.0.3`
- **rsl-rl-lib:** `5.4.0`

The project warning:

```text
VIRTUAL_ENV=/home/alaa-radhwan/venv-ardupilot does not match the project environment path .venv
```

appeared repeatedly when using `uv run`.

This warning does not affect the project because `uv` ignores the unrelated active virtual environment and uses the repository's own `.venv`.

The important API target from this point onward is therefore:

```text
MjLab 1.5.3
MuJoCo 3.10.0
Python 3.13.x
```

This version pin mattered immediately because MuJoCo 3.10 uses the Python signature:

```python
mujoco.mj_fullM(model, data, M)
```

rather than the older form that had originally been placed in the V3 audit script.

---

# 3. Starting Point of the Chat

At the beginning of the chat, the project was already understood to be at a **plant-definition stage**, not a reward-tuning stage.

The immediate questions were:

1. What rigid-body Spot model should V3 use?
2. What low-level actuator/servo abstraction should be placed on that model?

The latest research before this chat had identified the current Isaac Sim Spot asset as the strongest practical public source for rigid-body inertial properties.

The previously used V2 model had approximately:

- total mass: **50.34 kg**
- body mass: **32.86 kg**
- each hip: **1.68 kg**
- each upper leg: **2.34 kg**
- each lower leg: **0.35 kg**

and the torso inertia looked suspiciously isotropic for an elongated quadruped body.

The Isaac asset instead contained explicit:

- link masses,
- center-of-mass positions,
- principal inertias,
- principal-axis orientations,
- joint data,
- collision information,
- and authored joint-drive parameters.

The user clarified an important design constraint early in the chat:

> The model does not need to be highly accurate. It needs to be reasonable, internally consistent, and useful for training and later sim-to-sim transfer.

This changed the scope substantially.

We therefore explicitly decided **not** to perform a forensic Boston Dynamics reconstruction.

---

# 4. V3 Rigid-Body Modeling Philosophy

The original idea was to reconstruct a complete Isaac-derived MJCF from the flattened USDA asset.

That plan was simplified.

The project already had a working Spot MJCF scaffold with:

- correct 12-joint hierarchy,
- existing visual meshes,
- collision geoms,
- foot contact geoms,
- sensors,
- joint names,
- joint ordering,
- and home pose.

Therefore, the final V3 modeling strategy became:

> **Keep the existing project geometry and kinematics, but replace the old V2 inertial properties with the Isaac inertials.**

This was deliberately a causal A/B change:

```text
same geometry
same collision layout
same joint hierarchy
same names
same home pose

but:

new mass
new COM
new inertias
```

This avoids changing many variables at once.

---

# 5. Isaac-Derived V3 Mass Model

The flattened Isaac Spot file contained the following primary rigid-body masses.

## 5.1 Body

```text
body = 16.707651 kg
```

## 5.2 Hip links

Each hip:

```text
1.1368834 kg
```

Four hips total:

```text
4 × 1.1368834 kg
```

## 5.3 Upper legs

Each upper leg:

```text
2.2562037 kg
```

Four upper legs total:

```text
4 × 2.2562037 kg
```

## 5.4 Lower legs

Each lower leg:

```text
0.33 kg
```

Four lower legs total:

```text
4 × 0.33 kg
```

## 5.5 Foot helper bodies

The Isaac asset also included approximately massless foot helper bodies:

```text
~1e-6 kg each
```

These were intentionally **not reproduced as separate rigid bodies** because the existing project already models the feet as contact geoms attached to the lower-leg bodies.

Therefore the expected articulated V3 MJCF mass became:

```text
31.5999994 kg
```

instead of the Isaac asset's approximately:

```text
31.6000034 kg
```

The difference is only the omitted four `~1e-6 kg` foot helper bodies.

No scaling to 33.8 kg or any other target mass was performed.

---

# 6. V3 Inertia Model

The Isaac asset supplied principal inertias and principal-axis orientations.

The torso principal inertia used in V3 became:

```text
Ixx = 0.08154379 kg·m²
Iyy = 0.54923120 kg·m²
Izz = 0.56945440 kg·m²
```

This was significantly more plausible than the old V2 torso inertia because it correctly reflects the anisotropy of an elongated robot body.

Representative leg inertias used were:

## Hip

```text
0.001149959
0.0016612653
0.0017290359
```

## Upper leg

```text
0.026939424
0.026025372
0.0029866563
```

## Lower leg

```text
0.0070141945
0.00709946
0.00014465585
```

A deliberate rule during conversion was:

> Do not throw away the principal-axis orientation quaternion.

The V3 builder transferred:

- mass,
- COM position,
- principal inertia,
- and principal-axis quaternion

for every link.

This was the one part of the model conversion that was treated carefully even though exact Spot fidelity was not required.

---

# 7. V3 Model File Structure

The existing V2/reference model was left untouched.

The V3 model was placed in a separate path:

```text
src/spot_locomotion_v2/robot/mjcf/v3/
    spot.xml
    scene.xml
```

The existing paths remained conceptually:

```text
mjcf/reference/        # untouched reference
mjcf/experiments/      # V2 experimental models
mjcf/v3/               # V3 nominal plant
```

A local builder was created rather than editing GitHub.

The user explicitly requested:

> Do not edit anything on GitHub. Give me the files and I will run them locally.

No GitHub modification was therefore made.

---

# 8. V3 Model Builder

A local script was created:

```text
tools/build_spot_v3_model.py
```

Its job was to:

1. load the existing reference `spot.xml`;
2. preserve:
   - body hierarchy,
   - joint frames,
   - visual meshes,
   - collision geometry,
   - sensors,
   - foot geoms,
   - joint names,
   - joint ordering;
3. replace the old inertial values with Isaac-derived values;
4. remove the reference model actuator section;
5. remove actuator-only default properties;
6. keep the home keyframe but remove invalid `ctrl` data because V3 was initially actuator-free;
7. point the V3 compiler at the existing reference mesh directory;
8. write:
   - `mjcf/v3/spot.xml`
   - `mjcf/v3/scene.xml`

The first V3 MJCF was intentionally actuator-free.

This was important because the first question was:

> Is the new mechanical plant itself correct?

not:

> Can a particular controller make it work?

---

# 9. V3 Mechanical Audit

A second local script was created:

```text
tools/audit_spot_v3_model.py
```

Its checks included:

- model compilation;
- body count;
- joint count;
- `nq`;
- `nv`;
- `nu`;
- exact joint ordering;
- total mass;
- per-link mass;
- principal inertia validity;
- V2 vs V3 mass comparison;
- apparent joint inertia at the home pose:
  - floating base;
  - base constrained.

---

# 10. MuJoCo 3.10 `mj_fullM` API Fix

The first audit run reached the apparent-inertia section and failed with:

```text
TypeError: mj_fullM(): incompatible function arguments
```

The original code had:

```python
mujoco.mj_fullM(model, M, data.qM)
```

but MuJoCo 3.10 expects:

```python
mujoco.mj_fullM(model, data, M)
```

The fixed function became:

```python
def full_mass_matrix(model, data):
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, data, M)
    return M
```

This was purely an API-version issue.

The model itself had already passed all preceding checks.

---

# 11. V3 Mechanical Audit Results

The corrected audit passed completely.

## 11.1 Total mass

```text
Total compiled mass: 31.599999400 kg
Expected mass:       31.599999400 kg
Error:              -1.066e-14 kg
```

This confirmed the builder reproduced the intended V3 mass exactly.

---

# 12. Per-Link Mass Comparison: V2 vs V3

The audit produced:

| Body | V2 [kg] | V3 [kg] | Change |
|---|---:|---:|---:|
| body | 32.86000 | 16.70765 | -49.16% |
| each hip | 1.68000 | 1.13688 | -32.33% |
| each upper leg | 2.34000 | 2.25620 | -3.58% |
| each lower leg | 0.35000 | 0.33000 | -5.71% |
| **TOTAL** | **50.34000** | **31.60000** | **-37.23%** |

The major change was therefore not uniformly distributed.

The torso changed dramatically while the upper and lower legs remained relatively close to their old masses.

This was one reason the change could not be understood as simple uniform mass scaling.

---

# 13. Principal Inertia Audit

Every V3 body passed the physical inertia sanity checks.

The script checked:

- each principal inertia > 0;
- triangle inequality validity.

All 13 physical bodies passed.

The torso values were:

```text
0.08154379
0.54923120
0.56945440
```

and the leg links also passed.

This confirmed there was no invalid or numerically impossible inertia tensor introduced during the conversion.

---

# 14. Apparent Joint Inertia Audit

The most important result from the mechanical audit was the change in effective/apparent joint inertia.

At the existing home pose:

```text
HX = 0
HY = 1.04 rad
KN = -1.80 rad
```

the results were:

| Joint | V2 floating | V3 floating | V2 base-fixed | V3 base-fixed |
|---|---:|---:|---:|---:|
| front HX | ~0.04506 | ~0.03537 | ~0.07566 | ~0.06596 |
| front HY | ~0.07957 | ~0.06178 | ~0.08318 | ~0.06449 |
| front knee | ~0.01280 | ~0.01367 | ~0.01315 | ~0.01399 |
| rear HX | ~0.04184 | ~0.03415 | ~0.07566 | ~0.06596 |
| rear HY | ~0.06323 | ~0.05432 | ~0.08318 | ~0.06449 |
| rear knee | ~0.01273 | ~0.01364 | ~0.01315 | ~0.01399 |

The crucial observation was:

> The robot became 37% lighter overall, but the knee apparent inertia actually increased slightly.

For the front knee:

```text
V2 ~0.01280 kg·m²
V3 ~0.01367 kg·m²
```

Meanwhile front HY dropped from approximately:

```text
0.07957 → 0.06178 kg·m²
```

This proved that V3 represented a genuine redistribution of mass/inertia, not simple scaling.

---

# 15. Joint Bandwidth Ratio Insight

For equal stiffness:

```math
ω_n ∝ sqrt(Kp / J)
```

The approximate HY/knee inertia ratio changed from:

```text
V2: 0.07957 / 0.01280 ≈ 6.22
V3: 0.06178 / 0.01367 ≈ 4.52
```

Therefore the equal-gain knee/HY natural-frequency ratio changed roughly from:

```text
V2: sqrt(6.22) ≈ 2.49
V3: sqrt(4.52) ≈ 2.13
```

The mismatch is still substantial, but less extreme than V2.

This was one early indication that the new plant should be easier to control with one general joint-gain region.

---

# 16. V3 Model Acceptance

After:

- exact mass validation,
- valid inertias,
- correct joint ordering,
- correct model dimensions,
- sensible apparent inertia,
- and successful visual inspection,

the V3 rigid-body model was considered provisionally frozen.

The user visually checked the generated model and reported that it looked correct.

The old home pose was intentionally left unchanged to preserve a clean causal comparison:

```text
HX = 0.0
HY = 1.04
KN = -1.80
```

No new home pose was introduced during actuator characterization.

---

# 17. Actuator Design Stage

Once the V3 rigid body passed, the next task was to select a low-level servo baseline.

MjLab 1.5.3 provides native MuJoCo actuator configurations including:

```text
BuiltinPositionActuatorCfg
BuiltinPdActuatorCfg
IdealPdActuatorCfg
```

Because the policy output is a joint-position reference only, the cleanest native candidate was:

```text
BuiltinPositionActuatorCfg
```

The planned policy interface remained:

```text
policy @ 50 Hz
      ↓
q_ref
      ↓
zero-order hold for 20 ms
      ↓
native position servo
      ↓
MuJoCo @ 500 Hz
```

No trajectory interpolation was introduced.

No desired joint velocity was introduced.

No feedforward torque was introduced.

---

# 18. Initial V3 Actuator Baseline

The first baseline tested was:

```text
Kp = 60
Kd = 1.5

HX/HY effort limit = 45 N·m
knee effort limit  = 115 N·m
```

These values were motivated by modern Isaac/Spot-style controller scales and were much softer than the previous V2 explicit controller.

The project timing remained:

```text
physics = 500 Hz
policy  = 50 Hz
decimation = 10
```

No passive joint armature, friction, actuator delay, or torque-speed envelope was added at this stage.

The philosophy was:

> Test the simplest defensible servo first.

---

# 19. MjLab V3 Robot Configuration

A new V3 robot configuration file was created:

```text
src/spot_locomotion_v2/robot/spot_v3_constants.py
```

The existing V2:

```text
spot_constants.py
```

was intentionally left untouched.

The V3 constants file defined:

- V3 model path;
- 12 joint names;
- unchanged home pose;
- 500 Hz physics;
- 50 Hz policy;
- decimation 10;
- `BuiltinPositionActuatorCfg`;
- hip effort limit;
- knee effort limit;
- MjLab entity configuration;
- helper to compile a test model with ground plane.

The first actuator values were:

```python
V3_KP = 60.0
V3_KD = 1.5

V3_HIP_EFFORT_LIMIT = 45.0
V3_KNEE_EFFORT_LIMIT = 115.0
```

---

# 20. Visual Test 2: Free Standing, 60 / 1.5

A visual standing script was created:

```text
tools/visualize_v3_standing.py
```

It:

- built the actual MjLab 1.5.3 V3 entity;
- compiled the native position actuator into MuJoCo;
- reset to home pose;
- commanded constant home references;
- ran approximately real-time;
- opened the MuJoCo passive viewer;
- logged:
  - body height,
  - roll,
  - pitch,
  - foot contacts,
  - joint error,
  - joint torque.

## Visual result

The user reported:

> Spot immediately rested on its hind upper legs while the front legs extended.

This immediately showed that 60/1.5 was not a viable standing controller for the selected home pose.

## Numerical result

Final standing state:

```text
Final base height:      0.326846 m
Minimum base height:    0.317536 m

Final roll:             +0.022 deg
Final pitch:            -7.908 deg

Maximum |pitch|:        10.779 deg

Final feet on ground:   4

Max final joint error:  0.478316 rad
Peak joint torque:      34.797 N·m
```

Final rear-knee errors were approximately:

```text
hl_kn: +0.476715 rad
hr_kn: +0.478316 rad
```

The important diagnosis was:

> This was not torque saturation. It was static stiffness deficiency.

At zero velocity, with:

```text
Kp = 60
error ≈ 0.478 rad
```

the generated holding torque is approximately:

```text
60 × 0.478 ≈ 28.7 N·m
```

which matched the logged standing torque.

The knee effort limit was 115 N·m, so the controller was simply too compliant to generate gravity-support torque without a very large position error.

---

# 21. Visual Test 3: 250 N Push, 60 / 1.5

A visual push script was also created:

```text
tools/visualize_v3_push.py
```

The protocol was:

```text
0.00 → 2.00 s   settle
2.00 → 2.25 s   250 N downward torso push
2.25 → 6.25 s   recovery
```

The push was applied directly through:

```python
data.xfrc_applied
```

on the main body.

## Visual result

The user reported that the robot behaved like a spring:

- compressed heavily,
- then sprang back upward,
- while already resting incorrectly on its rear legs.

## Numerical result

```text
Height before push:       0.327793 m
Minimum height in push:   0.242438 m
Push compression:         85.36 mm

Peak downward vz:         -0.7175 m/s
Peak upward vz:           +0.7136 m/s

Final base height:        0.348225 m
Final pitch:              -9.011 deg
Peak joint torque:        46.418 N·m
```

Because the robot was already in the wrong support mode before the push, this test was not considered a clean actuator-compliance measurement.

The push could also move the robot from one contact equilibrium into another.

Therefore the next action was **not** to tune damping.

First the standing stiffness had to be fixed.

---

# 22. Controlled Stiffness Test: Kp 300, Kd 1.5

Only stiffness was changed:

```text
Kp: 60 → 300
Kd: 1.5 unchanged
```

Everything else remained fixed:

```text
hip limit 45 N·m
knee limit 115 N·m
500 Hz physics
50 Hz policy timing
same home pose
```

The reason for using 300 rather than slowly sweeping gains was analytical.

If approximately 29 N·m of static support torque is required:

```math
e ≈ τ / Kp
```

Then:

```text
Kp 60  → e ~0.48 rad
Kp 100 → e ~0.29 rad
Kp 200 → e ~0.145 rad
Kp 300 → e ~0.097 rad
Kp 500 → e ~0.058 rad
```

So 300 was a meaningful single test point rather than a blind sweep.

---

# 23. Standing Test: 300 / 1.5

The robot now stood correctly.

The user explicitly reported:

> Spot now stands correctly.

## Results

```text
Final base height:      0.433553 m
Minimum base height:    0.421009 m

Final pitch:            -0.313 deg
Maximum |pitch|:        1.658 deg

Final feet on ground:   4

Max final joint error:  0.069133 rad
Peak joint torque:      41.805 N·m
```

Final joint errors were approximately:

```text
front knees: ~0.0565 rad
rear knees:  ~0.0691 rad
```

The huge rear-knee error from the 60/1.5 case disappeared.

Static standing was therefore considered a pass.

The remaining visible behavior was a modest damped oscillation during initial settling.

---

# 24. Push Test: 300 / 1.5

With the robot now standing correctly, the same 250 N push test became meaningful.

The user reported:

> Spot stayed standing. The push did not compress it much, but it still oscillated a little.

## Results

```text
Height before push:       0.434498 m
Minimum height in push:   0.410980 m
Push compression:         23.52 mm

Peak downward vz:         -0.4794 m/s
Peak upward vz:           +0.3180 m/s

Final base height:        0.433913 m
Final pitch:              -0.356 deg

Peak joint torque:        48.001 N·m
```

This was considered a healthy but underdamped response.

The robot:

- did not collapse;
- did not lose its equilibrium;
- returned to essentially the same standing height;
- showed only moderate rebound.

This meant the remaining question was genuinely damping.

---

# 25. Controlled Damping Test: Kp 300, Kd 4

Only damping was changed:

```text
Kp = 300
Kd: 1.5 → 4.0
```

The approximate critical-damping estimate using:

```math
Kd_crit = 2 sqrt(Kp J)
```

gave rough scales around:

```text
HX   ~6.5
HY   ~8.6
knee ~4.0
```

So `Kd = 4` was a meaningful next test:

- roughly critical for the knee;
- still moderately damped for HX/HY;
- not an arbitrary sweep.

---

# 26. Standing Test: 300 / 4

Standing improved further.

## Results

```text
Final base height:      0.433603 m
Minimum base height:    0.424544 m

Final pitch:            -0.121 deg
Maximum |pitch|:        1.285 deg

Final feet on ground:   4

Max final joint error:  0.064587 rad
Peak joint torque:      37.048 N·m
```

Compared with 300/1.5:

- static equilibrium remained essentially unchanged;
- minimum height improved;
- settling oscillation reduced;
- peak torque reduced.

This was exactly what a damping-only change should do.

---

# 27. Push Test: 300 / 4

The push response also improved.

## Results

```text
Height before push:       0.434618 m
Minimum height in push:   0.413945 m
Push compression:         20.67 mm

Peak downward vz:         -0.4454 m/s
Peak upward vz:           +0.2159 m/s

Final base height:        0.433987 m
Final pitch:              -0.140 deg

Peak joint torque:        46.121 N·m
```

Compared with 300/1.5:

| Metric | 300 / 1.5 | 300 / 4 |
|---|---:|---:|
| Standing minimum height | 0.4210 m | 0.4245 m |
| Max standing pitch | 1.66° | 1.29° |
| Push compression | 23.52 mm | 20.67 mm |
| Peak downward velocity | -0.479 m/s | -0.445 m/s |
| Peak rebound velocity | +0.318 m/s | +0.216 m/s |

The rebound velocity dropped by roughly one third while preserving the correct static equilibrium.

At this stage the decision was made:

> Stop standing/push gain tuning unless a swing or policy-interface test exposes a problem.

---

# 28. Provisional V3 Servo After Standing/Push Tests

The nominal actuator became:

```text
MjLab actuator:
BuiltinPositionActuatorCfg

Kp = 300
Kd = 4

HX/HY effort limit = 45 N·m
knee effort limit  = 115 N·m
```

No extra actuator complexity was added.

Specifically, V3 still did **not** include:

- actuator delay;
- armature;
- friction loss;
- full motor electrical model;
- torque-speed envelope;
- desired joint velocity;
- spline interpolation;
- feedforward;
- angle-dependent knee limit.

These were intentionally deferred until evidence justified them.

---

# 29. Fixed-Base Step-Response Test

Standing and push tests primarily exercise contact/stance behavior.

The next required question was:

> Does 300/4 behave sensibly for unloaded/swing-like leg motion?

A new test was created:

```text
tools/test_v3_fixed_base_step.py
```

This script:

- imported the current V3 constants;
- built the actual MjLab actuator;
- removed only the floating-base free joint;
- disabled gravity;
- preserved all 12 native MjLab position actuators;
- tested small policy-representative joint steps.

The two tests were:

## FL HY

```text
1.04 → 1.09 rad
Δq = +0.05 rad
```

## FL knee

```text
-1.80 → -1.72 rad
Δq = +0.08 rad
```

Measured quantities included:

- first-step torque;
- 10–90% rise time;
- overshoot;
- 2% settling time;
- peak joint velocity;
- peak joint torque;
- effort saturation;
- cross-joint disturbance;
- final error.

---

# 30. First Fixed-Base Run and Integration Warning

The first fixed-base run generated an important MjSpec attach warning:

```text
Attach conflict when attaching 'spot_v3_isaac_dynamics'
impratio: parent has 1, child has 100, keeping parent value
integrator: parent has 0, child has 3, keeping parent value
cone: parent has 0, child has 1, keeping parent value
```

The key issue was:

```text
integrator: parent has 0, child has 3, keeping parent value
```

This meant the child V3 MJCF requested:

```text
implicitfast
```

but the MjLab parent spec initially retained its own default option.

Because the native MuJoCo PD actuator is specifically sensitive to the integration scheme, this was not ignored.

The test harness was corrected so the final spec explicitly set:

```python
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
spec.option.impratio = 100.0
```

The test also printed the compiled final values.

---

# 31. Verified MuJoCo Integration Options

The rerun confirmed:

```text
Integrator enum: 3
Cone enum:       1
Impratio:        100.0
```

So the actual compiled deterministic test was now definitely using:

```text
integrator = implicitfast
cone = elliptic
impratio = 100
```

This matched the intended model/environment configuration.

---

# 32. Fixed-Base HY Step Result

With the corrected integration options:

```text
Target:                 +1.040000 → +1.090000 rad
Step size:              +0.050000 rad

First-step torque:      +15.000000 N·m
Expected pure P torque: +15.000000 N·m

10–90% rise time:       0.026000 s
Overshoot:              24.191 %
2% settling time:       0.154000 s

Peak |velocity|:        1.713671 rad/s
Peak |joint torque|:    15.000000 N·m

Effort limit:           45.000 N·m
Hit effort limit:       no

Max other-joint motion: 0.004857 rad
Worst coupled joint:    fl_hx

Final error:            ~0 rad
```

Interpretation:

- no saturation;
- moderate underdamping;
- bounded velocity;
- low coupling;
- exact final tracking;
- no numerical pathology.

The HY overshoot was noticeable but not considered a reason to keep tuning damping yet.

---

# 33. Fixed-Base Knee Step Result

The corrected knee test produced:

```text
Target:                 -1.800000 → -1.720000 rad
Step size:              +0.080000 rad

First-step torque:      +24.000000 N·m
Expected pure P torque: +24.000000 N·m

10–90% rise time:       0.022000 s
Overshoot:              1.819 %
2% settling time:       0.034000 s

Peak |velocity|:        4.124244 rad/s
Peak |joint torque|:    24.000000 N·m

Effort limit:           115.000 N·m
Hit effort limit:       no

Max other-joint motion: 0.005829 rad
Worst coupled joint:    fl_hy

Final error:            ~0 rad
```

This was a particularly strong result.

The old V2 explicit-PD knee characterization had produced velocities around roughly 20 rad/s in some tests.

The new V3 native 300/4 knee step remained around:

```text
4.12 rad/s
```

with:

```text
~1.8% overshoot
34 ms settling
no saturation
```

This placed the new actuator behavior in a completely different and much healthier regime.

---

# 34. Decision After Step Tests

The step tests passed.

The HY remained more underdamped than the knee, which is expected because the joint effective inertias differ substantially.

The decision was:

> Do not keep changing hip damping based only on one isolated step.

Instead, test the interface the RL policy will actually use.

That interface is:

```text
50 Hz target updates
20 ms zero-order hold
500 Hz MuJoCo physics
native position actuator
```

---

# 35. 50 Hz ZOH Policy-Style Test

A new script was created:

```text
tools/test_v3_50hz_zoh.py
```

It used the same validated fixed-base V3 model.

The test intentionally used **zero-order-held sampled references**.

No interpolation was used.

No desired velocity was used.

The reference was:

```text
frequency = 2 Hz

FL HY:
home ± 0.10 rad

FL knee:
home ± 0.16 rad
```

The target was updated only every:

```text
20 ms
```

with physics running at:

```text
2 ms
```

Therefore each policy reference was held for exactly:

```text
10 physics steps
```

This was considered the most relevant deterministic test before returning to RL.

---

# 36. 50 Hz ZOH Test Configuration

The output confirmed:

```text
Actuator: BuiltinPositionActuatorCfg

Kp = 300
Kd = 4

Integrator enum = 3

Physics:
dt = 0.002 s
500 Hz

Policy:
dt = 0.020 s
50 Hz

Decimation = 10

Reference frequency = 2 Hz

FL HY amplitude   = ±0.100 rad
FL knee amplitude = ±0.160 rad
```

---

# 37. 50 Hz ZOH Tracking Results

The test executed:

```text
51 policy updates
```

Maximum target jumps between consecutive 50 Hz commands were:

```text
HY:   0.025067 rad
knee: 0.040107 rad
```

## FL HY

```text
RMS tracking error:      0.013506 rad
Peak tracking error:     0.032967 rad
Peak velocity:           1.462832 rad/s
Peak torque:             7.460697 N·m
Effort-limit samples:    0 / 501
Saturation fraction:     0.000%
```

## FL knee

```text
RMS tracking error:      0.020371 rad
Peak tracking error:     0.046966 rad
Peak velocity:           2.633023 rad/s
Peak torque:             11.937115 N·m
Effort-limit samples:    0 / 501
Saturation fraction:     0.000%
```

## Cross-joint coupling

```text
Max untouched-joint motion:
0.004022 rad

Worst untouched joint:
fl_hx
```

---

# 38. Interpretation of the ZOH Torque Spikes

The torque plot showed visible 50 Hz spikes/teeth.

These were not interpreted as instability.

They matched the expected instantaneous P response almost exactly.

For HY:

```text
Δq_max ≈ 0.025067 rad
Kp = 300
```

Expected P jump:

```text
300 × 0.025067 ≈ 7.52 N·m
```

Measured peak:

```text
7.46 N·m
```

For the knee:

```text
Δq_max ≈ 0.040107 rad
```

Expected P jump:

```text
300 × 0.040107 ≈ 12.03 N·m
```

Measured peak:

```text
11.94 N·m
```

This is exactly what should happen when a native position servo receives a new zero-order-held reference.

Therefore the torque teeth were understood as:

> normal response to 50 Hz reference updates,

not:

> pathological actuator chatter.

---

# 39. Position and Velocity Behavior in the ZOH Test

The position plot showed:

- staircase targets;
- smooth actual motion;
- good amplitude tracking;
- small lag;
- no growing oscillation.

The knee moved faster than HY, but remained controlled.

Approximate maximum velocities were:

```text
HY:   1.46 rad/s
knee: 2.63 rad/s
```

Again, these values are much lower than the problematic V2 knee spikes.

The ZOH test was therefore considered a pass.

---

# 40. Visual 50 Hz ZOH Test

After the numerical ZOH test passed, the user asked to visualize the same motion.

A visual script was created to:

- use the exact same fixed-base V3 model;
- use the same 300/4 actuator;
- use the same 2 Hz reference;
- use the same 50 Hz ZOH target update;
- run in real time in the MuJoCo viewer;
- extend the duration so the motion could be watched clearly.

An import collision initially occurred because a visual script ended up trying to import constants from a file with the same module name:

```text
ImportError:
cannot import name 'HY_AMPLITUDE_RAD' from 'test_v3_50hz_zoh'
```

The corrected visualization script was made completely independent from `test_v3_50hz_zoh.py`.

The user then reported:

> Everything seems to be correct.

That was the final visual confirmation of the actuator characterization phase.

---

# 41. Current Frozen V3 Nominal Plant

At the end of this chat, the current V3 nominal plant is:

## Rigid Body

```text
source:
Isaac-derived inertials

total articulated mass:
31.5999994 kg

geometry:
existing project Spot geometry

collisions:
existing project collisions

joint hierarchy:
unchanged

joint names/order:
unchanged

home pose:
HX = 0.0
HY = 1.04
KN = -1.80
```

## Actuator

```text
MjLab:
BuiltinPositionActuatorCfg

Kp = 300
Kd = 4

HX/HY effort limit:
45 N·m

knee effort limit:
115 N·m
```

## Timing

```text
physics:
500 Hz

physics timestep:
0.002 s

policy:
50 Hz

policy timestep:
0.020 s

decimation:
10

reference semantics:
zero-order hold
```

## MuJoCo solver/integration intent

```text
integrator:
implicitfast

cone:
elliptic

impratio:
100
```

The existing flat environment also uses the broader solver configuration:

```text
solver = newton
iterations = 100
ls_iterations = 50
ccd_iterations = 50
```

These should remain consistent when building the V3 RL task.

---

# 42. Current V3 Features Intentionally NOT Added

The following were deliberately left out of the nominal V3 model:

```text
actuator delay
joint armature
frictionloss
additional passive viscous damping
full DC motor model
torque-speed motor envelope
feedforward torque
desired joint velocity
trajectory interpolation
JTC-style spline interpolation
angle-dependent knee torque limit
learned actuator dynamics
```

The current philosophy remains:

> Add complexity only when a measured behavior gives a reason to add it.

The deterministic actuator baseline already behaves well without these additions.

---

# 43. Deterministic Validation Summary

The completed V3 validation sequence is:

```text
Isaac-derived 31.6 kg rigid body
        ↓
mechanical audit
        PASS
        ↓
visual model inspection
        PASS
        ↓
standing with 60/1.5
        FAIL
        ↓
diagnosis: static stiffness too low
        ↓
standing with 300/1.5
        PASS
        ↓
250 N push with 300/1.5
        PASS but underdamped
        ↓
standing with 300/4
        PASS
        ↓
250 N push with 300/4
        PASS
        ↓
fixed-base HY step
        PASS
        ↓
fixed-base knee step
        PASS
        ↓
integration-option verification
        PASS: implicitfast confirmed
        ↓
50 Hz ZOH policy-style test
        PASS
        ↓
visual 50 Hz ZOH test
        PASS
```

---

# 44. Key Numerical Results at a Glance

## Mechanical plant

```text
V2 total mass:
50.34 kg

V3 total mass:
31.60 kg

change:
-37.23%
```

## V3 apparent inertia near home

```text
front HX:
~0.0354 kg·m² floating

front HY:
~0.0618 kg·m² floating

front knee:
~0.0137 kg·m² floating
```

## Standing, 60/1.5

```text
final height:
0.3268 m

rear knee error:
~0.48 rad

pitch:
~ -7.9 deg

result:
FAIL
```

## Standing, 300/1.5

```text
final height:
0.4336 m

max final joint error:
0.069 rad

result:
PASS
```

## Push, 300/1.5

```text
compression:
23.52 mm

peak rebound velocity:
+0.318 m/s

result:
PASS, somewhat underdamped
```

## Standing, 300/4

```text
final height:
0.4336 m

max final joint error:
0.0646 rad

result:
PASS
```

## Push, 300/4

```text
compression:
20.67 mm

peak rebound velocity:
+0.216 m/s

result:
PASS
```

## Fixed-base HY step, 300/4

```text
step:
+0.05 rad

rise:
26 ms

overshoot:
24.2%

settling:
154 ms

peak qdot:
1.71 rad/s

peak torque:
15 N·m

saturation:
0%
```

## Fixed-base knee step, 300/4

```text
step:
+0.08 rad

rise:
22 ms

overshoot:
1.82%

settling:
34 ms

peak qdot:
4.12 rad/s

peak torque:
24 N·m

saturation:
0%
```

## 50 Hz ZOH, 300/4

```text
HY RMS error:
0.0135 rad

HY peak velocity:
1.46 rad/s

HY peak torque:
7.46 N·m

knee RMS error:
0.0204 rad

knee peak velocity:
2.63 rad/s

knee peak torque:
11.94 N·m

saturation:
0%

cross-joint motion:
0.0040 rad
```

---

# 45. Files Created During This Chat

The following local tooling/files were created during this work.

## V3 model generation

```text
tools/build_spot_v3_model.py
tools/audit_spot_v3_model.py
```

Generated:

```text
src/spot_locomotion_v2/robot/mjcf/v3/spot.xml
src/spot_locomotion_v2/robot/mjcf/v3/scene.xml
```

## V3 robot/controller configuration

```text
src/spot_locomotion_v2/robot/spot_v3_constants.py
```

## Shared visual test utility

```text
tools/v3_visual_test_common.py
```

## Standing test

```text
tools/visualize_v3_standing.py
```

## Push test

```text
tools/visualize_v3_push.py
```

## Fixed-base step test

```text
tools/test_v3_fixed_base_step.py
```

## 50 Hz ZOH numerical test

```text
tools/test_v3_50hz_zoh.py
```

## 50 Hz ZOH visual test

```text
tools/visualize_v3_50hz_zoh.py
```

---

# 46. Reports Generated During Testing

The deterministic scripts generated reports under:

```text
reports/v3_actuator/
```

Representative outputs include:

```text
fixed_base_fl_hy_step_kp300_kd4.csv
fixed_base_fl_hy_step_kp300_kd4.png

fixed_base_fl_kn_step_kp300_kd4.csv
fixed_base_fl_kn_step_kp300_kd4.png

zoh50hz_kp300_kd4.csv
zoh50hz_kp300_kd4_positions.png
zoh50hz_kp300_kd4_velocities.png
zoh50hz_kp300_kd4_torques.png
```

The standing and push scripts initially used hard-coded old filenames:

```text
standing_builtin_position_kp60_kd1p5.csv
push250_builtin_position_kp60_kd1p5.csv
```

even after the gains had changed to 300/1.5 and then 300/4.

This is only a logging-name issue.

The actual printed configuration correctly showed the current gains.

It should be cleaned up before future characterization so report filenames are always generated from the actual active gains.

---

# 47. Important Lessons From This Chat

## 47.1 A controller can fail without saturating

The 60/1.5 standing failure was not caused by a torque ceiling.

It was caused by insufficient static stiffness.

The controller had to develop a large position error before producing enough gravity-support torque.

---

## 47.2 Static and dynamic tuning must be separated

The work followed a useful order:

```text
first:
static stiffness

then:
damping

then:
swing/joint step response

then:
policy-rate interface
```

Changing stiffness and damping simultaneously would have made the results much harder to interpret.

---

## 47.3 The same gain can behave differently across joints

Even on the new model:

```text
HY effective inertia
>>
knee effective inertia
```

Therefore equal `Kp/Kd` does not imply equal bandwidth or damping ratio.

The fixed-base step tests demonstrated this clearly:

```text
HY:
24% overshoot
154 ms settling

knee:
1.8% overshoot
34 ms settling
```

with the same:

```text
Kp = 300
Kd = 4
```

---

## 47.4 A one-step test is not the same as the RL interface

The HY looked noticeably underdamped in an isolated position step.

However, the real 50 Hz ZOH test showed smooth and well-bounded behavior.

Therefore the correct engineering question is:

> Does the actuator behave well under the command semantics the policy will actually use?

not simply:

> Does one isolated step have perfect textbook damping?

---

## 47.5 Native MuJoCo actuator behavior depends on integration settings

The MjSpec attachment warning revealed that the intended child MJCF integration settings could be overridden by the parent spec.

For `BuiltinPositionActuatorCfg`, this matters.

The test harness was therefore updated to explicitly verify:

```text
integrator = implicitfast
```

rather than assuming the child MJCF option would survive attachment.

This is a critical implementation lesson for the eventual MjLab task.

---

## 47.6 ZOH torque teeth are expected

The 50 Hz torque spikes matched:

```math
τ ≈ Kp Δq
```

almost exactly.

Therefore they are not evidence of bad numerical behavior.

They are the correct response of a position servo to discrete 50 Hz position-reference changes.

---

## 47.7 The V3 plant is now much more defensible than V2

The current V3 plant has:

- plausible inertial distribution;
- reasonable mass;
- valid inertia tensors;
- mechanically verified joint dynamics;
- a stable native actuator;
- controlled standing response;
- reasonable external-load response;
- clean swing-like step behavior;
- clean 50 Hz command behavior.

This is a much stronger basis for RL than the previous V2 loop in which reward tuning and plant uncertainty were mixed together.

---

# 48. Current Decision: Freeze the Nominal V3 Actuator

The current decision is to freeze:

```text
BuiltinPositionActuatorCfg
Kp = 300
Kd = 4

hip limit = 45 N·m
knee limit = 115 N·m
```

for the first V3 locomotion training baseline.

This does not mean these are claimed to be exact Boston Dynamics Spot gains.

They are simply:

- stable,
- plausible,
- internally consistent,
- mechanically characterized,
- and appropriate for the chosen simulated plant.

That is enough for the project goal.

---

# 49. Next Engineering Step

The next step is **not more actuator tuning**.

The next step is to integrate the frozen V3 plant into the actual MjLab velocity task.

The existing V2 task should remain untouched.

A new V3 task should be created, conceptually:

```text
src/spot_locomotion_v2/tasks/velocity/
    flat_env_cfg.py       # current V2
    flat_env_cfg_v3.py    # new V3
```

The new task should initially preserve the current task structure as much as possible.

The primary plant-related changes should be:

```text
get_spot_robot_cfg()
        ↓
get_spot_v3_robot_cfg()
```

and old torque-limit constants should be replaced by the V3 constants.

---

# 50. What Should Remain Unchanged for the First V3 RL Run

The first V3 RL run should not redesign the entire task.

Keep:

```text
physics = 500 Hz
policy = 50 Hz
decimation = 10

joint ordering
joint-position action type
home pose

actor observations
critic structure
command structure

existing simple reward philosophy
```

The existing action scales should also initially remain:

```python
".*_hx": 0.30
".*_hy": 0.40
".*_kn": 0.50
```

This allows the first fresh PPO run to primarily test the effect of:

```text
new V3 rigid body
+
new native servo
```

rather than changing plant, action semantics, reward structure, and command distribution simultaneously.

---

# 51. Reward Philosophy for the First V3 Run

The project had already concluded before this chat that the complex gait-reward stacking should not be reintroduced initially.

The first V3 training should retain the simpler reward philosophy:

```text
track linear velocity
track angular/yaw velocity
upright
pose

plus modest regularization:
action rate
foot clearance
foot swing height
foot slip
soft landing
```

Do not initially add:

```text
diagonal gait reward
support-balance reward
phase synchronization reward
strong air-time reward
```

The first question should be:

> What locomotion does PPO naturally discover on the new plant?

---

# 52. Environment-Level Smoke Test Before PPO

Before launching a large fresh training run, the V3 task should be tested as an actual MjLab environment.

Recommended smoke test:

```text
1 environment
viewer enabled
zero action
~10 seconds
```

This should verify all of the pieces together:

- V3 `EntityCfg`;
- MjLab manager construction;
- native built-in position actuator;
- `JointPositionActionCfg`;
- action ordering;
- zero-action home target;
- contact sensors;
- observations;
- integration settings;
- collision behavior;
- standing equilibrium.

This is the final systems-integration test before PPO.

---

# 53. Important Environment Integration Detail

The actual environment should explicitly set:

```python
cfg.sim.mujoco.timestep = 0.002
cfg.sim.mujoco.integrator = "implicitfast"
cfg.sim.mujoco.cone = "elliptic"
cfg.sim.mujoco.impratio = 100.0
```

and should keep the intended solver settings.

Do not rely only on the child robot MJCF to carry these global simulation options into the final MjLab parent spec.

The fixed-base characterization demonstrated why this must be explicit.

---

# 54. What Not to Do Next

The following should still be avoided until there is evidence that they are needed:

```text
do not return to random Kp/Kd sweeps

do not change the rigid-body model again without a clear reason

do not add trajectory interpolation

do not add desired joint velocity to the policy

do not add arbitrary actuator delay

do not add arbitrary armature/friction

do not add a full DC motor model

do not add angle-dependent knee torque yet

do not stack gait rewards

do not continue old V2 checkpoints

do not train on the unresolved old V2 plant
```

The next PPO policy should be trained **from scratch**.

---

# 55. Overall Project State at the End of This Chat

At this point the V3 plant-definition phase has essentially succeeded.

The current state is:

```text
V3 rigid-body model
    DONE

Isaac-derived inertials
    DONE

31.6 kg articulated model
    VERIFIED

mechanical audit
    PASS

visual model inspection
    PASS

MjLab 1.5.3 native actuator selection
    DONE

static stiffness characterization
    DONE

damping characterization
    DONE

visual standing
    PASS

250 N push
    PASS

fixed-base swing-like step response
    PASS

implicitfast integration verification
    PASS

50 Hz ZOH policy-interface test
    PASS

visual 50 Hz ZOH test
    PASS

nominal V3 plant
    READY TO FREEZE

next:
V3 MjLab velocity environment integration
```

---

# 56. Final V3 Baseline Specification

For clarity, the current nominal baseline is reproduced here in one place.

```text
ROBOT
=====
Spot-like quadruped
12 actuated joints
31.5999994 kg

Rigid-body inertials:
Isaac-derived

Geometry/collisions:
existing project MJCF scaffold

Joint order:
fl_hx
fl_hy
fl_kn
fr_hx
fr_hy
fr_kn
hl_hx
hl_hy
hl_kn
hr_hx
hr_hy
hr_kn


HOME
====
HX = 0.0 rad
HY = 1.04 rad
KN = -1.80 rad


ACTUATOR
========
MjLab 1.5.3
BuiltinPositionActuatorCfg

Kp = 300
Kd = 4

HX/HY effort limit = 45 N·m
knee effort limit  = 115 N·m


TIMING
======
physics = 500 Hz
physics dt = 0.002 s

policy = 50 Hz
policy dt = 0.020 s

decimation = 10

reference semantics:
position reference
zero-order hold


MUJOCO
======
MuJoCo 3.10.0

integrator = implicitfast
cone = elliptic
impratio = 100

environment solver target:
newton
iterations = 100
ls_iterations = 50
ccd_iterations = 50
```

---

# 57. Recommended Immediate Continuation

The next work session should begin by creating the V3 velocity task and a one-environment visual smoke test.

Recommended order:

```text
1. Create flat_env_cfg_v3.py

2. Point it at get_spot_v3_robot_cfg()

3. Replace V2 actuator-limit constants in diagnostics
   with V3_HIP_EFFORT_LIMIT and V3_KNEE_EFFORT_LIMIT

4. Preserve current action scale initially

5. Preserve 500/50 Hz timing

6. Explicitly set implicitfast / elliptic / impratio 100

7. Register a V3 task ID

8. Launch one visual environment with zero action

9. Confirm:
   - standing
   - contacts
   - sensors
   - observations
   - action order
   - solver options

10. Only then launch fresh PPO
```

---

# 58. Closing Engineering Position

The major outcome of this chat is that the project now has a credible nominal V3 plant instead of an uncertain combination of:

```text
old 50.34 kg dynamics
+
aggressive explicit PD
+
reward tuning
```

The new baseline is:

```text
Isaac-derived 31.6 kg rigid body
+
MjLab 1.5.3 BuiltinPositionActuator
+
300 / 4 PD behavior
+
45 / 115 N·m effort limits
+
500 Hz MuJoCo implicitfast
+
50 Hz ZOH position policy interface
```

and every deterministic test performed so far supports continuing with this baseline.

The most important discipline from here is:

> Do not reopen the plant unless the integrated MjLab environment or fresh PPO training gives specific evidence that something is wrong.

The next question is no longer:

> What actuator should Spot use?

It is now:

> Does PPO discover clean locomotion when trained from scratch on this frozen V3 plant?

That is the next phase.

---

# CONTINUATION — 2026-09-07

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

