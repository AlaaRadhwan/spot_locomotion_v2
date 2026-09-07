# Spot Locomotion V2 → V3
## Comprehensive Findings: Robot Model, Actuator Dynamics, RL, and Sim-to-Sim Research

**Project:** `spot_locomotion_v2`  
**Repository:** `~/Projects/spot_locomotion_v2`  
**Date:** 2026-09-05  
**Status:** V2 investigation complete enough to redefine the V3 architecture; PPO training intentionally paused until the robot model and low-level actuator abstraction are frozen.

---

# 1. Purpose of This Document

This document consolidates the main findings from:

- the Spot V2 locomotion training history,
- the native-vs-explicit actuator investigation,
- the deterministic joint-dynamics audit,
- the Boston Dynamics / RAI / MuJoCo Menagerie / Isaac Sim model comparison,
- the research into real Spot RL actuator modeling,
- the research into other legged-robot actuator models,
- and the deep sim-to-sim review covering Isaac Lab, Isaac Gym, MuJoCo, Gazebo, and real-robot deployment pipelines.

The goal is to prevent the project from repeatedly reopening already-settled questions or tuning PPO around unresolved plant-model issues.

The two questions that must now be answered before more locomotion training are:

```text
QUESTION 1
What rigid-body Spot model should V3 use?

QUESTION 2
What low-level actuator / servo abstraction should V3 use?
```

---

# 2. Original Engineering Goal

The long-term goal is not merely to make a policy walk in one simulator.

The target is a Spot-like locomotion stack that is:

- physically plausible,
- internally consistent,
- reproducible,
- trainable with RL,
- transferable from MuJoCo/MjLab to ROS 2 + Gazebo,
- and conceptually compatible with later hardware deployment.

Exact proprietary Boston Dynamics fidelity is not required.

The desired high-level contract remains:

```text
velocity command
      ↓
RL policy
      ↓
12 normalized actions
      ↓
joint position references
      ↓
low-level joint servo / actuator model
      ↓
rigid-body plant
```

The critical requirement is that the policy must not depend on an accidental feature of one simulator that cannot be reproduced elsewhere.

---

# 3. The Problem That Started the V2 Actuator Investigation

The important empirical observation was:

```text
Old/native MuJoCo position servo
        ↓
natural locomotion learned successfully

Current explicit IdealPd + realistic torque limits
        ↓
tapping / rapid contact switching
asymmetric support
weak or ineffective foot lift
poor locomotion quality
```

The task, observations, policy structure, and general training philosophy were sufficiently similar that the actuator/control layer became the main suspect.

The project then explored:

- reward changes,
- gait / air-time shaping,
- slip penalties,
- torque-limit changes,
- gain changes,
- native-vs-explicit A/B tests,
- fixed-base actuator characterization,
- standing tests,
- and finally a full rigid-body / mass-matrix audit.

The key lesson is:

> PPO should not be used as the system-identification tool for the low-level plant.

---

# 4. V2 Baseline at the End of the Investigation

## 4.1 Timing

```text
Physics frequency: 500 Hz
Physics timestep:  0.002 s

Policy frequency:  50 Hz
Policy timestep:   0.020 s

Decimation:        10
```

## 4.2 Policy output

The policy outputs 12 normalized joint-position offsets around a home posture.

Conceptually:

```text
q_ref = q_home + action_scale * action
```

with a 50 Hz policy.

## 4.3 Explicit V2 actuator baseline

```text
HIP:
Kp = 500
Kd = 10
torque limit = 69.52 Nm

KNEE:
Kp = 500
Kd = 10
torque limit = 164.15 Nm
```

The knee limit was only a local constant approximation around the home knee angle; the real Spot knee transmission is configuration-dependent.

---

# 5. What Was Ruled Out

Several hypotheses were tested and did not explain the native-vs-explicit locomotion gap by themselves.

## 5.1 Reward shaping was not the root cause

Changes to:

- air-time reward,
- diagonal gait shaping,
- foot clearance,
- slip penalty,
- support balance,
- excessive-air-time logic,

changed the details of the exploit but did not consistently produce the natural locomotion seen with the native actuator.

Therefore:

> the tapping gait should not be treated primarily as a reward-design problem.

## 5.2 Torque authority was not the root cause

Increasing the explicit torque authority substantially did not restore native-like locomotion.

High-authority explicit runs became more violent, not more natural.

Therefore:

> insufficient maximum torque is not the main explanation.

## 5.3 A softer knee alone was not the solution

Changing the knee from approximately:

```text
500 / 10
```

to:

```text
300 / 8
```

reduced unloaded knee speed and torque reversals, but worsened standing stiffness and RL locomotion.

That experiment mixed static stiffness and transient dynamics, so it was not a clean physical solution.

---

# 6. Native MuJoCo Servo vs Explicit PD: What Was Actually Different

This is one of the central resolved findings.

## 6.1 Explicit `IdealPdActuator`

MjLab `IdealPdActuator` computes the PD torque outside MuJoCo:

```text
tau = Kp * (q_des - q)
    + Kd * (qdot_des - qdot)
```

and sends that result through a MuJoCo motor actuator.

From MuJoCo's point of view, that torque is an externally computed force for the current physics step.

For the current V2 policy:

```text
qdot_des = 0
```

so the controller is effectively:

```text
tau = Kp * (q_des - q) - Kd * qdot
```

## 6.2 MjLab native built-in servo

`BuiltinPositionActuator` creates a native MuJoCo position actuator.

`BuiltinPdActuator` creates native MuJoCo position + velocity actuator elements.

MjLab v1.5.3 describes `BuiltinPdActuator` as the implicit-integration counterpart of `IdealPdActuator`.

The key numerical difference is:

```text
IdealPd:
PD force is calculated externally
→ MuJoCo sees the resulting motor torque

BuiltinPosition / BuiltinPd:
MuJoCo knows the servo stiffness / damping derivatives
→ implicit / implicitfast integration can include them
```

MuJoCo's documentation explicitly recommends `implicitfast` or `implicit` when using actuator damping, and its modern integration formulation incorporates stabilizing damping/stiffness derivatives into the effective inertia used for the step.

This explains why two controllers that look similar algebraically can behave very differently numerically at the same timestep.

---

# 7. Joint-Dynamics Audit: What It Actually Established

A comprehensive deterministic audit was run on the V2 rigid-body model.

## 7.1 Current V2 physical model

Measured total mass:

```text
50.3400 kg
```

Representative passive joint parameters:

```text
armature      = 0
joint damping = 0
friction loss = 0
```

for the tested HX, HY, and knee joints.

Therefore the V2 plant underneath the PD controller is very close to:

```text
PD torque
   ↓
static torque clip
   ↓
ideal joint torque
   ↓
rigid-body inertia
```

with no modeled drivetrain inertia, passive viscous damping, Coulomb friction, or communication delay.

## 7.2 HOME effective inertia

From the audit:

| Model | Joint | Free apparent inertia (kg·m²) | Predicted natural frequency with current gains |
|---|---:|---:|---:|
| Fixed | HX | 0.075658 | 12.94 Hz |
| Fixed | HY | 0.083176 | 12.34 Hz |
| Fixed | Knee | 0.013150 | 31.03 Hz |
| Floating | HX | 0.045058 | 16.77 Hz |
| Floating | HY | 0.079568 | 12.62 Hz |
| Floating | Knee | 0.012797 | 31.46 Hz |

At floating HOME:

```text
J_HY / J_KN ≈ 6.218
```

With equal `Kp`, that alone predicts approximately:

```text
knee natural frequency / HY natural frequency ≈ 2.49
```

So equal gains do **not** create equal closed-loop joint behavior.

## 7.3 The previous ~19 rad/s knee spike was quantitatively explained

For the old 0.25 rad knee target step:

```text
Kp = 500
initial position error = 0.25 rad

initial P torque:
500 * 0.25 = 125 Nm
```

Using the measured fixed-base knee apparent inertia:

```text
J ≈ 0.01315 kg·m²
```

the initial acceleration is approximately:

```text
125 / 0.01315 ≈ 9500 rad/s²
```

Over one 2 ms physics/control interval:

```text
Δqdot ≈ 9500 * 0.002 ≈ 19 rad/s
```

which is essentially the previously measured value.

Then the next PD update sees roughly:

```text
-Kd * qdot ≈ -10 * 19 ≈ -190 Nm
```

before clipping / interaction with the proportional term.

This explained the large torque reversal without requiring a hypothetical hidden bug.

## 7.4 Mass-matrix extraction was validated

Small direct-torque pulse tests produced effectively zero error between:

- mass-matrix predicted acceleration,
- MuJoCo forward acceleration,
- and one-step measured acceleration.

Therefore the extracted effective inertias and coupling values were trustworthy representations of the V2 MuJoCo model.

## 7.5 Contact changes the apparent dynamics substantially

The stance perturbation tests showed that a joint can have much larger apparent inertia when the foot is constrained by the ground.

This is important because:

> one set of gains can behave very differently during swing and stance.

---

# 8. What the Audit Did **Not** Prove

The audit only characterized dynamics already present in the current MuJoCo model.

It could not identify real Spot:

- rotor inertia,
- gearbox drag,
- motor electrical constants,
- current-loop bandwidth,
- communication latency,
- friction,
- backlash,
- compliance,
- or true mass distribution.

The audit therefore answered:

> "Why does the current model behave this way?"

It did **not** answer:

> "Is the current model physically correct?"

That distinction triggered the V3 rigid-body model investigation.

---

# 9. Public Spot Model Investigation

Four main model sources were considered:

```text
1. Boston Dynamics official armless URDF
2. RAI / spot_description
3. MuJoCo Menagerie Spot
4. Isaac Sim Spot USD
```

---

# 10. Boston Dynamics Official Armless URDF

The uploaded Boston Dynamics ZIP contained:

```text
model.urdf
13 OBJ meshes
```

The URDF contains:

```text
13 links
12 revolute joints
13 visual meshes

0 inertial elements
0 collision elements
```

The joint effort / velocity limits are set to placeholder-like values:

```text
effort = 1000
velocity = 1000
```

Therefore this file is **not a complete rigid-body dynamics model**.

Its value is as a high-authority source for:

- topology,
- joint naming,
- joint axes,
- joint locations,
- geometric link layout,
- visual geometry,
- and joint angular ranges.

It should not be treated as a source for:

- mass,
- inertia,
- collision behavior,
- or actuator limits.

---

# 11. RAI `spot_description` and MuJoCo Menagerie

The RAI Spot description is the source lineage behind the Menagerie Spot model.

RAI explicitly documents that its inertial properties were extracted from Isaac Sim using collision geometry and a uniform mass-distribution assumption.

The RAI / Menagerie mass allocation used by V2 is approximately:

```text
body:       32.86 kg
hip:         1.68 kg each
upper leg:   2.34 kg each
lower leg:   0.35 kg each

total:      50.34 kg
```

The current V2 torso inertia is:

```text
Ixx = Iyy = Izz = 0.13144 kg·m²
```

For a long quadruped torso, equal principal inertias are a major warning sign.

The Menagerie README also states that its position actuators were added but were not tuned.

Therefore the Menagerie model is still useful as:

- a convenient MuJoCo model,
- a geometry/collision reference,
- and a reproducible baseline,

but it should not be treated as physical truth.

---

# 12. Isaac Sim Spot USD: Major New Finding

The current Isaac Sim Spot asset was downloaded and flattened into USDA for inspection.

The flattened asset contains a substantially different mass distribution from V2.

## 12.1 Total mass

Computed sum of rigid-body masses:

```text
Isaac Spot USD total ≈ 31.600003 kg
```

The current Boston Dynamics production specification states:

```text
Spot with battery = 33.8 kg
```

Difference:

```text
2.200 kg
≈ 6.5% of the published current production mass
```

The Isaac asset is therefore far closer to the real Spot mass scale than the V2 50.34 kg model.

However, the remaining difference should **not** be blindly removed by rescaling until the asset's version / battery representation / omitted components are understood.

## 12.2 Isaac per-link masses

The flattened USD contains approximately:

| Link type | Isaac mass |
|---|---:|
| Body | 16.707651 kg |
| Each hip | 1.1368834 kg |
| Each upper leg | 2.2562037 kg |
| Each lower leg | 0.3300000 kg |
| Each foot helper | 0.000001 kg |

Compared with V2:

| Link type | V2 | Isaac | Change |
|---|---:|---:|---:|
| Body | 32.860000 | 16.707651 | -49.2% |
| Hip | 1.680000 | 1.136883 | -32.3% |
| Upper leg | 2.340000 | 2.256204 | -3.6% |
| Lower leg | 0.350000 | 0.330000 | -5.7% |
| Total | 50.340000 | 31.600003 | -37.2% |

The largest difference is in the torso.

This is especially important because body mass and inertia strongly affect:

- pitch response,
- roll response,
- yaw response,
- load transfer,
- stance coupling,
- and contact-driven body motion.

## 12.3 Isaac body inertia

The flattened USD body principal inertia is:

```text
Ixx = 0.08154379
Iyy = 0.54923120
Izz = 0.56945440 kg·m²
```

This is strongly anisotropic, as expected for an elongated quadruped body.

It is qualitatively much more plausible than:

```text
0.13144, 0.13144, 0.13144
```

from V2.

This does not prove that the Isaac values are perfect, but it is a major reason to prefer the Isaac rigid-body model as the leading V3 candidate.

## 12.4 Isaac leg inertias

Representative values from the flattened USD:

```text
HIP:
mass = 1.1368834 kg
principal inertia ≈
(0.001149959, 0.0016612653, 0.0017290359)

UPPER LEG:
mass = 2.2562037 kg
principal inertia ≈
(0.026939424, 0.026025372, 0.0029866563)

LOWER LEG:
mass = 0.33 kg
principal inertia ≈
(0.0070141945, 0.00709946, 0.00014465585)
```

These should be preserved with their corresponding COM positions and principal-axis orientations during conversion, not copied only as diagonal numbers into arbitrary frames.

---

# 13. Isaac USD Native Joint Drives

The flattened USD also contains native angular drive settings for all 12 joints.

Representative values:

```text
stiffness = 60
damping   = 1.5

hip max force  = 45 Nm
knee max force = 115 Nm
```

These values are extremely interesting because they closely match the explicit Spot actuator settings used by Isaac Lab / real-Spot RL work.

However, an important distinction must be preserved:

> the drive parameters authored in the USD are not automatically the same thing as the actuator model Isaac Lab uses at runtime.

Isaac Lab replaces / augments the plant with explicit actuator classes, including delayed and remotized PD models.

Also, the USD's `maxJointVelocity` value is very high and appears to be a simulator-side numerical cap rather than a physical joint speed limit.

Therefore:

```text
USE:
mass / COM / inertia / kinematics as strong V3 evidence

DO NOT BLINDLY USE:
every authored USD drive or velocity-limit value as hardware truth
```

---

# 14. Boston Dynamics Public Physical Constraints

Boston Dynamics' public Joint Control documentation gives high-authority actuator information.

## 14.1 Joint control law

The public low-level API exposes a PD + feed-forward structure:

```text
load_servo =
    Kp * (q_des - q_measured)
  + Kd * (qd_des - qd_measured)
  + load_desired
```

The robot state stream is available at high rate, documented at 333 Hz.

This is important because it validates the general architecture:

```text
high-level policy
    ↓
joint position / velocity references
    ↓
closed-loop joint servo
```

## 14.2 Hip transmission

Public Spot data:

```text
HX gear ratio = 51
HY gear ratio = 51
max motor torque = 0.88 Nm
```

A simple ideal-output estimate is:

```text
51 * 0.88 ≈ 44.88 Nm
```

which aligns closely with the 45 Nm hip limits used in Isaac Lab / RAI full descriptions.

## 14.3 Knee transmission

Public Spot data:

```text
max knee motor torque = 1.50 Nm
transmission ratio = configuration dependent
```

The knee is actuated through a ball-screw / push-rod mechanism.

Therefore:

```text
tau_knee_max = f(q_knee)
```

is physically better than one constant knee limit.

Boston Dynamics publishes the knee torque-vs-angle table publicly.

This means the previous V2 constant knee limit should not survive into a physical V3 model.

---

# 15. Real Spot RL: The Most Direct Precedent

The 2025 work:

**High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures**

is one of the most directly relevant references to this project.

The published architecture is approximately:

```text
RL policy: 50 Hz
      ↓
joint-position offsets
      ↓
q_ref = q_default + 0.2 * action
      ↓
explicit PD control
      ↓
joint torque
```

Published actuator/controller values include approximately:

```text
Kp = 60
Kd = 1.5
```

with a lower-level control rate reported around 200 Hz.

The work deployed policies directly to Spot's low-level joint-control interface and demonstrated aggressive locomotion.

The implication is very important:

> explicit PD is not inherently unsuitable for Spot RL.

Therefore the failure of V2 cannot be reduced to:

```text
"explicit PD is bad"
```

A much stronger hypothesis is:

```text
V2 combined:
high gains (500 / 10)
+ sparse actuator physics
+ questionable 50.34 kg rigid-body model
```

and treated that combination as the realistic baseline.

---

# 16. Isaac Lab Spot Actuator Model

Current Isaac Lab Spot configuration uses two actuator classes.

## 16.1 Hips

```text
DelayedPDActuator
Kp = 60
Kd = 1.5
effort limit = 45 Nm
delay = 0–8 ms
```

## 16.2 Knees

```text
RemotizedPDActuator
Kp = 60
Kd = 1.5
angle-dependent torque lookup
delay = 0–8 ms
```

The knee lookup table is explicitly documented as data collected by the Boston Dynamics AI Institute.

This architecture is important because it models the actuator as more than:

```text
clip(Kp*e - Kd*qdot)
```

It includes at least:

- closed-loop PD behavior,
- command delay,
- realistic hip effort limits,
- and nonlinear knee transmission capability.

---

# 17. MuJoCo Playground Spot

Google DeepMind's MuJoCo Playground contains a modern Spot locomotion task.

Representative configuration:

```text
control dt = 0.020 s  → 50 Hz high-level control
simulation dt = 0.004 s
Kp = 300
Kd = 1
action scale = 0.3
```

The underlying Menagerie-derived model also contains native joint properties such as:

- armature,
- joint damping,
- friction loss,
- native position actuation.

Important interpretation:

> a successful modern MuJoCo Spot RL stack still uses a simulator-native closed-loop servo abstraction rather than requiring every motor electrical effect to be modeled explicitly.

The exact MuJoCo Playground parameters should not be copied as Spot truth, but the architecture is relevant.

---

# 18. Unitree RL Lab: Strong Sim-to-Sim Precedent

Unitree's official `unitree_rl_lab` is a particularly important reference.

Its workflow is explicitly:

```text
Isaac Lab training
      ↓
MuJoCo sim-to-sim verification
      ↓
physical Unitree deployment
```

The repository documentation instructs users to test the trained policy in `unitree_mujoco` before sim-to-real deployment.

This directly supports using a second simulator as a deployment gate.

## 18.1 Unitree actuator modeling

Unitree's actuator implementation includes effects such as:

- delayed PD behavior,
- torque-speed operating region,
- static friction,
- dynamic friction,
- velocity-dependent friction activation,
- reflected drivetrain/armature inertia.

The reflected inertia is not inserted arbitrarily; the code includes a gearbox/rotor inertia relation.

This provides a strong architectural example:

```text
policy target
   ↓
PD servo
   ↓
delay
   ↓
torque-speed limits
   ↓
friction
   ↓
joint torque
```

This structure is portable across physics engines because the semantics are defined above the plant.

---

# 19. Unitree RL MjLab: Native Actuation Is Not Automatically "Cheating"

Unitree also maintains an official MuJoCo/MjLab RL stack.

That project uses MjLab / MuJoCo for training and then deploys to real Unitree hardware.

Its robot configurations make substantial use of MjLab's native built-in position actuators rather than exclusively relying on Python-computed explicit PD.

This is a major conceptual correction for this project:

> a native simulator position servo can legitimately represent a closed-loop embedded joint servo.

The question is not whether the implementation is "native."

The question is:

> Does the closed-loop mapping from joint reference to joint motion / torque approximate the target real servo well enough?

---

# 20. Humanoid-Gym: Isaac Gym → MuJoCo Sim-to-Sim

`humanoid-gym` provides another strong precedent.

Its workflow includes:

```text
Isaac Gym RL training
      ↓
exported policy
      ↓
MuJoCo sim-to-sim execution
      ↓
real humanoid deployment
```

The project explicitly describes sim-to-sim as a way to verify robustness and generalization in a different physics simulator before hardware deployment.

It reports real zero-shot sim-to-real deployment on RobotEra humanoids.

This demonstrates that cross-engine execution is not merely a debugging trick; it can be a formal part of a deployment pipeline.

---

# 21. Isaac Gym → ROS 2 Gazebo Quadruped Sim-to-Sim

The community project `quadruped_sim2sim` implements:

```text
Isaac Gym-trained quadruped policy
      ↓
ROS 2 controller
      ↓
Gazebo
```

with policies including DreamWaQ, Walk-These-Ways, and HIMLoco.

This project is less authoritative than the official Unitree pipeline and does not provide the same level of hardware validation, but it is directly relevant to our intended:

```text
MuJoCo/MjLab
      ↓
ROS 2
      ↓
Gazebo Harmonic
```

deployment architecture.

It demonstrates that transferring a learned low-level quadruped policy into ROS 2 + Gazebo is practically achievable.

---

# 22. Minitaur: A Very Similar Actuator-Numerics Lesson

The 2018 Minitaur sim-to-real work is historically important.

The authors found that the simulator's built-in motor/servo behavior did not sufficiently match the real actuator.

Their solution included:

- system identification,
- an improved actuator model,
- and simulated latency.

The important general lesson is:

> do not make the explicit controller imitate a convenient simulator servo merely because the simulator servo trains well.

Instead:

> model the closed-loop actuator behavior that the deployed robot actually has.

This is very close to the issue encountered in V2.

---

# 23. ANYmal: Learned Actuator Dynamics

The ANYmal work by Hwangbo et al. treated actuator dynamics as a major sim-to-real issue.

Rather than relying only on a simple analytical servo, hardware data were used to train an actuator network that maps state/error history to actuator torque.

Conceptually:

```text
q_ref + state history
      ↓
learned actuator model
      ↓
joint torque
```

This approach is especially useful when the actuator contains dynamics that are difficult to model analytically.

For our current Spot project, this is **not yet the preferred method** because we do not have the required real-robot actuator telemetry.

The lesson is architectural:

> actuator dynamics deserve their own identification/modeling layer.

---

# 24. KAIST Hound: Torque-Speed Constraints Matter

The Hound high-speed locomotion work explicitly includes the actuator torque-speed operating region in RL.

A real electric actuator cannot normally provide maximum torque at arbitrary speed.

The Hound work showed that constraining the policy to physically feasible motor operating regions improves deployability and prevents the policy from exploiting impossible actuator states.

This matters because the V2 static torque clip effectively assumed:

```text
same max torque at:
0 rad/s
10 rad/s
20 rad/s
...
```

until clipping.

That is not generally representative of a real electric actuator.

---

# 25. Three Successful Actuator Modeling Philosophies Found in the Literature

The research revealed three broad approaches.

## 25.1 Native closed-loop servo

Examples:

- MuJoCo Playground,
- Unitree RL MjLab,
- many MuJoCo / MjLab RL stacks.

Conceptually:

```text
policy q_ref
    ↓
simulator-native closed-loop joint servo
    ↓
joint / drivetrain constraints
    ↓
rigid-body dynamics
```

Advantages:

- numerically stable,
- simple,
- efficient,
- often highly trainable.

Risk:

- hidden simulator-specific dynamics if not characterized.

## 25.2 Explicit servo + physical actuator constraints

Examples:

- Isaac Lab Spot,
- real Spot RL work,
- Unitree RL Lab,
- Minitaur,
- KAIST Hound.

Conceptually:

```text
q_ref
  ↓
PD
  ↓
delay
  ↓
friction / torque-speed / transmission constraints
  ↓
joint torque
```

Advantages:

- portable semantics,
- easier to reproduce across simulators,
- direct physical interpretation.

Risk:

- numerically stiff if gains, timestep, and omitted drivetrain dynamics are poorly chosen.

## 25.3 Learned actuator model

Example:

- ANYmal.

Best when good hardware identification data exist.

Not currently justified for this project.

---

# 26. Deep Sim-to-Sim Research: Main Findings

The sim-to-sim research changes how this project should define portability.

## 26.1 Simulator internals do not need to be identical

A common earlier assumption was:

```text
MuJoCo and Gazebo must use exactly the same internal actuator implementation.
```

The reviewed systems do not support such a strict requirement.

A better requirement is:

> the policy-facing closed-loop plant should be sufficiently equivalent.

That means the important mapping is:

```text
(q_ref, robot state)
        ↓
q(t), qdot(t), torque/load, contact response
```

not whether both simulators use the same integrator or actuator primitive internally.

## 26.2 Sim-to-sim is a diagnostic gate

A second simulator is valuable because a policy that only works in the training engine may be exploiting:

- contact solver details,
- actuator integration details,
- numerical damping,
- collision approximations,
- timing assumptions,
- or state ordering/scaling errors.

Passing a second simulator is stronger evidence that the learned behavior depends on meaningful dynamics rather than one engine's quirks.

## 26.3 Sim-to-sim is not proof of real-world correctness

Cross-engine success is useful, but two simulators can still share the same wrong assumptions.

Therefore:

```text
sim-to-sim pass
≠
proof of physical fidelity
```

It is a robustness/deployment check, not a substitute for physical parameter validation.

---

# 27. Common Sources of Sim-to-Sim Failure

The deep review suggests that cross-engine failures often come from the interface contract rather than "RL" itself.

The following must be frozen explicitly.

## 27.1 Joint ordering

```text
policy output index
↔
joint name
```

must be identical.

## 27.2 Joint sign conventions

A positive joint command must mean the same physical direction.

## 27.3 Reference posture

The same home/default joint angles must be used.

## 27.4 Action scaling

For example:

```text
q_ref = q_home + scale * action
```

must use identical scaling.

## 27.5 Observation definition and ordering

Including:

- joint positions,
- joint velocities,
- body angular velocity,
- projected gravity,
- commands,
- previous actions.

Normalization must also match.

## 27.6 Timing

The policy update period and reference hold/interpolation semantics matter.

## 27.7 Servo semantics

The deployed simulator must approximate the training simulator's closed-loop joint behavior.

## 27.8 Effort / velocity constraints

Different physical limits can completely change locomotion.

## 27.9 Collision/contact properties

Especially:

- foot radius,
- friction,
- contact softness,
- body collision geometry,
- lower-leg collisions.

## 27.10 Base inertial properties

Cross-engine equality is impossible if the two robot descriptions do not represent the same mass distribution.

---

# 28. Reconsidering `BuiltinPositionActuator`

The deep research substantially changed the earlier position on native MuJoCo actuation.

Previously the project leaned toward:

```text
explicit effort controller everywhere
```

because this seemed maximally portable to Gazebo.

That view was too strict.

## 28.1 Why `BuiltinPositionActuator` is now a legitimate candidate

A real Spot already exposes a closed-loop joint-servo interface through the Joint Control API.

A simulator-native position servo can therefore be interpreted as:

> a numerical representation of the robot's embedded closed-loop joint servo.

It is not necessarily pretending that the bare electric motor is an ideal position source.

## 28.2 `BuiltinPositionActuator` capabilities

MjLab supports:

- stiffness,
- damping,
- effort limit,
- armature,
- friction loss,
- viscous damping.

For the current policy, where:

```text
qdot_des = 0
```

`BuiltinPositionActuator` is sufficient to represent a standard zero-reference-velocity PD position servo.

## 28.3 `BuiltinPdActuator`

`BuiltinPdActuator` is more general.

It accepts:

```text
position target
velocity target
```

using native MuJoCo position + velocity actuator elements.

It is especially relevant if V3 later wants:

```text
q_des
qdot_des
```

rather than only position references.

## 28.4 Important restriction

Reconsidering native actuation does **not** mean returning to:

```text
Kp = 500
Kd = 40
±1000 Nm
```

The old native actuator was a successful behavioral reference, not a physically grounded final design.

---

# 29. Reconsidering `IdealPdActuator`

`IdealPdActuator` should remain a reference implementation.

Its strengths:

- transparent equation,
- easy effort-level interpretation,
- conceptually portable to Gazebo,
- easy to add nonlinear torque constraints around.

Its weaknesses in the V2 configuration:

- the controller was extremely stiff relative to the low-inertia knee,
- the PD was evaluated outside the MuJoCo implicit solve,
- the plant lacked many drivetrain effects,
- and the rigid-body model itself is now questionable.

Therefore V2 does **not** demonstrate that `IdealPdActuator` is fundamentally bad.

It demonstrates that the particular V2 combination was poorly specified.

---

# 30. Current Actuator Candidates for V3

At this stage the sensible candidates are:

| Candidate | Strength | Main concern |
|---|---|---|
| `BuiltinPositionActuator` | Simple, stable, proven in MuJoCo RL, fits current position-only policy | Native MuJoCo servo must be characterized for Gazebo equivalence |
| `BuiltinPdActuator` | Native implicit PD with explicit position + velocity targets | More complex policy/controller contract if velocity targets are used |
| `IdealPdActuator` | Transparent portable effort law | Requires carefully chosen gains/rate/plant model to avoid V2 problems |

Current working expectation:

> `BuiltinPositionActuator` or `BuiltinPdActuator` may become the V3 baseline, provided the closed-loop behavior is physically grounded and reproducible in Gazebo.

This is deliberately **not yet a final decision**.

---

# 31. Gazebo Portability: Revised Interpretation

A strict internal one-to-one actuator implementation is not necessary.

A legitimate architecture could be:

## MuJoCo training

```text
50 Hz RL policy
      ↓
q_ref
      ↓
MuJoCo BuiltinPosition / BuiltinPd
      ↓
MuJoCo rigid-body plant
```

## Gazebo sim-to-sim

```text
50 Hz RL policy
      ↓
same q_ref
      ↓
ROS 2 low-level servo
      ↓
Gazebo effort interface
      ↓
Gazebo rigid-body plant
```

The Gazebo servo would be tuned / implemented to reproduce the relevant closed-loop behavior of the training servo.

The matching targets should include:

- static stiffness,
- step response,
- damping,
- rise time,
- bandwidth,
- saturation behavior,
- delay,
- and configuration-dependent knee torque capability.

The important equivalence is behavioral, not implementation-level.

---

# 32. Current Rigid-Body Model Decision

The investigation has now moved the model ranking.

## 32.1 No longer preferred as V3 truth

```text
RAI / Menagerie 50.34 kg dynamics
```

because:

- total mass is far from current production Spot mass,
- the body inertia is suspicious,
- and the inertia provenance is synthetic.

## 32.2 Authoritative geometry reference

```text
Boston Dynamics official armless URDF
```

because it provides high-authority kinematics and geometry.

## 32.3 Leading V3 dynamics candidate

```text
current Isaac Sim Spot USD
```

because:

- total mass is on the correct Spot scale,
- the mass distribution is substantially more plausible,
- body inertia is anisotropic,
- and its joint/drive ecosystem aligns with current Isaac Lab Spot work.

This is not yet the same as declaring every Isaac parameter correct.

---

# 33. Proposed V3 Model Construction Strategy

The preferred next model workflow is:

```text
Boston Dynamics URDF
        ↓
validate topology / joint axes / joint ranges

Isaac Sim flattened USD
        ↓
extract mass / COM / inertia / geometry / collision data

Boston Dynamics public specifications
        ↓
check global mass scale / joint capabilities

Boston Dynamics + BDAI actuator data
        ↓
hip effort + nonlinear knee transmission

convert to clean MjLab/MuJoCo model
```

Do **not** rescale the Isaac asset to 33.8 kg automatically.

First determine why the current Isaac asset is approximately 31.6 kg.

Possible causes include:

- asset version,
- battery representation,
- omitted sensors/components,
- or different production generation.

Until that provenance is resolved, 31.6 kg should be treated as a strong model candidate, not exact ground truth.

---

# 34. V3 Model Validation Before RL

Once the Isaac-derived V3 MJCF is built, the robot should be validated deterministically.

The validation suite should include:

```text
compiled total mass
link-by-link mass table
center-of-mass audit
inertia validity
symmetry checks
joint limits
foot location
collision geometry
fixed-base effective joint inertia
floating-base effective joint inertia
standing equilibrium
support torque
small stance perturbations
```

The goal is to freeze the rigid-body plant before actuator tuning.

---

# 35. V3 Actuator Validation Before RL

After the rigid-body model is frozen, actuator candidates should be compared with deterministic reference commands.

The test should include:

```text
small position steps
policy-like target changes
standing disturbances
swing-joint response
stance-joint response
torque saturation
velocity behavior
delay response
```

The comparison should answer:

```text
BuiltinPosition?
BuiltinPd?
IdealPd?
```

using engineering metrics rather than PPO return.

---

# 36. Sim-to-Sim Should Become a Formal Deployment Gate

The V3 workflow should include Gazebo **before** any real hardware consideration.

Recommended sequence:

```text
1. Train fresh policy in MjLab/MuJoCo
        ↓
2. Export policy
        ↓
3. Run same observation/action contract in Gazebo Harmonic
        ↓
4. Compare commanded and measured behavior
        ↓
5. Resolve sim-to-sim mismatch
        ↓
6. Only then consider hardware
```

Useful cross-simulator metrics:

- command tracking error,
- body velocity tracking,
- joint-position RMS error,
- joint-velocity distributions,
- contact timing,
- duty factor,
- foot slip,
- torque/load distribution,
- base pitch/roll dynamics,
- mechanical power,
- termination/failure rate.

---

# 37. What Should Be Randomized During RL

The literature supports randomization, but only around a credible nominal plant.

Randomization should not be used to hide a badly specified model.

Potential V3 domain-randomization candidates:

```text
small mass variation
small COM variation
friction variation
small actuator delay variation
small gain variation
terrain/contact variation
sensor noise
```

Parameters should be randomized around evidence-based nominal values.

---

# 38. What Should Not Be Done Next

The following should remain paused:

```text
new gait reward experiments
more air-time tuning
torque-limit sweeps
blind Kp/Kd sweeps
1 kHz / 2 kHz rate experiments without a defined target actuator
PPO training on the unresolved V2 plant
arbitrary armature insertion
arbitrary friction insertion
```

Those experiments would generate more data without resolving the model architecture.

---

# 39. Major Assumptions That Have Been Reversed

## Old assumption

```text
50.34 kg model is internally consistent, so keep it.
```

## Current conclusion

Internal consistency is not enough. Its dynamics are not authoritative and should be replaced or revalidated.

---

## Old assumption

```text
native MuJoCo position servo is inherently unsuitable for sim-to-sim.
```

## Current conclusion

A native servo can legitimately model an embedded closed-loop joint controller if its behavior is characterized and reproduced in Gazebo.

---

## Old assumption

```text
explicit effort control is automatically more realistic.
```

## Current conclusion

Explicit control is only more physically meaningful if its gains, limits, delay, friction, transmission, and rigid-body plant are actually grounded.

---

## Old assumption

```text
the knee spike suggests missing actuator inertia is probably the main problem.
```

## Current conclusion

The V2 rigid-body inertia + sampled high-gain explicit PD already explains the knee spike quantitatively. Missing drivetrain physics may still matter for realism, but it is not required to explain that measurement.

---

## Old assumption

```text
more controller frequency may be the main fix.
```

## Current conclusion

Real Spot RL precedent shows successful explicit PD at lower rates with much softer gains. Controller frequency is not the main design question until the actuator target is defined.

---

# 40. Current Best Engineering Interpretation

The failure of V2 is best understood as a **plant-definition problem**, not a single reward or control bug.

V2 combined:

```text
questionable 50.34 kg rigid-body model
        +
very stiff 500/10 explicit PD
        +
static effort clipping
        +
no armature
        +
no friction
        +
no delay
        +
constant knee torque approximation
```

and called that the realistic actuator baseline.

The research now shows that serious legged-robot systems tend to use one of two better-defined alternatives:

```text
well-characterized native closed-loop servo
```

or:

```text
explicit servo + identified actuator constraints
```

V3 should deliberately choose one of these architectures.

---

# 41. Current Two-Question V3 Plan

## Question 1 — Rigid-body model

Current leading direction:

```text
Isaac Sim Spot dynamics
+
Boston Dynamics authoritative kinematics/reference
+
public physical constraints
```

Next task:

- convert / reproduce the Isaac rigid-body properties cleanly in MjLab,
- verify mass / COM / inertia / geometry,
- run the deterministic model audit,
- freeze the model.

## Question 2 — Actuator abstraction

After the model is frozen, compare:

```text
BuiltinPositionActuator
BuiltinPdActuator
IdealPdActuator
```

against a defined closed-loop target.

The target should draw from:

- real Spot joint-control semantics,
- Isaac Lab Spot,
- real Spot RL literature,
- and sim-to-sim portability requirements.

Only then should one actuator become the V3 training baseline.

---

# 42. Recommended Concrete Next Steps

```text
STEP 1
Extract all relevant rigid-body data from spot_flattened.usda.

STEP 2
Create an Isaac-derived Spot MJCF in a new V3 model path.
Do not overwrite the old reference model.

STEP 3
Preserve:
- masses
- COMs
- principal inertias
- principal-axis orientations
- kinematics
- joint limits
- appropriate collision geometry

STEP 4
Compile and run one deterministic rigid-body audit.

STEP 5
Compare the new model against:
- official BD kinematics
- Isaac source values
- current production mass specification

STEP 6
Freeze the V3 rigid-body model.

STEP 7
Design one deterministic actuator comparison:
- BuiltinPosition
- BuiltinPd
- IdealPd

STEP 8
Freeze the low-level servo abstraction.

STEP 9
Train a completely fresh locomotion policy.

STEP 10
Use Gazebo Harmonic sim-to-sim as a required deployment gate.
```

---

# 43. Source and Evidence Hierarchy

When sources disagree, use the following priority.

## Tier 1 — Boston Dynamics

Use for:

- current product mass,
- joint morphology,
- Joint Control API semantics,
- gear ratios,
- motor torque,
- nonlinear knee torque capability.

## Tier 2 — Boston Dynamics AI Institute / Isaac Lab Spot data

Use for:

- knee transmission lookup,
- practical RL actuator architecture,
- delay assumptions,
- actuator limits used in modern Spot RL.

## Tier 3 — Isaac Sim Spot asset

Use as the leading public rigid-body dynamics candidate, subject to validation.

## Tier 4 — RAI `spot_description`

Useful for:

- full ROS model,
- collision geometry,
- historical model provenance.

Do not treat its 50.34 kg dynamics as authoritative.

## Tier 5 — MuJoCo Menagerie / Playground

Excellent implementation references and behavioral baselines, but their model parameters should not automatically override higher-authority physical evidence.

---

# 44. Key External References

## Boston Dynamics

- Spot specifications:  
  https://support.bostondynamics.com/articles/Knowledge/Spot-Specifications-49916

- Joint Control API:  
  https://dev.bostondynamics.com/docs/concepts/joint_control/readme

- Supplemental joint / transmission data:  
  https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html

- Knee torque lookup:  
  https://dev.bostondynamics.com/docs/concepts/joint_control/knee_torque_limits.html

## Spot RL / Isaac Lab

- High-Performance Reinforcement Learning on Spot:  
  https://arxiv.org/abs/2504.17857

- Isaac Lab Spot configuration:  
  https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/spot.py

## MuJoCo / MjLab

- MuJoCo actuator XML reference:  
  https://mujoco.readthedocs.io/en/latest/XMLreference.html

- MuJoCo integration/computation reference:  
  https://mujoco.readthedocs.io/en/latest/computation/

- MjLab v1.5.3 built-in actuator implementation:  
  https://github.com/mujocolab/mjlab/blob/v1.5.3/src/mjlab/actuator/builtin_actuator.py

- MuJoCo Playground:  
  https://github.com/google-deepmind/mujoco_playground

## Sim-to-Sim

- Unitree RL Lab — Isaac Lab → MuJoCo → hardware:  
  https://github.com/unitreerobotics/unitree_rl_lab

- Unitree RL MjLab:  
  https://github.com/unitreerobotics/unitree_rl_mjlab

- Humanoid-Gym — Isaac Gym → MuJoCo sim-to-sim:  
  https://github.com/roboterax/humanoid-gym

- Quadruped Isaac Gym → ROS 2 Gazebo sim-to-sim:  
  https://github.com/evronix/quadruped_sim2sim

## Actuator-modeling literature

- Minitaur sim-to-real:  
  https://arxiv.org/abs/1804.10332

- ANYmal actuator / locomotion work:  
  https://arxiv.org/abs/1901.08652

- KAIST Hound actuator-constrained RL:  
  https://arxiv.org/abs/2312.17507

---

# 45. Local Project Evidence Used

The conclusions in this report also use the following project artifacts generated or supplied during the investigation:

```text
spot_constants.py

spot_floating_base_explicit_pd.xml

joint_dynamics_audit.zip
    model_audit.csv
    joint_physical_properties.csv
    effective_inertia.csv
    coupling.csv
    modal_analysis.csv
    pulse_validation.csv
    standing_support.csv
    stance_perturbation_summary.csv
    summary.md

spot_base_urdf.zip

spot.usd

spot_flattened.usda

Spot_V2_Actuator_Investigation_Update_and_V3_Rethink.md

Spot_V2_Gazebo_ROS2_Low_Level_Control_Architecture.md
```

---

# 46. Final Current Decision

The project should **not** continue by tuning the V2 controller or PPO.

The next phase is:

```text
V3 RIGID-BODY MODEL
        ↓
freeze and validate
        ↓
V3 LOW-LEVEL SERVO
        ↓
freeze and validate
        ↓
fresh RL training
        ↓
MuJoCo → Gazebo sim-to-sim
        ↓
future hardware
```

The current strongest rigid-body candidate is the **Isaac Sim Spot asset**, validated against Boston Dynamics geometry and specifications.

The current actuator decision is deliberately still open, with **BuiltinPositionActuator / BuiltinPdActuator now restored as serious candidates** alongside a better-specified explicit actuator.

The most important methodological rule going forward is:

> Do not optimize the policy around a plant whose mechanics and low-level control semantics have not yet been frozen.

---

# 47. One-Sentence State of the Project

> V2 showed that a policy can learn locomotion on Spot, but V3 now aims to replace the questionable 50.34 kg/high-gain actuator baseline with an evidence-based rigid-body model and a deliberately chosen closed-loop servo, then use sim-to-sim transfer as a formal robustness and deployment test.
