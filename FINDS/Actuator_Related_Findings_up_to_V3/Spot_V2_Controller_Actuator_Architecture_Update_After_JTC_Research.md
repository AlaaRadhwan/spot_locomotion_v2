# Spot Locomotion V2 — Controller / Actuator Architecture Update After JTC Research

## Status

This document is a continuation of the following project reports:

1. `Spot_Locomotion_V2_Engineering_Retrospective.md`
2. `Spot_V2_Actuator_Characterization_Report.md`
3. `Spot_V2_Actuator_Investigation_Update_and_V3_Rethink.md`
4. `Spot_V2_Gazebo_ROS2_Low_Level_Control_Architecture.md`

It records the work, experiments, research, corrections, and revised technical conclusions reached **after** those documents.

This document is especially important because several assumptions in the previous V3 design were tested directly and did not survive the experiments.

The main changes in understanding are:

- the original zero-order-held PD position-control structure is **not inherently wrong** for a real torque-controlled quadruped,
- a 50 Hz RL position command should be treated primarily as a **setpoint**, not automatically as a trajectory waypoint,
- naïve linear JTC interpolation creates derivative-torque impulses,
- cubic JTC interpolation removes velocity discontinuity but creates a new problem when waypoint velocities are inferred poorly,
- `JointTrajectoryController` is therefore no longer the preferred ROS controller for the normal RL joint-setpoint loop,
- ROS 2's standard `pid_controller/PidController` is now the leading standard-controller candidate for Gazebo deployment,
- the main unresolved problem has shifted back toward the **physical actuator plant beneath the PD controller**,
- reflected drivetrain inertia, effective joint inertia, physical joint/transmission dynamics, and mechanical-model parity now deserve higher priority than more trajectory shaping,
- and the next phase should begin with a quantitative dynamics audit before any more PPO training.

The purpose of this report is not merely to list those conclusions.

It records **how we reached them**, what was disproven, what remains uncertain, and how the project direction should change.

---

# 1. Starting Point From the Previous Reports

The previous reports had already established several strong results.

## 1.1 Reward design was no longer the leading suspect

Repeated reward experiments had shown:

- strong slip penalties could suppress useful stepping,
- air-time rewards could improve cadence but worsen asymmetry,
- explicit diagonal gait rewards could create high-frequency reward exploits,
- timing rewards could produce more reasonable cadence while still allowing unequal support participation,
- support-balance rewards detected asymmetry but did not remove the underlying actuator-dependent behavior,
- and returning to the original simple reward structure reproduced the same V2 tapping/support pattern.

This strongly weakened the idea that the main failure was simply:

```text
"bad reward shaping"
```

---

## 1.2 The same robot/task could learn good locomotion with the old native actuator

The strongest A/B result from the earlier phase was:

```text
current V2 robot/task
+
native MuJoCo position servo
+
old-style high-authority actuation
```

returned much more natural locomotion.

Representative differences included:

```text
explicit V2:
switch rate  ~10.6 Hz
power        ~237 W
vertical RMS ~0.144 m/s
slip         ~0.309 m/s
landing      ~198 N

native actuator:
switch rate  ~4.5 Hz
power        ~115 W
vertical RMS ~0.075 m/s
slip         ~0.173 m/s
landing      ~64 N
```

The native case also used both diagonal pairs much more evenly.

Therefore the basic:

- robot geometry,
- action space,
- observation structure,
- command structure,
- and simple reward function

were all capable of producing good locomotion.

The actuator/control implementation remained the dominant difference.

---

## 1.3 Torque authority was tested and strongly weakened as the root cause

The explicit actuator was tested with:

```text
1× physical torque limits
2× physical torque limits
~±1000 Nm authority
```

The central tapping/support pathology did not disappear.

Instead, high authority made the explicit plant much more energetic.

The ±1000 Nm case produced several-kilowatt mechanical power, severe slip, high landing forces, and poor episode survival.

Therefore:

> The physical torque ceilings are not the primary cause of the V2 gait failure.

The current limits remained useful and physically interpretable.

---

## 1.4 Lower knee gains did not solve locomotion

The knee controller was changed from:

```text
500/10
```

to:

```text
300/8
```

while the hips remained:

```text
500/10
```

The deterministic fixed-base test improved:

- peak knee velocity decreased by about 40%,
- torque reversal decreased substantially,
- cross-joint disturbance decreased,
- tracking remained fast.

But the free-standing test became softer:

- more standing sag,
- more displacement under the same push,
- more coupled hip motion.

The RL run was worse:

- weaker stepping,
- foot dragging,
- poorer-looking locomotion.

Therefore:

> Excessive unloaded knee bandwidth alone was not the root cause.

This was a critical negative result.

---

# 2. Previous V3 Direction Before the New Experiments

At the end of the earlier V3 rethink, one proposed architecture was:

```text
RL policy @ 50 Hz
      |
      | q_target
      v

trajectory / command reconstruction
      |
      | q_des
      | qdot_des
      v

PD / PDFF controller @ 500 Hz
      |
      | tau_cmd
      v

actuator constraints
      |
      | tau_actual
      v

physics engine
```

The idea was partly motivated by Boston Dynamics' public joint-control interface, which supports:

```text
q_des
qdot_des
Kp
Kd
tau_ff
```

and by public examples that stream interpolated joint trajectories at high rate.

A specific candidate was therefore:

> Interpolate the 50 Hz RL joint-position targets into 500 Hz `q_des/qdot_des` trajectories.

This was an attractive hypothesis because the current explicit controller uses:

```text
qdot_des = 0
```

and therefore the derivative term opposes joint motion.

However, this hypothesis had not yet been tested under policy-like command changes.

That became the next experiment.

---

# 3. Important Reassessment: The Original PD Law Is Not Fundamentally Wrong

Before discussing the trajectory experiments, one major conceptual correction should be stated clearly.

The current low-level law:

```text
tau =
Kp(q_des - q)
-
Kd qdot
```

is simply the special case of a standard PD/impedance controller:

```text
tau =
Kp(q_des - q)
+
Kd(qdot_des - qdot)
+
tau_ff
```

with:

```text
qdot_des = 0
tau_ff   = 0
```

This structure is widely used in real quadruped low-level interfaces.

Examples include public Unitree motor commands and the Boston Dynamics Spot Joint Control API.

Therefore:

> The fact that the D term opposes joint motion is not itself a controller defect.

That is precisely what damping does when the desired velocity is zero.

The relevant engineering question is:

```text
Is the resulting closed-loop response physically appropriate
for the actual joint plant?
```

not:

```text
Does the D term oppose motion?
```

This correction matters because it changes the role of trajectory interpolation.

Interpolation is no longer considered necessary merely to "fix" the derivative term.

It must justify itself through actual behavior.

---

# 4. Why the Policy Output Should Be Viewed as a Setpoint

The RL policy produces a new vector every:

```text
20 ms
```

at:

```text
50 Hz
```

The output represents:

```text
12 desired joint-position offsets
```

from the home pose.

The policy does **not** explicitly output:

```text
desired joint velocity
desired joint acceleration
future trajectory
```

Therefore each action is more naturally interpreted as:

> "This is the joint position I want now."

rather than:

> "This is a waypoint on a known smooth trajectory."

This distinction became central.

A trajectory controller must infer additional information when the policy gives only positions.

That inference is not guaranteed to be physically meaningful.

---

# 5. Policy-Rate Controller A/B Test

A targeted deterministic test was built to answer:

> Does JTC-like interpolation produce a better low-level response to 50 Hz policy position commands than zero-order-held position targets?

The test deliberately held constant:

```text
robot model
physics rate
controller rate
policy command rate
Kp/Kd
torque limits
target sequence
```

Only the command semantics changed.

---

# 6. Test Configuration

Physics and controller:

```text
500 Hz
dt = 0.002 s
```

Policy command rate:

```text
50 Hz
20 ms per policy target
```

Controller:

```text
Kp = 500
Kd = 10
```

Physical limits:

```text
HY   = ±69.52 Nm
Knee = ±164.15 Nm
```

Representative test joints:

```text
FL_HY
FL_KN
```

The policy-like target sequence repeatedly moved the joints through moderate position changes rather than using the much larger instantaneous steps from the original characterization.

Representative per-policy target changes:

```text
HY:
up to 0.05 rad per 20 ms

Knee:
up to 0.08 rad per 20 ms
```

---

# 7. A/B Case 1 — Old Zero-Order-Hold Controller

The baseline used:

```text
q_des = latest 50 Hz policy target
qdot_des = 0
```

held constant for 20 ms.

Control law:

```text
tau =
Kp(q_des - q)
-
Kd qdot
```

This represents the existing conceptual V2 action-to-torque structure.

---

# 8. A/B Case 2 — JTC-Like Linear Interpolation

The second case linearly interpolated from the previous policy target to the new one over 20 ms.

For each interval:

```text
q_des(t) = linear segment
```

with:

```text
qdot_des =
(q_new - q_old) / 0.02
```

Control law:

```text
tau =
Kp(q_des - q)
+
Kd(qdot_des - qdot)
```

The hypothesis was:

> The controller would follow the desired movement rather than damping all movement toward zero velocity.

---

# 9. Linear Interpolation Result

The position plots initially looked encouraging.

The actual joint followed the changing reference more directly.

However, the torque plots exposed a serious issue.

---

# 10. Why Linear JTC Produced Large Torque Impulses

For HY, a typical desired velocity was:

```text
+2.5 rad/s
```

or:

```text
-2.5 rad/s
```

because:

```text
0.05 / 0.02 = 2.5 rad/s
```

At a reversal, desired velocity changed approximately:

```text
+2.5 -> -2.5 rad/s
```

therefore:

```text
Δqdot_des ≈ -5 rad/s
```

With:

```text
Kd = 10
```

the derivative contribution changes by approximately:

```text
10 * 5 = 50 Nm
```

almost instantaneously.

This matched the measured HY torque impulses.

---

# 11. Knee Linear-Interpolation Impulse

The knee command had:

```text
0.08 / 0.02 = 4 rad/s
```

desired segment speed.

At reversal:

```text
+4 -> -4 rad/s
```

so:

```text
Δqdot_des ≈ -8 rad/s
```

With:

```text
Kd = 10
```

the expected derivative-torque change is:

```text
~80 Nm
```

The knee torque plot showed spikes around this magnitude.

This was a very clean validation of the mechanism.

---

# 12. Linear A/B Conclusion

Linear interpolation did not solve the controller problem.

It merely moved the discontinuity.

Old ZOH:

```text
position reference discontinuous
velocity reference = 0
```

Linear interpolation:

```text
position reference continuous
velocity reference discontinuous
```

Because the controller contains a derivative term, the velocity discontinuity created deterministic torque impulses.

The linear test therefore established:

> Position-only linear trajectory interpolation is not appropriate for the 50 Hz RL command stream under the tested PD structure.

This result also aligned with ROS 2 JointTrajectoryController documentation, which warns that position-only linear interpolation has discontinuous velocity at waypoints.

---

# 13. Why the Linear Test Was Valuable

This test prevented a potentially costly mistake.

Without the deterministic A/B, it would have been easy to:

- implement linear JTC-like interpolation in MjLab,
- train a full PPO policy,
- observe some new behavior,
- then struggle to determine whether the policy, reward, or controller caused it.

Instead, the low-level torque problem was visible immediately.

This reinforced the project engineering rule:

> Test low-level controller semantics deterministically before letting PPO adapt to them.

---

# 14. Cubic JTC Hypothesis

The next hypothesis was narrow.

If the problem with linear interpolation was discontinuous desired velocity, then cubic interpolation could preserve:

```text
continuous q_des
continuous qdot_des
```

and remove the derivative kicks.

The goal was **not** to continue trying interpolation variants indefinitely.

One cubic test was justified because it directly tested the diagnosed weakness of the linear method.

---

# 15. Cubic Waypoint Velocity Choice

The policy still supplied only positions.

Therefore a causal velocity estimate was required.

The test used:

```text
v_k =
(q_k - q_{k-1}) / 0.02
```

for each policy waypoint.

These position and velocity endpoints were then joined by cubic Hermite interpolation.

Thus each segment satisfied:

```text
q(0)    = q_0
q(T)    = q_1
qdot(0) = v_0
qdot(T) = v_1
```

with:

```text
T = 20 ms
```

This ensured velocity continuity across intervals.

---

# 16. Cubic JTC Result — Knee

The knee showed a real improvement.

Approximate linear torque spikes were around:

```text
+85 to +90 Nm
-80 Nm
```

while the cubic case reduced the largest repeated spikes to roughly:

```text
+30 Nm
-40 Nm
```

depending on the event.

The knee velocity trace also became somewhat less extreme.

The main conclusion was:

> The large linear-JTC knee torque spikes were indeed caused largely by desired-velocity discontinuity.

This validated the diagnosis from the linear test.

---

# 17. Cubic JTC Result — HY

The HY improvement was more modest.

Approximate largest impulses changed from roughly:

```text
linear:
±64–65 Nm
```

to approximately:

```text
cubic:
±55–58 Nm
```

The desired velocity inside the cubic segments also became larger than the simple linear segment velocity in some regions.

Therefore:

> Velocity continuity alone does not guarantee a gentler trajectory.

A smooth spline can still demand aggressive motion.

---

# 18. The Critical Cubic Problem: Overshoot From Bad Waypoint Slopes

The position plot exposed the decisive weakness.

At local maxima and minima, the cubic desired trajectory overshot beyond the policy's commanded position range.

This did not originate primarily from the PD controller.

It originated from the chosen waypoint velocity estimates.

Consider three target values:

```text
1.14
1.19
1.14
```

The middle point is clearly a local maximum.

A physically sensible desired velocity at that exact peak would likely be approximately:

```text
0
```

However, the causal backward difference uses only:

```text
1.14 -> 1.19
```

and assigns:

```text
v_k =
(1.19 - 1.14) / 0.02
=
+2.5 rad/s
```

Therefore the cubic spline is told:

> Arrive at the highest position while still moving upward quickly.

The next segment immediately requires the trajectory to reverse.

Overshoot is therefore unavoidable.

The knee showed the same underlying issue.

---

# 19. Cubic A/B Conclusion

The cubic controller fixed one problem:

```text
velocity discontinuity
```

but introduced another:

```text
poorly inferred waypoint velocity
```

This produced:

- position overshoot,
- larger internal desired velocities in some segments,
- a new modeling assumption not supplied by the policy.

Therefore:

> The tested cubic interpolation should not become the V3 policy-command interface.

The result did not show that cubic splines are generally bad.

It showed that:

> A position-only RL policy does not provide enough information to define a physically meaningful cubic trajectory without adding another assumption.

---

# 20. Why We Stopped the Spline Investigation

At this point the progression was:

```text
ZOH
-> position discontinuity

linear
-> velocity discontinuity and derivative impulses

cubic
-> continuous velocity but artificial waypoint-slope assumptions and overshoot
```

Continuing into:

```text
quintic
filters
Catmull-Rom
velocity limiters
trajectory predictors
```

would have moved farther away from the actual engineering question.

The policy is not a motion planner.

It is a feedback controller producing position setpoints.

Therefore the correct conclusion was:

> Stop trying to turn every policy output into a preplanned trajectory.

No further interpolation variants are currently justified.

---

# 21. Correction to the Previous Gazebo Architecture Note

The previous Gazebo/ROS 2 report leaned toward:

```text
JointTrajectoryController
```

or eventually a custom controller.

That recommendation is now revised.

JTC is still useful when the system genuinely has a trajectory to execute.

Examples:

```text
sit / stand sequence
calibration motion
scripted pose transition
planned joint-space motion
maintenance routines
```

But ordinary RL locomotion does not naturally provide a future joint trajectory.

It provides rapidly updated setpoints.

This makes JTC an awkward abstraction for the normal locomotion loop.

---

# 22. New ROS 2 Standard-Controller Candidate: PidController

ROS 2 Jazzy provides a standard:

```text
pid_controller/PidController
```

inside the `ros2_controllers` ecosystem.

This controller is better aligned with the locomotion interface because it is designed for continuously updated references rather than requiring the input to be interpreted as a preplanned trajectory.

The standard controller can use:

```text
position
velocity
```

as reference/state information and produce an effort command.

Conceptually:

```text
tau_cmd =
Kp(q_ref - q)
+
Ki integral(error)
+
Kd(qdot_ref - qdot)
```

For our initial baseline:

```text
Ki       = 0
qdot_ref = 0
```

therefore:

```text
tau_cmd =
Kp(q_ref - q)
-
Kd qdot
```

This matches the simple RL joint impedance law directly.

---

# 23. Why This Standard ROS Controller Is Preferable Right Now

Using the standard ROS 2 PID controller satisfies the project preference:

> Avoid writing a custom controller until a standard controller demonstrably fails to provide something necessary.

Advantages:

- standard ROS 2 controller lifecycle,
- integrates with `controller_manager`,
- effort output,
- no hidden Gazebo position servo,
- no need to interpret setpoints as trajectories,
- easier initial Gazebo deployment,
- lower implementation uncertainty,
- easier comparison with the MjLab PD law.

A custom controller remains an option later if required by:

```text
angle-dependent nonlinear actuator limits
non-standard feed-forward structure
motor torque-speed constraints
special safety logic
hardware-specific requirements
```

but it is no longer the default first choice.

---

# 24. Revised Deployment Architecture

The preferred initial architecture is now:

```text
                 COMMAND
               ──────────

                 /cmd_vel
                    |
                    v

             RL policy @ 50 Hz
                    |
                    | q_ref[12]
                    v


             LOW-LEVEL CONTROL
            ───────────────────

        ROS 2 PidController @ ~500 Hz
                    |
                    | qdot_ref = 0 initially
                    |
                    | tau_cmd =
                    | Kp(q_ref-q)
                    | - Kd*qdot
                    v


            EFFORT INTERFACE
           ──────────────────

              gz_ros2_control
                    |
                    | effort
                    v


               ACTUATOR /
             ROBOT PHYSICS
           ─────────────────

              Gazebo Harmonic
```

Training should reproduce the same control semantics:

```text
RL policy @ 50 Hz
      |
      | q_ref
      v

explicit PD @ 500 Hz
      |
      | tau_cmd
      v

actuator model
      |
      v

MuJoCo
```

---

# 25. Important Principle: Physics Engine Receives Torque

One conclusion from the previous Gazebo research remains unchanged.

The simulator-native position command interface should not be the final low-level boundary.

Preferred:

```text
controller computes torque
-> effort interface
-> physics engine
```

rather than:

```text
controller sends position
-> simulator-specific hidden position servo
-> physics
```

This avoids unnecessary MuJoCo-vs-Gazebo servo mismatch.

The project should own the low-level position/impedance law.

---

# 26. Therefore the Main Problem Is No Longer "How Should We Interpolate?"

The new experiments narrowed the problem.

The main question is now:

> What physical joint plant should exist underneath the PD controller?

This is the key conceptual reset.

A real torque-controlled geared quadruped joint is not simply:

```text
PD equation
-> static torque clip
-> massless ideal torque source
```

The physical actuator stack includes some combination of:

```text
motor torque/current dynamics
rotor inertia
transmission inertia
gear or linkage ratio
friction
efficiency
torque-speed limits
position-dependent torque capability
control delay / bandwidth
mechanical compliance
```

The exact required fidelity depends on the robot and transfer target.

---

# 27. Current Explicit Plant Is Extremely Simple

The current explicit MjLab model is conceptually close to:

```text
q_ref
  |
  v
PD controller
  |
  | tau_cmd
  v
static torque clipping
  |
  | tau_actual = clipped tau_cmd
  v
MuJoCo joint/link inertia
```

The current main experimental MJCF uses direct `<motor>` torque actuators.

The joints do not currently include a carefully identified actuator-side model such as:

```text
reflected motor inertia
gear friction
motor torque-speed envelope
actuator time constant
```

This simplicity is useful for understanding.

But it may also be the main reason the explicit plant behaves unlike a real geared quadruped.

---

# 28. Effective Joint Inertia Is Now a High-Priority Quantity

The earlier step-response tests showed:

```text
HY peak velocity:
~6.25 rad/s

Knee peak velocity:
~19–21 rad/s
```

under the same nominal:

```text
Kp=500
Kd=10
```

The correct explanation is not simply:

```text
"knee gain is too high"
```

Closed-loop response depends on effective inertia:

```text
J_eff(q)
```

as well as gain.

For a simplified joint:

```text
J_eff qdd
+
Kd qdot
+
Kp(q-q_ref)
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
Kd / (2 sqrt(Kp J_eff))
```

Therefore the same gain pair can produce very different dynamics if:

```text
J_eff_HY
!=
J_eff_KN
```

This is already expected in a coupled leg.

We should calculate it rather than infer it only from the time response.

---

# 29. MuJoCo Can Give Us the Effective Inertia

MuJoCo provides the generalized mass matrix.

That lets us evaluate the configuration-dependent inertia seen at each DOF.

The next analysis should extract representative effective inertias for:

```text
HX
HY
KN
```

at multiple poses.

At minimum:

```text
home pose

compressed stance

more extended stance

forward leg configuration

rearward leg configuration

multiple knee angles
```

This will give quantitative answers to:

- why the knee accelerates faster,
- how much joint response changes across the workspace,
- whether uniform gains are physically defensible,
- what gain ratios would produce comparable bandwidth,
- how much room remains for reflected actuator inertia.

---

# 30. Reflected Rotor / Drivetrain Inertia

MuJoCo's `armature` parameter represents additional inertia reflected through an actuator/transmission.

For a simple geared rotational transmission:

```text
J_ref =
N^2 * J_rotor
```

For Spot-like HX/HY:

```text
N = 51
```

therefore:

```text
N^2 = 2601
```

Even a small rotor inertia can therefore become significant at the output joint.

This is one of the strongest physical candidates for the unexpectedly fast explicit-joint acceleration.

---

# 31. Why Missing Reflected Inertia Is Plausibly Important

The current direct-torque model is close to:

```text
ideal torque
-> link inertia
```

while a real geared actuator sees:

```text
motor rotor inertia
gear/transmission inertia
link inertia
```

The reflected rotor inertia can materially slow acceleration for the same applied joint torque.

That maps directly onto one of the original V2 observations:

> Some joints, especially the knee, could accelerate to very large velocities under modest target commands without requiring saturation.

This is exactly the direction of error expected from an actuator drivetrain with too little reflected inertia.

However, this remains a **hypothesis**, not yet a confirmed cause.

---

# 32. Why We Should Not Simply Add an Arbitrary MuJoCo Armature

The project goal is not just:

```text
make MuJoCo behave nicely
```

The goal is:

```text
MuJoCo
-> ROS 2 / Gazebo
-> later possible hardware
```

MuJoCo has a direct joint `armature` parameter.

A standard Gazebo/SDFormat joint does not provide an obviously identical one-line reflected-rotor-inertia parameter.

Therefore inserting an arbitrary armature in MuJoCo would create a new sim-to-sim problem unless we also define how the same physical effect is represented in Gazebo.

Possible future solutions include:

```text
explicit rotor/transmission mechanics

custom simulated hardware dynamics

equivalent identified joint-side model

modified equivalent link/joint inertia representation
```

Each has tradeoffs.

Therefore:

> Reflected inertia is a priority research topic, not yet a parameter to guess.

---

# 33. Joint Damping and Friction Also Need an Audit

A real quadruped drivetrain contains losses.

Possible effects include:

```text
viscous drag
Coulomb friction
gearbox friction
ball-screw friction
seal/bearing friction
efficiency losses
```

The current explicit model captures little of this in a deliberately identified way.

Manufacturer-provided quadruped MuJoCo models such as Unitree's include nonzero joint:

```text
armature
damping
friction loss
```

This does **not** justify copying their values onto Spot.

It shows that a completely ideal torque-source joint is not the only common simulation assumption.

These parameters should be investigated only with defensible estimates or identification targets.

---

# 34. The Existing Torque Limits Remain One of the Better-Justified Parts

The current hip torque limit:

```text
69.52 Nm
```

has a strong Spot-derived interpretation.

Boston Dynamics publishes:

```text
HX/HY gear ratio = 51
max motor torque = 0.88 Nm
```

giving approximately:

```text
44.88 Nm
```

joint-side reference torque for real Spot before our scaling assumption.

The current model mass is approximately:

```text
50.34 kg
```

versus a public Spot reference around:

```text
32.5 kg
```

Scaling torque by the mass ratio produces approximately:

```text
69.5 Nm
```

which matches the current limit.

Thus the hip limit remains defensible as:

> A Spot-derived torque-to-weight-preserving scaling assumption for the heavier model.

---

# 35. Knee Torque Limit Must Eventually Become Configuration Dependent

Boston Dynamics documents the knee as a ball-screw / push-rod transmission.

Therefore joint torque capability varies with:

```text
q_knee
```

The current:

```text
164.15 Nm
```

limit corresponds approximately to one reference configuration near the current home pose after the same mass scaling.

It should eventually become:

```text
tau_kn_max =
f(q_kn)
```

This is one of the most attractive future improvements because:

- it is physically meaningful,
- public transmission data exists,
- it does not require inventing unknown electrical parameters,
- it can be reproduced in both training and deployment logic.

However, it is probably **not** the main cause of the current tapping because torque-limit sweeps did not remove the pathology.

---

# 36. Torque-Speed Behavior Is Physically Important but Underidentified

Real electric actuators cannot usually produce the same torque at:

```text
0 rad/s
```

and:

```text
very high speed
```

A DC/PMSM-like actuator has a feasible torque-speed operating region.

This could be relevant to the knee because the current explicit model allows large speed while still providing static torque authority up to the same hard ceiling.

However, we currently lack trusted Spot values for:

```text
motor Kt
motor Ke
winding resistance
bus voltage
no-load speed
current-controller saturation
gear efficiency
```

Therefore a full motor model would currently require too many guessed values.

The correct current decision remains:

> Do not create false precision with an underidentified DC motor model.

Torque-speed modeling should be added only when enough information exists to parameterize it sensibly.

---

# 37. Actuator Delay and Bandwidth Are Also Important but Not Yet Known

Real low-level torque tracking is not instantaneous.

Relevant dynamics can include:

```text
motor current-loop bandwidth
torque sensor/control bandwidth
communication latency
software scheduling delay
filtering
```

Sim-to-real literature repeatedly identifies latency and actuator dynamics as important.

However, public Spot command-stream frequency is not the same thing as internal torque-loop bandwidth.

Therefore:

```text
external API at ~100–333 Hz
```

must **not** be interpreted as:

```text
motor torque loop = 100–333 Hz
```

The internal servo may be much faster.

We therefore should not invent a low-pass filter or fixed delay simply because it seems realistic.

---

# 38. Feed-Forward Remains an Option, Not a Requirement

The earlier standing tests showed:

```text
pure PD
```

supports gravity through position error.

At approximately:

```text
Kp=500
```

the knee required roughly:

```text
0.06 rad
```

of static sag to create around:

```text
30 Nm
```

of support torque.

At:

```text
Kp=300
```

the required error increased toward:

```text
0.10 rad
```

which matched the observed softer stance.

Feed-forward torque could decouple:

```text
static load support
```

from:

```text
virtual spring stiffness
```

However, this does not mean feed-forward is necessary for locomotion.

Many successful RL systems use:

```text
q_ref
qdot_ref = 0
tau_ff = 0
```

with an appropriate PD/actuator plant.

Therefore:

> Do not add feed-forward until we can state exactly which problem it is intended to solve.

---

# 39. The Mechanical Robot Model Also Needs a Transfer-Focused Audit

Because Gazebo transfer is now an explicit requirement, the mechanical model should be reviewed separately from the actuator.

One item worth checking is the torso inertia.

The current main body uses approximately:

```text
mass = 32.86 kg

diagonal inertia =
0.13144
0.13144
0.13144 kg m²
```

Equal principal inertias for an elongated quadruped torso are unusual enough to deserve source tracing and validation.

This does **not** prove the values are wrong.

The correct action is:

```text
find origin
check consistency
compare against geometry / reference model
```

before copying the same inertias into URDF/SDF.

This is especially important because:

> Sim-to-sim transfer requires the same mechanical plant in both simulators before differences in contact and actuator behavior can be interpreted.

---

# 40. Current Problem Ranking

After all experiments and research to date, the problems should now be ranked approximately as follows.

| Priority | Problem | Current confidence |
|---|---|---|
| 1 | Incomplete / underidentified physical actuator plant beneath PD | Very high |
| 2 | Controller gains not designed from effective joint dynamics | Very high |
| 3 | MuJoCo ↔ Gazebo mechanical-model parity not yet established | High |
| 4 | ROS low-level control interface needs final baseline implementation | High, but conceptually straightforward |
| 5 | Reflected drivetrain inertia missing / unresolved | Potentially very high |
| 6 | Angle-dependent knee torque limit missing | High for realism, likely secondary to tapping |
| 7 | Torque-speed behavior absent | Likely important, parameters currently weak |
| 8 | Joint/transmission friction underidentified | Likely relevant |
| 9 | Actuator delay / bandwidth underidentified | Relevant later |
| 10 | Feed-forward / gravity compensation | Optional until a concrete need appears |
| 11 | Reward/gait shaping | Not current priority |

This table is a major update from the earlier reports.

---

# 41. What Has Now Been Strongly Ruled Out or Downgraded

## 41.1 "The reward is the main problem"

Strongly downgraded.

Reward changes changed the symptom but not the actuator-dependent family of bad solutions.

---

## 41.2 "The torque limits are too small"

Strongly downgraded.

Higher limits did not restore natural locomotion.

---

## 41.3 "The knee just needs lower gains"

Strongly downgraded.

`300/8` improved one deterministic metric but made locomotion worse.

---

## 41.4 "Target interpolation is the missing fix"

Strongly downgraded.

Linear and cubic tests exposed new problems rather than a compelling solution.

---

## 41.5 "JTC should be the normal RL low-level controller"

Downgraded.

JTC is better suited to actual trajectory execution.

A standard ROS PID controller is a better match for continuously updated RL setpoints.

---

## 41.6 "The D term fighting motion means the controller is conceptually wrong"

Rejected.

A zero-velocity-reference impedance controller is common and physically valid.

The issue is the closed-loop plant/gain combination, not the existence of damping.

---

# 42. What Remains Strongly Supported

## 42.1 Policy rate

```text
50 Hz
```

remains a reasonable learned-policy frequency.

---

## 42.2 Low-level controller rate

```text
~500 Hz
```

remains a reasonable baseline.

It matches the current MjLab decimation structure and fits within common quadruped low-level command/control rates.

---

## 42.3 Policy action type

```text
12 joint-position offsets
```

remains portable and practical.

There is no current evidence that the policy must output torque directly.

---

## 42.4 Effort interface in Gazebo

This remains strongly preferred.

The low-level controller should calculate torque explicitly and send effort to Gazebo.

---

## 42.5 Standard controller first

Use ROS standard controllers before implementing a custom one.

The new leading candidate is:

```text
pid_controller/PidController
```

rather than JTC.

---

# 43. Revised V3 Baseline Candidate

The simplest defensible controller architecture now looks like:

```text
Policy @ 50 Hz
      |
      | action
      v

q_ref =
home_pose + action_scale * action
      |
      | held for 20 ms
      v

PD @ 500 Hz

tau_cmd =
Kp(q_ref-q)
-
Kd qdot
      |
      v

physical actuator model
      |
      v

tau_actual
      |
      v

physics engine
```

Initial:

```text
qdot_ref = 0
tau_ff   = 0
Ki       = 0
```

This is intentionally simple.

The important work now lies in defining:

```text
Kp/Kd
and
physical actuator model
```

rather than adding trajectory machinery.

---

# 44. What Belongs in the Controller

Controller-level quantities include:

```text
q_ref
qdot_ref
Kp
Kd
Ki if ever used
tau_ff if ever used
controller update rate
safety/reference limits
```

These define desired behavior.

---

# 45. What Belongs in the Physical Actuator / Robot Model

Physical-model quantities include:

```text
motor/transmission torque capability
angle-dependent knee transmission
torque-speed envelope
rotor/transmission inertia
joint friction
viscous damping
actuator response bandwidth
delay where physical
gear efficiency
mechanical compliance if modeled
joint limits
```

These define what the plant can physically do.

This separation is now one of the core project rules.

---

# 46. Revised ROS 2 / Gazebo Architecture

The first deployment prototype should target:

```text
/cmd_vel
    |
    v

policy node @ 50 Hz
    |
    | q_ref
    v

ros2_control PidController @ ~500 Hz
    |
    | effort
    v

gz_ros2_control
    |
    v

Gazebo Harmonic
```

State interfaces:

```text
position
velocity
```

Command interface:

```text
effort
```

The policy and PD controller should not depend on Gazebo-specific native position-control behavior.

---

# 47. Where JTC Still Belongs

JTC should not be discarded.

It remains a good standard ROS controller for motions that genuinely have trajectory semantics.

Likely future uses:

```text
stand-up sequence
sit-down sequence
initial pose move
calibration
maintenance motion
scripted gait test
safe transition between controllers
```

The key correction is simply:

> Do not force 50 Hz RL position setpoints into JTC trajectory semantics when the policy does not provide a trajectory.

---

# 48. Revised Development Sequence

The next development sequence should now be:

## Phase 1 — Freeze controller semantics

For the next plant investigation, use:

```text
50 Hz policy-style position setpoint
zero-order hold
500 Hz PD
qdot_ref = 0
tau_ff = 0
physical torque limits
```

No more spline work.

---

## Phase 2 — Audit mechanical dynamics

Extract:

```text
mass matrix
effective joint inertia
body/link inertias
joint configuration dependence
```

for representative poses.

Trace suspicious mechanical parameters to their source.

---

## Phase 3 — Design controller bandwidth from inertia

Instead of guessing `Kp/Kd`, choose desired:

```text
closed-loop natural frequency
damping ratio
```

then derive gains from:

```text
J_eff
```

for HX/HY/KN.

This may lead to joint-specific gains.

---

## Phase 4 — Verify standard ROS PidController

Build one minimal Gazebo joint or reduced robot test using:

```text
pid_controller/PidController
effort command interface
position + velocity state
```

Confirm the actual Jazzy behavior matches the intended PD equation.

This is a deployment-interface validation, not a locomotion experiment.

---

## Phase 5 — Establish MuJoCo ↔ Gazebo deterministic parity

Run identical simple commands in both engines and compare:

```text
q
qdot
tau_cmd
free response
loaded response
```

Do this before PPO transfer.

---

## Phase 6 — Investigate drivetrain inertia representation

Once the base mechanics and gain design are understood, determine how to represent the same reflected-inertia effect in:

```text
MuJoCo
Gazebo
```

Do not enable a MuJoCo-only effect without a deployment plan.

---

## Phase 7 — Add configuration-dependent knee torque limit

Use the known Spot-derived transmission relation.

Make the exact same limiter available in both simulator/control stacks.

---

## Phase 8 — Add additional actuator physics only when justified

Possible later additions:

```text
friction
torque-speed envelope
actuator bandwidth
delay
feed-forward
```

Each should solve a measured discrepancy.

---

## Phase 9 — Freeze V3 plant

Once deterministic behavior is coherent and portable, freeze:

```text
robot mechanics
actuator model
controller
action semantics
```

---

## Phase 10 — Restart PPO

Use:

```text
simple reward
deployment-compatible observations
same 50 Hz action interface
```

and train from scratch.

At that point PPO is finally learning on a plant we understand.

---

# 49. Immediate Next Technical Task

The next task should **not** be:

```text
another RL run
another reward
another interpolation
another arbitrary gain
```

The immediate task should be:

> Compute and audit the effective joint dynamics of the current model.

Specifically:

```text
J_eff_HX(q)
J_eff_HY(q)
J_eff_KN(q)
```

at representative configurations.

This should be paired with a torso/link inertia audit.

From those numbers we can answer:

- how different the joints actually are,
- what `500/10` means dynamically,
- what joint-specific gains would imply,
- whether missing drivetrain inertia is likely large enough to matter,
- what values the Gazebo model must reproduce.

This is the highest-value next step.

---

# 50. Updated Acceptance Criteria Before PPO

A future V3 actuator/controller should satisfy explicit deterministic criteria.

## Mechanical model

- mass and inertia values have traceable sources,
- MuJoCo and Gazebo descriptions correspond physically,
- no unexplained simulator-only inertia effects.

## Controller

- explicit mathematical definition,
- same setpoint semantics in both simulators,
- known controller frequency,
- gains designed from desired dynamics rather than guesswork.

## Actuator

- torque limits physically justified,
- no unexplained multi-kW ordinary motion,
- no extreme free-joint acceleration without a physical explanation,
- knee torque capability eventually configuration-dependent,
- added friction/inertia/delay parameters justified.

## Cross-simulator behavior

For equivalent deterministic inputs:

```text
MuJoCo and Gazebo should show comparable:
joint acceleration
velocity
settling
static load response
torque clipping
```

Exact contact behavior is not expected to match perfectly.

## RL interface

- actor observations available in ROS,
- actions have identical meaning in training and deployment,
- no simulator-only hidden servo.

---

# 51. Engineering Lessons From the JTC Investigation

## 51.1 Smoothness is not the same as physical correctness

Linear interpolation looked smoother in position but created derivative torque impulses.

Cubic interpolation made velocity continuous but created overshoot from bad derivative assumptions.

A smooth curve can still be dynamically wrong.

---

## 51.2 A controller should not invent information the policy does not provide without a good reason

The policy gives:

```text
position setpoint
```

not:

```text
future path
velocity plan
acceleration plan
```

Any interpolation method that requires those missing values adds modeling assumptions.

Those assumptions must be justified.

---

## 51.3 Deterministic controller tests are high value when tied to a specific hypothesis

The linear and cubic tests were useful because each answered one narrow question.

They prevented full RL runs on unsuitable controller semantics.

The lesson is not:

```text
"always do many tests"
```

It is:

> Test only when the result can accept or reject a concrete engineering hypothesis.

---

## 51.4 Standard ROS infrastructure can reduce deployment uncertainty

The earlier design leaned toward a custom controller because it offered complete control.

The later research showed that standard ROS 2 controllers already cover more of the required functionality than initially assumed.

Using standard components first reduces the number of independent implementation variables in the sim-to-sim transfer.

---

# 52. Controls Engineering Lessons

## 52.1 Gains are meaningless without the plant

`500/10` is not a complete controller description.

The real question is:

```text
What closed-loop bandwidth and damping does 500/10 produce
with the actual effective joint inertia?
```

This now becomes a quantitative design problem.

---

## 52.2 Static support and dynamic response remain separate concerns

The `300/8` experiment still matters.

It showed that reducing Kp to calm motion increases static sag.

This does not imply feed-forward is required.

It means controller design must account for both:

```text
load support
dynamic response
```

instead of optimizing only one standalone metric.

---

## 52.3 The actuator is part of the plant

A realistic joint is:

```text
controller
+
motor
+
transmission
+
inertia
+
friction
+
limits
+
delay
```

not just a torque clamp.

PPO learns the whole closed-loop plant.

---

# 53. Sim-to-Sim Lessons

The correct transfer order should be:

```text
robot mechanics parity
        ↓
controller parity
        ↓
actuator-response parity
        ↓
simple deterministic motion parity
        ↓
policy transfer
```

not:

```text
train policy
        ↓
export policy
        ↓
put into Gazebo
        ↓
debug all discrepancies simultaneously
```

This is now a core project methodology.

---

# 54. Updated "Do Not Do" List

For the current phase:

```text
DO NOT:
add more gait rewards
run more torque-limit sweeps
run another arbitrary Kp/Kd sweep
continue spline experiments
build a full guessed DC motor model
add arbitrary armature
add arbitrary delay
write a custom ROS controller prematurely
train another PPO policy before the plant audit
```

These are not permanently forbidden.

They are simply not justified by the current evidence.

---

# 55. Updated "Keep" List

```text
KEEP:
50 Hz policy
500 Hz low-level baseline
12 joint-position actions
deployment-compatible actor observations
effort-controlled Gazebo boundary
simple reward structure
current torque-limit scaling assumption
rich per-foot diagnostics
old native-actuator run as behavioral reference
deterministic actuator characterization tools
```

---

# 56. Updated "Investigate Next" List

```text
START:
effective joint inertia extraction
mass/inertia model audit
joint-specific closed-loop gain design
minimal ROS PidController + Gazebo effort test
MuJoCo/Gazebo mechanical parity work
reflected-inertia representation research
angle-dependent knee-limit implementation planning
```

---

# 57. Final Current Technical Interpretation

The strongest current interpretation is now:

> The V2 tapping/support pathology is unlikely to be solved by another reward term, another torque ceiling, another arbitrary knee gain, or another reference interpolation method.

The most likely remaining issue is that the explicit torque-controlled simulation does not yet represent the physical joint plant well enough, and the controller gains have never been designed relative to that plant.

The policy interface itself is probably much less exotic than we previously feared.

A perfectly reasonable baseline is:

```text
50 Hz position setpoints
+
500 Hz joint PD
+
realistic physical actuator dynamics
```

This is consistent with standard quadruped control practice and is portable to ROS 2 / Gazebo.

The main engineering work now lies below the PD law.

---

# 58. Core Question Going Forward

The project question has evolved through several stages.

Initially:

> How do we make PPO stop tapping?

Then:

> Which reward creates a proper trot?

Then:

> Which Kp/Kd makes the explicit actuator behave better?

Then:

> Should we interpolate the policy targets into a trajectory?

The strongest current question is:

> **What deterministic joint dynamics should a physically defensible Spot-like torque-controlled actuator have, and how do we represent those dynamics consistently in MuJoCo and Gazebo?**

That is the correct controls and sim-to-sim problem to solve next.

---

# 59. Current Recommended Next Step

The next concrete engineering task is:

```text
1. Extract MuJoCo generalized/effective joint inertias.
2. Evaluate HX/HY/KN across representative configurations.
3. Audit the torso and link inertia values.
4. Use those results to define target closed-loop bandwidth/damping.
5. Only then choose a new gain set or actuator-inertia model.
```

Once those dynamics are understood, the next Gazebo task should be a minimal standard ROS `PidController` effort test.

Only after deterministic MuJoCo and Gazebo behavior is coherent should the project resume full PPO locomotion training.

---

# 60. Reference Sources Relevant to This Update

## ROS 2

### PID Controller

ROS 2 Jazzy `pid_controller/PidController` documentation:

`https://control.ros.org/jazzy/doc/ros2_controllers/pid_controller/doc/userdoc.html`

Relevant point:

- designed for direct continuously updated references,
- can use reference/state interface pairs,
- can output effort,
- avoids using trajectory semantics when a trajectory is not actually required.

### JointTrajectoryController

`https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html`

Trajectory behavior:

`https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html`

Relevant points:

- position-only interpolation can produce discontinuous velocity,
- cubic interpolation depends on supplied velocity constraints,
- inconsistent derivatives can create overshoot.

### gz_ros2_control

`https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html`

Relevant point:

- effort command interface provides the cleanest boundary for an explicitly owned low-level controller.

---

## Boston Dynamics Spot

### Joint Control

`https://dev.bostondynamics.com/docs/concepts/joint_control/readme`

Relevant structure:

```text
q_des
qdot_des
Kp
Kd
tau_ff
```

### Transmission Data

`https://dev.bostondynamics.com/docs/concepts/joint_control/supplemental_data.html`

Relevant points:

```text
HX/HY gear ratio = 51
knee transmission is configuration dependent
```

### Knee Torque Limits

`https://dev.bostondynamics.com/docs/concepts/joint_control/knee_torque_limits.html`

Relevant point:

- knee joint torque capability varies strongly with knee configuration.

---

## MuJoCo

### Joint armature / inertia

`https://mujoco.readthedocs.io/en/latest/XMLreference.html`

Relevant relationship:

```text
J_reflected ≈ N² J_rotor
```

for simple geared rotational transmissions.

---

## Sim-to-Real / Quadruped Actuator Modeling

### Hwangbo et al.

`https://arxiv.org/abs/1901.08652`

Relevant lesson:

- actuator dynamics can be important enough to require learned actuator models for successful transfer.

### Tan et al.

`https://arxiv.org/abs/1804.10332`

Relevant lesson:

- motor behavior and latency materially affect quadruped sim-to-real transfer.

---

# 61. Final Summary

Since the previous four reports, the project tested a major V3 assumption directly.

The idea that the 50 Hz policy targets should be reconstructed into a smooth JTC-like trajectory was investigated.

### Linear interpolation

Result:

```text
continuous position
but discontinuous desired velocity
-> large derivative torque impulses
```

### Cubic interpolation

Result:

```text
continuous velocity
smaller knee derivative spikes
but inferred waypoint slopes
-> overshoot and artificial trajectory behavior
```

Therefore:

> Target interpolation is not the fundamental fix.

The deeper conceptual correction is:

> A 50 Hz RL position command is a continuously updated setpoint, not necessarily a trajectory waypoint.

This makes the simple:

```text
q_ref
qdot_ref = 0
PD
```

architecture valid again as the baseline.

For Gazebo deployment, the preferred first standard ROS controller should now be:

```text
pid_controller/PidController
```

with:

```text
position + velocity state
effort command
```

rather than JTC for the normal locomotion loop.

The dominant unresolved problems are now:

```text
effective joint inertia
reflected drivetrain inertia
joint-specific bandwidth/gain design
mechanical-model parity
angle-dependent knee transmission
friction / torque-speed / delay as later identified effects
```

The next phase should therefore begin with a **quantitative dynamics audit**, not another PPO run.

The project has now reached a much cleaner engineering position:

```text
understand the plant
-> design the controller
-> reproduce it in MuJoCo and Gazebo
-> then train the policy
```

That is the current technical direction.
