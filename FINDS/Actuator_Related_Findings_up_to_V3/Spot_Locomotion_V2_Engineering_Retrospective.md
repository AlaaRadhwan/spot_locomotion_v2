# Spot Locomotion V2 — Engineering Retrospective and Current Technical Understanding

## 1. Purpose of This Document

This document records the development path of the Spot Locomotion V2 project up to the current actuator A/B experiment.

The goal is not only to remember what was changed, but to preserve the **engineering reasoning** behind the changes:

- what behavior was observed,
- what we thought might be causing it,
- what experiment was designed to test that hypothesis,
- what the result actually showed,
- what was disproven,
- what remains uncertain,
- and what we learned as engineers and controls engineers.

This is important because reinforcement-learning locomotion can easily become a process of adding rewards until the robot appears to behave correctly. That can produce a policy that looks good while hiding a deeper modeling or control problem.

A major lesson of this project has been that a good-looking final policy is not enough. We want to understand **why it works**.

---

# 2. Project Goal

The long-term objective is to build a Spot-like quadruped locomotion pipeline that is:

- physically plausible,
- internally consistent,
- reproducible,
- compatible with sim-to-sim transfer,
- compatible with later ROS 2 / Gazebo deployment,
- and eventually suitable as the basis for a real low-level controller.

Exact Boston Dynamics fidelity is not required.

The engineering target is instead:

> A robot model, actuator model, observation space, action space, reward function, and training environment that agree with each other physically and behave predictably across simulators.

---

# 3. Current V2 Technical Baseline

## Simulation

- MjLab: 1.5.3
- MuJoCo: 3.10
- Python: 3.13.6
- Physics frequency: 500 Hz
- Physics timestep: 0.002 s
- Policy frequency: 50 Hz
- Policy timestep: 0.02 s
- Decimation: 10

## Robot

- Approximate total mass: 50.34 kg
- 12 actuated joints
- Joint order:

```text
fl_hx, fl_hy, fl_kn,
fr_hx, fr_hy, fr_kn,
hl_hx, hl_hy, hl_kn,
hr_hx, hr_hy, hr_kn
```

## Home Pose

```text
HX = 0.0 rad
HY = 1.04 rad
KN = -1.80 rad
```

for all four legs.

## Action Space

The policy outputs desired joint-position offsets from the home pose.

Action scales:

```text
HX = 0.30 rad
HY = 0.40 rad
KN = 0.50 rad
```

## Current Realistic/Explicit V2 Actuator

The current main V2 model uses MjLab `IdealPdActuatorCfg`.

Conceptually:

```text
tau = clip(
    Kp * (q_des - q) - Kd * qdot,
    -tau_max,
    +tau_max
)
```

with:

```text
Kp = 500
Kd = 10
Hip torque limit  = 69.52 Nm
Knee torque limit = 164.15 Nm
```

The torque limits were introduced deliberately to move away from the unrealistically high-authority actuator used in the old project.

---

# 4. Why V2 Was Started

The old Spot locomotion project successfully learned locomotion.

However, the actuator setup was not considered sufficiently realistic for deployment.

The old model used MuJoCo native position actuators with approximately:

```text
Kp = 500
Kd = 40
effort limit = +/-1000 Nm
```

This gave the policy an extremely high-authority servo.

That was useful for learning locomotion, but it created concerns for sim-to-sim transfer because the policy was trained in a system that could produce unrealistically large corrective torques.

V2 was therefore built from scratch around a more explicit and constrained actuator model.

The assumption was:

> If the robot model and actuator limits are more physically realistic, the learned locomotion should also become more transferable.

This assumption was reasonable, but the project eventually showed that changing the actuator changed the learned locomotion behavior much more profoundly than expected.

---

# 5. Early V2 Validation

Before serious RL training, we validated basic model behavior.

Tests included:

- fixed-base joint-step response,
- free-standing stability,
- explicit actuator checks,
- torque-limit checks,
- joint-order checks,
- zero-action standing,
- lower-leg contact checks.

The standing model could remain upright.

The basic joint/actuator ordering was correct.

The robot model was therefore not obviously broken at the level of:

- mass,
- joint indexing,
- home pose,
- actuator mapping,
- or gross numerical instability.

This was important because it meant later locomotion failures could not simply be dismissed as a completely invalid model.

---

# 6. First Major Locomotion Problem: Collapse and Failure to Lift the Feet

During early fresh RL runs, several bad behaviors appeared.

Observed behaviors included:

- legs barely lifting,
- feet skidding across the ground,
- rapid tapping,
- policies collapsing onto the belly,
- policies that remained upright but did not develop clear stepping.

At one point there was concern that action clipping or actuator configuration might be suppressing useful leg motion.

We removed or changed some of these constraints.

The collapse behavior improved, but the locomotion was still not good.

This was the first indication that there was no single obvious bug.

---

# 7. Foot-Slip Penalty Investigation

One important early finding involved the foot-slip penalty.

## Strong Foot-Slip Penalty

With:

```text
foot_slip = -0.25
```

the policy became overly conservative.

Observed behavior:

- weak leg movement,
- little useful stepping,
- poor command tracking,
- feet stayed close to the ground.

Interpretation:

> The policy learned that moving the feet aggressively was too expensive.

We reduced the penalty to:

```text
foot_slip = -0.10
```

This restored useful movement.

However, a new failure mode appeared:

> rapid foot tapping.

This was an important lesson.

A reward weight that appears physically sensible can still eliminate the exploratory motion required for PPO to discover locomotion.

---

# 8. Diagnosing the Tapping

We did not want to immediately add a gait reward.

Instead, diagnostics were added.

These included:

- mean completed air time,
- mean completed stance time,
- foot contact switch rate,
- mean mechanical power,
- actuator utilization,
- actuator saturation fraction,
- vertical base velocity RMS,
- landing force,
- foot slip,
- action acceleration,
- tracking errors.

A mature early baseline showed approximately:

```text
mean air time       ~ 0.062 s
mean stance time    ~ 0.120 s
switch rate         ~ 10.8 Hz
power               ~ 139 W
saturation          ~ 0.2%
vertical RMS        ~ 0.113 m/s
slip                ~ 0.20 m/s
```

The key inference was:

> The rapid cadence was not being caused by actuator saturation.

The actuator saturation percentage was extremely small.

It was also not simply a whole-body pogo-stick behavior because the base vertical velocity was moderate.

The robot had genuinely learned a high-cadence local optimum.

---

# 9. Literature and Existing Quadruped Reward Research

At this point we compared our reward structure with published and open-source locomotion systems.

Important references included:

- Legged Gym,
- IsaacLab Spot,
- Unitree-style locomotion environments,
- ETH quadruped locomotion work,
- Rapid Locomotion via Reinforcement Learning.

A recurring lesson from the literature was:

> Reward functions are not universal recipes.

The same reward term can produce very different behavior depending on:

- actuator model,
- robot mass,
- foot geometry,
- control frequency,
- action representation,
- observation space,
- contact solver,
- command distribution.

We also saw examples in the literature where policies exploited apparently reasonable reward terms.

This made us more cautious about solving every visible behavior with another reward.

---

# 10. Native MjLab Air-Time Reward Experiment

MjLab's native air-time reward rewards the foot while its current air time lies within a given interval.

A test was made using approximately:

```text
air-time reward weight = +0.10
air-time interval       = 0.05 to 0.50 s
```

This improved cadence substantially.

A mature run reached roughly:

```text
air time        ~112 ms
stance          ~145 ms
switch rate     ~7.56 Hz
power           ~140 W
slip            ~0.166 m/s
```

This was a real improvement over the rapid tapping baseline.

However, visual inspection showed growing diagonal asymmetry.

One diagonal pair was participating differently from the other.

This became the next major failure mode.

---

# 11. Adding an Explicit Diagonal-Trot Reward

We then studied the official IsaacLab Spot gait reward.

It explicitly encourages synchronization between:

```text
FL + HR
FR + HL
```

and opposite phase between those diagonal pairs.

This was attractive because it encoded the expected trot structure.

## Strong Gait Reward Test

A relatively strong gait reward produced:

```text
air time        ~58 ms
stance          ~43 ms
switch rate     ~20.3 Hz
power           ~674 W
saturation      ~7.9%
slip            ~0.67 m/s
```

The diagonal structure looked correct, but the robot discovered an extreme high-frequency gait.

The policy was maximizing the gait reward by switching contacts very rapidly.

This was a classic reward exploit.

### Lesson

> Correct phase relationships do not imply physically reasonable cadence.

The reward told the robot **which feet should agree**, but not how slowly or naturally they should move.

---

# 12. Rebalancing Gait and Air-Time Rewards

The gait reward was weakened and air-time reward increased.

A later run reached approximately:

```text
air time        ~208 ms
stance          ~182 ms
switch rate     ~5.1 Hz
power           ~236 W
slip            ~0.146 m/s
```

This looked much calmer numerically.

However, the robot found a new exploit:

> One diagonal pair supported the robot while the opposite pair remained airborne for long periods and tapped the ground briefly.

The reward was technically satisfied, but the gait was not balanced.

This was a major turning point.

The problem was no longer simply cadence.

It was **support participation**.

---

# 13. Faithful IsaacLab Spot Air/Contact Timing Reward

To avoid inventing another reward, we ported the official IsaacLab Spot timing reward more faithfully.

The reward encourages both stance and swing duration up to a `mode_time`, rather than simply rewarding air time.

Initial result:

```text
air time        ~51 ms
stance          ~64 ms
switch rate     ~18.4 Hz
power           ~175 W
```

Again, diagonal coordination looked correct, but cadence was too high.

The gait reward was still contributing much more than the timing reward.

Increasing the timing reward improved things:

```text
air time        ~142 ms
stance          ~173 ms
switch rate     ~6.8 Hz
power           ~151 W
slip            ~0.126 m/s
```

Visually this was one of the first clearly improved policies.

However, the user noticed that:

> One diagonal pair was doing the real support/walking while the opposite pair mostly stayed unloaded and tapped periodically.

This observation turned out to be correct.

---

# 14. Why Aggregate Metrics Were Misleading

At first we only logged four-foot averages.

For example:

```text
mean air time
mean stance time
mean switch rate
```

These looked reasonably healthy.

But they could hide a situation like:

```text
FL/HR: very high stance participation
FR/HL: very high air participation
```

The average could still look normal.

We therefore added per-foot diagnostics:

```text
air_time_FL_s
air_time_FR_s
air_time_HL_s
air_time_HR_s

stance_time_FL_s
stance_time_FR_s
stance_time_HL_s
stance_time_HR_s
```

and contact-pattern diagnostics:

```text
diag_FL_HR_fraction
diag_FR_HL_fraction
all_four_contact_fraction
flight_fraction
other_contact_pattern_fraction
```

This was one of the most important diagnostic improvements in the project.

---

# 15. Confirmed Support-Pair Asymmetry

With the new diagnostics, the asymmetry became quantitative.

A representative run showed approximately:

```text
FL duty factor ~78%
HR duty factor ~79%

FR duty factor ~35%
HL duty factor ~37%
```

The exact FL+HR-only support pattern appeared roughly four times as often as FR+HL-only.

This confirmed the visual observation.

The policy was not producing a balanced trot.

It was assigning very different roles to the two diagonal pairs.

---

# 16. Support-Balance Reward Experiment

A targeted support-balance penalty was introduced.

The idea was deliberately not to penalize instantaneous diagonal differences because a proper trot should alternate between diagonal pairs.

Instead, a roughly one-second exponential moving average of pair contact occupancy was compared.

Penalty:

```text
persistent mismatch between pair A and pair B
```

with:

```text
pair A = FL + HR
pair B = FR + HL
```

The first test used a small weight.

The result showed that the reward correctly detected the asymmetry.

However, the cost was too small compared with the positive gait and timing rewards.

At mature behavior:

```text
gait + timing reward  ~ +1.09
support balance cost  ~ -0.046
```

The imbalance was therefore still cheap.

Before increasing the support-balance reward further, we decided to reconsider whether reward shaping itself had become the problem.

---

# 17. Decision to Go Back to Basics

At this point the reward function had accumulated several terms:

- air-time shaping,
- diagonal gait,
- timing reward,
- air-time variance,
- support balance.

Each new term fixed one exploit while opening another.

This led to the decision:

> Stop solving every visible gait problem with another reward.

We returned to the simple reward architecture used in the old successful Spot project.

The simplified reward structure was:

```text
track_linear_velocity   +2.0
track_angular_velocity  +2.0
upright                  +1.0
pose                     +1.0

air_time                  0.0
foot_clearance           -2.0
foot_swing_height        -0.25
action_rate_l2           -0.10
foot_slip                -0.10
soft_landing             -1e-5
```

with optional:

```text
excessive_air_time       -0.35
```

only as a guardrail against extreme long airborne phases.

All diagnostic metrics were retained.

---

# 18. Back-to-Basics Result

This experiment was extremely informative.

The simplified V2 reward reproduced almost exactly the original V2 failure mode.

Typical values again became roughly:

```text
air time        ~62 ms
stance          ~120–130 ms
switch rate     ~10–11 Hz
power           ~135–140 W
vertical RMS    ~0.11 m/s
```

Most importantly, the per-foot diagnostics again showed:

```text
FL / HR = dominant support pair
FR / HL = more active stepping pair
```

This proved:

> The asymmetry was not created by the complicated gait rewards.

The explicit gait rewards had influenced the shape of the failure, but they were not the root cause.

---

# 19. Removing the Excessive-Air-Time Guardrail

The `excessive_air_time` penalty was then removed completely.

This was another useful A/B test.

Without the penalty:

- FR and HL air time became much longer,
- FL and HR remained heavily grounded,
- the FL+HR support dominance became stronger.

The reward contribution of `excessive_air_time` had previously looked almost negligible, but removing it showed that it was still acting as a useful guardrail against extreme foot abandonment.

### Lesson

> A reward term can have a small average contribution while still strongly shaping rare but important behavior.

A sparse penalty should not be judged only by its mean reward value.

---

# 20. Comparing V2 With the Old Successful Project

The old `anymal_c_velocity` repository was then inspected directly.

This comparison was one of the most important parts of the project.

Many things turned out to be essentially the same:

## Same or Very Similar

- robot mass/inertias,
- home pose,
- joint ordering,
- action scales,
- 500 Hz physics,
- 50 Hz policy,
- command distribution,
- basic tracking rewards,
- upright reward,
- pose reward,
- clearance reward weight,
- swing-height reward weight,
- action-rate penalty,
- slip penalty,
- soft-landing penalty.

This meant the current V2 locomotion task was already much closer to the old successful task than expected.

The major remaining difference was the actuator.

---

# 21. Major Actuator Difference

## Old Successful Project

The old project used MuJoCo native position actuators with approximately:

```xml
<joint actuatorfrcrange="-1000 1000"/>
<position kp="500" kv="40"/>
```

In other words:

```text
Kp = 500
Kd = 40
torque authority ~ +/-1000 Nm
```

The actuator was a **native MuJoCo position servo**.

## Current V2

The current V2 uses MjLab `IdealPdActuatorCfg`.

Conceptually:

```text
tau = Kp(q_des - q) - Kd*qdot
```

then torque clipping.

Parameters:

```text
Kp = 500
Kd = 10
hip limit  = 69.52 Nm
knee limit = 164.15 Nm
```

This changed three major things at once:

1. actuator implementation,
2. damping,
3. available torque authority.

This became the strongest remaining causal hypothesis.

---

# 22. Hypothesis: The New Actuator Changed the RL Optimization Landscape

The central hypothesis became:

> The realistic explicit actuator made some locomotion strategies much harder or more expensive, so PPO discovered a different local optimum.

One possible strategy is:

```text
FL + HR = stable support/anchors
FR + HL = active movement pair
```

From PPO's perspective, this can still:

- keep the body upright,
- track velocity,
- use little power,
- avoid saturation,
- avoid falling.

If the reward does not explicitly require equal support participation, this can be a perfectly valid solution.

The old high-authority servo may have made normal alternating four-leg locomotion much easier to discover.

This was still only a hypothesis.

It needed an A/B test.

---

# 23. First Legacy-Actuator A/B Attempt — Failed

We first tried to reproduce the old actuator capability while keeping the current `IdealPdActuatorCfg`.

Test configuration:

```text
IdealPdActuatorCfg
Kp = 500
Kd = 40
effort limit = +/-1000 Nm
```

This test exploded numerically.

Symptoms:

```text
mean episode length       ~4 steps
mechanical power          ~521 kW
foot slip                 ~15 m/s
flight fraction           ~59%
large illegal contacts
NaN observations
```

The run terminated with:

```text
ValueError:
actor observation group contains NaN values
```

This was not a locomotion result.

It was a numerical/dynamic failure.

---

# 24. What the Failed A/B Test Taught Us

The failed test revealed an important actuator-modeling detail.

The old actuator was **not simply an explicit PD controller with large torque limits**.

The old system used MuJoCo's native position actuator.

The current `IdealPdActuatorCfg` computes the PD torque externally and sends it through a motor.

A high-stiffness native actuator and a high-stiffness explicit PD motor are not numerically equivalent.

MuJoCo's native actuator participates differently in the implicit dynamics solve.

### Important Controls Lesson

Two controllers with equations that look similar on paper can behave very differently when implemented at different locations in the simulation/control stack.

Implementation matters.

---

# 25. Correct Legacy-Actuator A/B Test

We then switched the test to MjLab's native/builtin position actuator.

Configuration:

```text
native MuJoCo position actuator
Kp = 500
Kd = 40
effort limit = +/-1000 Nm
```

Everything else remained the current V2 environment:

- MjLab 1.5.3,
- same robot geometry,
- same masses,
- same observations,
- same action scaling,
- same rewards,
- same commands,
- same PPO,
- same contact diagnostics.

This was the cleanest actuator experiment so far.

---

# 26. Legacy Native-Position Actuator Smoke Test

The smoke test remained numerically stable.

Early values were still rough because the policy was nearly random:

```text
power           ~1–2 kW
slip            ~1.5 m/s
landing force   ~600 N
episode length  ~16–18 steps
```

However:

- there were no NaNs,
- the simulation did not explode,
- all feet showed comparable early timing,
- training could continue.

This confirmed that the actuator implementation, not just the gains, mattered.

---

# 27. Major Current Result: Natural Locomotion Returned

During the full legacy native-position actuator training run, Spot began behaving visually much like the old successful MjLab 1.1 policy.

The TensorBoard data supported the visual observation.

At roughly iterations 180–280:

```text
contact switch rate    ~4.5 Hz
mechanical power       ~115 W
vertical RMS           ~0.075 m/s
slip                   ~0.173 m/s
landing force          ~64 N
XY velocity error      ~0.326
yaw error              ~0.384
```

Compared with the current explicit-PD V2 baseline at a similar training stage, the legacy native actuator was:

- slower in cadence,
- lower in mechanical power,
- smoother vertically,
- lower in slip,
- much softer in landing,
- better in linear tracking,
- better in angular tracking,
- smoother in action changes.

Most importantly, diagonal support participation became much more balanced.

Representative contact fractions:

```text
FL + HR only  ~24%
FR + HL only  ~17%
```

instead of the extreme imbalance seen in the explicit-PD V2 policy.

This is the strongest causal evidence obtained so far.

---

# 28. What We Can Say With Confidence Now

## Confirmed

### 1. The complicated gait rewards were not the root cause.

Removing them did not remove the asymmetric support behavior.

### 2. The basic robot model is capable of learning good locomotion.

The same current V2 robot geometry and mass model behaves very well when paired with the legacy native actuator.

### 3. The explicit V2 actuator changes the learned locomotion solution substantially.

Changing the actuator while keeping most of the RL environment fixed produced a major behavioral change.

### 4. The old high-authority gains cannot simply be copied into the explicit PD actuator.

Doing so caused numerical/dynamic instability.

### 5. Native MuJoCo position actuation and explicit PD actuation are not equivalent.

Even when their nominal gains look similar.

### 6. Aggregate gait metrics are insufficient.

Per-foot and per-contact-pattern diagnostics were necessary to expose the true failure mode.

---

# 29. What Is Still Not Proven

The successful legacy actuator test changed three important variables together:

```text
IdealPdActuator      -> native position actuator
Kd = 10              -> Kd = 40
realistic limits     -> +/-1000 Nm
```

Therefore we do **not yet know** which factor is primarily responsible for the improved gait.

Possible explanations:

1. native implicit actuator implementation,
2. higher damping,
3. greater torque authority,
4. interaction among all three.

We should not conclude:

> "Spot needs +/-1000 Nm."

That is not supported by the experiment.

The next experiments must isolate these variables.

---

# 30. Recommended Next A/B Experiments

## Experiment A — Native Position Actuator With Realistic Limits

Keep:

```text
native MuJoCo position actuator
```

but use:

```text
Kp = 500
Kd = 10
hip limit  = 69.52 Nm
knee limit = 164.15 Nm
```

Question:

> Does changing only the actuator implementation restore the good gait?

If yes, the main issue is likely explicit-vs-native actuator dynamics rather than torque authority.

---

## Experiment B — Native Position Actuator With Kd = 40 and Realistic Limits

```text
native actuator
Kp = 500
Kd = 40
realistic torque limits
```

Question:

> Does higher damping matter when torque authority is realistic?

---

## Experiment C — Native Position Actuator With Kd = 10 and High Limits

```text
native actuator
Kp = 500
Kd = 10
+/-1000 Nm
```

Question:

> Is high torque authority more important than damping?

---

## Experiment D — Explicit PD With Carefully Tuned Gains

Only after the native-actuator experiments should we return to the explicit actuator and systematically tune:

- Kp,
- Kd,
- torque limits,
- possibly velocity-dependent or angle-dependent limits.

The goal would be to recover the favorable locomotion optimization landscape without returning to an unrealistic actuator.

---

# 31. Controls Engineering Reflections

## 31.1 A Stable Standing Controller Is Not Automatically a Good Locomotion Controller

The V2 explicit PD actuator could stand reliably.

That did not mean it produced a good RL locomotion landscape.

Standing tests mostly examine:

- static stiffness,
- damping around one equilibrium,
- disturbance recovery.

Locomotion additionally requires:

- large joint excursions,
- fast swing motion,
- impact recovery,
- dynamic load transfer,
- repeated contact transitions,
- rapid changes in desired joint position.

A gain set that works well for standing may still be poor for learned dynamic locomotion.

---

## 31.2 Actuator Dynamics Are Part of the Plant

In control theory it is tempting to think:

```text
robot mechanics = plant
controller = separate
```

In practice, for RL locomotion:

```text
effective plant =
robot mechanics
+ actuator model
+ torque limits
+ servo dynamics
+ numerical integration
+ contact model
```

Changing the actuator changes the system that PPO is learning to control.

The policy is not simply learning "how Spot walks."

It is learning:

> how this particular Spot + servo + contact solver + action interface moves.

---

## 31.3 Saturation Is Not Just a Safety Constraint

Torque limits also shape which trajectories are easy for the optimizer to discover.

Even when mature saturation percentages are small, the limits can affect:

- early exploration,
- recovery from bad actions,
- contact transitions,
- swing acceleration,
- how quickly the robot can correct body motion.

Therefore:

> Low final saturation does not prove that torque limits were irrelevant during learning.

---

## 31.4 Controller Implementation Matters as Much as the Equation

Two actuator models can both appear to implement:

```text
tau = Kp*error - Kd*velocity
```

but still behave differently because one is:

- integrated implicitly inside MuJoCo,

while the other is:

- evaluated explicitly and applied as motor torque.

A controls engineer must care not only about the control law, but also:

- sample rate,
- computation point,
- delay,
- actuator representation,
- saturation location,
- solver coupling,
- discretization.

---

## 31.5 Do Not Tune Around an Unknown Plant

Much of the early reward tuning was effectively trying to compensate for behavior created by an actuator model we had not yet fully understood.

This is a common engineering mistake:

> tuning the controller or objective before characterizing the plant.

The project improved significantly when we reversed the order:

1. observe behavior,
2. instrument it,
3. characterize the actuator,
4. isolate variables,
5. then decide whether the reward actually needs to change.

---

# 32. Reinforcement Learning Reflections

## 32.1 PPO Is Not Trying to Produce a Human-Recognizable Gait

PPO maximizes reward.

If the easiest way to achieve:

- velocity tracking,
- upright stability,
- low action rate,
- low slip,

is to keep two feet as anchors and use the other two aggressively, PPO can legitimately discover that solution.

A visually strange gait is not necessarily a failed optimizer.

It may be an excellent optimizer exploiting an incomplete objective or unusual plant.

---

## 32.2 Every New Reward Creates a New Optimization Surface

We repeatedly observed:

```text
slip penalty too strong
-> feet barely move

air-time reward
-> longer swing, but asymmetry

strong gait reward
-> extremely high cadence

timing reward
-> better cadence, but support-pair exploit

support-balance reward
-> detects imbalance, but competes with other rewards
```

This reinforced an important principle:

> Reward shaping should be used only after the physical/control system is understood well enough that we know what behavior needs shaping.

---

## 32.3 A Reward Can Be Correct and Still Be Exploitable

The diagonal gait reward correctly encoded trot phase relationships.

The policy still exploited it.

The timing reward correctly encouraged reasonable phase durations.

The policy still found a way to satisfy it with unequal support participation.

This is why reward design must be considered as an optimization problem, not as a list of desired behaviors.

---

## 32.4 Average Metrics Can Hide Structured Failure

The four-foot mean air time looked reasonable.

The actual robot had:

```text
two feet mostly grounded
two feet mostly airborne
```

The average was misleading.

Engineering diagnostics should preserve structure whenever possible.

For quadrupeds this means per-foot metrics are usually more informative than global averages.

---

# 33. General Engineering Reflections

## 33.1 Change One Variable at a Time

The most useful experiments in this project were the clean A/B tests.

Examples:

- foot-slip -0.25 -> -0.10,
- excessive-air-time on -> off,
- complicated reward -> simple reward,
- explicit actuator -> native actuator.

When several things changed together, interpretation became much harder.

---

## 33.2 A Failed Experiment Can Still Be Valuable

The explicit-PD `Kp=500, Kd=40, +/-1000 Nm` test failed catastrophically.

But it taught us something fundamental:

> the old actuator could not be represented simply by copying its gain numbers into the new actuator class.

This prevented us from drawing a false conclusion from an invalid comparison.

---

## 33.3 Preserve Diagnostics Even When Simplifying the Controller

When we went back to a simple reward function, we kept the new diagnostics.

That was crucial.

Simplifying the objective should not mean reducing observability.

A strong engineering workflow is:

```text
simple system
+
rich instrumentation
```

rather than:

```text
complex system
+
poor instrumentation
```

---

## 33.4 Visual Inspection Is Still Important

TensorBoard did not initially tell us:

> "FL and HR are doing most of the support while FR and HL tap."

The visual observation came first.

Then diagnostics were designed to test it quantitatively.

The best workflow was:

```text
observe
-> form hypothesis
-> add measurement
-> test
-> quantify
```

not:

```text
look only at scalar reward
-> tune blindly
```

---

# 34. How Our Engineering Process Improved During the Project

Early in the project the loop sometimes looked like:

```text
bad gait
-> adjust reward
-> train
-> inspect video
-> adjust another reward
```

Later it became:

```text
bad gait
-> identify exact failure mode
-> form physical/control hypothesis
-> add targeted diagnostics
-> make one isolated change
-> compare quantitatively
-> accept/reject hypothesis
```

That transition is arguably one of the most important outcomes of the project.

The technical result matters.

The improvement in engineering method matters even more.

---

# 35. Current Best Technical Interpretation

As of now, the strongest evidence suggests:

> The asymmetric tapping/support behavior in Spot V2 is primarily associated with the actuator/control implementation, not with the reward function alone.

The current V2 robot can learn natural locomotion when paired with the old-style native high-authority position servo.

Therefore:

- the robot geometry is not fundamentally incapable of walking,
- the RL task is not fundamentally incapable of learning walking,
- the simple reward is not fundamentally insufficient,
- the main investigation should now focus on actuator dynamics and torque authority.

The next goal is **not** to keep the old +/-1000 Nm actuator.

The next goal is:

> determine what property of the old actuator made locomotion easy to learn, then reproduce the useful dynamic behavior using a physically plausible actuator model.

That is the core controls problem now.

---

# 36. Current Engineering Question

The project has progressed from:

> "How do we make PPO learn a better gait?"

to the much better question:

> "What actuator dynamics make physically realistic locomotion both achievable and learnable?"

That is a more fundamental problem.

And it is much closer to the real sim-to-real challenge.

---

# 37. Final Reflection

This project began as a reinforcement-learning locomotion task.

It has gradually become an exercise in:

- system identification,
- actuator modeling,
- contact dynamics,
- numerical simulation,
- experimental design,
- reward optimization,
- control architecture,
- and engineering reasoning.

One of the most important lessons is that reinforcement learning does not remove the need for controls engineering.

It makes good controls engineering even more important.

A learned controller can only optimize the system that we give it.

If the actuator model, contact model, limits, observations, or objectives are wrong, PPO can become extremely good at solving the wrong problem.

The engineering responsibility is therefore not merely to train a policy.

It is to construct an environment in which:

```text
the easiest solution for the optimizer
is also a solution we would accept physically.
```

That is the real standard this project should continue aiming for.
