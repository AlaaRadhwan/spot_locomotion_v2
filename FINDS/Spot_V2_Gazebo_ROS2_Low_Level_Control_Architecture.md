# Spot Locomotion V2 — Gazebo / ROS 2 Low-Level Control Architecture

## Purpose

This document records the current understanding of how the Spot V2 locomotion policy should ultimately be deployed from MjLab / MuJoCo into ROS 2 Jazzy + Gazebo Harmonic.

The main goal is to make sure that the actuator/controller model used during RL training can be reproduced deliberately in Gazebo rather than relying on simulator-specific hidden position servos.

The long-term architecture should also remain compatible with eventual hardware deployment.

The central design requirement is:

> The learned policy should not depend on a MuJoCo-specific actuator implementation that cannot be reproduced in ROS 2 / Gazebo.

The low-level controller and actuator constraints should therefore be explicit and simulator-independent wherever possible.

---

# 1. Important Clarification About the Old Successful MjLab 1.1 Run

The successful older Spot locomotion run did not literally use a class called `BuiltinPdActuator`.

It used:

```python
XmlPositionActuatorCfg
```

which loaded native MuJoCo `<position>` actuators directly from the MJCF.

The old Spot XML contained approximately:

```xml
<option integrator="implicitfast"/>

<joint actuatorfrcrange="-1000 1000"/>

<position
    kp="500"
    kv="40"
    inheritrange="1"
/>
```

for the leg joints.

Therefore the old successful actuator was:

```text
native MuJoCo position servo
Kp = 500
Kd = 40
very high torque authority
implicitfast integration
```

This belongs to the same broad family as MjLab's newer `BuiltinPdActuator`, because both rely on MuJoCo-native position / velocity actuator behavior.

The newer `BuiltinPdActuator` is more flexible because it can represent both:

```text
q_des
qdot_des
```

using native MuJoCo actuator terms.

However, for `qdot_des = 0`, the old native position actuator already gave us a closely related experiment.

Therefore:

> Repeating the old `500/40 ±1000 Nm` setup using `BuiltinPdActuator` is not a useful priority.

We already know that native MuJoCo position-servo actuation can learn natural locomotion.

The more important question is whether that type of actuator can be reproduced faithfully in Gazebo.

---

# 2. Deployment Goal

The desired final system is:

```text
velocity command
      |
      v

RL policy
50 Hz
      |
      | 12 joint targets
      v

low-level controller
~500 Hz
      |
      | 12 joint torques
      v

ROS 2 control layer
      |
      v

Gazebo Harmonic
```

and later, ideally:

```text
same RL policy
+
same low-level controller
      |
      v

real robot hardware interface
```

The learned policy should therefore be simulator-agnostic.

---

# 3. ROS 2 / Gazebo Control Stack

For Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic, the relevant integration package is:

```text
gz_ros2_control
```

not the older `gazebo_ros2_control` package used with Gazebo Classic.

Conceptually:

```text
ROS 2 node / controller
        |
        v
controller_manager
        |
        v
ros2_control command interface
        |
        v
gz_ros2_control
        |
        v
Gazebo simulated joint
        |
        v
Gazebo physics
```

`gz_ros2_control` exposes simulated Gazebo joints through the standard `ros2_control` hardware interface abstraction.

---

# 4. The Three Main Joint Command Interfaces

Gazebo / `gz_ros2_control` can expose joints using:

```text
position
velocity
effort
```

These interfaces do not mean the same thing physically.

This distinction is critical for sim-to-sim transfer.

---

# 5. Why the Position Interface Is Not the Right Final Interface

A Gazebo `position` command interface does not necessarily implement the same dynamics as MuJoCo's native `<position>` actuator.

The Gazebo ROS-control integration uses its own internal mechanism to move the simulated joint toward the commanded position.

This means the architecture:

```text
RL policy
    |
    v
position target
    |
    v
Gazebo position interface
```

would allow Gazebo's internal controller implementation to become part of the plant.

Meanwhile training in MuJoCo would use a different internal native servo.

The system would effectively become:

```text
TRAINING:
MuJoCo position servo

DEPLOYMENT:
Gazebo position servo
```

with the assumption that the two behave similarly.

That is exactly the type of hidden simulator dependency the project is trying to avoid.

Therefore:

> The Gazebo position interface should not be the primary actuator interface for the final Spot V2 locomotion architecture.

---

# 6. Why the Velocity Interface Is Also Not Ideal

The Gazebo velocity interface directly commands joint velocity behavior.

This can be significantly more idealized than a real motor / torque-controlled joint.

For quadruped locomotion, the policy should experience:

- inertia,
- torque limitations,
- contact loads,
- acceleration limits,
- actuator dynamics.

A direct velocity interface hides too much of that behavior.

Therefore the velocity interface is also not a good foundation for the final low-level quadruped actuator model.

---

# 7. Why the Effort Interface Is the Preferred Boundary

The `effort` interface is the cleanest simulator-independent boundary.

The controller computes a joint torque:

```text
tau_cmd
```

and Gazebo applies that torque to the simulated joint.

Conceptually:

```text
low-level controller
       |
       | tau_cmd
       v
effort interface
       |
       v
Gazebo joint
       |
       v
Gazebo rigid-body dynamics
```

This gives us direct control over the low-level servo law.

The same conceptual interface can be reproduced in MuJoCo:

```text
low-level controller
       |
       | tau_cmd
       v
MuJoCo torque actuator
       |
       v
MuJoCo rigid-body dynamics
```

This is the most important architectural conclusion from the Gazebo research so far.

---

# 8. Preferred Sim-to-Sim Architecture

## MuJoCo training

```text
                    POLICY
                     50 Hz
                       |
                       | q_target
                       v

              LOW-LEVEL CONTROLLER
                     500 Hz
                       |
                       | tau_cmd
                       v

              ACTUATOR CONSTRAINTS
                       |
                       | tau_actual
                       v

                  MUJOCO PHYSICS
```

## Gazebo deployment

```text
                    POLICY
                     50 Hz
                       |
                       | q_target
                       v

              LOW-LEVEL CONTROLLER
                     500 Hz
                       |
                       | tau_cmd
                       v

              ACTUATOR CONSTRAINTS
                       |
                       | tau_actual
                       v

              ros2_control effort interface
                       |
                       v

                  GAZEBO PHYSICS
```

The equations above the physics engine should be kept as identical as practical.

---

# 9. ros2_control Runtime Loop

`ros2_control` uses a standard control loop:

```text
READ
  |
  | read joint position / velocity / effort
  v

UPDATE
  |
  | controller computes new commands
  v

WRITE
  |
  | write command interfaces
  v
```

The `controller_manager` executes the controller update at a configured frequency.

For Spot V2, a candidate controller update rate is:

```text
500 Hz
```

to match the current MuJoCo low-level servo rate.

The policy can remain:

```text
50 Hz
```

This gives:

```text
policy period     = 20 ms
controller period =  2 ms
```

---

# 10. Gazebo Physics Rate

The controller update frequency cannot be considered independently of the Gazebo physics timestep.

A clean initial target is either:

```text
Option A

Gazebo physics:       500 Hz
controller:           500 Hz
policy:                50 Hz
```

or:

```text
Option B

Gazebo physics:      1000 Hz
controller:           500 Hz
policy:                50 Hz
```

Option A exactly mirrors the current MjLab timing:

```text
physics dt = 0.002 s
controller dt = 0.002 s
policy dt = 0.020 s
```

Option B gives the Gazebo physics engine one extra integration step per controller update.

The final choice should be based on:

- contact stability,
- performance,
- real-time factor,
- controller scheduling behavior.

It should not be chosen arbitrarily.

---

# 11. Possible ROS 2 Low-Level Controller Implementations

There are three serious options.

---

# 12. Option A — JointGroupEffortController

A `JointGroupEffortController` forwards torque values to effort command interfaces.

Architecture:

```text
custom ROS node
      |
      | calculate PD / PDFF torque
      v
JointGroupEffortController
      |
      v
Gazebo effort interface
```

Advantages:

- simple,
- easy to understand,
- full control over torque calculation.

Disadvantages:

- the 500 Hz control loop would live in a separate ROS node,
- ordinary ROS pub/sub is not ideal for the most deterministic low-level servo,
- more timing / latency dependence may enter the loop.

This could be useful for early experiments but is probably not the best final architecture.

---

# 13. Option B — JointTrajectoryController With Effort Output

ROS 2's standard `joint_trajectory_controller` can use:

```text
state interfaces:
position
velocity

command interface:
effort
```

In effort mode it can implement a control law of the form:

```text
tau =
Kp * (q_des - q)
+
Ki * integral(error)
+
Kd * (qdot_des - qdot)
+
feedforward terms
```

Setting:

```text
Ki = 0
```

gives a PD / PDFF-style controller.

It also supports trajectory interpolation.

This makes it conceptually similar to the public Boston Dynamics joint-control architecture:

```text
q_des
qdot_des
tau_ff
Kp
Kd
```

Advantages:

- standard ROS 2 controller,
- effort interface,
- built-in trajectory interpolation,
- mature ROS infrastructure,
- less custom code.

Disadvantages:

- more awkward to implement:
  - angle-dependent knee limits,
  - custom torque-speed envelopes,
  - actuator delays,
  - Spot-specific nonlinear actuator behavior,
  - custom policy-target handling.

This remains a legitimate option.

---

# 14. Option C — Custom ros2_control Controller

A custom Spot controller could run directly inside the `controller_manager`.

Conceptually:

```text
SpotLowLevelController
```

would claim:

```text
STATE INTERFACES

12 x position
12 x velocity

COMMAND INTERFACES

12 x effort
```

Its `update()` method could execute the exact low-level law we define for training.

Example conceptual law:

```text
q_des, qdot_des
       |
       v

tau_cmd =
    Kp * (q_des - q)
  + Kd * (qdot_des - qdot)
  + tau_ff

       |
       v

actuator limits
       |
       v

effort command
```

Advantages:

- exact control over equations,
- easiest way to match MuJoCo implementation,
- easy to add Spot-specific behavior,
- easy to introduce:
  - angle-dependent knee torque limit,
  - torque-speed envelope,
  - interpolation,
  - delay,
  - rate limits,
  - actuator response models,
  - feed-forward.

Disadvantages:

- more code to maintain,
- must be implemented carefully as a real-time controller.

Current preferred direction:

> A small custom `ros2_control` effort controller is likely the best long-term architecture if we need actuator behavior beyond simple PD.

---

# 15. Controller vs Actuator Physics

The V3 redesign should explicitly separate the controller from actuator physics.

They are not the same thing.

---

# 16. Low-Level Controller Layer

The controller receives target motion and computes requested torque.

Example:

```text
q_target
   |
   v
target interpolation
   |
   +--> q_des
   |
   +--> qdot_des
   |
   v

tau_cmd =
    Kp(q_des - q)
  + Kd(qdot_des - qdot)
  + tau_ff
```

These are primarily control-system decisions.

---

# 17. Actuator Physical Layer

The requested torque then passes through physical actuator constraints:

```text
tau_cmd
   |
   v

torque limit
torque-speed limit
transmission ratio
actuator delay
motor response
efficiency
etc.
   |
   v

tau_actual
```

This layer represents the motor / transmission rather than the controller.

---

# 18. Physics Layer

Finally:

```text
tau_actual
    |
    v

rigid-body dynamics
contact
friction
gravity
```

are handled by MuJoCo during training or Gazebo during sim-to-sim deployment.

---

# 19. Proposed V3 Layer Separation

The complete conceptual stack should be:

```text
RL POLICY
50 Hz
    |
    | action
    v

POLICY ACTION MAPPING
    |
    | q_target
    v

COMMAND / TRAJECTORY LAYER
    |
    | q_des
    | qdot_des
    v

LOW-LEVEL CONTROLLER
~500 Hz
    |
    | tau_cmd
    v

ACTUATOR / TRANSMISSION MODEL
    |
    | tau_actual
    v

PHYSICS ENGINE
```

This architecture should be reproducible in both simulators.

---

# 20. Why MuJoCo BuiltinPD Is Not the Final Target

`BuiltinPdActuator` is useful as a MuJoCo-native servo.

However, it relies on MuJoCo's internal actuator representation and implicit numerical integration.

Gazebo's native position interface is not guaranteed to reproduce the same system.

Therefore:

> Even if `BuiltinPdActuator` produces excellent locomotion, relying on it as the final training actuator would create a simulator-specific low-level controller.

For this project, that is undesirable.

The deployment target should instead use explicit torque output with the low-level control equations owned by us.

---

# 21. What Should Be Shared Between MuJoCo and Gazebo

The following should ideally be simulator-independent.

## Policy

```text
policy frequency
observation order
observation scaling
action order
action scaling
home pose
```

## Low-level controller

```text
controller rate
Kp
Kd
feed-forward definition
target interpolation
desired velocity generation
command delay
torque clipping logic
```

## Actuator constraints

```text
hip torque limits
knee torque limits
angle-dependent knee limit
torque-speed envelope
rate limits
```

## Robot mechanics

As closely as possible:

```text
mass
center of mass
link inertia
joint axes
joint limits
geometry
joint damping
joint friction
reflected inertia
```

---

# 22. What Will Inevitably Differ

Some elements cannot be made exactly identical.

For example:

```text
contact solver
constraint solver
friction implementation
collision handling
numerical integration
solver tolerances
```

MuJoCo and Gazebo are different physics engines.

The purpose of the shared actuator/controller architecture is not to eliminate all simulator differences.

The goal is:

> Prevent the low-level controller itself from becoming another unnecessary source of sim-to-sim mismatch.

---

# 23. Actor Observations Must Remain Deployment-Compatible

The actor policy should use only information that can be obtained in ROS / Gazebo and later hardware.

Current suitable categories include:

```text
base linear velocity
base angular velocity
projected gravity / orientation
joint positions
joint velocities
previous policy action
velocity command
```

Privileged information can remain critic-only during training.

Examples of information that should not become necessary actor inputs:

```text
perfect MuJoCo contact forces
internal solver variables
perfect terrain state
simulation-only ground truth
```

The critic disappears after training, so critic-only privileged observations do not prevent deployment.

---

# 24. Policy / Controller Contract

A clean interface should be frozen eventually.

Candidate:

```text
POLICY INPUT
deployment-compatible observation vector

POLICY OUTPUT
12 normalized actions

ACTION MAPPING

q_target =
home_pose
+
action_scale * action
```

The low-level controller then owns all faster servo behavior.

This means the neural network does not need to contain simulator-specific torque logic.

---

# 25. 50 Hz Policy and 500 Hz Controller

The policy should not be required to run at motor-control frequency.

Candidate timing:

```text
policy:
50 Hz

controller:
500 Hz
```

Between policy updates, target interpolation can reconstruct a 500 Hz desired trajectory.

Instead of:

```text
q_target_A
-------------------- 20 ms
instant jump
q_target_B
```

the low-level controller could generate:

```text
q_des(t)
qdot_des(t)
```

between targets.

This is portable to both MuJoCo and ROS 2.

---

# 26. Candidate Low-Level Control Law

A current candidate is:

```text
tau_cmd =
    Kp * (q_des - q)
  + Kd * (qdot_des - qdot)
  + tau_ff
```

then:

```text
tau_actual =
actuator_constraints(
    tau_cmd,
    q,
    qdot
)
```

Important:

> This is a candidate architecture, not yet the final tuned controller.

The exact gains and feed-forward design must still be derived and characterized.

---

# 27. Candidate Physical Constraints

Current physically motivated constraints include:

```text
HX/HY limit:
approximately ±69.52 Nm

knee:
configuration-dependent limit
```

The 69.52 Nm hip value has a defensible interpretation as a Spot-derived torque capability scaled for the heavier ~50.34 kg robot model.

The current 164.15 Nm knee value is only a local approximation because the knee transmission ratio varies with joint angle.

A V3 actuator should eventually use:

```text
tau_knee_max = f(q_knee)
```

rather than one constant knee limit.

---

# 28. Possible Future Motor-Level Effects

Possible later additions include:

```text
torque-speed envelope
reflected rotor inertia
joint friction
motor / controller delay
torque slew rate
actuator bandwidth
```

Each should be added only when it can be justified physically or identified experimentally.

Do not add complex motor physics merely because MuJoCo or Gazebo supports it.

---

# 29. Gazebo Custom Hardware Layer

`gz_ros2_control` can also support a custom simulated hardware implementation.

This is potentially useful if some actuator behavior belongs below the controller.

For example:

```text
controller
   |
   | tau_cmd
   v

custom Gazebo simulated hardware layer
   |
   | actuator dynamics
   v

tau_actual
   |
   v

Gazebo
```

This may be a suitable place for:

```text
actuator low-pass response
motor delay
nonlinear transmission
torque-speed limitations
```

if those effects should conceptually belong to the hardware rather than the controller.

However, the architecture should remain easy to reproduce in MuJoCo.

---

# 30. Current Preferred ROS 2 Joint Definition

Each joint should likely expose:

```xml
<joint name="...">

    <command_interface name="effort"/>

    <state_interface name="position"/>

    <state_interface name="velocity"/>

    <state_interface name="effort"/>

</joint>
```

The essential command interface is:

```text
effort
```

because that prevents hidden simulator-native position controllers from entering the deployment plant.

---

# 31. Current Preferred ROS Architecture

The current preferred architecture is:

```text
/cmd_vel
    |
    v

Policy node
50 Hz
    |
    | q_target[12]
    v

SpotLowLevelController
ros2_control plugin
500 Hz
    |
    | effort[12]
    v

gz_ros2_control
    |
    v

Gazebo Harmonic
```

The policy node and low-level controller can remain separate.

This keeps machine learning separate from deterministic real-time joint control, which is desirable for both simulation and future hardware.

---

# 32. Why This Architecture Helps Future Hardware Deployment

Later, the Gazebo-specific hardware interface can be replaced.

Simulation:

```text
SpotLowLevelController
       |
       v
gz_ros2_control
       |
       v
Gazebo
```

Hardware:

```text
SpotLowLevelController
       |
       v
real ros2_control hardware interface
       |
       v
motor drivers
```

The policy and low-level controller could remain largely unchanged.

This is one of the strongest reasons to use the ROS 2 effort interface as the deployment boundary.

---

# 33. Current Decision About BuiltinPD

Current decision:

```text
DO NOT PRIORITIZE:
another BuiltinPD reproduction of the old successful run
```

Reason:

- the old successful run already used native MuJoCo position actuation,
- we already know that family of actuation can produce natural locomotion,
- a MuJoCo-native position servo is not the architecture we want to depend on for Gazebo transfer.

BuiltinPD can still be used later as a diagnostic reference if a specific numerical question arises.

It is no longer on the critical path.

---

# 34. Current Critical Path

The next work should focus on defining the portable actuator/controller interface.

Recommended order:

```text
1. Freeze policy/action interface.

2. Define the exact low-level control law.

3. Define controller frequency.

4. Decide how 50 Hz targets become 500 Hz q_des/qdot_des.

5. Define torque constraints.

6. Separate controller behavior from actuator physical behavior.

7. Implement the same deterministic controller in MuJoCo.

8. Implement the corresponding ros2_control effort controller.

9. Create a minimal Gazebo Spot model.

10. Compare deterministic actuator tests between MuJoCo and Gazebo.

11. Only then resume serious RL training.
```

---

# 35. Important Sim-to-Sim Validation Strategy

Before transferring a trained policy, first transfer the actuator characterization tests.

For example, run the same tests in both simulators:

```text
fixed-base joint step
loaded joint step
gravity hold
250 N body push
sine / chirp response
50 Hz target-sequence response
```

Compare:

```text
joint position
joint velocity
joint torque
body motion
settling time
frequency response
```

This is much more informative than immediately deploying PPO and asking whether the robot walks.

The order should be:

```text
controller transfer
        ↓
actuator response transfer
        ↓
robot dynamics comparison
        ↓
policy transfer
```

not:

```text
train policy
        ↓
put it in Gazebo
        ↓
debug everything at once
```

---

# 36. Engineering Principle Going Forward

For every new actuator/controller feature added to MjLab, ask:

> How exactly will this same feature exist in ROS 2 / Gazebo?

Examples:

```text
Kp/Kd?
-> custom ros2_control controller

interpolation?
-> same algorithm in both implementations

torque clipping?
-> same mathematical function

angle-dependent knee limit?
-> same lookup/equation

delay?
-> same explicit delay model

armature?
-> equivalent physical parameter in both robot models

friction?
-> matched joint model where possible
```

If there is no clear Gazebo equivalent, the feature should not quietly become part of the training plant.

---

# 37. Final Current Architecture

The present best architectural target is:

```text
                     HIGH LEVEL
                   ─────────────

                    /cmd_vel
                       |
                       v

                RL policy @ 50 Hz
                       |
                       | q_target
                       v


                   LOW LEVEL
                 ────────────

            target reconstruction
                       |
                q_des, qdot_des
                       |
                       v

         PD / PDFF controller @ 500 Hz
                       |
                       | tau_cmd
                       v

             actuator constraints
                       |
                       | tau_actual
                       v


                    PHYSICS
                  ─────────

        MuJoCo torque input during training

                       OR

      ros2_control effort interface in Gazebo
```

The key design philosophy is:

> The physics engine should receive torque.

The simulator should not secretly own the joint position controller.

The low-level controller should be ours.

---

# 38. Final Summary

The Gazebo research changed one important aspect of the actuator V3 direction.

The goal is not:

```text
find the best MuJoCo actuator class
```

The goal is:

```text
define a low-level actuator/controller architecture
that behaves coherently in:

MuJoCo
Gazebo
and later real hardware.
```

The most promising interface is:

```text
effort-controlled joints
```

through:

```text
ros2_control
+
gz_ros2_control
```

with an explicit low-level controller running at approximately:

```text
500 Hz
```

and the learned policy remaining around:

```text
50 Hz
```

The current preferred long-term solution is likely a custom Spot `ros2_control` effort controller because it gives full control over:

```text
PD / PDFF
target interpolation
joint-specific gains
torque limits
knee transmission limits
future motor dynamics
```

and makes it much easier to keep the MjLab and Gazebo implementations mathematically aligned.

Before more serious RL work, the low-level controller should therefore be designed as a portable control specification, not as a MuJoCo-specific actuator configuration.

---

# 39. Reference Sources Used During This Research

## ROS 2 / Gazebo

- gz_ros2_control documentation:
  https://control.ros.org/jazzy/doc/gz_ros2_control/doc/index.html

- ros2_control Controller Manager:
  https://control.ros.org/jazzy/doc/ros2_control/controller_manager/doc/userdoc.html

- JointTrajectoryController:
  https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html

- Effort controllers:
  https://control.ros.org/jazzy/doc/ros2_controllers/effort_controllers/doc/userdoc.html

- ros2_control getting started / controller interface:
  https://control.ros.org/jazzy/doc/getting_started/getting_started.html

- Gazebo SDF world / physics configuration:
  https://gazebosim.org/docs/latest/sdf_worlds/

## Existing Spot Project Reference

The old successful MjLab 1.1 Spot setup used:

```text
XmlPositionActuatorCfg
native MuJoCo <position> actuators
Kp = 500
Kd = 40
actuatorfrcrange = ±1000 Nm
implicitfast
```

That setup remains useful as a behavioral reference, but it is no longer the desired final deployment architecture.
