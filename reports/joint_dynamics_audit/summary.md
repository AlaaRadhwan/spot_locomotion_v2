# Spot V2 Joint Dynamics Audit — Auto Summary

## Active baseline

```text
HIP:  Kp=500, Kd=10, limit=69.52 Nm
KNEE: Kp=500, Kd=10, limit=164.15 Nm
Physics=500 Hz, policy=50 Hz
```

## Mechanical audit

Total body mass: **50.3400 kg**

Flags:

- Torso principal inertias are nearly identical; source-audit this for the elongated torso.

### Representative compiled passive joint properties

| Joint | Armature | Damping | Friction loss |
| --- | --- | --- | --- |
| fl_hx | 0 | 0 | 0 |
| fl_hy | 0 | 0 | 0 |
| fl_kn | 0 | 0 | 0 |

## HOME inertia and current-PD prediction

| Model | Joint | J locked | J free | Pred fn Hz | Pred ζ |
| --- | --- | --- | --- | --- | --- |
| FIXED | fl_hx | 0.089808 | 0.075658 | 12.94 | 0.813 |
| FIXED | fl_hy | 0.106546 | 0.083176 | 12.34 | 0.775 |
| FIXED | fl_kn | 0.015402 | 0.013150 | 31.03 | 1.950 |
| FLOATING | fl_hx | 0.089808 | 0.045058 | 16.77 | 1.053 |
| FLOATING | fl_hy | 0.106546 | 0.079568 | 12.62 | 0.793 |
| FLOATING | fl_kn | 0.015402 | 0.012797 | 31.46 | 1.977 |

Floating HOME `J_HY/J_KN = 6.218`. Under equal Kp, the simplest inertia-only model predicts a knee/HY natural-frequency ratio of **2.494×**.

## Coupled 3-DOF leg modes at HOME

| Model | Mode | Natural Hz | Damped Hz | ζ |
| --- | --- | --- | --- | --- |
| FIXED | 1 | 9.84 | 7.73 | 0.618 |
| FIXED | 2 | 13.62 | 7.05 | 0.856 |
| FLOATING | 1 | 11.24 | 7.96 | 0.706 |

## Direct-torque validation

| Model | Joint | Pulse Nm | Pred Δqdd | Forward Δqdd | 1-step Δqdd | 1-step err % |
| --- | --- | --- | --- | --- | --- | --- |
| FIXED | fl_hx | 2.2697 | 30.000 | 30.000 | 30.000 | 0.000 |
| FIXED | fl_hy | 2.4953 | 30.000 | 30.000 | 30.000 | 0.000 |
| FIXED | fl_kn | 0.3945 | 30.000 | 30.000 | 30.000 | 0.000 |
| FLOATING | fl_hx | 1.3517 | 30.000 | 30.000 | 30.000 | 0.000 |
| FLOATING | fl_hy | 2.3870 | 30.000 | 30.000 | 30.000 | 0.000 |
| FLOATING | fl_kn | 0.3839 | 30.000 | 30.000 | 30.000 | 0.000 |

## Standing support

Mean settled base height over final 0.50 s: **0.43405 m**

| Perturbed joint | Added torque Nm | Δqdot at pulse end | Peak |Δqdot| | Peak base Δvel norm |
| --- | --- | --- | --- | --- |
| fl_hx | 3.476 | 0.025013 | 0.073387 | 0.065721 |
| fl_hy | 3.476 | 0.015861 | 0.039650 | 0.018236 |
| fl_kn | 8.208 | 0.118332 | 0.292580 | 0.065747 |

## Interpretation limits

- This audit only identifies dynamics already present in the current MuJoCo model.
- It cannot infer real Spot rotor inertia, gearbox friction, motor electrical constants, torque-loop bandwidth, or latency if those are not modeled.
- Fixed-base zero-velocity bias force is not the same as whole-body standing support torque; standing support is measured separately under contact.
