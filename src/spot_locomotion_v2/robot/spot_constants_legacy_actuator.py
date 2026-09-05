"""Experimental explicit-PD actuator configuration.

Current A/B experiment:
    Same IdealPdActuator implementation as V2
    Same Kp = 500
    Same Kd = 10
    2x current V2 torque limits

Diagnostic only.
"""

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .spot_constants import (
    KP,
    KD,
    SPOT_INIT_STATE,
    get_spec,
)


EXPERIMENT_HIP_TORQUE_LIMIT = 139.04
EXPERIMENT_KNEE_TORQUE_LIMIT = 328.30


EXPERIMENT_HIP_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=(
        ".*_hx",
        ".*_hy",
    ),
    stiffness=KP,
    damping=KD,
    effort_limit=EXPERIMENT_HIP_TORQUE_LIMIT,
)


EXPERIMENT_KNEE_ACTUATOR_CFG = IdealPdActuatorCfg(
    target_names_expr=(".*_kn",),
    stiffness=KP,
    damping=KD,
    effort_limit=EXPERIMENT_KNEE_TORQUE_LIMIT,
)


EXPERIMENT_SPOT_ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(
        EXPERIMENT_HIP_ACTUATOR_CFG,
        EXPERIMENT_KNEE_ACTUATOR_CFG,
    ),
)


def get_spot_legacy_actuator_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=SPOT_INIT_STATE,
        spec_fn=get_spec,
        articulation=EXPERIMENT_SPOT_ARTICULATION_CFG,
        sort_actuators=True,
    )