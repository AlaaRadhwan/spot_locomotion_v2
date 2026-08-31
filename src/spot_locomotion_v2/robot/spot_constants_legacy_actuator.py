"""Legacy native-position actuator configuration for A/B diagnosis only.

This reproduces the high-authority actuator behavior used by the older
successful Spot locomotion project as closely as practical in MjLab 1.5.3.

Unlike the normal V2 robot, this intentionally uses MuJoCo's native
position actuator rather than an explicit IdealPdActuator.
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .spot_constants import (
    SPOT_INIT_STATE,
    get_spec,
)


LEGACY_KP = 500.0
LEGACY_KD = 40.0
LEGACY_EFFORT_LIMIT = 1000.0


LEGACY_SPOT_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_hx",
        ".*_hy",
        ".*_kn",
    ),
    stiffness=LEGACY_KP,
    damping=LEGACY_KD,
    effort_limit=LEGACY_EFFORT_LIMIT,
)


LEGACY_SPOT_ARTICULATION_CFG = EntityArticulationInfoCfg(
    actuators=(LEGACY_SPOT_ACTUATOR_CFG,),
)


def get_spot_legacy_actuator_robot_cfg() -> EntityCfg:
    """Return Spot with the legacy native MuJoCo position actuator."""

    return EntityCfg(
        init_state=SPOT_INIT_STATE,
        spec_fn=get_spec,
        articulation=LEGACY_SPOT_ARTICULATION_CFG,
        sort_actuators=True,
    )