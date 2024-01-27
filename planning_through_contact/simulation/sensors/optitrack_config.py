from dataclasses import dataclass
from typing import List, Optional

from pydrake.math import RigidTransform, RollPitchYaw


@dataclass
class OptitrackConfig:
    iiwa_id: int
    slider_id: int
    X_optitrackBody_plantBody: Optional[RigidTransform] = None
    # Optionally provide instead
    p_W_pB: Optional[List[float]] = None
    # Rotation defined in terms of RPY
    R_W_oB: Optional[List[float]] = None
    p_W_oB: Optional[List[float]] = None

    def __post_init__(self):
        if self.X_optitrackBody_plantBody is None:
            assert (
                self.p_W_pB is not None
                and self.R_W_oB is not None
                and self.p_W_oB is not None
            ), "Must provide either X_optitrackBody_plantBody or p_W_pB, R_W_oB, and p_W_oB"
            X_W_oB = RigidTransform(
                RollPitchYaw(*self.R_W_oB),
                self.p_W_oB,
            )
            X_W_pB = RigidTransform(
                self.p_W_pB,
            )
            X_oB_pB = X_W_oB.inverse() @ X_W_pB
            self.X_optitrackBody_plantBody = X_oB_pB
