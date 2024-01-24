from dataclasses import dataclass

from pydrake.math import RigidTransform


@dataclass
class OptitrackConfig:
    iiwa_id: int
    slider_id: int
    X_optitrackBody_plantBody: RigidTransform
