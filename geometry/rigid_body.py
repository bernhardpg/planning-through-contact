from dataclasses import dataclass
from typing import Literal

import numpy.typing as npt
import pydrake.symbolic as sym

from geometry.bezier import BezierVariable


@dataclass
class RigidBody:
    name: str
    dim: int
    geometry: Literal["point", "box"]
    width: float = 0  # TODO generalize
    height: float = 0
    position_curve_order: int = 2
    actuated: bool = False

    def __post_init__(self) -> None:
        self.pos = BezierVariable(
            self.dim, self.position_curve_order, name=f"{self.name}_pos"
        )

    @property
    def vel(self) -> BezierVariable:
        return self.pos.get_derivative()

    @property
    def pos_x(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[0, :]

    @property
    def pos_y(self) -> npt.NDArray[sym.Expression]:
        return self.pos.x[1, :]
