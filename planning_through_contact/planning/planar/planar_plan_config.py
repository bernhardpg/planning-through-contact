from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class BoxWorkspace:
    width: float = 0.5
    height: float = 0.5
    center: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )

    @property
    def x_min(self) -> float:
        return self.center[0] - self.width / 2

    @property
    def x_max(self) -> float:
        return self.center[0] + self.width / 2

    @property
    def y_min(self) -> float:
        return self.center[0] - self.height / 2

    @property
    def y_max(self) -> float:
        return self.center[0] + self.height / 2

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lb = np.array([self.x_min, self.y_min], dtype=np.float64)
        ub = np.array([self.x_max, self.y_max], dtype=np.float64)
        return lb, ub


@dataclass
class PlanarPushingWorkspace:
    # Currently we don't use the workspace constraints for the slider, because
    # the full Semidefinite relaxation of the QCQP creates a ton of
    # constraints, which really slows down the optimization!
    slider: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(
            width=1.0, height=1.0, center=np.array([0.4, 0.0])
        )
    )
    pusher: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(width=0.8, height=0.8)
    )


@dataclass
class PlanarPlanConfig:
    num_knot_points_contact: int = 4
    num_knot_points_non_collision: int = 2
    time_in_contact: float = 2
    time_non_collision: float = 0.5
    pusher_radius: float = 0.01
    avoid_object: bool = False
    allow_teleportation: bool = False
    avoidance_cost: Literal["linear", "quadratic", "socp"] = "quadratic"
    minimize_squared_eucl_dist: bool = True
    use_eq_elimination: bool = False
    use_redundant_dynamic_constraints: bool = (
        True  # TODO(bernhardpg): This sometimes makes rounding not work
    )
    penalize_mode_transitions: bool = False
    no_cycles: bool = False
    workspace: PlanarPushingWorkspace = field(
        default_factory=lambda: PlanarPushingWorkspace()
    )
