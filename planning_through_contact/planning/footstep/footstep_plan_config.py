from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as npt


# TODO(bernhardpg): Move to another location?
@dataclass
class PotatoRobot:
    # TODO(bernhardpg): set reasonable inertia properties
    # This is from a sphere with mass = 50 kg and all axis = 0.5 meter
    mass: float = 50.0  # kg
    # TODO(bernhardpg): compute inertia from dimensions
    inertia: float = 5.0  # kg m**2
    foot_length: float = 0.3  # m
    foot_height: float = 0.15  # m
    step_span: float = 0.8  # m
    desired_com_height: float = 1.5  # m
    size: Tuple[float, float, float] = (0.5, 0.5, 1.0)

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def depth(self) -> float:
        return self.size[1]

    @property
    def height(self) -> float:
        return self.size[2]

    def get_nominal_pose(self) -> npt.NDArray:
        return np.array([0, self.desired_com_height, 0])


@dataclass
class FootstepPlanningConfig:
    robot: PotatoRobot = field(default_factory=lambda: PotatoRobot())
    period: float = 1.0
    period_steps: int = 3

    @property
    def dt(self) -> float:
        return self.period / self.period_steps
