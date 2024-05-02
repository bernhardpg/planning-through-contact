from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.geometry.optimization import GraphOfConvexSets
from underactuated.exercises.humanoids.footstep_planning_gcs_utils import plot_rectangle


@dataclass
class InPlaneSteppingStone:
    x_pos: float
    z_pos: float
    width: float
    name: str

    @property
    def com(self) -> npt.NDArray[np.float64]:  # (2,)
        return np.array([self.x_pos, self.z_pos])

    @property
    def height(self) -> float:
        return self.z_pos

    @property
    def center(self) -> npt.NDArray[np.float64]:
        """
        Returns the surface center of the stone.
        """
        return np.array([self.x_pos, self.z_pos])

    @property
    def x_min(self) -> float:
        return self.x_pos - self.width / 2

    @property
    def x_max(self) -> float:
        return self.x_pos + self.width / 2

    def plot(self, **kwargs):
        center = np.array([self.x_pos, self.z_pos / 2])

        return plot_rectangle(center, self.width, self.height, **kwargs)


class InPlaneTerrain:
    def __init__(self) -> None:
        self.stepping_stones = []

    def add_stone(
        self,
        x_pos: float,
        z_pos: float,
        width: float,
        name: str,
    ) -> InPlaneSteppingStone:
        stone = InPlaneSteppingStone(x_pos, z_pos, width, name)
        self.stepping_stones.append(stone)
        return stone

    def get_stone_by_name(self, name):
        for stone in self.stepping_stones:
            if stone.name == name:
                return stone

        raise ValueError(f"No stone in the terrain has name {name}.")

    @property
    def max_height(self) -> float:
        return max([s.height for s in self.stepping_stones])

    def plot(
        self, title: Optional[str] = None, max_height: Optional[float] = None, **kwargs
    ):
        # make light green the default facecolor
        if not "facecolor" in kwargs:
            kwargs["facecolor"] = [0, 1, 0, 0.1]

        # plot stepping stones disposition
        labels = ["Stepping stone", None]
        for i, stone in enumerate(self.stepping_stones):
            stone.plot(label=labels[min(i, 1)], **kwargs)

        if title is not None:
            plt.title(title)
            # get current plot axis if one is not given
        ax = plt.gca()

        if max_height:
            ax.set_ylim((0.0, max_height))
