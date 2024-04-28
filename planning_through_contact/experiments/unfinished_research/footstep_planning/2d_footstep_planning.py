from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    HPolyhedron,
    Point,
    Spectrahedron,
)
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
)
from underactuated.exercises.humanoids.footstep_planning_gcs_utils import plot_rectangle

from planning_through_contact.geometry.utilities import cross_2d


@dataclass
class InPlaneSteppingStone:
    x_pos: float
    z_pos: float
    width: float
    name: Optional[str] = None

    @property
    def height(self) -> float:
        return self.z_pos

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
        name: Optional[str] = None,
    ) -> InPlaneSteppingStone:
        stone = InPlaneSteppingStone(x_pos, z_pos, width)
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

    def plot(self, title: Optional[str] = None, **kwargs):
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
        ax.set_ylim((0.0, self.max_height * 2.0))


@dataclass
class Robot:
    mass: float = 50.0  # kg
    inertia: float = 5.0  # kg m**2
    foot_length: float = 0.3  # m


def create_set(stone: InPlaneSteppingStone, robot: Robot) -> Spectrahedron:
    # Assume we always only have one foot in contact
    prog = MathematicalProgram()

    # declare states
    p_WB = prog.NewContinuousVariables(2, "p_WB")
    v_WB = prog.NewContinuousVariables(2, "v_WB")
    theta_WB = prog.NewContinuousVariables(1, "theta_WB")[0]
    omega_WB = prog.NewContinuousVariables(1, "omega_WB")[0]

    # declare inputs
    p_BF_W = prog.NewContinuousVariables(2, "p_BF_W")
    f_F_1W = prog.NewContinuousVariables(2, "f_F_1W")
    f_F_2W = prog.NewContinuousVariables(2, "f_F_2W")

    # auxilliary vars
    # TODO(bernhardpg): we might be able to get around this once we
    # have SDP constraints over the edges
    tau_F_1 = prog.NewContinuousVariables(1, "tau_F_1")[0]
    tau_F_2 = prog.NewContinuousVariables(1, "tau_F_2")[0]

    # linear acceleration
    x_ddot = (1 / robot.mass) * (f_F_1W[0] + f_F_2W[0])
    z_ddot = (1 / robot.mass) * (f_F_1W[1] + f_F_2W[1])

    # angular acceleration
    theta_ddot = (1 / robot.inertia) * (tau_F_1 + tau_F_2)

    # torque = arm x force
    p_BF_1W = p_BF_W + np.array([robot.foot_length / 2, 0])
    p_BF_2W = p_BF_W - np.array([robot.foot_length / 2, 0])
    tau_F_1 = cross_2d(p_BF_1W, f_F_1W)
    tau_F_2 = cross_2d(p_BF_2W, f_F_2W)

    relaxed_prog = MakeSemidefiniteRelaxation(prog)

    spectrahedron = Spectrahedron(relaxed_prog)

    return spectrahedron


def main():
    terrain = InPlaneTerrain()
    initial_stone = terrain.add_stone(x_pos=0.5, width=1.0, z_pos=0.2, name="initial")
    target_stone = terrain.add_stone(x_pos=1.5, width=1.0, z_pos=0.3, name="target")

    robot = Robot()

    spect = create_set(initial_stone, robot)

    # terrain.plot()
    # plt.show()


if __name__ == "__main__":
    main()
