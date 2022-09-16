from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from pydrake.math import ge, eq
import pydrake.symbolic as sym

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult

# TODO: Plan
# 1. Implement Bezier Curves
# 2. Implement vertices
# 3. Implement edges
# 4. Implement edge lengths
# 5. Formulate the math prog
# ...


@dataclass
class BezierCurve:
    order: int
    dim: int

    def __post_init__(self):  # TODO add initial conditions here
        self.prog = MathematicalProgram()

        self.num_ctrl_points = self.order + 1
        self.ctrl_points = self.prog.NewContinuousVariables(
            self.dim, self.num_ctrl_points, "gamma"
        )

    def constrain_to_polyhedron(self, A: npt.NDArray[float], b: npt.NDArray[float]):
        # NOTE: Would like to do:
        # self.prog.AddLinearConstraint(A.dot(self.ctrl_points) >= b)
        # But this doesnt work, see https://github.com/RobotLocomotion/drake/issues/16025
        constraint = ge(A.dot(self.ctrl_points), b)
        self.prog.AddLinearConstraint(constraint)

    def constrain_start_pos(self, x0: npt.NDArray[float]) -> None:
        constraint = eq(self.ctrl_points[:, 0:1], x0)
        self.prog.AddLinearConstraint(constraint)

    def constrain_end_pos(self, xf: npt.NDArray[float]) -> None:
        constraint = eq(self.ctrl_points[:, -1:], xf)
        self.prog.AddLinearConstraint(constraint)

    def solve(self) -> MathematicalProgramResult:
        self.result = Solve(self.prog)
        assert self.result.is_success
        return self.result

    def get_coeff_values(self) -> npt.NDArray[np.float64]:
        assert self.result is not None
        return self.result.GetSolution(self.ctrl_points)


# @dataclass
# class BernsteinPolynomial:
#    order: int
#    control_point_number: int
#
#    def __post_init__(self):


def main():
    order = 2
    dim = 2

    x0 = np.array([0, 0.5]).reshape((-1, 1))
    xf = np.array([4, 1]).reshape((-1, 1))

    path = BezierCurve(order, dim)
    A = np.array([[1, -1], [-1, -1], [0, 1], [1, 0]], dtype=np.float64)
    b = np.array([-1, -5, 0, 0], dtype=np.float64).reshape((-1, 1))
    path.constrain_to_polyhedron(A, b)
    path.constrain_start_pos(x0)
    path.constrain_end_pos(xf)
    path.solve()
    coeffs = path.get_coeff_values()
    breakpoint()

    return 0


if __name__ == "__main__":
    main()
