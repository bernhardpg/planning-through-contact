import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

import math
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
    deg: int
    dim: int

    def __post_init__(self):  # TODO add initial conditions here
        self.prog = MathematicalProgram()

        self.num_ctrl_points = self.deg + 1
        self.ctrl_points = self.prog.NewContinuousVariables(
            self.dim, self.num_ctrl_points, "gamma"
        )

        self.coeffs = [
            BernsteinPolynomial(deg=self.deg, k=k) for k in range(0, self.deg + 1)
        ]

    def constrain_to_polyhedron(
        self, A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
    ):
        # NOTE: Would like to do:
        # self.prog.AddLinearConstraint(A.dot(self.ctrl_points) >= b)
        # But this doesnt work, see https://github.com/RobotLocomotion/drake/issues/16025
        constraint = ge(A.dot(self.ctrl_points), b)
        self.prog.AddLinearConstraint(constraint)

    def constrain_start_pos(self, x0: npt.NDArray[np.float64]) -> None:
        constraint = eq(self.ctrl_points[:, 0:1], x0)
        self.prog.AddLinearConstraint(constraint)

    def constrain_end_pos(self, xf: npt.NDArray[np.float64]) -> None:
        constraint = eq(self.ctrl_points[:, -1:], xf)
        self.prog.AddLinearConstraint(constraint)

    def solve(self):
        self.result = Solve(self.prog)
        assert self.result.is_success

    def calc_ctrl_points(self):
        self.solve()
        assert self.result is not None
        self.ctrl_point_values = self.result.GetSolution(self.ctrl_points)

    def eval_coeffs(self, at_s: float) -> npt.NDArray[np.float64]:
        evaluated_coeffs = np.array(
            [coeff.eval(at_s) for coeff in self.coeffs]
        ).reshape((-1, 1))
        return evaluated_coeffs

    def eval(self, at_s: float) -> npt.NDArray[np.float64]:
        assert self.ctrl_point_values is not None
        evaluated_coeffs = self.eval_coeffs(at_s)
        path_value = self.ctrl_point_values.dot(evaluated_coeffs)
        return path_value


@dataclass
class BernsteinPolynomial:
    deg: int
    k: int

    def __post_init__(self) -> sym.Expression:
        self.s = sym.Variable("s")
        self.poly = (
            math.comb(self.deg, self.k)
            * np.power(self.s, self.k)
            * np.power((1 - self.s), (self.deg - self.k))
        )

    def eval(self, at_s: float) -> float:
        assert 0 <= at_s and at_s <= 1.0
        env = {self.s: at_s}
        value_at_s = self.poly.Evaluate(env)
        return value_at_s


@dataclass
class Polyhedron:
    A: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]

    def __post_init__(self):
        self.dim = self.A.shape[1]

    def get_vertices(self) -> npt.NDArray[np.float64]:  # [N, 2]
        # NOTE: Use cdd to calculate vertices
        # cdd expects: [b -A], where A'x <= b
        # We have A'x >= b ==> -A'x <= -b
        # Hence we need [-b A]
        cdd_matrix = cdd.Matrix(np.hstack((-self.b, self.A)))
        cdd_matrix.rep_type = cdd.RepType.INEQUALITY
        cdd_poly = cdd.Polyhedron(cdd_matrix)
        generators = np.array(cdd_poly.get_generators())
        # cdd specific, see https://pycddlib.readthedocs.io/en/latest/polyhedron.html
        vertices = generators[:, 1 : 1 + self.dim]
        return vertices


def test_bezier_curve() -> None:
    deg = 2
    dim = 2

    A = np.array([[1, -1], [-1, -1], [0, 1], [1, 0]], dtype=np.float64)
    b = np.array([-1, -5, 0, 0], dtype=np.float64).reshape((-1, 1))

    poly = Polyhedron(A, b)
    vertices = poly.get_vertices()
    plt.fill(vertices[:, 0], vertices[:, 1], alpha=0.3)

    x0 = np.array([0, 0.5]).reshape((-1, 1))
    xf = np.array([4, 1]).reshape((-1, 1))

    bezier_curve = BezierCurve(deg, dim)
    bezier_curve.constrain_to_polyhedron(A, b)
    bezier_curve.constrain_start_pos(x0)
    bezier_curve.constrain_end_pos(xf)
    bezier_curve.calc_ctrl_points()
    path = np.concatenate(
        [bezier_curve.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
    ).T

    plt.plot(path[:, 0], path[:, 1])
    plt.scatter(x0[0], x0[1])
    plt.scatter(xf[0], xf[1])

    plt.show()


def test_bernstein_polynomial() -> None:
    deg = 2
    k = 0

    bp = BernsteinPolynomial(deg, k)


def main():
    test_bezier_curve()

    return 0


if __name__ == "__main__":
    main()
