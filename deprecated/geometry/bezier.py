from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve


@dataclass
class BezierVariable:
    dim: int
    order: int  # Bezier curve order
    x: Optional[npt.NDArray[sym.Expression]] = None  # TODO rename to ctrl points
    name: Optional[str] = None

    def __post_init__(self):
        self.n_vars = self.order + 1
        if self.x is None:
            self.x = sym.MakeMatrixContinuousVariable(
                self.dim, self.order + 1, self.name
            )

    def get_derivative(self) -> "BezierVariable":
        assert self.order >= 1
        der_ctrl_points = self.order * (self.x[:, 1:] - self.x[:, 0:-1])
        derivative = BezierVariable(self.dim, self.order - 1, der_ctrl_points)
        return derivative

    @property
    def first(self) -> "BezierVariable":
        new_x = self.x[:, 0]
        return BezierVariable(self.dim, 0, new_x)

    @property
    def last(self) -> "BezierVariable":
        new_x = self.x[:, -1]
        return BezierVariable(self.dim, 0, new_x)

    def __add__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x + other.x
        else:
            new_ctrl_points = self.x + other
        return BezierVariable(self.dim, self.order, new_ctrl_points)

    def __radd__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        return self + other

    def __sub__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x - other.x
        else:
            new_ctrl_points = self.x - other
        return BezierVariable(self.dim, self.order, new_ctrl_points)

    def __rsub__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        return self - other

    def __mul__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            new_ctrl_points = self.x * other.x
        else:
            new_ctrl_points = self.x * other
        return BezierVariable(self.dim, self.order, new_ctrl_points)

    def __rmul__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        return self * other

    def __le__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            return le(self.x, other.x)
        else:
            return le(self.x, other)

    def __ge__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            return ge(self.x, other.x)
        else:
            return ge(self.x, other)

    def __eq__(
        self, other: Union["BezierVariable", npt.NDArray[np.float64], float, int]
    ) -> "BezierVariable":
        if type(other) == BezierVariable:
            assert other.dim == self.dim
            assert other.order == self.order
            return eq(self.x, other.x)
        else:
            return eq(self.x, other)


# TODO remove this
@dataclass
class BezierCurveMathProgram:
    order: int
    dim: int

    def __post_init__(self):  # TODO add initial conditions here
        self.prog = MathematicalProgram()

        self.num_ctrl_points = self.order + 1
        self.ctrl_points = self.prog.NewContinuousVariables(
            self.dim, self.num_ctrl_points, "gamma"
        )

        self.coeffs = [
            BernsteinPolynomial(order=self.order, k=k) for k in range(0, self.order + 1)
        ]

    def constrain_to_polyhedron(self, poly: opt.HPolyhedron):
        A = poly.A()
        b = poly.b().reshape((-1, 1))
        # NOTE: Would like to do:
        # self.prog.AddLinearConstraint(A.dot(self.ctrl_points) <= b)
        # But this doesnt work, see https://github.com/RobotLocomotion/drake/issues/16025
        constraint = le(A.dot(self.ctrl_points), b)
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
