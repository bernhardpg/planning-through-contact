import math
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve


# TODO: At some point this will be deprecated
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


@dataclass
class BernsteinPolynomial:
    order: int
    k: int

    def __post_init__(self) -> sym.Expression:
        self.s = sym.Variable("s")
        self.poly = (
            math.comb(self.order, self.k)
            * sym.pow(self.s, self.k)
            * sym.pow((1 - self.s), (self.order - self.k))
        )

    def eval(self, at_s: float) -> float:
        assert 0 <= at_s and at_s <= 1.0
        env = {self.s: at_s}
        value_at_s = self.poly.Evaluate(env)
        return value_at_s


# TODO use this everywhere instead of just a numpy array
@dataclass
class BezierCtrlPoints:
    ctrl_points: npt.NDArray[np.float64]

    @property
    def dim(self) -> int:
        return self.ctrl_points.shape[0]

    @property
    def order(self) -> int:
        return self.ctrl_points.shape[1] - 1

    @classmethod
    def from_list(cls, points: List[npt.NDArray[np.float64]]) -> "BezierCtrlPoints":
        return cls(np.hstack(points))


@dataclass
class BezierCurve:
    order: int
    dim: int

    def __post_init__(self):  # TODO add initial conditions here
        self.num_ctrl_points = self.order + 1
        self.coeffs = [
            BernsteinPolynomial(order=self.order, k=k) for k in range(0, self.order + 1)
        ]

    @classmethod
    def create_from_ctrl_points(
        cls, ctrl_points: npt.NDArray[np.float64]  # (num_dims, num_ctrl_points)
    ) -> "BezierCurve":
        dim = ctrl_points.shape[0]
        order = ctrl_points.shape[1] - 1
        curve = cls(order, dim)
        curve.set_ctrl_points(ctrl_points)
        return curve

    def set_ctrl_points(self, ctrl_points: npt.NDArray[np.float64]) -> None:
        assert ctrl_points.shape[0] == self.dim
        assert ctrl_points.shape[1] == self.order + 1
        self.ctrl_points = ctrl_points

    def eval_coeffs(self, at_s: float) -> npt.NDArray[np.float64]:
        evaluated_coeffs = np.array(
            [coeff.eval(at_s) for coeff in self.coeffs]
        ).reshape((-1, 1))
        return evaluated_coeffs

    def eval(self, at_s: float) -> npt.NDArray[np.float64]:
        assert self.ctrl_points is not None
        evaluated_coeffs = self.eval_coeffs(at_s)
        path_value = self.ctrl_points.dot(evaluated_coeffs)
        return path_value

    def eval_entire_interval(self) -> npt.NDArray[np.float64]:
        values = np.concatenate(
            [self.eval(s) for s in np.arange(0.0, 1.01, 0.01)], axis=1
        ).T
        return values


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
