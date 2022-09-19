import matplotlib.pyplot as plt
import cdd

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Literal

import math
from pydrake.math import le, ge, eq
import pydrake.symbolic as sym
import pydrake.geometry.optimization as opt

from pydrake.geometry.optimization import GraphOfConvexSets
from pydrake.solvers import MathematicalProgram, Solve, MathematicalProgramResult


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
        cls, dim: int, ctrl_points: npt.NDArray[np.float64]
    ) -> "BezierCurve":
        assert ctrl_points.size % dim == 0
        order = ctrl_points.size // dim - 1

        ctrl_points_reshaped = ctrl_points.reshape((order + 1), dim).T  # TODO ugly code
        curve = cls(order, dim)
        curve.set_ctrl_points(ctrl_points_reshaped)
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
