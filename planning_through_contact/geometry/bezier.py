import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq, ge, le
from pydrake.solvers import MathematicalProgram, Solve


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
