from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram


@dataclass
class LinearComplementaritySystem:
    """
    ẋ = Ax + Bu + Dλ + a
    0 ≤ λ ⊥ Ex + Fλ + Hu + c ≥ 0
    """

    A: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    D: npt.NDArray[np.float64]
    d: npt.NDArray[np.float64]
    E: npt.NDArray[np.float64]
    F: npt.NDArray[np.float64]
    H: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]

    @property
    def num_states(self) -> int:
        return self.A.shape[1]

    @property
    def num_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def num_forces(self) -> int:
        return self.D.shape[1]

    def get_x_dot(self, x: np.ndarray, u: np.ndarray, λ: np.ndarray) -> np.ndarray:
        return self.A @ x + self.B @ u + self.D @ λ + self.d

    def get_complementarity_rhs(
        self, x: np.ndarray, u: np.ndarray, λ: np.ndarray
    ) -> np.ndarray:
        return self.E @ x + self.F @ λ + self.H @ u + self.c


class CartPoleWithWalls(LinearComplementaritySystem):
    """
    Cart-pole with walls system from the paper:

    A. Aydinoglu, P. Sieg, V. M. Preciado, and M. Posa,
    “Stabilization of Complementarity Systems via
    Contact-Aware Controllers.” 2021
    """

    def __init__(self):
        # fmt: off
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 3.51, 0, 0],
            [0, 22.2, 0, 0]
        ])
        
        B = np.array([
            [0],
            [0],
            [1.02],
            [1.7]
        ])
        
        D = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [4.7619, -4.7619]
        ])
        
        d = np.zeros((4,))

        E = np.array([
            [-1, 0.6, 0, 0],
            [1, -0.6, 0, 0]
        ])
        
        F = np.array([
            [0.0014, 0],
            [0, 0.0014]
        ])
        
        H = np.zeros((2,1))
        
        c = np.array([0.35, 0.35])
        # fmt: on

        super().__init__(A, B, D, d, E, F, H, c)


# TODO: Move to unit test
def test_cart_pole_w_walls():
    sys = CartPoleWithWalls()
    x = np.zeros((sys.num_states,))
    u = np.zeros((sys.num_inputs,))
    λ = np.zeros((sys.num_forces,))
    assert isinstance(sys.get_x_dot(x, u, λ), type(np.array([])))
    assert sys.get_x_dot(x, u, λ).shape == (sys.num_states,)

    assert isinstance(sys.get_complementarity_rhs(x, u, λ), type(np.array([])))
    assert sys.get_complementarity_rhs(x, u, λ).shape == (sys.num_forces,)


def main() -> None:
    test_cart_pole_w_walls()

    sys = CartPoleWithWalls()

    N = 10
    T_s = 0.01

    prog = MathematicalProgram()
    xs = prog.NewContinuousVariables(N, sys.num_states, "x")
    us = prog.NewContinuousVariables(N, sys.num_inputs, "u")
    λs = prog.NewContinuousVariables(N, sys.num_forces, "λ")

    # Dynamics
    for k in range(N - 1):
        x, u, λ = xs[k], us[k], λs[k]
        x_next = xs[k + 1]

        x_dot = sys.get_x_dot(x, u, λ)
        forward_euler = eq(x_next, x + T_s * x_dot)
        for c in forward_euler:
            prog.AddLinearEqualityConstraint(c)

    # RHS nonnegativity of complementarity constraint
    for k in range(N - 1):
        x, u, λ = xs[k], us[k], λs[k]

        rhs = sys.get_complementarity_rhs(x, u, λ)
        prog.AddLinearConstraint(ge(rhs, 0))

    # LHs nonnegativity of complementarity constraint
    for k in range(N - 1):
        λ = λs[k]
        prog.AddLinearConstraint(ge(λ, 0))

    # Complementarity constraint (element-wise)
    for k in range(N - 1):
        x, u, λ = xs[k], us[k], λs[k]
        rhs = sys.get_complementarity_rhs(x, u, λ)

        elementwise_product = λ * rhs
        for p in elementwise_product:
            prog.AddQuadraticConstraint(p, 0, 0)  # p == 0

    # Input limits
    # TODO

    # State limits
    # TODO


if __name__ == "__main__":
    main()
