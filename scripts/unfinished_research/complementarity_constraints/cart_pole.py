from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    SemidefiniteRelaxationOptions,
    Solve,
    SolverOptions,
)


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

    x_1 = cart position
    x_2 = pole position
    x_3 = cart velocity
    x_4 = pole velocity

    u_1 = force applied to cart

    λ_1 = contact force from right wall
    λ_2 = contact force from left wall

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


@dataclass
class TrajectoryOptimizationParameters:
    N: int
    T_s: float
    Q: npt.NDArray[np.float64]
    Q_N: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]


class LcsTrajectoryOptimization:
    def __init__(
        self,
        sys: LinearComplementaritySystem,
        params: TrajectoryOptimizationParameters,
        x0: npt.NDArray[np.float64],
    ):

        prog = MathematicalProgram()
        xs = prog.NewContinuousVariables(params.N, sys.num_states, "x")
        xs = np.vstack([x0, xs])  # First entry of xs is x0
        us = prog.NewContinuousVariables(params.N, sys.num_inputs, "u")
        λs = prog.NewContinuousVariables(params.N, sys.num_forces, "λ")

        # Dynamics
        for k in range(params.N - 1):
            x, u, λ = xs[k], us[k], λs[k]
            x_next = xs[k + 1]

            x_dot = sys.get_x_dot(x, u, λ)
            forward_euler = eq(x_next, x + params.T_s * x_dot)
            for c in forward_euler:
                prog.AddLinearEqualityConstraint(c)

        # RHS nonnegativity of complementarity constraint
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]

            rhs = sys.get_complementarity_rhs(x, u, λ)
            prog.AddLinearConstraint(ge(rhs, 0))

        # LHs nonnegativity of complementarity constraint
        for k in range(params.N):
            λ = λs[k]
            prog.AddLinearConstraint(ge(λ, 0))

        # Complementarity constraint (element-wise)
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            rhs = sys.get_complementarity_rhs(x, u, λ)

            elementwise_product = λ * rhs
            for p in elementwise_product:
                prog.AddQuadraticConstraint(p, 0, 0)  # p == 0

        # Input limits
        # TODO

        # State limits
        # TODO

        # Cost
        for k in range(params.N - 1):
            x, u = xs[k], us[k]
            prog.AddQuadraticCost(x.T @ params.Q @ x)

            u = u.reshape((-1, 1))  # handle the case where u.shape = (1,)
            prog.AddQuadraticCost((u.T @ params.R @ u).item())

        # Terminal cost
        prog.AddQuadraticCost(xs[params.N].T @ params.Q_N @ xs[params.N])

        self.prog = prog
        self.xs = xs
        self.us = us
        self.λs = λs


# TODO: Move to unit test
def test_lcs_trajectory_optimization():
    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )

    x0 = np.array([0, 0, 0, 0])

    trajopt = LcsTrajectoryOptimization(sys, params, x0)

    assert trajopt.xs.shape == (params.N + 1, sys.num_states)
    assert trajopt.us.shape == (params.N, sys.num_inputs)
    assert trajopt.λs.shape == (params.N, sys.num_forces)

    # Dynamics
    assert (
        len(trajopt.prog.linear_equality_constraints())
        == (params.N - 1) * sys.num_states
    )

    # Complementarity LHS and RHS ≥ 0
    assert (
        len(trajopt.prog.linear_constraints())
        + len(trajopt.prog.bounding_box_constraints())
        == params.N * 2
    )

    # Complementarity constraints
    assert len(trajopt.prog.quadratic_constraints()) == params.N * sys.num_forces

    # Running costs + terminal cost
    assert len(trajopt.prog.quadratic_costs()) == params.N + params.N - 1


def main() -> None:
    test_cart_pole_w_walls()
    test_lcs_trajectory_optimization()

    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=10,
        T_s=0.01,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )
    x0 = np.array([0.35, 0, 0, 0])

    trajopt = LcsTrajectoryOptimization(sys, params, x0)

    options = SemidefiniteRelaxationOptions()
    options.set_to_weakest()

    relaxed_prog = MakeSemidefiniteRelaxation(trajopt.prog, options)

    solver_options = SolverOptions()
    solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    relaxed_result = Solve(relaxed_prog, solver_options=solver_options)
    assert relaxed_result.is_success()

    breakpoint()


if __name__ == "__main__":
    main()
