from dataclasses import dataclass
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, SnoptSolver

from planning_through_contact.convex_relaxation.sdp import (
    get_gaussian_from_sdp_relaxation_solution,
    solve_sdp_relaxation,
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

        qcqp = MathematicalProgram()
        xs = qcqp.NewContinuousVariables(params.N, sys.num_states, "x")
        xs = np.vstack([x0, xs])  # First entry of xs is x0
        us = qcqp.NewContinuousVariables(params.N, sys.num_inputs, "u")
        λs = qcqp.NewContinuousVariables(params.N, sys.num_forces, "λ")

        # Dynamics
        for k in range(params.N - 1):
            x, u, λ = xs[k], us[k], λs[k]
            x_next = xs[k + 1]

            x_dot = sys.get_x_dot(x, u, λ)
            forward_euler = eq(x_next, x + params.T_s * x_dot)
            for c in forward_euler:
                qcqp.AddLinearEqualityConstraint(c)

        # RHS nonnegativity of complementarity constraint
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]

            rhs = sys.get_complementarity_rhs(x, u, λ)
            qcqp.AddLinearConstraint(ge(rhs, 0))

        # LHs nonnegativity of complementarity constraint
        for k in range(params.N):
            λ = λs[k]
            qcqp.AddLinearConstraint(ge(λ, 0))

        # Complementarity constraint (element-wise)
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            rhs = sys.get_complementarity_rhs(x, u, λ)

            elementwise_product = λ * rhs
            for p in elementwise_product:
                qcqp.AddQuadraticConstraint(p, 0, 0)  # p == 0

        # Input limits
        # TODO

        # State limits
        # TODO

        # Cost
        for k in range(params.N):
            x, u = xs[k], us[k]
            qcqp.AddQuadraticCost(x.T @ params.Q @ x)

            u = u.reshape((-1, 1))  # handle the case where u.shape = (1,)
            qcqp.AddQuadraticCost((u.T @ params.R @ u).item())

        # Terminal cost
        qcqp.AddQuadraticCost(xs[params.N].T @ params.Q_N @ xs[params.N])

        self.qcqp = qcqp
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
        len(trajopt.qcqp.linear_equality_constraints())
        == (params.N - 1) * sys.num_states
    )

    # Complementarity LHS and RHS ≥ 0
    assert (
        len(trajopt.qcqp.linear_constraints())
        + len(trajopt.qcqp.bounding_box_constraints())
        == params.N * 2
    )

    # Complementarity constraints
    assert len(trajopt.qcqp.quadratic_constraints()) == params.N * sys.num_forces

    # Running costs + terminal cost
    assert len(trajopt.qcqp.quadratic_costs()) == 2 * params.N + 1


@dataclass
class RoundingTrial:
    success: bool
    time: float
    cost: float
    result: MathematicalProgramResult


def plot_rounding_trials(trials: list[RoundingTrial]) -> None:
    num_rounding_trials = len(trials)

    # Extract attributes for plotting
    success_values = [trial.success for trial in trials]
    time_values = [trial.time for trial in trials]
    cost_values = [trial.cost for trial in trials]

    # Plotting the attributes
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    # Ensure axs is a list of Axes
    if isinstance(axs, Axes):
        axs = [axs]

    # Plot success values
    axs[0].bar(range(num_rounding_trials), success_values)
    axs[0].set_xlabel("Trial Index")
    axs[0].set_ylabel("Success")
    axs[0].set_title("Rounding Trial Success")
    axs[0].set_xticks(range(num_rounding_trials))

    # Plot time values
    axs[1].bar(range(num_rounding_trials), time_values)
    axs[1].set_xlabel("Trial Index")
    axs[1].set_ylabel("Time (s)")
    axs[1].set_title("Rounding Trial Time")
    axs[1].set_xticks(range(num_rounding_trials))

    # Plot cost values
    axs[2].bar(range(num_rounding_trials), cost_values)
    axs[2].set_xlabel("Trial Index")
    axs[2].set_ylabel("Cost")
    axs[2].set_title("Rounding Trial Cost")
    axs[2].set_xticks(range(num_rounding_trials))

    plt.tight_layout()
    plt.show()


def cart_pole_experiment_1() -> None:
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

    Y = solve_sdp_relaxation(
        qcqp=trajopt.qcqp, plot_eigvals=False, print_eigvals=True, trace_cost=False
    )
    μ, Σ = get_gaussian_from_sdp_relaxation_solution(Y)

    num_rounding_trials = 10
    initial_guesses = np.random.multivariate_normal(
        mean=μ, cov=Σ, size=num_rounding_trials
    )
    trials = []
    for initial_guess in initial_guesses:
        snopt = SnoptSolver()

        start = time()
        result = snopt.Solve(trajopt.qcqp, initial_guess)
        end = time()
        rounding_time = end - start

        trial = RoundingTrial(
            result.is_success(), rounding_time, result.get_optimal_cost(), result
        )
        trials.append(trial)

    plot_rounding_trials(trials)


def main() -> None:
    test_cart_pole_w_walls()
    test_lcs_trajectory_optimization()

    cart_pole_experiment_1()


if __name__ == "__main__":
    main()
