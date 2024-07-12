from dataclasses import dataclass
from time import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, SnoptSolver

from planning_through_contact.convex_relaxation.sdp import (
    get_gaussian_from_sdp_relaxation_solution,
    solve_sdp_relaxation,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array


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

    # Find the index of the trial with the lowest cost
    best_trial_index = cost_values.index(min(cost_values))

    # Plotting the attributes
    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    # Ensure axs is a list of Axes
    if isinstance(axs, Axes):
        axs = [axs]

    # Plot success values
    axs[0].bar(range(num_rounding_trials), success_values, color="grey")
    axs[0].bar(
        best_trial_index, success_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[0].set_xlabel("Trial Index")
    axs[0].set_ylabel("Success")
    axs[0].set_title("Rounding Trial Success")
    axs[0].set_xticks(range(num_rounding_trials))

    # Plot time values
    axs[1].bar(range(num_rounding_trials), time_values, color="grey")
    axs[1].bar(
        best_trial_index, time_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[1].set_xlabel("Trial Index")
    axs[1].set_ylabel("Time (s)")
    axs[1].set_title("Rounding Trial Time")
    axs[1].set_xticks(range(num_rounding_trials))

    # Plot cost values
    axs[2].bar(range(num_rounding_trials), cost_values, color="grey")
    axs[2].bar(
        best_trial_index, cost_values[best_trial_index], color="red"
    )  # Highlight the best trial
    axs[2].set_xlabel("Trial Index")
    axs[2].set_ylabel("Cost")
    axs[2].set_title("Rounding Trial Cost")
    axs[2].set_xticks(range(num_rounding_trials))

    plt.tight_layout()
    plt.show()


def animate_cart_pole(
    cart_positions,
    pole_angles,
    applied_forces,
    right_contact_forces,
    left_contact_forces,
    interval_ms=20,
):
    """
    Animate the cart-pole system with wall forces.

    Parameters:
    cart_positions (list or array): The positions of the cart over time.
    pole_angles (list or array): The angles of the pole over time.
    applied_forces (list or array): The forces applied to the cart over time.
    left_contact_forces (list or array): The contact forces from the left wall over time.
    right_contact_forces (list or array): The contact forces from the right wall over time.
    interval (int): The delay between frames in milliseconds.
    """

    # Ensure inputs are numpy arrays
    cart_positions = np.array(cart_positions)
    pole_angles = np.array(pole_angles)
    applied_forces = np.array(applied_forces)
    left_contact_forces = np.array(left_contact_forces)
    right_contact_forces = np.array(right_contact_forces)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect("equal")
    ax.grid()

    # Initialize the cart and pole
    cart_width = 0.4
    cart_height = 0.2
    pole_length = 1.0

    cart = Rectangle((0, 0), cart_width, cart_height, fc="blue")
    ax.add_patch(cart)
    pole = Line2D([], [], lw=3, c="green")
    ax.add_line(pole)

    # Initialize force arrows using FancyArrowPatch
    FORCE_SCALE = 0.1

    def create_force_patch(color):
        return FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle="->", color=color, mutation_scale=10
        )

    applied_force_arrow = create_force_patch("red")
    left_contact_force_arrow = create_force_patch("purple")
    right_contact_force_arrow = create_force_patch("orange")

    ax.add_patch(applied_force_arrow)
    ax.add_patch(left_contact_force_arrow)
    ax.add_patch(right_contact_force_arrow)

    def init():
        cart.set_xy((-cart_width / 2, -cart_height / 2))
        pole.set_data([], [])
        applied_force_arrow.set_positions((0, 0), (0, 0))
        left_contact_force_arrow.set_positions((0, 0), (0, 0))
        right_contact_force_arrow.set_positions((0, 0), (0, 0))
        return (
            cart,
            pole,
            applied_force_arrow,
            left_contact_force_arrow,
            right_contact_force_arrow,
        )

    def update(frame):
        x = cart_positions[frame]
        theta = pole_angles[frame]
        applied_force = applied_forces[frame]
        left_contact_force = left_contact_forces[frame]
        right_contact_force = right_contact_forces[frame]

        cart.set_xy((x - cart_width / 2, -cart_height / 2))

        pole_x = [x, x + pole_length * np.sin(theta)]
        pole_y = [0, pole_length * np.cos(theta)]
        pole.set_data(pole_x, pole_y)

        # Update force arrows
        applied_force_arrow.set_positions((x, 0), (x + applied_force * FORCE_SCALE, 0))
        wall_height = 0.3
        wall_distance = 0.3
        left_contact_force_arrow.set_positions(
            (-wall_distance, wall_height),
            (-wall_distance + left_contact_force * FORCE_SCALE, wall_height),
        )
        right_contact_force_arrow.set_positions(
            (wall_distance, wall_height),
            (wall_distance - right_contact_force * FORCE_SCALE, wall_height),
        )

        return (
            cart,
            pole,
            applied_force_arrow,
            left_contact_force_arrow,
            right_contact_force_arrow,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(cart_positions),
        init_func=init,
        blit=True,
        interval=interval_ms,
    )
    plt.show()


def plot_cart_pole_trajectories(x: npt.NDArray[np.float64]) -> None:
    """
    Plots the trajectories for cart position, pole angle, cart velocity, and pole angular velocity.

    Parameters:
    x (numpy.ndarray): A 2D array with shape (N, 4) where N is the number of time steps. The columns represent:
                       [cart position, pole angle, cart velocity, pole angular velocity].
    """
    if x.shape[1] != 4:
        raise ValueError("Input array must have shape (N, 4)")

    # Extract individual trajectories
    cart_position = x[:, 0]
    pole_angle = x[:, 1]
    cart_velocity = x[:, 2]
    pole_angular_velocity = x[:, 3]

    # Create a figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Ensure axs is a list of Axes
    if isinstance(axs, Axes):
        axs = [axs]

    # Plot each trajectory and adjust y-axis limits
    def plot_with_dynamic_limits(ax, data, label, color):
        ax.plot(data, label=label, color=color)
        ax.scatter(range(len(data)), data, color=color, s=10)  # Plot points
        ax.set_ylabel(label)
        ax.legend(loc="upper right")

        data_range = data.max() - data.min()
        TOL = 1e-1
        if data_range < TOL:  # Adjust this threshold as needed
            ax.set_ylim(data.min() - TOL, data.max() + TOL)

    plot_with_dynamic_limits(axs[0], cart_position, "Cart Position", "blue")
    plot_with_dynamic_limits(axs[1], pole_angle, "Pole Angle", "orange")
    plot_with_dynamic_limits(axs[2], cart_velocity, "Cart Velocity", "green")
    plot_with_dynamic_limits(
        axs[3], pole_angular_velocity, "Pole Angular Velocity", "red"
    )

    # Set the x-axis label for the last subplot
    axs[3].set_xlabel("Time Step")

    # Display the plots
    plt.tight_layout()
    plt.show()


def cart_pole_experiment_1() -> None:
    sys = CartPoleWithWalls()
    params = TrajectoryOptimizationParameters(
        N=100,
        T_s=0.1,
        Q=np.diag([1, 1, 1, 1]),
        Q_N=np.diag([1, 1, 1, 1]),
        R=np.array([1]),
    )
    x0 = np.array([0.35, 0, 0, 0])

    trajopt = LcsTrajectoryOptimization(sys, params, x0)

    Y, cost_relaxed = solve_sdp_relaxation(
        qcqp=trajopt.qcqp, plot_eigvals=False, print_eigvals=False, trace_cost=False
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

    # plot_rounding_trials(trials)

    best_trial_idx = np.argmax([trial.cost for trial in trials])
    best_trial = trials[best_trial_idx]

    cost_rounded = best_trial.cost

    print(f"Optimality gap: {(cost_rounded - cost_relaxed) / cost_relaxed:.2f}%")

    xs_sol = evaluate_np_expressions_array(trajopt.xs, best_trial.result)
    us_sol = best_trial.result.GetSolution(trajopt.us)
    λs_sol = best_trial.result.GetSolution(trajopt.λs)

    # plot_cart_pole_trajectories(xs_sol)

    cart_pos = xs_sol[:, 0]
    pole_pos = xs_sol[:, 1]
    applied_force = us_sol
    right_wall_force = λs_sol[:, 0]
    left_wall_force = λs_sol[:, 1]

    # TODO something is wrong here!
    animate_cart_pole(
        cart_pos,
        pole_pos,
        applied_force,
        right_wall_force,
        left_wall_force,
        interval_ms=int(params.T_s * 1000),
    )


def main() -> None:
    test_cart_pole_w_walls()
    test_lcs_trajectory_optimization()

    cart_pole_experiment_1()


if __name__ == "__main__":
    main()
