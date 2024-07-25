from dataclasses import dataclass, fields
from functools import cached_property
from logging import Logger
from pathlib import Path
from time import time
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from pydrake.math import eq, ge
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult, SnoptSolver
from pydrake.symbolic import Variable, Variables
from pydrake.systems.controllers import DiscreteTimeLinearQuadraticRegulator
from tqdm import tqdm

from planning_through_contact.convex_relaxation.sdp import (
    ImpliedConstraintsType,
    compute_optimality_gap,
    get_gaussian_from_sdp_relaxation_solution,
    solve_sdp_relaxation,
)
from planning_through_contact.tools.script_utils import (
    YamlMixin,
    default_script_setup,
    get_current_git_commit,
)
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.colors import BROWN2, BURLYWOOD3, BURLYWOOD4


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

        self.distance_to_walls = 0.35
        self.pole_length = 0.6
        self.cart_mass = 0.978
        self.pole_mass = 0.35

    def get_linearized_pole_top_position(self, x: float, θ: float) -> float:
        """
        Returns the linearized pole position:
        x - l sin θ ≈ x - l θ

        @param x is the cart position.
        @param θ is the pole angle.
        """
        return x - self.pole_length * θ

    def get_linearized_distance_to_wall(
        self, x: float, θ: float, wall: Literal["right", "left"]
    ) -> float:
        """
        Returns the linearized distance to the wall, where the pole position is taken as
        x - l sin θ ≈ x - l θ

        @param x is the cart position.
        @param θ is the pole angle.
        """
        if wall == "right":
            return self.distance_to_walls - self.get_linearized_pole_top_position(x, θ)
        else:  # left:
            return self.distance_to_walls + self.get_linearized_pole_top_position(x, θ)


@dataclass
class CartPoleWithWallsTrajectory:
    cart_position: npt.NDArray[np.float64]
    pole_angle: npt.NDArray[np.float64]
    cart_velocity: npt.NDArray[np.float64]
    pole_velocity: npt.NDArray[np.float64]
    applied_force: npt.NDArray[np.float64]
    right_contact_force: npt.NDArray[np.float64]
    left_contact_force: npt.NDArray[np.float64]
    sys: CartPoleWithWalls
    T_s: float

    def __post_init__(self):
        states = "cart_position", "pole_angle", "cart_velocity", "pole_velocity"

        for field in fields(self):
            if field.name in ("sys", "T_s"):
                continue
            if field.name in states:
                if len(getattr(self, field.name)) != self.state_length:
                    raise ValueError(
                        f"All state trajectories must be of length {self.state_length}"
                    )
            else:
                if len(getattr(self, field.name)) != self.input_length:
                    raise ValueError(
                        f"All input/force trajectories must be of length {self.input_length}"
                    )

    @cached_property
    def linearized_pole_top_position(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.sys.get_linearized_pole_top_position(x, θ)
                for x, θ in zip(self.cart_position, self.pole_angle)
            ]
        )

    @classmethod
    def from_state_input_forces(
        cls,
        state: npt.NDArray[np.float64],
        input: npt.NDArray[np.float64],
        forces: npt.NDArray[np.float64],
        sys: CartPoleWithWalls,
        T_s: float,
    ) -> "CartPoleWithWallsTrajectory":
        if state.shape[1] != 4:
            raise ValueError("Input array must have shape (N, 4)")

        if len(input.shape) != 1:
            raise ValueError("Input array must have shape (N, )")

        if forces.shape[1] != 2:
            raise ValueError("Input array must have shape (N, 2)")

        cart_pos = state[:, 0]
        pole_pos = state[:, 1]
        cart_vel = state[:, 2]
        pole_vel = state[:, 3]
        applied_force = input
        right_wall_force = forces[:, 0]
        left_wall_force = forces[:, 1]

        return cls(
            cart_pos,
            pole_pos,
            cart_vel,
            pole_vel,
            applied_force,
            right_wall_force,
            left_wall_force,
            sys,
            T_s,
        )

    @property
    def state_length(self) -> int:
        return len(self.cart_position)

    @property
    def input_length(self) -> int:
        return len(self.cart_position) - 1

    def plot(self, filepath: Path | None = None) -> None:
        # Create a figure and subplots.
        fig, axs = plt.subplots(4, 2, figsize=(12, 6), sharex=True)

        # Ensure axs is a list of Axes.
        if isinstance(axs, Axes):
            axs = np.array([[axs]])
        elif len(axs.shape) == 1:
            axs = axs[:, np.newaxis]

        # Plot each trajectory and adjust y-axis limits.
        def plot_with_dynamic_limits(ax, data, label, color):
            ax.plot(data, label=label, color=color)
            ax.scatter(range(len(data)), data, color=color, s=10)  # Plot keypoints.
            ax.set_title(label)

            data_range = data.max() - data.min()
            TOL = 1e-1
            if data_range < TOL:  # Adjust this threshold as needed.
                ax.set_ylim(data.min() - TOL, data.max() + TOL)

        plot_with_dynamic_limits(
            axs[0][0], self.cart_position, "Cart Position [m]", "blue"
        )
        plot_with_dynamic_limits(
            axs[1][0], self.cart_velocity, "Cart Velocity [m/s]", "blue"
        )
        plot_with_dynamic_limits(
            axs[2][0], self.pole_angle * 180 / np.pi, "Pole Angle [deg]", "orange"
        )

        plot_with_dynamic_limits(
            axs[3][0],
            self.pole_velocity * 180 / np.pi,
            "Pole (angular) Velocity [deg/s]",
            "orange",
        )

        plot_with_dynamic_limits(
            axs[0][1], self.applied_force, "Applied force [N]", "green"
        )
        plot_with_dynamic_limits(
            axs[1][1], self.left_contact_force, "(Left) Contact force [N]", "red"
        )
        plot_with_dynamic_limits(
            axs[2][1], self.right_contact_force, "(Right) Contact force [N]", "red"
        )

        plot_with_dynamic_limits(
            axs[3][1],
            self.linearized_pole_top_position,
            "(Linearized) Pole top position [m]",
            "purple",
        )
        # Plot wall positions
        axs[3][1].axhline(y=self.sys.distance_to_walls, color="grey", linestyle="--")
        axs[3][1].axhline(y=-self.sys.distance_to_walls, color="grey", linestyle="--")

        # Set the x-axis label for the last subplots.
        axs[3][0].set_xlabel("Time Step")
        axs[3][1].set_xlabel("Time Step")

        # Display the plots
        plt.tight_layout()

        if filepath is not None:
            fig.savefig(filepath)
        else:
            plt.show()

    def animate(self, output_file: Path | None = None) -> None:
        # Set up the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")

        # Initialize the cart and pole
        cart_width = 0.4
        cart_height = 0.3
        pole_length = self.sys.pole_length

        cart = Rectangle(
            (0, 0), cart_width, cart_height, fc=BROWN2.diffuse(), ec="black"
        )
        ax.add_patch(cart)
        pole = Line2D(
            [],
            [],
            lw=3,
            color=BURLYWOOD3.diffuse(),  # type: ignore
            marker="o",
            markersize=9,
            markerfacecolor=BURLYWOOD4.diffuse(),  # type: ignore
            markeredgecolor="black",
        )
        ax.add_line(pole)

        # Initialize walls
        wall_distance = self.sys.distance_to_walls
        wall_height = 0.4
        wall_width = 0.1

        left_wall = Rectangle(
            (-wall_distance - wall_width / 2, 0.3),
            wall_width,
            wall_height,
            fc="grey",
            ec="black",
        )
        right_wall = Rectangle(
            (wall_distance + wall_width / 2, 0.3),
            wall_width,
            wall_height,
            fc="grey",
            ec="black",
        )
        ax.add_patch(left_wall)
        ax.add_patch(right_wall)

        # Initialize force arrows using FancyArrowPatch
        FORCE_SCALE = 1

        def create_force_patch(color):
            return FancyArrowPatch(
                (0, 0),
                (0, 0),
                arrowstyle="->",
                color=color,
                mutation_scale=8,
                linewidth=2,
                zorder=10,
            )

        applied_force_arrow = create_force_patch("blue")
        left_contact_force_arrow = create_force_patch("blue")
        right_contact_force_arrow = create_force_patch("blue")

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
            x = float(self.cart_position[frame])
            theta = float(self.pole_angle[frame])
            applied_force = float(self.applied_force[frame])
            left_contact_force = float(self.left_contact_force[frame])
            right_contact_force = float(self.right_contact_force[frame])

            cart.set_xy((x - cart_width / 2, -cart_height / 2))

            rod_x = x + pole_length * np.sin(theta)
            rod_y = pole_length * np.cos(theta)

            pole_x = [x, rod_x]
            pole_y = [0, rod_y]
            pole.set_data(pole_x, pole_y)

            # Define a threshold for plotting forces
            FORCE_THRESHOLD = 1e-3

            # Helper function to update force arrows conditionally
            def update_horizontal_force_arrow(arrow, start_pos, force):
                if abs(force) >= FORCE_THRESHOLD:
                    arrow.set_positions(
                        start_pos,
                        (start_pos[0] + force * FORCE_SCALE, start_pos[1]),
                    )
                else:
                    arrow.set_positions((0, 0), (0, 0))

            # Update force arrows
            update_horizontal_force_arrow(applied_force_arrow, (x, 0), applied_force)
            update_horizontal_force_arrow(
                left_contact_force_arrow,
                (rod_x, rod_y),
                left_contact_force,
            )
            update_horizontal_force_arrow(
                right_contact_force_arrow,
                (rod_x, rod_y),
                -right_contact_force,  # this force is along the negative x-direction.
            )

            return (
                cart,
                pole,
                applied_force_arrow,
                left_contact_force_arrow,
                right_contact_force_arrow,
            )

        interval_ms = int(self.T_s * 1000)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=self.input_length,
            init_func=init,
            blit=True,
            interval=interval_ms,
        )

        if output_file is not None:
            ani.save(output_file, writer="ffmpeg")
        else:
            plt.show()


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

    # Test distance functions
    # (Upright positions)
    distance = sys.get_linearized_distance_to_wall(0.35, 0, wall="right")
    assert distance == 0

    # (Upright positions)
    distance = sys.get_linearized_distance_to_wall(-0.35, 0, wall="left")
    assert distance == 0

    # (Some angle)
    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="left"
    ) > sys.get_linearized_distance_to_wall(0, 0.1, wall="left")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="left"
    ) < sys.get_linearized_distance_to_wall(0, -0.1, wall="left")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="right"
    ) < sys.get_linearized_distance_to_wall(0, 0.1, wall="right")

    assert sys.get_linearized_distance_to_wall(
        0, 0, wall="right"
    ) > sys.get_linearized_distance_to_wall(0, -0.1, wall="right")


@dataclass
class TrajectoryOptimizationParameters:
    N: int
    T_s: float
    Q: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    Q_N: npt.NDArray[np.float64] | None = None


class LcsTrajectoryOptimization:
    def __init__(
        self,
        sys: LinearComplementaritySystem,
        params: TrajectoryOptimizationParameters,
        x0: npt.NDArray[np.float64],
        integrator: Literal["forward_euler", "backward_euler"] = "forward_euler",
    ):

        qcqp = MathematicalProgram()
        xs = qcqp.NewContinuousVariables(params.N, sys.num_states, "x")
        xs = np.vstack([x0, xs])  # First entry of xs is x0
        us = qcqp.NewContinuousVariables(params.N, sys.num_inputs, "u")
        λs = qcqp.NewContinuousVariables(params.N, sys.num_forces, "λ")

        # Dynamics
        for k in range(params.N):
            x, u, λ = xs[k], us[k], λs[k]
            x_next = xs[k + 1]

            if integrator == "forward_euler":
                # Forward euler: x_next = x_curr + h * f(x_curr, u_curr)
                x_dot = sys.get_x_dot(x, u, λ)
                forward_euler = eq(x_next, x + params.T_s * x_dot)
                for c in forward_euler:
                    qcqp.AddLinearEqualityConstraint(c)
            else:  # "backward_euler"
                # Backward euler: x_next = x_curr + h * f(x_next, u_next)
                raise NotImplementedError()

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
        if params.Q_N is not None:
            Q_N = params.Q_N
        else:
            _, S = res = DiscreteTimeLinearQuadraticRegulator(
                sys.A, sys.B, params.Q, params.R
            )
            Q_N = S  # use the infinite-horizon optimal cost-to-go as the terminal cost

        qcqp.AddQuadraticCost(xs[params.N].T @ Q_N @ xs[params.N])

        self.qcqp = qcqp
        self.xs = xs
        self.us = us
        self.λs = λs
        self.params = params

    def evaluate_state_input_forces(
        self, result: MathematicalProgramResult
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        xs_sol = evaluate_np_expressions_array(self.xs, result)
        us_sol = result.GetSolution(self.us)
        λs_sol = result.GetSolution(self.λs)
        return xs_sol, us_sol, λs_sol

    def get_vars_at_time_step(self, k: int) -> np.ndarray:
        assert k <= self.params.N
        # First entry of self.xs is just x0 (which is a constant)
        return np.concatenate([self.xs[k + 1], self.us[k], self.λs[k]])

    def get_variable_groups(self) -> list[Variables]:
        variable_groups = [
            Variables(
                np.concatenate(
                    [self.get_vars_at_time_step(k), self.get_vars_at_time_step(k + 1)]
                )
            )
            for k in range(self.params.N - 1)
        ]
        return variable_groups


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
        len(trajopt.qcqp.linear_equality_constraints()) == (params.N) * sys.num_states
    )

    # Complementarity LHS and RHS ≥ 0
    assert (
        len(trajopt.qcqp.linear_constraints())
        + len(trajopt.qcqp.bounding_box_constraints())
        == params.N * 2
    )

    # Complementarity constraints
    assert len(trajopt.qcqp.quadratic_constraints()) == params.N * sys.num_forces

    # Running costs (2 per time step) + terminal cost
    assert len(trajopt.qcqp.quadratic_costs()) == 2 * params.N + 1

    for k in range(params.N):
        group = trajopt.get_vars_at_time_step(k)
        assert group.shape == (sys.num_states + sys.num_inputs + sys.num_forces,)
        for var in group:
            assert type(var) == Variable

    groups = trajopt.get_variable_groups()
    assert len(groups) == params.N - 1

    for group in groups:
        assert isinstance(group, Variables)
        assert len(group) == (sys.num_states + sys.num_inputs + sys.num_forces) * 2


@dataclass
class RoundingTrial:
    success: bool
    time: float
    cost: float
    result: MathematicalProgramResult

    def __str__(self) -> str:
        return f"success: {self.success}, time: {self.time}, cost: {self.cost}"


def plot_rounding_trials(
    trials: list[RoundingTrial], output_dir: Path | None = None
) -> None:
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

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / "rounding_trials.pdf")


@dataclass
class CartPoleConfig(YamlMixin):
    trajopt_params: TrajectoryOptimizationParameters
    x0: npt.NDArray[np.float64]
    implied_constraints: ImpliedConstraintsType
    trace_cost: float | None
    use_chain_sparsity: bool
    seed: int
    num_rounding_trials: int
    git_commit: str


def cart_pole_experiment_1(output_dir: Path, debug: bool, logger: Logger) -> None:
    sys = CartPoleWithWalls()
    Q = np.diag([10, 100, 0.1, 0.1])

    cfg = CartPoleConfig(
        trajopt_params=TrajectoryOptimizationParameters(
            N=20,
            T_s=0.1,
            Q=Q,
            R=np.array([1]),
        ),
        x0=np.array([0.2, 0, 0.1, 0]),
        implied_constraints="weakest",
        trace_cost=None,
        use_chain_sparsity=True,
        seed=0,
        num_rounding_trials=5,
        git_commit=get_current_git_commit(),
    )

    cfg.save(output_dir / "config.yaml")

    np.random.seed(cfg.seed)

    trajopt = LcsTrajectoryOptimization(sys, cfg.trajopt_params, cfg.x0)

    logger.info("Solving SDP relaxation...")
    Y, relaxed_cost, relaxed_result = solve_sdp_relaxation(
        qcqp=trajopt.qcqp,
        trace_cost=cfg.trace_cost,
        implied_constraints=cfg.implied_constraints,
        variable_groups=(
            trajopt.get_variable_groups() if cfg.use_chain_sparsity else None
        ),
        print_time=True,
        plot_eigvals=True,
        print_eigvals=True,
        logger=logger,
        output_dir=output_dir,
    )

    relaxed_trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
        *trajopt.evaluate_state_input_forces(relaxed_result),
        sys,
        cfg.trajopt_params.T_s,
    )
    relaxed_trajectory.plot(output_dir / "relaxed_trajectory.pdf")
    relaxed_trajectory.animate(output_dir / "relaxed_animation.mp4")

    # Rounding
    μ, Σ = get_gaussian_from_sdp_relaxation_solution(Y)

    initial_guesses = [μ]  # use the mean as an initial guess
    initial_guesses.extend(
        np.random.multivariate_normal(mean=μ, cov=Σ, size=cfg.num_rounding_trials)
    )

    trials = []
    logger.info(f"Rounding {len(initial_guesses)} trials...")
    for initial_guess in tqdm(initial_guesses):
        snopt = SnoptSolver()

        start = time()
        result = snopt.Solve(trajopt.qcqp, initial_guess)  # type: ignore
        end = time()
        rounding_time = end - start

        trial = RoundingTrial(
            result.is_success(), rounding_time, result.get_optimal_cost(), result
        )
        trials.append(trial)

    plot_rounding_trials(trials, output_dir)

    best_trial_idx = np.argmin([trial.cost for trial in trials])
    best_trial = trials[best_trial_idx]

    for idx, trial in enumerate(trials):
        logger.info(
            f"Trial {idx}: {trial}, optimality gap (upper bound): {compute_optimality_gap(trial.cost, relaxed_cost):.4f}"
        )

        trajectory = CartPoleWithWallsTrajectory.from_state_input_forces(
            *trajopt.evaluate_state_input_forces(trial.result),
            sys,
            cfg.trajopt_params.T_s,
        )

        trial_dir = output_dir / f"trial_{idx}"

        if idx == best_trial_idx:
            trial_dir = Path(str(trial_dir) + "_BEST")
        trial_dir.mkdir(exist_ok=True)
        trajectory.plot(trial_dir / "trajectory.pdf")
        trajectory.animate(trial_dir / "animation.mp4")

    logger.info(f"Best trial: {best_trial_idx}")
    logger.info(
        f"Best optimality gap: {compute_optimality_gap(best_trial.cost, relaxed_cost):.4f}%"
    )


def main(output_dir: Path, debug: bool, logger: Logger) -> None:
    test_cart_pole_w_walls()
    test_lcs_trajectory_optimization()

    cart_pole_experiment_1(output_dir, debug, logger)


if __name__ == "__main__":
    debug, output_dir, logger = default_script_setup()
    main(output_dir, debug, logger)
