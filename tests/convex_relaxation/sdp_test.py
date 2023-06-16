from typing import Callable, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
import pytest
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram, Solve

from convex_relaxation.sdp import (
    _collect_bounding_box_constraints,
    _linear_bindings_to_affine_terms,
    create_sdp_relaxation,
    eliminate_equality_constraints,
    find_solution,
    get_nullspace_matrix,
)
from visualize.analysis import plot_cos_sine_trajs


class LinearSystem(NamedTuple):
    A: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]
    x_sol: Optional[npt.NDArray[np.float64]] = None


@pytest.fixture
def fully_determined_linear_system() -> LinearSystem:
    A = np.array([[1, 0, 1], [0, -3, 1], [2, 1, 3]])
    b = np.array([6, 7, 15]).reshape(-1, 1)
    x = np.array([2, -1, 4]).reshape(-1, 1)
    return LinearSystem(A, b, x)


@pytest.fixture
def underdetermined_linear_system() -> LinearSystem:
    A = np.array([[1, 1, 1], [1, 1, 2]])
    b = np.array([1, 3]).reshape(-1, 1)
    return LinearSystem(A, b)


def test_find_solution_fully_determined(
    fully_determined_linear_system: LinearSystem,
) -> None:
    A, b, x_sol = fully_determined_linear_system
    assert x_sol is not None
    x_hat = find_solution(A, b)

    assert x_hat.shape == (3, 1)

    assert np.allclose(x_sol, x_hat)


def test_find_solution_underdetermined(
    underdetermined_linear_system: LinearSystem,
) -> None:
    A, b, _ = underdetermined_linear_system
    x_hat = find_solution(A, b)
    assert np.allclose(A.dot(x_hat), b)

    assert x_hat.shape == (3, 1)


def test_get_nullspace_matrix(underdetermined_linear_system: LinearSystem) -> None:
    A, b, _ = underdetermined_linear_system
    F = get_nullspace_matrix(A)

    assert np.allclose(A.dot(F), 0)


@pytest.fixture
def bounding_box_prog() -> MathematicalProgram:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddLinearConstraint(x >= -1)
    prog.AddLinearConstraint(x <= 1)
    return prog


def test_collect_bounding_box_constraints(
    bounding_box_prog: MathematicalProgram,
) -> None:
    prog = bounding_box_prog
    x = prog.decision_variables()[0]
    y = prog.decision_variables()[1]

    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    assert bounding_box_ineqs[0].EqualTo(1 + x)
    assert bounding_box_ineqs[1].EqualTo(1 - x)
    assert len(bounding_box_eqs) == 0


@pytest.fixture
def so_2_true_sol() -> npt.NDArray[np.float64]:
    return np.array([-0.70710678, -0.70710678])


@pytest.fixture
def so_2_prog() -> MathematicalProgram:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]

    prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
    prog.AddLinearConstraint(x == y)
    prog.AddQuadraticCost(
        x * y + x + y
    )  # add a cost with a linear term to make the relaxation tight

    return prog


def test_linear_bindings_to_affine_terms(so_2_prog: MathematicalProgram) -> None:
    prog = so_2_prog
    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    A, b = _linear_bindings_to_affine_terms(
        prog.linear_equality_constraints(),
        bounding_box_eqs,
        prog.decision_variables(),
    )

    assert np.allclose(A, np.array([[-1, 1]]))
    assert np.allclose(b, np.array([[0]]))
    assert len(b.shape) == 2


def test_sdp_relaxation(so_2_prog: MathematicalProgram) -> None:
    prog = so_2_prog
    N = len(so_2_prog.decision_variables())

    relaxed_prog, X, _ = create_sdp_relaxation(prog)

    assert X.shape == (N + 1, N + 1)  # 1 in upper left corner of X

    # All quadratic costs become linear in SDP relaxation
    assert len(relaxed_prog.quadratic_costs()) == 0
    assert len(relaxed_prog.linear_costs()) == len(
        prog.linear_costs() + prog.quadratic_costs()
    )

    # Quadratic constraint becomes linear in SDP relaxation
    assert len(relaxed_prog.quadratic_constraints()) == 0


def test_sdp_relaxation_so_2_tightness(
    so_2_prog: MathematicalProgram, so_2_true_sol: npt.NDArray[np.float64]
) -> None:
    prog = so_2_prog
    relaxed_prog, X, _ = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    sol = result.GetSolution(X[1:, 0])

    assert np.allclose(sol, so_2_true_sol)


class ProgSo2WithDetails(NamedTuple):
    prog: MathematicalProgram
    initial_angle: float
    target_angle: float
    get_rs_from_x: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    create_r_vec_from_angle: Callable[[float], npt.NDArray[np.float64]]


@pytest.fixture
def so_2_prog_multiple_points() -> ProgSo2WithDetails:
    # Initial conditions
    th_initial = 0
    th_final = np.pi - 0.2

    NUM_CTRL_POINTS = 8
    NUM_DIMS = 2

    prog = MathematicalProgram()

    r = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS, "r")

    # Constrain the points to lie on the unit circle
    for i in range(NUM_CTRL_POINTS):
        r_i = r[:, i]
        so_2_constraint = r_i.T.dot(r_i) - 1
        prog.AddQuadraticConstraint(so_2_constraint, 0, 0)

    # Minimize squared euclidean distances in rotaion parameters
    for i in range(NUM_CTRL_POINTS - 1):
        r_i = r[:, i]
        r_next = r[:, i + 1]
        r_dot_i = r_next - r_i

        rot_cost_i = r_dot_i.T.dot(r_dot_i)
        prog.AddQuadraticCost(rot_cost_i)

    minimize_squares = False
    if minimize_squares:
        for i in range(NUM_CTRL_POINTS):
            r_i = r[:, i]
            prog.AddQuadraticCost(r_i.T.dot(r_i))

    create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

    initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
    final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

    for c in initial_cond:
        prog.AddLinearConstraint(c)

    for c in final_cond:
        prog.AddLinearConstraint(c)

    def get_rs_from_x(x):
        r_val = x[: NUM_CTRL_POINTS * NUM_DIMS]
        r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")
        return r_val

    return ProgSo2WithDetails(
        prog, th_initial, th_final, get_rs_from_x, create_r_vec_from_angle
    )


def test_so_2_relaxation_multiple_points(
    so_2_prog_multiple_points: ProgSo2WithDetails,
) -> None:
    (
        prog,
        th_initial,
        th_target,
        get_r_from_x,
        create_r_vec_from_angle,
    ) = so_2_prog_multiple_points

    relaxed_prog, X, _ = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    assert result.is_success()

    X_val = result.GetSolution(X)

    tol = 1e-6
    num_nonzero_eigvals = len(
        [val for val in np.linalg.eigvals(X_val) if np.abs(val) >= tol]
    )
    assert num_nonzero_eigvals == 1
    assert np.isclose(result.get_optimal_cost(), 1.2180540144494847)

    x_val = X_val[1:, 0]
    rs = get_r_from_x(x_val)

    DEBUG = False
    if DEBUG:
        plot_cos_sine_trajs(rs.T)

    assert np.allclose(rs[:, 0], create_r_vec_from_angle(th_initial))
    assert np.allclose(rs[:, -1], create_r_vec_from_angle(th_target))


def test_eq_elimination(so_2_prog_multiple_points: ProgSo2WithDetails) -> None:
    (
        prog,
        th_initial,
        th_target,
        get_r_from_x,
        create_r_vec_from_angle,
    ) = so_2_prog_multiple_points

    # Find solution by eliminating equalities first
    smaller_prog, get_x = eliminate_equality_constraints(prog)
    relaxed_prog, Z, _ = create_sdp_relaxation(smaller_prog)
    result = Solve(relaxed_prog)
    assert result.is_success()

    Z_val = result.GetSolution(Z)

    tol = 1e-6
    num_nonzero_eigvals = len(
        [val for val in np.linalg.eigvals(Z_val) if np.abs(val) >= tol]
    )
    assert num_nonzero_eigvals == 1
    z = Z_val[1:, 0]
    x = get_x(z)

    # Find solution from relaxing the program directly
    relaxed_prog, X, _ = create_sdp_relaxation(prog)
    result = Solve(relaxed_prog)
    X_val = result.GetSolution(X)
    x_val_true = X_val[1:, 0].reshape((-1, 1))

    rs = get_r_from_x(x)

    DEBUG = False
    if DEBUG:
        plot_cos_sine_trajs(rs.T)

    assert np.allclose(rs[:, 0], create_r_vec_from_angle(th_initial))
    assert np.allclose(rs[:, -1], create_r_vec_from_angle(th_target))
    assert np.allclose(x_val_true, x, atol=1e-5)


if __name__ == "__main__":
    # test_so_2_relaxation_multiple_points(so_2_prog_multiple_points())
    test_eq_elimination(so_2_prog_multiple_points())
# test_equality_elimination_with_initial_guess(so_2_prog(), so_2_true_sol())
# test_equality_elimination_with_sdp_relaxation()


# TODO: Finish writing these unit tests


# def test_equality_elimination_with_initial_guess(
#     so_2_prog: MathematicalProgram, so_2_true_sol: npt.NDArray[np.float64]
# ) -> None:
#     prog = so_2_prog
#     true_sol = so_2_true_sol
#
#     prog.SetInitialGuess(prog.decision_variables(), np.array([0.7, 0.7]))
#
#     result = Solve(prog)
#     assert result.is_success()
#
#     sol = result.GetSolution(prog.decision_variables())
#
#     smaller_prog, retrieve_x, F, x_hat = eliminate_equality_constraints(prog)
#     z = smaller_prog.decision_variables()[0]
#     smaller_prog.SetInitialGuess(z, 0.7)
#     smaller_result = Solve(smaller_prog)
#     assert smaller_result.is_success()
#
#     sol = retrieve_x(smaller_result.GetSolution(z))
#     breakpoint()
#     assert np.all(sol == true_sol)
#

# def test_equality_elimination_with_sdp_relaxation(
#     so_2_test_data: Tuple[MathematicalProgram, npt.NDArray[np.float64]]
# ):
#     prog, true_sol = so_2_test_data
#
#     # TODO: Fix problem where sdp relaxation doesn't pick up quadratic constraints
#     prog = MathematicalProgram()
#     x = prog.NewContinuousVariables(1, "x")[0]
#     y = prog.NewContinuousVariables(1, "y")[0]
#
#     prog.AddQuadraticConstraint(x**2 + y**2 - 1, 0, 0)
#     prog.AddLinearConstraint(x == y)
#
#     # Solve with SDP relaxation
#     relaxed_prog, X, _ = create_sdp_relaxation(prog)
#     x = X[1:, 0]
#
#     result = Solve(relaxed_prog)
#     assert result.is_success()
#
#     sol_relaxation = result.GetSolution(x)
#
#     # Eliminate linear equality constraints, then solve relaxation
#     smaller_prog, retrieve_x = eliminate_equality_constraints(prog)
#     relaxed_prog, Z, _ = create_sdp_relaxation(smaller_prog)
#     relaxed_result = Solve(relaxed_prog)
#     assert relaxed_result.is_success()
#
#     z_sol = relaxed_result.GetSolution(Z[1:, 0])
#     eliminated_sol = retrieve_x(z_sol)
#     assert np.allclose(sol_relaxation, eliminated_sol, atol=1e-3)
#
#     assert np.allclose(eliminated_sol, TRUE_SOL, atol=1e-3)
