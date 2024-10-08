from typing import Callable, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
import pytest
from pydrake.math import eq
from pydrake.solvers import (
    BoundingBoxConstraint,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
)

from planning_through_contact.convex_relaxation.sdp import (
    _collect_bounding_box_constraints,
    _linear_bindings_to_affine_terms,
    add_trace_cost_on_psd_cones,
    approximate_sdp_cones_with_linear_cones,
    create_sdp_relaxation,
    eliminate_equality_constraints,
    find_solution,
    get_principal_minor,
    to_symmetric_matrix_from_lower_triangular_columns,
)
from planning_through_contact.tools.math import null_space_basis_svd
from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs


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


def test_null_space_basis_svd(underdetermined_linear_system: LinearSystem) -> None:
    A, b, _ = underdetermined_linear_system
    F = null_space_basis_svd(A)

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


###### DEPRECATED, we now use Drake's builtin MakeSemidefiniteRelaxation


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


###### DEPRECATED (end)


def test_eq_elimination_formulation() -> None:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]
    z = prog.NewContinuousVariables(1, "z")[0]

    prog.AddLinearConstraint(y == z)

    prog.AddBoundingBoxConstraint(-10, 10, x)
    prog.AddBoundingBoxConstraint(-5, np.inf, y)
    prog.AddBoundingBoxConstraint(-5, np.inf, z)

    smaller_prog, _, _ = eliminate_equality_constraints(prog, null_space_method="svd")

    num_linear_constraints = len(smaller_prog.linear_constraints()) + len(
        smaller_prog.bounding_box_constraints()
    )
    # we should have the same number of linear constraints
    assert num_linear_constraints == 3


def _get_x(
    F: npt.NDArray[np.float64],
    x_hat: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return F @ z + x_hat.flatten()


def test_eq_elimination_qp_solution() -> None:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(1, "x")[0]
    y = prog.NewContinuousVariables(1, "y")[0]
    z = prog.NewContinuousVariables(1, "z")[0]

    prog.AddLinearConstraint(y == z)

    prog.AddBoundingBoxConstraint(-10, 10, x)
    prog.AddBoundingBoxConstraint(-5, np.inf, y)
    prog.AddBoundingBoxConstraint(-5, np.inf, z)

    prog.AddCost(x**2 + y + z)

    smaller_prog, F, x_hat = eliminate_equality_constraints(
        prog, null_space_method="svd"
    )

    prog.SetInitialGuess(prog.decision_variables(), np.array([0.1, -3, -3]))
    result = Solve(prog)

    sol = result.GetSolution(prog.decision_variables())
    # solution is [0, -5, -5]

    smaller_initial_guess = np.array([-5, -5])
    # make sure we provide the right initial guess
    assert np.allclose(sol, _get_x(F, x_hat, smaller_initial_guess))

    smaller_prog.SetInitialGuess(
        smaller_prog.decision_variables(), smaller_initial_guess
    )
    smaller_result = Solve(smaller_prog)
    assert smaller_result.is_success()  # this should not fail
    smaller_sol = _get_x(
        F, x_hat, smaller_result.GetSolution(smaller_prog.decision_variables())
    )

    assert np.allclose(sol, smaller_sol)

    # Make sure the costs are the same
    assert np.isclose(smaller_result.get_optimal_cost(), result.get_optimal_cost())


def test_so2_equality_elimination_with_initial_guess(
    so_2_prog: MathematicalProgram, so_2_true_sol: npt.NDArray[np.float64]
) -> None:
    prog = so_2_prog
    true_sol = so_2_true_sol

    prog.SetInitialGuess(prog.decision_variables(), np.array([0.7, 0.7]))

    result = Solve(prog)
    assert result.is_success()

    sol = result.GetSolution(prog.decision_variables())

    smaller_prog, F, x_hat = eliminate_equality_constraints(prog)

    z = smaller_prog.decision_variables()
    smaller_prog.SetInitialGuess(z, np.array([-0.7]))
    smaller_result = Solve(smaller_prog)
    assert smaller_result.is_success()

    sol = _get_x(F, x_hat, smaller_result.GetSolution(z))
    # There are only two feasible points
    assert np.allclose(np.abs(sol), np.abs(true_sol))


def test_eq_elimination_with_relaxation(
    so_2_prog_multiple_points: ProgSo2WithDetails,
) -> None:
    (
        prog,
        th_initial,
        th_target,
        get_r_from_x,
        create_r_vec_from_angle,
    ) = so_2_prog_multiple_points

    # Find solution by eliminating equalities first
    smaller_prog, F, x_hat = eliminate_equality_constraints(prog)

    relaxed_prog = MakeSemidefiniteRelaxation(smaller_prog)
    result = Solve(relaxed_prog)
    assert result.is_success()

    N = prog.num_vars()

    def get_X(original_prog, relaxed_prog) -> NpVariableArray:
        N = original_prog.num_vars()
        X = (
            relaxed_prog.positive_semidefinite_constraints()[0]
            .variables()
            .reshape((N + 1, N + 1))
        )
        return X

    Z = get_X(smaller_prog, relaxed_prog)
    Z_val = result.GetSolution(Z)

    # Solution should be tight with this cost
    tol = 1e-6
    num_nonzero_eigvals = len(
        [val for val in np.linalg.eigvals(Z_val) if np.abs(val) >= tol]
    )
    assert num_nonzero_eigvals == 1
    z = result.GetSolution(smaller_prog.decision_variables())
    x = _get_x(F, x_hat, z)

    # Find solution from relaxing the program directly
    relaxed_prog = MakeSemidefiniteRelaxation(prog)
    result = Solve(relaxed_prog)
    x_val_true = result.GetSolution(prog.decision_variables())

    rs = get_r_from_x(x)

    DEBUG = False
    if DEBUG:
        plot_cos_sine_trajs(rs.T)

    # Make sure the solutions are the same
    assert np.allclose(rs[:, 0], create_r_vec_from_angle(th_initial))
    assert np.allclose(rs[:, -1], create_r_vec_from_angle(th_target))
    assert np.allclose(x_val_true, x, atol=1e-5)


def test_add_trace_cost(so_2_prog: MathematicalProgram):
    original_num_costs = len(so_2_prog.GetAllCosts())
    relaxed_prog = MakeSemidefiniteRelaxation(so_2_prog)
    num_psd_cones = len(relaxed_prog.positive_semidefinite_constraints())
    add_trace_cost_on_psd_cones(relaxed_prog)

    assert len(relaxed_prog.GetAllCosts()) == original_num_costs + num_psd_cones

    for b in relaxed_prog.GetAllCosts():
        const = b.evaluator()
        assert type(const) == LinearCost


def test_approx_psd_cone_with_linear_cone(so_2_prog: MathematicalProgram):
    relaxed_prog = MakeSemidefiniteRelaxation(so_2_prog)
    approximate_sdp_cones_with_linear_cones(relaxed_prog)

    def _is_linear_constraint(const) -> bool:
        return (
            type(const) == LinearConstraint
            or type(const) == LinearEqualityConstraint
            or type(const) == BoundingBoxConstraint
        )

    for b in relaxed_prog.GetAllConstraints():
        const = b.evaluator()
        assert _is_linear_constraint(const)

    for b in relaxed_prog.GetAllCosts():
        const = b.evaluator()
        assert type(const) == LinearCost


def test_to_symmetric_from_tril_columns():
    prog = MathematicalProgram()
    X = prog.NewSymmetricContinuousVariables(3, "X")

    new_X = to_symmetric_matrix_from_lower_triangular_columns(prog.decision_variables())
    entries_equal = np.vectorize(lambda x: x.Evaluate())(eq(X, new_X))
    assert np.all(entries_equal)


def test_get_principal_minor_valid():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = [0, 2]
    expected_result = np.array([[1, 3], [7, 9]])
    result = get_principal_minor(matrix, indices)
    np.testing.assert_array_equal(result, expected_result)


def test_get_principal_minor_non_square_matrix():
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    indices = [0, 1]
    with pytest.raises(ValueError, match="Input matrix must be square"):
        get_principal_minor(matrix, indices)


def test_get_principal_minor_invalid_indices():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = [0, 3]
    with pytest.raises(
        ValueError, match="All indices must be integers within the matrix dimensions"
    ):
        get_principal_minor(matrix, indices)


def test_get_principal_minor_duplicate_indices():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = [0, 0]
    with pytest.raises(ValueError, match="Indices must not contain duplicates"):
        get_principal_minor(matrix, indices)


def test_get_principal_minor_non_numpy_array():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    indices = [0, 1]
    with pytest.raises(ValueError, match="Input matrix must be a NumPy array"):
        get_principal_minor(matrix, indices)
