from enum import Enum
from itertools import permutations
from logging import Logger
from pathlib import Path
from typing import Any, List, Literal, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.common.containers import EqualToDict
from pydrake.math import eq, ge, le
from pydrake.solvers import (
    Binding,
    CommonSolverOption,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    MosekSolver,
    PositiveSemidefiniteConstraint,
    QuadraticConstraint,
    SemidefiniteRelaxationOptions,
    SnoptSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.geometry.utilities import unit_vector
from planning_through_contact.tools.math import (
    null_space_basis_qr_pivot,
    null_space_basis_svd,
)
from planning_through_contact.tools.script_utils import make_default_logger
from planning_through_contact.tools.types import (
    NpExpressionArray,
    NpFormulaArray,
    NpMonomialArray,
    NpPolynomialArray,
    NpVariableArray,
)


class BoundType(Enum):
    UPPER = 0
    LOWER = 1


# TODO there is definitely a much more efficient way of doing this
def _linear_binding_to_expressions(binding: Binding) -> NpExpressionArray:
    """
    Takes in a binding and returns a polynomial p that should satisfy\
    p(x) = 0 for equality constraints, p(x) >= for inequality constraints
    
    """
    # NOTE: I cannot use binding.evaluator().Eval(binding.variables())
    # here, because it ignores the constant term for linear constraints! Is this a bug?
    A = binding.evaluator().GetDenseA()
    x = binding.variables()
    A_x = A.dot(x)
    b_upper = binding.evaluator().upper_bound()
    b_lower = binding.evaluator().lower_bound()

    formulas = []
    for a_i_x, b_i_upper, b_i_lower in zip(A_x, b_upper, b_lower):
        if b_i_upper == b_i_lower:  # eq constraint
            formulas.append(b_i_upper - a_i_x)
        elif not np.isinf(b_i_upper):
            formulas.append(b_i_upper - a_i_x)
        elif not np.isinf(b_i_lower):
            formulas.append(a_i_x - b_i_lower)

    return np.array(formulas)


def _linear_bindings_to_affine_terms(
    linear_bindings: List[Binding],
    bounding_box_expressions: NpExpressionArray,
    vars: NpVariableArray,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Returns A and b satisfying
    Ax + b = vector of expressions (bindings)
    """
    if len(linear_bindings) > 0:
        binding_type = type(linear_bindings[0].evaluator())
        if not all([isinstance(b.evaluator(), binding_type) for b in linear_bindings]):
            raise ValueError(
                "When converting to homogenous form, all bindings must be either eq or ineqs."
            )

        linear_exprs = np.concatenate(
            [
                _linear_binding_to_expressions(b)
                for b in linear_bindings
                if b.variables().size
                > 0  # some bindings are empty? This fixes it. I will have to rewrite this whole thing either way
            ]
        )
    else:
        linear_exprs = []
    all_linear_exprs = np.concatenate([linear_exprs, bounding_box_expressions])

    A, b = sym.DecomposeAffineExpressions(all_linear_exprs.flatten(), vars)
    return A, b.reshape((-1, 1))


def _affine_terms_to_homogenous_form(
    A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Assumes Ax + b = 0 should hold (note the sign difference from Ax = b)
    """
    A_homogenous = np.hstack((b.reshape(-1, 1), A))
    return A_homogenous


def linear_bindings_to_homogenuous_form(
    linear_bindings: List[Binding],
    bounding_box_expressions: NpExpressionArray,
    vars: NpVariableArray,
) -> npt.NDArray[np.float64]:
    """
    Returns the matrix that satisfies [b A][1 x]' = 0
    """
    A, b = _linear_bindings_to_affine_terms(
        linear_bindings, bounding_box_expressions, vars
    )
    A_homogenous = _affine_terms_to_homogenous_form(A, b)
    return A_homogenous


# TODO temporary
class ConstraintType(Enum):
    EQ = 0
    INEQ = 1


def _generic_constraint_binding_to_polynomials(
    binding: Binding,
) -> List[Tuple[sym.Polynomial, ConstraintType]]:
    # TODO replace with QuadraticConstraint
    poly = sym.Polynomial(binding.evaluator().Eval(binding.variables())[0])
    b_upper = binding.evaluator().upper_bound()
    b_lower = binding.evaluator().lower_bound()

    polys = []
    for b_u, b_l in zip(b_upper, b_lower):
        if b_u == b_l:  # eq constraint
            polys.append((b_u - poly, ConstraintType.EQ))
        else:
            if not np.isinf(b_l):
                polys.append((poly - b_l, ConstraintType.INEQ))
            if not np.isinf(b_u):
                polys.append((b_u - poly, ConstraintType.INEQ))
    return polys


def _quadratic_binding_to_homogenuous_form(
    binding: Binding, basis: NpMonomialArray, num_vars: int
) -> npt.NDArray[np.float64]:
    Q = binding.evaluator().Q()
    b = binding.evaluator().b()
    c = binding.evaluator().c()
    x = binding.variables()
    poly = sym.Polynomial(0.5 * x.T.dot(Q.dot(x)) + b.T.dot(x) + c)
    Q_hom = _quadratic_polynomial_to_homoenuous_form(poly, basis, num_vars)
    return Q_hom


def _get_monomial_coeffs(
    poly: sym.Polynomial, basis: NpMonomialArray
) -> npt.NDArray[np.float64]:
    coeff_map = poly.monomial_to_coefficient_map()
    coeffs = np.array([coeff_map.get(m, sym.Expression(0)).Evaluate() for m in basis])
    return coeffs


def _construct_symmetric_matrix_from_triang(
    triang_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return triang_matrix + triang_matrix.T


def _quadratic_polynomial_to_homoenuous_form(
    poly: sym.Polynomial, basis: NpMonomialArray, num_vars: int
) -> npt.NDArray[np.float64]:
    coeffs = _get_monomial_coeffs(poly, basis)
    upper_triangular = np.zeros((num_vars, num_vars))
    upper_triangular[np.triu_indices(num_vars)] = coeffs
    Q = _construct_symmetric_matrix_from_triang(upper_triangular)
    return Q * 0.5


def _generic_constraint_bindings_to_polynomials(
    generic_bindings: List[Binding],
) -> Tuple[NpPolynomialArray, NpPolynomialArray]:
    generic_constraints_as_polynomials = sum(
        [_generic_constraint_binding_to_polynomials(b) for b in generic_bindings], []
    )
    eq_polynomials = np.array(
        [p for p, t in generic_constraints_as_polynomials if t == ConstraintType.EQ]
    )
    ineq_polynomials = np.array(
        [p for p, t in generic_constraints_as_polynomials if t == ConstraintType.INEQ]
    )

    return (eq_polynomials, ineq_polynomials)


def _assert_max_degree(polys: NpPolynomialArray, degree: int) -> None:
    max_degree = max([p.TotalDegree() for p in polys])
    min_degree = min([p.TotalDegree() for p in polys])
    # if max_degree > degree or min_degree < degree:
    #     breakpoint()
    #     raise ValueError(
    #         "Can only create SDP relaxation for (possibly non-convex) Quadratically Constrainted Quadratic Programs (QCQP)"
    #     )  # TODO for now we don't allow lower degree or higher degree


def _collect_bounding_box_constraints(
    bounding_box_bindings: List[Binding],
) -> Tuple[NpExpressionArray, NpExpressionArray]:
    bounding_box_constraints = []
    for b in bounding_box_bindings:
        x = b.variables()
        b_upper = b.evaluator().upper_bound()
        b_lower = b.evaluator().lower_bound()

        for x_i, b_u, b_l in zip(x, b_upper, b_lower):
            if b_u == b_l:  # eq constraint
                # TODO: Remove this part
                raise ValueError("Bounding box equalities are not supported!")
                bounding_box_constraints.append((x_i - b_u, ConstraintType.EQ))
            else:
                if not np.isinf(b_u):
                    bounding_box_constraints.append((b_u - x_i, ConstraintType.INEQ))
                if not np.isinf(b_l):
                    bounding_box_constraints.append((x_i - b_l, ConstraintType.INEQ))

    bounding_box_eqs = np.array(
        [c for c, t in bounding_box_constraints if t == ConstraintType.EQ]
    )
    bounding_box_ineqs = np.array(
        [c for c, t in bounding_box_constraints if t == ConstraintType.INEQ]
    )

    return bounding_box_eqs, bounding_box_ineqs


def find_solution(
    A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


EqualityEliminationType = Literal["svd", "qr_pivot"]


def eliminate_equality_constraints(
    prog: MathematicalProgram,
    sparsity_viz_output_dir: Path | None = None,
    logger: Logger | None = None,
    null_space_method: EqualityEliminationType = "qr_pivot",
) -> Tuple[MathematicalProgram, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if logger is None:
        logger = make_default_logger()

    decision_vars = np.array(
        sorted(prog.decision_variables(), key=lambda x: x.get_id())
    )  # Not really necessary, they are sorted in this order in the prog
    old_dim = len(decision_vars)
    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    has_linear_eq_constraints = (
        len(prog.linear_equality_constraints()) > 0 or len(bounding_box_eqs) > 0
    )
    if not has_linear_eq_constraints:
        raise ValueError("There are no linear equality constraints to eliminate.")

    A_eq, b_eq = _linear_bindings_to_affine_terms(
        prog.linear_equality_constraints(), bounding_box_eqs, decision_vars
    )
    logger.info(f"Number of equality constraints: {A_eq.shape[0]}")
    if np.linalg.matrix_rank(A_eq, tol=1e-4) != A_eq.shape[0]:
        raise RuntimeError("Equality constraints are linearly dependent!")

    if null_space_method == "qr_pivot":
        F = null_space_basis_qr_pivot(A_eq, tol=1e-8)
    elif null_space_method == "svd":
        F = null_space_basis_svd(A_eq)
    else:
        raise NotImplementedError

    LIN_INDEP_TOL = 1e-4
    if not np.linalg.matrix_rank(A_eq, tol=LIN_INDEP_TOL) == A_eq.shape[0]:
        raise ValueError(
            f"A_eq has linearly independent rows (up to tolerance {LIN_INDEP_TOL})."
        )

    if sparsity_viz_output_dir is not None:
        visualize_sparsity(A_eq, output_dir=sparsity_viz_output_dir, postfix="_A_eq")
        visualize_sparsity(F, output_dir=sparsity_viz_output_dir, postfix="_F")

    x_hat = find_solution(
        A_eq, -b_eq
    )  # TODO: Sign must be flipped because of the way _linear_bindings_to_affine_terms returns A and b

    new_dim = F.shape[1]
    new_prog = MathematicalProgram()
    new_decision_vars = new_prog.NewContinuousVariables(new_dim, "x")

    # In SDP relaxation we will have:
    # (N^2 - N) / 2 + N variables
    # (all entries - diagonal entries)/2 (because X symmetric) + add back diagonal)
    calc_num_vars = lambda N: int(((N + 1) ** 2 - (N + 1)) / 2 + (N + 1))
    num_vars_without_elimination = calc_num_vars(old_dim)
    num_vars_with_elimination = calc_num_vars(new_dim)
    diff = num_vars_without_elimination - num_vars_with_elimination
    logger.info(f"Total number of vars in original problem: {len(decision_vars)}")
    logger.info(
        f"Total number of vars in original problem after elimination: {len(new_decision_vars)}"
    )
    logger.info(
        f"Total number of vars in original problem liminated: {len(decision_vars) - len(new_decision_vars)}"
    )
    logger.info(
        f"Total number of vars in SDP relaxation of original problem: {num_vars_without_elimination}"
    )
    logger.info(
        f"Total number of vars in SDP relaxation after elimination: {num_vars_with_elimination}"
    )
    logger.info(f"Total number of vars in SDP relaxation eliminated: {diff}")

    for idx, b in enumerate(prog.bounding_box_constraints()):
        e = b.evaluator()
        num_constraints = len(b.variables())
        var_idxs = prog.FindDecisionVariableIndices(b.variables())

        # Construct a matrix that picks out the corresponding bbox variables
        A = np.zeros((num_constraints, old_dim))
        for constraint_i, prog_var_j in enumerate(var_idxs):
            A[constraint_i, prog_var_j] = 1.0

        lb = e.lower_bound()
        ub = e.upper_bound()

        # lb <= A x <= ub becomes lb - A x_hat <= A F z <= ub - A x_hat
        # (Remember that x = Fx + x_hat)
        A_x_hat = (A @ x_hat).flatten()
        new_lb = lb - A_x_hat
        new_ub = ub - A_x_hat
        new_A = A @ F

        # NOTE: We multiply out result when adding the constraint to make sure that
        # Drake does not add the constraint as a dense constraint.
        b_new = new_prog.AddLinearConstraint(new_A @ new_decision_vars, new_lb, new_ub)

        if sparsity_viz_output_dir is not None:
            visualize_sparsity(
                A, output_dir=sparsity_viz_output_dir, postfix=f"_bbox_{idx}"
            )
            visualize_sparsity(
                new_A, output_dir=sparsity_viz_output_dir, postfix=f"_new_bbox_{idx}"
            )

    for idx, b in enumerate(prog.linear_constraints()):
        e = b.evaluator()
        binding_A = e.GetDenseA()
        num_constraints = binding_A.shape[0]
        A = np.zeros((num_constraints, old_dim))
        var_idxs = prog.FindDecisionVariableIndices(b.variables())

        for constraint_i in range(num_constraints):
            for binding_var_j, prog_var_j in enumerate(var_idxs):
                A[constraint_i, prog_var_j] = binding_A[constraint_i, binding_var_j]

        lb = e.lower_bound()
        ub = e.upper_bound()

        # lb <= A x <= ub becomes lb - A x_hat <= A F z <= ub - A x_hat
        # (Remember that x = Fx + x_hat)
        A_x_hat = (A @ x_hat).flatten()
        new_lb = lb - A_x_hat
        new_ub = ub - A_x_hat
        new_A = A @ F

        # NOTE: We multiply out result when adding the constraint to make sure that
        # Drake does not add the constraint as a dense constraint.
        new_prog.AddLinearConstraint(new_A @ new_decision_vars, new_lb, new_ub)

        if sparsity_viz_output_dir is not None:
            visualize_sparsity(
                A, output_dir=sparsity_viz_output_dir, postfix=f"_A_{idx}"
            )
            visualize_sparsity(
                new_A, output_dir=sparsity_viz_output_dir, postfix=f"_new_A_{idx}"
            )

    has_generic_constaints = len(prog.generic_constraints()) > 0
    if has_generic_constaints:
        raise ValueError(
            "Cannot eliminate equality constraints for program with generic constraints."
        )

    if len(prog.quadratic_constraints()) > 0:
        for idx, binding in enumerate(prog.quadratic_constraints()):
            e = binding.evaluator()
            Q = np.zeros((old_dim, old_dim))
            binding_Q = e.Q()
            var_idxs = prog.FindDecisionVariableIndices(binding.variables())

            for binding_i, prog_i in enumerate(var_idxs):
                for binding_j, prog_j in enumerate(var_idxs):
                    Q[prog_i, prog_j] = binding_Q[binding_i, binding_j]

            b = np.zeros(old_dim)
            b[var_idxs] = e.b()

            lb = e.lower_bound().item()
            ub = e.upper_bound().item()

            new_Q = F.T.dot(Q).dot(F)
            new_b = (x_hat.T.dot(Q).dot(F) + b.T.dot(F)).T

            new_lb = lb - (0.5 * x_hat.T.dot(Q).dot(x_hat) + b.T.dot(x_hat)).item()
            new_ub = ub - (0.5 * x_hat.T.dot(Q).dot(x_hat) + b.T.dot(x_hat)).item()

            constraint_empty = (
                np.allclose(new_Q, 0)
                and np.allclose(new_b, 0)
                and np.isclose(new_lb, 0)
                and np.isclose(new_ub, 0)
            )
            if constraint_empty:
                continue

            new_prog.AddQuadraticConstraint(
                new_Q, new_b, new_lb, new_ub, new_decision_vars
            )

            if sparsity_viz_output_dir is not None:
                if not new_lb == new_ub:
                    raise NotImplementedError(
                        "Quadratic inequality constraints are not supported."
                    )

                def _make_homogenuous(Q, b, c):
                    b = b.reshape((-1, 1))
                    # fmt: off
                    Q_hom = np.block([[0.5 * Q, 0.5 * b],
                                      [0.5 * b.T, c]])
                    # fmt: on
                    return Q_hom

                visualize_sparsity(
                    _make_homogenuous(Q, b, lb),
                    output_dir=sparsity_viz_output_dir,
                    postfix=f"_Q_{idx}",
                )
                visualize_sparsity(
                    _make_homogenuous(new_Q, new_b, new_lb),
                    output_dir=sparsity_viz_output_dir,
                    postfix=f"_new_Q_{idx}",
                )
            # Better way of doing this:
            # Q = binding.evaluator().Q()
            # b = binding.evaluator().b()
            # var_idxs = prog.FindDecisionVariableIndices(binding.variables())
            #
            # F_rows = F[var_idxs, :]
            # new_Q = F_rows.T.dot(Q).dot(F_rows)

    has_linear_costs = len(prog.linear_costs()) > 0
    if has_linear_costs:
        raise NotImplementedError("Linear costs not yet implemented!")

    if len(prog.quadratic_costs()) > 0:
        for binding in prog.quadratic_costs():
            e = binding.evaluator()
            Q = np.zeros((old_dim, old_dim))
            binding_Q = e.Q()
            var_idxs = prog.FindDecisionVariableIndices(binding.variables())

            for binding_i, prog_i in enumerate(var_idxs):
                for binding_j, prog_j in enumerate(var_idxs):
                    Q[prog_i, prog_j] = binding_Q[binding_i, binding_j]

            b = np.zeros(old_dim)
            b[var_idxs] = e.b()

            c = e.c()

            new_Q = F.T.dot(Q).dot(F)
            new_b = (x_hat.T.dot(Q).dot(F) + b.T.dot(F)).T

            new_c = (
                0.5 * x_hat.T.dot(Q).dot(x_hat)
                + b.T.dot(x_hat)
                + (0.5 * x_hat.T @ Q @ x_hat + b.T @ x_hat + c)
            )

            new_prog.AddQuadraticCost(
                new_Q, new_b, new_c, new_decision_vars, e.is_convex()
            )

    return new_prog, F, x_hat


# WARNING: This is no longer used. Instead, we use the builtin Drake function
# `MakeSemidefiniteRelaxation`.
def create_sdp_relaxation(
    prog: MathematicalProgram,
    use_linear_relaxation: bool = False,
    MULTIPLY_EQS_AND_INEQS: bool = False,
) -> Tuple[MathematicalProgram, NpVariableArray, NpMonomialArray]:
    DEGREE_QUADRATIC = 2  # We are only relaxing (non-convex) quadratic programs

    decision_vars = np.array(
        sorted(prog.decision_variables(), key=lambda x: x.get_id())
    )
    num_vars = (
        len(decision_vars) + 1
    )  # 1 will also be a decision variable in the relaxation

    basis = np.flip(sym.MonomialBasis(decision_vars, DEGREE_QUADRATIC))

    relaxed_prog = MathematicalProgram()
    X = relaxed_prog.NewSymmetricContinuousVariables(num_vars, "X")
    if use_linear_relaxation == False:
        relaxed_prog.AddPositiveSemidefiniteConstraint(X)

    relaxed_prog.AddLinearConstraint(X[0, 0] == 1)  # First variable is 1

    bounding_box_eqs, bounding_box_ineqs = _collect_bounding_box_constraints(
        prog.bounding_box_constraints()
    )

    has_linear_costs = len(prog.linear_costs()) > 0
    if has_linear_costs:
        raise NotImplementedError("Linear costs not yet implemented!")

    has_quadratic_costs = len(prog.quadratic_costs()) > 0
    if has_quadratic_costs:
        quadratic_costs = prog.quadratic_costs()
        Q_cost = [
            _quadratic_binding_to_homogenuous_form(c, basis, num_vars)
            for c in quadratic_costs
        ]
        for Q in Q_cost:
            c = np.sum(X * Q)
            relaxed_prog.AddCost(c)

    has_linear_eq_constraints = (
        len(prog.linear_equality_constraints()) > 0 or len(bounding_box_eqs) > 0
    )
    A_eq = None
    if has_linear_eq_constraints:
        A_eq = linear_bindings_to_homogenuous_form(
            prog.linear_equality_constraints(), bounding_box_eqs, decision_vars
        )
        multiplied_constraints = eq(A_eq.dot(X).flatten(), 0)
        for c in multiplied_constraints:
            relaxed_prog.AddLinearConstraint(c)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = eq(A_eq.dot(X).dot(e_1), 0)
        for c in linear_constraints:
            relaxed_prog.AddLinearConstraint(c)

    has_linear_ineq_constraints = (
        len(prog.linear_constraints()) > 0 or len(bounding_box_ineqs) > 0
    )
    A_ineq = None
    if has_linear_ineq_constraints:
        A_ineq = linear_bindings_to_homogenuous_form(
            prog.linear_constraints(), bounding_box_ineqs, decision_vars
        )
        multiplied_constraints = ge(A_ineq.dot(X).dot(A_ineq.T), 0)
        for c in multiplied_constraints.flatten():
            relaxed_prog.AddLinearConstraint(c)

        e_1 = unit_vector(0, X.shape[0])
        linear_constraints = ge(A_ineq.dot(X).dot(e_1), 0)
        for c in linear_constraints:
            relaxed_prog.AddLinearConstraint(c)

    # Multiply equality and inequality constraints together.
    # In theory, this should help, but it doesn't seem to make a
    # difference. Commented out for now as it is very slow, and
    # I will not need it once I implement null-space projections
    if MULTIPLY_EQS_AND_INEQS:
        if A_ineq is not None and A_eq is not None:
            for a_i in A_ineq:
                for a_j in A_eq:
                    outer_product = np.outer(a_j, a_i)
                    c = np.sum(outer_product * X)
                    relaxed_prog.AddLinearConstraint(c == 0)

    has_generic_constaints = len(prog.generic_constraints()) > 0
    # TODO: I can use Hongkai's PR once that is merged
    if has_generic_constaints:
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_constraint_bindings_to_polynomials(prog.generic_constraints())

        generic_constraints_as_polynomials = np.concatenate(
            (
                generic_eq_constraints_as_polynomials.flatten(),
                generic_ineq_constraints_as_polynomials.flatten(),
            )
        )
        _assert_max_degree(generic_constraints_as_polynomials, DEGREE_QUADRATIC)

        Q_eqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_eq_constraints_as_polynomials
        ]
        for Q in Q_eqs:
            constraints = eq(np.sum(X * Q), 0).flatten()

            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)
        Q_ineqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_ineq_constraints_as_polynomials
        ]
        for Q in Q_ineqs:
            constraints = ge(np.sum(X * Q), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)

    # TODO: clean up
    if len(prog.quadratic_constraints()) > 0:
        (
            generic_eq_constraints_as_polynomials,
            generic_ineq_constraints_as_polynomials,
        ) = _generic_constraint_bindings_to_polynomials(prog.quadratic_constraints())

        generic_constraints_as_polynomials = np.concatenate(
            (
                generic_eq_constraints_as_polynomials.flatten(),
                generic_ineq_constraints_as_polynomials.flatten(),
            )
        )
        # Don't add degree 0 polynomials (these should just be equal to 0)
        generic_constraints_as_polynomials = [
            c for c in generic_constraints_as_polynomials if c.TotalDegree() > 0
        ]
        _assert_max_degree(generic_constraints_as_polynomials, DEGREE_QUADRATIC)

        Q_eqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_eq_constraints_as_polynomials
        ]
        for Q in Q_eqs:
            constraints = eq(np.sum(X * Q), 0).flatten()

            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)
        Q_ineqs = [
            _quadratic_polynomial_to_homoenuous_form(p, basis, num_vars)
            for p in generic_ineq_constraints_as_polynomials
        ]
        for Q in Q_ineqs:
            constraints = ge(np.sum(X * Q), 0).flatten()
            for c in constraints:  # Drake requires us to add one constraint at the time
                relaxed_prog.AddLinearConstraint(c)

    return relaxed_prog, X, basis


def get_X_from_semidefinite_relaxation(relaxation: MathematicalProgram):
    assert len(relaxation.positive_semidefinite_constraints()) == 1
    X = relaxation.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


def get_Xs_from_semidefinite_relaxation(
    relaxation: MathematicalProgram,
) -> list[np.ndarray]:
    Xs = [
        get_X_from_psd_constraint(c)
        for c in relaxation.positive_semidefinite_constraints()
    ]
    return Xs


def get_X_from_psd_constraint(binding) -> npt.NDArray:
    assert type(binding.evaluator()) == PositiveSemidefiniteConstraint
    X = binding.variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))

    return X


def approximate_sdp_cones_with_linear_cones(prog: MathematicalProgram) -> None:
    """
    Iterates through all the PSD constraints on `prog` and replaces the PSD cone constraints
    X ≽ 0 with with the constraints X_ii ≥ 0 for i = 1, …, N (which must be true for all PSD matrices)
    """

    for psd_constraint in prog.positive_semidefinite_constraints():
        X = get_X_from_psd_constraint(psd_constraint)
        prog.RemoveConstraint(psd_constraint)  # type: ignore
        N = X.shape[0]
        for i in range(N):
            X_i = X[i, i]
            # if i == N - 1:
            #     continue  # skip 'one' == 1
            prog.AddLinearConstraint(X_i >= 0)


def add_trace_cost_on_psd_cones(
    prog: MathematicalProgram, eps: float = 1e-6
) -> List[Binding[LinearCost]]:
    added_costs = []
    for psd_constraint in prog.positive_semidefinite_constraints():
        X = get_X_from_psd_constraint(psd_constraint)
        c = prog.AddLinearCost(eps * np.trace(X))
        added_costs.append(c)

    return added_costs


def visualize_sparsity(
    matrix: npt.NDArray[np.float64],
    output_dir: Path | None = None,
    precision: float = 1e-6,
    postfix: str = "",
    color: bool = True,
):
    """
    Visualize the sparsity pattern of a given matrix, color-coded with a color bar.

    Values close to zero will appear white.

    Parameters:
    matrix (numpy.ndarray): The matrix to visualize.
    output_dir (Path, optional): Directory to save the output figure, if provided.
    precision (float): Threshold to define "sparsity" in the visualization.
    postfix (str): A string to add to the saved file name, if saving the plot.
    """

    if color:
        # Define a colormap where zero values are white
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white")

        plt.figure(figsize=(10, 10))

        max_val = (np.abs(matrix)).max()

        # Normalize the color map to ensure proper scaling (vmin, vmax exclude small values)
        norm = mcolors.Normalize(vmin=-max_val, vmax=max_val)

        # Plot the matrix using imshow with masked values and a color map
        plt.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        plt.colorbar(label="Matrix Values")
    else:
        # Plot the sparsity pattern
        plt.figure(figsize=(10, 10))
        plt.spy(matrix, precision=precision)

    if output_dir is None:
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"sparsity{postfix}.pdf")
        plt.close()


def plot_eigenvalues(
    X: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
    output_dir: Path | None = None,
    postfix: str = "",
) -> None:
    def compute_and_sort_eigenvalues(
        matrix: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.sort(eigenvalues)[::-1]

    if isinstance(X, list):
        assert all(
            matrix.shape == X[0].shape for matrix in X
        ), "All matrices must have the same shape."
        N = X[0].shape[0]
        assert all(
            matrix.shape == (N, N) for matrix in X
        ), "Each matrix must be square."

        sorted_eigenvalues_list = [compute_and_sort_eigenvalues(matrix) for matrix in X]
        eigenvalues_array = np.array(sorted_eigenvalues_list)

        mean_eigenvalues = np.mean(eigenvalues_array, axis=0)
        std_eigenvalues = np.std(eigenvalues_array, axis=0)

        sorted_eigenvalues = mean_eigenvalues
        yerr = std_eigenvalues
        title = f"Eigenvalues of the {len(X)} PSD Matrices (sorted)"
    else:
        N = X.shape[0]
        assert X.shape == (N, N), "The input matrix must be square."

        sorted_eigenvalues = compute_and_sort_eigenvalues(X)
        yerr = None
        title = "Eigenvalues of the PSD Matrix (sorted)"

    indices = np.arange(len(sorted_eigenvalues))
    plt.bar(
        indices, sorted_eigenvalues, yerr=yerr, capsize=5 if yerr is not None else 0
    )
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title(title)

    if output_dir is None:
        plt.show()
    else:
        plt.savefig(output_dir / f"eigenvalues{postfix}.pdf")
        plt.close()


def print_eigenvalues(
    X: npt.NDArray[np.float64], threshold: float = 1e-4, logger: Logger | None = None
) -> None:
    if logger is None:
        logger = make_default_logger()

    N = X.shape[0]
    assert X.shape == (N, N)

    # Compute eigenvalues
    # (note: eigvalsh is more stable for real symmetric matrices)
    eigenvalues = np.linalg.eigvalsh(X)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Filter eigenvalues above the threshold
    filtered_eigenvalues = sorted_eigenvalues[sorted_eigenvalues > threshold]

    # Print the filtered eigenvalues
    logger.info("Eigenvalues above the threshold of {}: ".format(threshold))
    for i, eigenvalue in enumerate(filtered_eigenvalues, start=1):
        logger.info("    Eigenvalue {}: {:.4f}".format(i, eigenvalue))

    if len(filtered_eigenvalues) == 1:
        logger.info("Solution is rank-tight!")


def get_principal_minor(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    """
    Returns the principal minor of a matrix for a given list of indices.

    Parameters:
    matrix (np.ndarray): The input square matrix.
    indices (list of int): The list of indices for the principal minor.

    Returns:
    np.ndarray: The principal minor submatrix.

    Raises:
    ValueError: If the matrix is not square or indices are invalid.
    """
    # Check if the input is a NumPy array
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input matrix must be a NumPy array.")

    # Check if the matrix is square
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Input matrix must be square.")

    # Check if all indices are valid
    if not all(isinstance(i, int) and 0 <= i < rows for i in indices):
        raise ValueError("All indices must be integers within the matrix dimensions.")

    # Check for duplicate indices
    if len(indices) != len(set(indices)):
        raise ValueError("Indices must not contain duplicates.")

    # Convert the indices to a numpy array for advanced indexing
    indices = np.array(indices)  # type: ignore

    # Get the submatrix corresponding to the given indices
    principal_minor = matrix[np.ix_(indices, indices)]

    return principal_minor


def solve_psd_completion(
    original_vars: npt.NDArray[Any],
    sparse_sdp: MathematicalProgram,
    sparse_result: MathematicalProgramResult,
    variable_groups: list[sym.Variables],
    print_solver_output: bool = False,
    logger: Logger | None = None,
) -> npt.NDArray[np.float64]:
    """
    Given the original decision variables, a Semidefinite Program where sparsity is exploited,
    the solution result for the sparse program, and a list of variable groups used to generate
    the sparse program, this solves the PSD completion problem for the sparse program.

    I.e. it solves a feasiblity program to find Y = [X x; xᵀ 1] ≽ 0.
    """

    if logger is None:
        logger = make_default_logger()

    sparse_Xs = get_Xs_from_semidefinite_relaxation(sparse_sdp)
    sparse_X_vals = [sparse_result.GetSolution(X) for X in sparse_Xs]

    # Check that the dimensions in the provided prog are correct.
    for group, sparse_X, sparse_X_val in zip(variable_groups, sparse_Xs, sparse_X_vals):
        if not len(group) == sparse_X.shape[0] - 1:
            raise RuntimeError(
                "Dimensions of variable groups and sparse PSD matrices do not match. \
                Something must be wrong."
            )

    one = sparse_Xs[0][-1, -1]
    assert str(one) == "one"

    # Create a new SDP with one big Y matrix.
    psd_completion = MathematicalProgram()
    psd_completion.AddDecisionVariables(np.array([one]))
    # NOTE: If we enforce one = 1.0, then we get infeasible.
    # It seems that this program is barely feasible as it is!
    psd_completion.AddLinearEqualityConstraint(one == sparse_result.GetSolution(one))

    # Add all (original) variables to the new program.
    psd_completion.AddDecisionVariables(original_vars)

    # Add the constraint X ≽ xxᵀ as Y = [X x; xᵀ 1] ≽ 0
    # (using Schur complement).
    N = len(original_vars)
    x = original_vars.reshape((-1, 1))
    X = psd_completion.NewSymmetricContinuousVariables(N, "X")
    Y = np.block([[X, x], [x.T, np.array([[one]])]])
    psd_completion.AddPositiveSemidefiniteConstraint(Y)

    # Tolerance for checking that variables are in fact equal when they should be equal.
    TOL = 1e-5

    # Check that the overlapping entries in the sparse program are in fact equal.
    for i in range(len(variable_groups) - 1):
        group = variable_groups[i]
        group_next = variable_groups[i + 1]
        common_variables = sym.intersect(group, group_next)

        sparse_X = sparse_Xs[i]
        sparse_X_next = sparse_Xs[i + 1]

        def _get_submatrix_of_common_variables_in_matrix(M: np.ndarray):
            """
            Gets the submatrix (principal minor) in M associated with all variables in common_variables.
            """
            submatrix_idxs = []
            for v in common_variables:
                for var_idx in range(M.shape[0]):
                    if v.EqualTo(M[var_idx, -1]):
                        submatrix_idxs.append(var_idx)
            return get_principal_minor(M, submatrix_idxs)

        shared_submatrix = _get_submatrix_of_common_variables_in_matrix(sparse_X)
        shared_submatrix_next = _get_submatrix_of_common_variables_in_matrix(
            sparse_X_next
        )

        shared_submatrix_val = sparse_result.GetSolution(shared_submatrix)
        shared_submatrix_next_val = sparse_result.GetSolution(shared_submatrix_next)
        # The overlapping entries should be equal in the result.
        if not np.allclose(shared_submatrix_val, shared_submatrix_next_val, atol=TOL):
            raise RuntimeError("Overlapping entries in PSD matrix are NOT equal.")

    # Collect the values for the variables that should be fixed.
    monomials_to_idx = EqualToDict({var: i for i, var in enumerate(original_vars)})
    monomials_to_idx[one] = N

    var_values = EqualToDict()
    for sparse_X, sparse_X_val in zip(sparse_Xs, sparse_X_vals):
        assert sparse_X[-1, -1].EqualTo(one)
        last_col = sparse_X[:, -1]
        last_row = sparse_X[-1, :]
        for i in range(sparse_X.shape[0]):
            for j in range(sparse_X.shape[1]):
                val = sparse_X_val[i, j]

                # Find the corresponding entry in Y
                i_monomial, j_monomial = last_row[i], last_col[j]

                if i_monomial not in monomials_to_idx:
                    raise RuntimeError(
                        f"Could not find {i_monomial} in decision variables.\
                    Something is likely wrong."
                    )

                if j_monomial not in monomials_to_idx:
                    raise RuntimeError(
                        f"Could not find {j_monomial} in decision variables.\
                    Something is likely wrong."
                    )

                i_in_Y = monomials_to_idx[i_monomial]
                j_in_Y = monomials_to_idx[j_monomial]
                var = Y[i_in_Y, j_in_Y]

                if var in var_values:
                    if not np.isclose(var_values[var], val, atol=TOL):
                        raise RuntimeError(
                            f"Found conflicting values for {var},\
                            old value = {var_values[var]}, new value = {val}"
                        )
                else:
                    var_values[var] = val

    # Constrain the values in Y to be equal to those from the sparse program solution.
    for var, val in var_values.items():
        psd_completion.AddLinearConstraint(var <= val + TOL)
        psd_completion.AddLinearConstraint(var >= val - TOL)
        # psd_completion.AddLinearEqualityConstraint(var == val)

    # Minimize the trace of Y
    psd_completion.AddLinearCost(np.trace(Y))

    mosek = MosekSolver()
    solver_options = SolverOptions()
    if print_solver_output:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
    result = mosek.Solve(psd_completion, solver_options=solver_options)  # type: ignore

    assert result.is_success()

    logger.info(
        f"Solved PSD completion in {result.get_solver_details().optimizer_time:.2f} s"
    )
    logger.info(f" -- Solution status: {result.get_solution_result()}")

    return result.GetSolution(Y)


ImpliedConstraintsType = Literal["weakest", "strongest"]


def solve_sdp_relaxation(
    qcqp: MathematicalProgram,
    trace_cost: float | None = None,
    implied_constraints: ImpliedConstraintsType = "weakest",
    variable_groups: list[sym.Variables] | None = None,
    print_solver_output: bool = False,
    plot_eigvals: bool = False,
    print_eigvals: bool = False,
    print_time: bool = False,
    logger: Logger | None = None,
    output_dir: Path | None = None,
    eq_elimination_method: EqualityEliminationType | None = None,
) -> tuple[npt.NDArray[np.float64], float, MathematicalProgramResult]:
    """
    @return Y, cost (without trace penalty), MathematicalProgramResult
    where Y = [X x; xᵀ 1] ≽ 0 ⇔ X ≽ xxᵀ
    """
    if logger is None:
        logger = make_default_logger()

    options = SemidefiniteRelaxationOptions()
    if implied_constraints == "weakest":
        options.set_to_weakest()
    else:
        options.set_to_strongest()

    SPARSITY_OUTPUT_DIR = (
        output_dir / "sparsity_patterns" if output_dir is not None else None
    )
    if eq_elimination_method is not None:
        if variable_groups:
            raise NotImplementedError(
                "Cannot use variable groups when using equality elimination"
            )
        logger.info(f"Eliminating equality constraints with {eq_elimination_method}")
        qcqp, F, x_hat = eliminate_equality_constraints(
            qcqp,
            sparsity_viz_output_dir=SPARSITY_OUTPUT_DIR,
            logger=logger,
            null_space_method=eq_elimination_method,
        )
    else:
        F, x_hat = None, None

    if variable_groups is None:
        sdp_relaxation = MakeSemidefiniteRelaxation(qcqp, options)
    else:
        sdp_relaxation = MakeSemidefiniteRelaxation(qcqp, variable_groups, options)

    solver_options = SolverOptions()
    if print_solver_output:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore
    else:
        solver_options.SetOption(CommonSolverOption.kPrintFileName, str(output_dir / "solver_log.txt"))  # type: ignore

    trace_costs = None
    if trace_cost is not None:
        trace_costs = add_trace_cost_on_psd_cones(sdp_relaxation, eps=trace_cost)

    relaxed_result = Solve(sdp_relaxation, solver_options=solver_options)
    # assert relaxed_result.is_success()
    logger.info("Found solution.")
    if print_time:
        logger.info(
            f"Elapsed solver time: {relaxed_result.get_solver_details().optimizer_time:.2f} s"  # type: ignore
        )
        logger.info(
            f"Relaxed cost: {relaxed_result.get_optimal_cost():.4f}"  # type: ignore
        )

    if variable_groups is None:
        Y = get_X_from_semidefinite_relaxation(sdp_relaxation)
        Y_val = relaxed_result.GetSolution(Y)
        Y_vals = None

        if eq_elimination_method is not None:
            assert F is not None and x_hat is not None
            Z = Y_val[:-1, :-1]
            z = Y_val[-1, :-1]
            assert Z.shape == (F.shape[1], F.shape[1])
            assert z.shape == (Z.shape[0],)
            # X = F Z Fᵀ + Fzx̂ᵀ + x̂(Fz)ᵀ + x̂x̂ᵀ
            X_val = (
                F @ Z @ F.T
                + np.outer(F @ z, x_hat)
                + np.outer(x_hat, F @ z)
                + np.outer(x_hat, x_hat)
            )
            x_val = (F @ z + x_hat.flatten()).reshape((-1, 1))
            Y_val = np.block([[X_val, x_val], [x_val.T, 1]])

    else:
        Ys = get_Xs_from_semidefinite_relaxation(sdp_relaxation)
        Y_vals = [relaxed_result.GetSolution(X_k) for X_k in Ys]

        plot_eigenvalues(Y_vals, output_dir, postfix="_sparse_Xs")

        Y_val = solve_psd_completion(
            qcqp.decision_variables(), sdp_relaxation, relaxed_result, variable_groups
        )

    if plot_eigvals:
        if Y_vals is not None:
            plot_eigenvalues(Y_vals, output_dir, postfix="_sparse_Xs")
        plot_eigenvalues(Y_val, output_dir)

    if print_eigvals:
        print_eigenvalues(Y_val, logger=logger)

    if trace_cost and trace_costs is not None:

        def eval_cost(cost: Binding[LinearCost]) -> float:
            return cost.evaluator().Eval(relaxed_result.GetSolution(cost.variables()))

        trace_cost_val = np.sum([eval_cost(cost) for cost in trace_costs])
        logger.info(f"Total trace cost: ε * Tr X = {trace_cost_val:.4f}")

        optimal_cost = relaxed_result.get_optimal_cost() - trace_cost_val

    else:
        optimal_cost = relaxed_result.get_optimal_cost()

    return Y_val, optimal_cost, relaxed_result


def compute_optimality_gap_pct(rounded_cost: float, relaxed_cost: float) -> float:
    return ((rounded_cost - relaxed_cost) / relaxed_cost) * 100


def get_gaussian_from_sdp_relaxation_solution(
    Y: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Follows the sceme from Section 3.3 in
    J. Park and S. P. Boyd, “General Heuristics for
    Nonconvex Quadratically Constrained Quadratic Programming,” 2017

    Y = [X  x
         xᵀ 1]

    @return (mean vector, covariance matrix)
    """
    N = Y.shape[0]
    assert Y.shape == (N, N)
    assert Y.dtype == float

    # NOTE: This tolerance is very low, because solving the PSD
    # completion program gives numerical instability.
    assert np.isclose(Y[-1, -1], 1, atol=1e-3)
    X = Y[:-1, :-1]
    x = Y[:-1, -1]

    μ = x
    Σ = X - np.outer(x, x)

    return μ, Σ


def to_symmetric_matrix_from_lower_triangular_columns(
    vec: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    # Determine the size of the matrix
    n = int(np.sqrt(2 * len(vec) + 0.25) - 0.5)

    if len(vec) != (n * (n + 1)) // 2:
        raise ValueError(
            "The length of the vector is not appropriate for forming a symmetric matrix."
        )

    # Create an empty symmetric matrix
    symm_matrix = np.zeros((n, n), dtype=vec.dtype)

    # Fill the lower triangle and diagonal
    index = 0
    for j in range(n):
        for i in range(j, n):
            symm_matrix[i, j] = vec[index]
            if i != j:
                symm_matrix[j, i] = vec[index]
            index += 1

    return symm_matrix
