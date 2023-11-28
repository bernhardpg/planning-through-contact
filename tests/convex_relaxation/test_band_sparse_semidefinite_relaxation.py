import numpy as np
import numpy.typing as npt
from pydrake.math import eq, le
from pydrake.solvers import (
    Binding,
    BoundingBoxConstraint,
    CommonSolverOption,
    LinearConstraint,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
    SolverOptions,
)
from pydrake.symbolic import (
    DecomposeAffineExpression,
    DecomposeAffineExpressions,
    Expression,
    Variable,
    Variables,
)

from planning_through_contact.convex_relaxation.band_sparse_semidefinite_relaxation import (
    BandSparseSemidefiniteRelaxation,
)
from planning_through_contact.convex_relaxation.sdp import (
    create_sdp_relaxation,
    linear_bindings_to_homogenuous_form,
)
from planning_through_contact.tools.utils import convert_formula_to_lhs_expression
from planning_through_contact.visualize.analysis import plot_cos_sine_trajs

DEBUG = False


def test_band_sparse_sdp_formulation() -> None:
    NUM_CTRL_POINTS = 20
    NUM_DIMS = 2

    prog = BandSparseSemidefiniteRelaxation(NUM_CTRL_POINTS)

    rs = [
        prog.new_variables(idx, NUM_DIMS, f"r_{idx}") for idx in range(NUM_CTRL_POINTS)
    ]

    # Constrain the points to lie on the unit circle
    for i in range(NUM_CTRL_POINTS):
        r_i = rs[i]
        so_2_constraint = r_i.T.dot(r_i) - 1
        prog.add_quadratic_constraint(i, i, so_2_constraint, 0, 0)

    # Constrain the cosines and sines
    for i in range(NUM_CTRL_POINTS):
        r_i = rs[i]
        prog.add_linear_inequality_constraint(i, le(r_i, 1))
        prog.add_linear_inequality_constraint(i, le(-1, r_i))

    # Initial conditions
    th_initial = 0
    th_final = np.pi - 0.1

    create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

    initial_cond = eq(rs[0], create_r_vec_from_angle(th_initial))
    final_cond = eq(rs[-1], create_r_vec_from_angle(th_final))

    for c in initial_cond:
        prog.add_linear_equality_constraint(0, c)

    for c in final_cond:
        prog.add_linear_equality_constraint(-1, c)

    # Add in angular velocity
    th_dots = [
        prog.new_variables(idx, 1, f"th_dot_{idx}")[0]
        for idx in range(NUM_CTRL_POINTS - 1)
    ]

    def skew_symmetric(a):
        return np.array([[0, -a], [a, 0]])

    def approximate_exp_map(omega_hat):
        return np.eye(NUM_DIMS) + omega_hat + 0.5 * omega_hat @ omega_hat

    def rot_matrix(r):
        return np.array([[r[0], -r[1]], [r[1], r[0]]])

    ang_vel_constraints = []
    for idx in range(NUM_CTRL_POINTS - 1):
        th_dot_k = th_dots[idx]
        R_k = rot_matrix(rs[idx])
        R_k_next = rot_matrix(rs[idx + 1])
        omega_hat_k = skew_symmetric(th_dot_k)

        exp_om_dt = approximate_exp_map(omega_hat_k)
        constraint = exp_om_dt - R_k.T @ R_k_next
        for c in constraint.flatten():
            prog.add_quadratic_constraint(idx, idx + 1, c, 0, 0)

        ang_vel_constraints.append([expr for expr in constraint.flatten()])

    for i in range(NUM_CTRL_POINTS - 1):
        prog.add_quadratic_cost(i, i, pow(th_dots[i], 2))

    # Solve SDP relaxation
    relaxed_prog = prog.make_relaxation()
    print("Finished formulating SDP relaxation")

    solver_options = SolverOptions()
    if DEBUG:
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

    from time import time

    assert len(prog.linear_costs) == NUM_CTRL_POINTS
    assert len(prog.linear_inequality_constraints) == NUM_CTRL_POINTS
    assert len(prog.linear_inequality_constraints) == NUM_CTRL_POINTS

    assert len(prog.quadratic_costs) == 2 * NUM_CTRL_POINTS - 1
    assert len(prog.quadratic_constraints) == 2 * NUM_CTRL_POINTS - 1

    assert prog.num_groups == NUM_CTRL_POINTS

    start = time()
    result = Solve(relaxed_prog, solver_options=solver_options)
    elapsed_time = time() - start
    assert result.is_success()

    # Just make sure that if anything changes we test it!
    assert len(relaxed_prog.decision_variables()) == 977

    assert len(prog.prog.decision_variables()) == 59

    r_val = result.GetSolution(rs)  # type: ignore
    if DEBUG:
        print(f"Cost: {result.get_optimal_cost()}")
        print(f"Elapsed time: {elapsed_time}")

        plot_cos_sine_trajs(r_val)
