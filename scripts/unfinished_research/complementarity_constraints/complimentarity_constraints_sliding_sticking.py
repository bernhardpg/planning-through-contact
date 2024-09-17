from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydrake.math import eq, ge
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    L1NormCost,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MosekSolver,
    SnoptSolver,
    Solve,
    SolverOptions,
)

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.tools.utils import evaluate_np_expressions_array

OUTPUT_DIR = Path("output/complimentarity_constraints/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dt = 0.1
N = 10
mu = 0.5
mass = 1.0

prog = MathematicalProgram()

# Box / table variables
gamma = prog.NewContinuousVariables(N, "gamma")
# First component is along negative x axis, second along positive x axis
lambda_f_comps = prog.NewContinuousVariables(N, 2, "lambda_f")
lambda_n = mass * 9.81
v_rel = prog.NewContinuousVariables(N, "v_rel")

v_rel_comps = np.vstack([-v_rel, v_rel]).T  # (N, 2)

lambda_f = lambda_f_comps[:, 1] - lambda_f_comps[:, 0]
pos_0 = 0
end_pos = pos_0 + np.sum(v_rel)

# Finger variables
phi = prog.NewContinuousVariables(N + 1, "phi")
finger_lambda_n = prog.NewContinuousVariables(N, "finger_lambda_n")

e = np.ones((2,))


def get_X_from_relaxation(relaxed_prog: MathematicalProgram) -> npt.NDArray:
    assert len(relaxed_prog.positive_semidefinite_constraints()) == 1
    # We can just get the one PSD constraint matrix
    X = relaxed_prog.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))
    return X


def plot_eigvals(mat: npt.NDArray) -> None:
    eigs, _ = np.linalg.eig(mat)
    norms = np.abs(eigs)

    plt.bar(range(len(norms)), norms)
    plt.xlabel("Index of Eigenvalue")
    plt.ylabel("Norm of Eigenvalue")
    plt.title("Norms of the Eigenvalues of the Matrix")


def add_complimentarity_constraint(prog, lhs, rhs) -> List:
    """
    Adds the constraint 0 <= lhs ⊥ rhs >= 0 element-wise.
    """
    if isinstance(lhs, type(np.array([]))):
        prog.AddLinearConstraint(ge(lhs, 0))
        prog.AddLinearConstraint(ge(rhs, 0))

        product = lhs * rhs
        bindings = []
        for c in product:
            bindings.append(prog.AddQuadraticConstraint(c, 0, 0))

        return bindings  # type: ignore
    else:  # scalar
        prog.AddLinearConstraint(lhs >= 0)
        prog.AddLinearConstraint(rhs >= 0)

        product = lhs * rhs
        return [prog.AddQuadraticConstraint(product, 0, 0)]


sliding_comp_constraints = []
contact_comp_constraints = []

for i in range(N):
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e + v_rel_comps[i]
    rhs = lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * lambda_n - np.sum(lambda_f_comps[i])
    rhs = gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = phi[i]
    rhs = finger_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add force balance constraint
    prog.AddLinearConstraint(lambda_f[i] + finger_lambda_n[i] == 0)

# Initial conditions
prog.AddLinearConstraint(phi[0] == 1)
prog.AddLinearConstraint(phi[N] == 1)
prog.AddLinearConstraint(end_pos == 1)


prog.AddQuadraticCost(v_rel.T @ v_rel)  # type: ignore
# prog.AddQuadraticCost(finger_lambda_n.T @ finger_lambda_n)  # type: ignore
# prog.AddQuadraticCost(lambda_f_comps.flatten() @ lambda_f_comps.flatten())  # type: ignore
# prog.AddQuadraticCost(gamma.T @ gamma)  # type: ignore

# phi_diffs = phi[1:] - phi[:-1]
# prog.AddQuadraticCost(phi_diffs.T @ phi_diffs)  # type: ignore

# prog.AddLinearCost(np.sum(finger_lambda_n))  # type: ignore
# prog.AddLinearCost(np.sum(lambda_f_comps))  # type: ignore

relaxed_prog = MakeSemidefiniteRelaxation(prog)
X = get_X_from_relaxation(relaxed_prog)
x = prog.decision_variables()

# Use my SDP relaxation code:
# relaxed_prog = create_sdp_relaxation(prog)[0]
# X = get_X_from_relaxation(relaxed_prog)
# x = X[1:, 0]

# This seems to actually make nonlinear rounding easier.
# Would be achieve the same effect if we processed all the numbers
# and made the small numbers equal to 0?
trace_cost = relaxed_prog.AddLinearCost(1e-6 * np.trace(X))


# Set solver options to be equal
# TOL = 1e-5
mosek = MosekSolver()
mosek_options = SolverOptions()
mosek_options.SetOption(CommonSolverOption.kPrintFileName, str(OUTPUT_DIR / "mosek_log.txt"))  # type: ignore
# mosek_options.SetOption(mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_PFEAS", TOL)
# mosek_options.SetOption(mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_DFEAS", TOL)
# mosek_options.SetOption(mosek.solver_id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", TOL)

snopt = SnoptSolver()
snopt_options = SolverOptions()
snopt_options.SetOption(
    snopt.solver_id(), "Print file", str(OUTPUT_DIR / "snopt_log.txt")
)
# snopt_options.SetOption(snopt.solver_id(), "Major Feasibility Tolerance", TOL)
# snopt_options.SetOption(snopt.solver_id(), "Major Optimality Tolerance", TOL)
# snopt_options.SetOption(snopt.solver_id(), "Minor Feasibility Tolerance", TOL)
# snopt_options.SetOption(snopt.solver_id(), "Minor Optimality Tolerance", TOL)

# NOTE: We sometimes get a negative optimality gap of ~0.01%, which seems to be because
# of numerical tolerances. To investigate this properly, I would have to look at the solver
# parameters in more detail.


result = mosek.Solve(relaxed_prog, solver_options=mosek_options)  # type: ignore

assert result.is_success()

relaxed_cost = np.sum(
    relaxed_prog.EvalBindings(
        relaxed_prog.GetAllCosts()[:-1],  # skip trace_cost
        result.GetSolution(relaxed_prog.decision_variables()),
    )
)

X_sol = result.GetSolution(X)
# print(f"Rank(X): {np.linalg.matrix_rank(X_sol, tol=1e-2)}")
plot_eigvals(X_sol)

do_rounding = True
use_first_row_as_initial_guess = True
if do_rounding:
    if use_first_row_as_initial_guess:
        x_sol = result.GetSolution(x)
    else:  # use biggest eigenvector (does not work)
        eigvals, eigvecs = np.linalg.eig(X_sol)
        # get the idxs that sort the eigvals in descending order
        sorted_idxs, _ = zip(*reversed(sorted(enumerate(eigvals), key=lambda x: x[1])))
        biggest_eigvec_normalized = eigvecs[:, sorted_idxs[0]]  # eigvecs are columns
        biggest_eigval = eigvals[sorted_idxs[0]]
        biggest_eigvec = biggest_eigvec_normalized
        x_sol = biggest_eigvec[:-1]

    feasible_result = snopt.Solve(prog, initial_guess=x_sol, solver_options=snopt_options)  # type: ignore

    assert feasible_result.is_success()
    feasible_cost = feasible_result.get_optimal_cost()

    optimality_gap = ((feasible_cost - relaxed_cost) / relaxed_cost) * 100
    print(f"Global optimality gap: {optimality_gap} %")


plot = True
if plot:
    fig, axs = plt.subplots(6, 2)
    fig.set_size_inches(16, 10)  # type: ignore

    v_rel_sol = result.GetSolution(v_rel)
    phi_sol = result.GetSolution(phi)
    finger_lambda_n_sol = result.GetSolution(finger_lambda_n)
    gamma_sol = result.GetSolution(gamma)
    lambda_f_comps_sol = result.GetSolution(lambda_f_comps)

    axs[0, 0].plot(v_rel_sol)  # type: ignore
    axs[0, 0].set_title("v_rel")
    axs[0, 0].set_xlim(0, N + 1)

    axs[1, 0].plot(phi_sol)
    axs[1, 0].set_title("phi")
    axs[1, 0].set_ylim([0, max(phi_sol)])
    axs[1, 0].set_xlim(0, N + 1)

    axs[2, 0].plot(finger_lambda_n_sol)
    axs[2, 0].set_title("finger_lambda_n")
    axs[2, 0].set_xlim(0, N + 1)

    axs[3, 0].plot(lambda_f_comps_sol[:, 0])
    axs[3, 0].set_title("lambda_f 1")
    axs[3, 0].set_xlim(0, N + 1)

    axs[4, 0].plot(lambda_f_comps_sol[:, 1])
    axs[4, 0].set_title("lambda_f 2")
    axs[4, 0].set_xlim(0, N + 1)

    axs[5, 0].plot(gamma_sol)
    axs[5, 0].set_title("gamma")
    axs[5, 0].set_xlim(0, N + 1)

    if do_rounding:
        v_rel_sol = feasible_result.GetSolution(v_rel)
        phi_sol = feasible_result.GetSolution(phi)
        finger_lambda_n_sol = feasible_result.GetSolution(finger_lambda_n)
        gamma_sol = feasible_result.GetSolution(gamma)
        lambda_f_comps_sol = feasible_result.GetSolution(lambda_f_comps)

        axs[0, 1].plot(v_rel_sol)  # type: ignore
        axs[0, 1].set_title("v_rel")
        axs[0, 1].set_xlim(0, N + 1)

        axs[1, 1].plot(phi_sol)
        axs[1, 1].set_title("phi")
        axs[1, 1].set_ylim([0, max(phi_sol)])
        axs[1, 1].set_xlim(0, N + 1)

        axs[2, 1].plot(finger_lambda_n_sol)
        axs[2, 1].set_title("finger_lambda_n")
        axs[2, 1].set_xlim(0, N + 1)

        axs[3, 1].plot(lambda_f_comps_sol[:, 0])
        axs[3, 1].set_title("lambda_f 1")
        axs[3, 1].set_xlim(0, N + 1)

        axs[4, 1].plot(lambda_f_comps_sol[:, 1])
        axs[4, 1].set_title("lambda_f 2")
        axs[4, 1].set_xlim(0, N + 1)

        axs[5, 1].plot(gamma_sol)
        axs[5, 1].set_title("gamma")
        axs[5, 1].set_xlim(0, N + 1)
        fig.suptitle("Relaxation | Feasible")

    plt.tight_layout()
    plt.show()
