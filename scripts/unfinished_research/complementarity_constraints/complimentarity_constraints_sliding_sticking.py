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

from planning_through_contact.tools.utils import evaluate_np_expressions_array

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
v_rel_comps = prog.NewContinuousVariables(N, 2, "v_rel")

v_rel = v_rel_comps[:, 1]
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

    for norm in norms:
        print(norm)

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
    prog.AddLinearConstraint(v_rel_comps[i, 0] == -v_rel_comps[i, 1])

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
# trace_cost = relaxed_prog.AddLinearCost(1e-6 * np.trace(X))
result = Solve(relaxed_prog)

assert result.is_success()

# trace_cost_sol = result.GetSolution(trace_cost.evaluator().Eval(trace_cost.variables()))
# Remove the trace from the relaxed cost so it doesn't impact the optimality gap
# relaxed_cost = result.get_optimal_cost() - trace_cost_sol
relaxed_cost = result.get_optimal_cost()

x_sol = result.GetSolution(prog.decision_variables())
prog.SetInitialGuess(prog.decision_variables(), x_sol)

snopt = SnoptSolver()
feasible_result = snopt.Solve(prog)  # type: ignore

assert feasible_result.is_success()
feasible_cost = feasible_result.get_optimal_cost()

optimality_gap = ((feasible_cost - relaxed_cost) / relaxed_cost) * 100

breakpoint()

print(f"Global optimality gap: {optimality_gap} %")

X_sol = result.GetSolution(X)
# print(f"Rank(X): {np.linalg.matrix_rank(X_sol, tol=1e-4)}")
plot_eigvals(X_sol)

fig, axs = plt.subplots(6, 1)

v_rel_sol = result.GetSolution(v_rel)
v_rel_comps_sol = result.GetSolution(v_rel_comps)
phi_sol = result.GetSolution(phi)
finger_lambda_n_sol = result.GetSolution(finger_lambda_n)
gamma_sol = result.GetSolution(gamma)
lambda_f_comps_sol = result.GetSolution(lambda_f_comps)

axs[0].plot(v_rel_sol)  # type: ignore
axs[0].set_title("v_rel")

axs[1].plot(phi_sol)
axs[1].set_title("phi")
axs[1].set_ylim([0, max(phi_sol)])

axs[2].plot(finger_lambda_n_sol)
axs[2].set_title("finger_lambda_n")

axs[3].plot(lambda_f_comps_sol[:, 0])
axs[3].set_title("lambda_f 1")

axs[4].plot(lambda_f_comps_sol[:, 1])
axs[4].set_title("lambda_f 2")

axs[5].plot(gamma_sol)
axs[5].set_title("gamma")
fig.suptitle("Relaxation")

plt.tight_layout()

fig, axs = plt.subplots(6, 1)

v_rel_sol = feasible_result.GetSolution(v_rel)
v_rel_comps_sol = feasible_result.GetSolution(v_rel_comps)
phi_sol = feasible_result.GetSolution(phi)
finger_lambda_n_sol = feasible_result.GetSolution(finger_lambda_n)
gamma_sol = feasible_result.GetSolution(gamma)
lambda_f_comps_sol = feasible_result.GetSolution(lambda_f_comps)

axs[0].plot(v_rel_sol)  # type: ignore
axs[0].set_title("v_rel")

axs[1].plot(phi_sol)
axs[1].set_title("phi")
axs[1].set_ylim([0, max(phi_sol)])

axs[2].plot(finger_lambda_n_sol)
axs[2].set_title("finger_lambda_n")

axs[3].plot(lambda_f_comps_sol[:, 0])
axs[3].set_title("lambda_f 1")

axs[4].plot(lambda_f_comps_sol[:, 1])
axs[4].set_title("lambda_f 2")

axs[5].plot(gamma_sol)
axs[5].set_title("gamma")
fig.suptitle("Feasible")

plt.tight_layout()
plt.show()
