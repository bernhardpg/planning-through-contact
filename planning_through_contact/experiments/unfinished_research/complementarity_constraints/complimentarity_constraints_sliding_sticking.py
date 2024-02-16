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

dt = 0.1
N = 10
mu = 0.5

prog = MathematicalProgram()

# Box / table variables
gamma = prog.NewContinuousVariables(N, "gamma")
lambda_f = prog.NewContinuousVariables(N, 2, "lambda_f")
lambda_n = prog.NewContinuousVariables(N, "lambda_n")
box_pos = prog.NewContinuousVariables(N + 1, "p")
v_rel = (1 / dt) * (box_pos[1:] - box_pos[:-1])

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


def add_complimentarity_constraint(prog, lhs, rhs) -> npt.NDArray:
    """
    Adds the constraint 0 <= lhs ⊥ rhs >= 0
    element-wise.
    """
    if isinstance(lhs, type(np.array([]))):
        prog.AddLinearConstraint(ge(lhs, 0))
        prog.AddLinearConstraint(ge(rhs, 0))

        product = lhs * rhs
        bindings = []
        for c in product:
            bindings.append(prog.AddQuadraticConstraint(c, 0, 0))

        return np.array(bindings)  # type: ignore
    else:  # scalar
        prog.AddLinearConstraint(lhs >= 0)
        prog.AddLinearConstraint(rhs >= 0)

        product = lhs * rhs
        return np.array(prog.AddQuadraticConstraint(product, 0, 0))


sliding_comp_constraints = []
contact_comp_constraints = []

for i in range(N):
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e + v_rel[i]
    rhs = lambda_f[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = phi[i]
    rhs = finger_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add force balance constraint
    prog.AddLinearConstraint(-lambda_f[i, 0] + lambda_f[i, 1] + finger_lambda_n[i] == 0)

# Initial conditions
prog.AddLinearConstraint(phi[0] == 1)
prog.AddLinearConstraint(phi[N] == 1)
prog.AddLinearConstraint(box_pos[0] == 0)
prog.AddLinearConstraint(box_pos[N] == 1)

prog.AddQuadraticCost(v_rel.T @ v_rel)  # type: ignore

relaxed_prog = MakeSemidefiniteRelaxation(prog)
X = get_X_from_relaxation(relaxed_prog)
relaxed_prog.AddLinearCost(1e-6 * np.trace(X))
result = Solve(relaxed_prog)

assert result.is_success()

analyze_relaxed = False
if analyze_relaxed:
    X_sol = result.GetSolution(X)
    # print(f"Rank(X): {np.linalg.matrix_rank(X_sol, tol=1e-4)}")
    plot_eigvals(X_sol)

    fig, axs = plt.subplots(5, 1)

    axs[0].plot(result.GetSolution(box_pos))
    axs[0].set_title("Box position")

    axs[1].plot(result.GetSolution(phi))
    axs[1].set_title("SDF")
    axs[1].set_ylim([0, max(result.GetSolution(phi))])

    axs[2].plot(result.GetSolution(finger_lambda_n))
    axs[2].set_title("Finger normal force")

    plt.show()

x_sol = result.GetSolution(prog.decision_variables())
prog.SetInitialGuess(prog.decision_variables(), x_sol)

snopt = SnoptSolver()
feasible_result = snopt.Solve(prog)  # type: ignore

fig, axs = plt.subplots(5, 1)

axs[0].plot(result.GetSolution(box_pos))
axs[0].set_title("Box position")

axs[1].plot(result.GetSolution(phi))
axs[1].set_title("SDF")
axs[1].set_ylim([0, max(result.GetSolution(phi))])

axs[2].plot(result.GetSolution(finger_lambda_n))
axs[2].set_title("Finger normal force")

plt.show()
