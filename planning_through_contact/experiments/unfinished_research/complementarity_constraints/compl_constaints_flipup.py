from dataclasses import dataclass
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
from pydrake.symbolic import Variable

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.tools.utils import evaluate_np_expressions_array

OUTPUT_DIR = Path("output/complimentarity_constraints/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


@dataclass
class Config:
    dt: float = 0.1
    N: int = 10
    mu: float = 0.5
    box_mass: float = 1.0


# @dataclass
# class ContactPoint:
#     cfg: Config
#     gamma: npt.NDArray
#     lambda_n: npt.NDArray
#     lambda_f_comps: npt.NDArray
#     v_rel: npt.NDArray
#     phi: npt.NDArray
#
#     @classmethod
#     def create(
#         cls, name: str, prog: MathematicalProgram, cfg: Config
#     ) -> "ContactPoint":
#         lambda_n = cfg.box_mass * 9.81
#         gamma = prog.NewContinuousVariables(cfg.N - 1, f"{name}_gamma")
#         # First component is along negative x axis, second along positive x axis
#         lambda_f_comps = prog.NewContinuousVariables(cfg.N - 1, 2, f"{name}_lambda_f")
#         lambda_n = prog.NewContinuousVariables(cfg.N - 1, 2, f"{name}_lambda_n")
#         v_rel = prog.NewContinuousVariables(cfg.N - 1, f"{name}_v_rel")
#         phi = prog.NewContinuousVariables(cfg.N, f"{name}_phi")
#
#         return cls(cfg, gamma, lambda_n, lambda_f_comps, v_rel, phi)
#
#     @property
#     def v_rel_comps(self) -> npt.NDArray:
#         return np.vstack([-self.v_rel, self.v_rel]).T  # (N, 2)
#
#     @property
#     def lambda_f(self) -> npt.NDArray:
#         # First component is along negative x axis, second along positive x axis
#         return self.lambda_f_comps[:, 1] - self.lambda_f_comps[:, 0]
#
#     def sticking_sliding_comp(self, i: int) -> List[Tuple]:
#         lhs_1 = self.gamma[i] * e + self.v_rel_comps[i]
#         rhs_1 = self.lambda_f_comps[i]
#
#         lhs_2 = self.cfg.mu * self.lambda_n - np.sum(self.lambda_f_comps[i])
#         rhs_2 = self.gamma[i]
#         return [(lhs_1, rhs_1), (lhs_2, rhs_2)]


# # Box / table variables
# pos_0 = 0
# end_pos = pos_0 + np.sum(v_rel)

# Friction cone: First component is along negative x axis, second along positive x axis
cfg = Config(N=5)
prog = MathematicalProgram()

gravity_force = cfg.box_mass * 9.81

box = Box2d(width=0.2, height=0.1)

p_WB = prog.NewContinuousVariables(cfg.N, 2, "p_WB")
p_BF_x = prog.NewContinuousVariables(cfg.N, "p_BF_x")
cos_th = prog.NewContinuousVariables(cfg.N, "cos_th")
sin_th = prog.NewContinuousVariables(cfg.N, "sin_th")
R_WB = [np.array([[c, -s], [s, c]]) for s, c in zip(cos_th, sin_th)]

# Box/finger
name = "bf"
bf_phi = -p_BF_x - box.width
bf_lambda_n = prog.NewContinuousVariables(cfg.N - 1, f"{name}_lambda_n")
bf_lambda_f_comps = prog.NewContinuousVariables(cfg.N - 1, 2, f"{name}_lambda_f")
bf_lambda_f = bf_lambda_f_comps[:, 1] - bf_lambda_f_comps[:, 0]
f_F_B = [np.array([n, -f]) for n, f in zip(bf_lambda_n, bf_lambda_f)]
f_F_W = [R @ f for R, f in zip(R_WB, f_F_B)]
# no velocity (we fix sticking here)

# Box/table corner 3
name = "bt3"
bt3_gamma = prog.NewContinuousVariables(cfg.N - 1, f"{name}_gamma")
bt3_lambda_n = prog.NewContinuousVariables(cfg.N - 1, f"{name}_lambda_n")
bt3_lambda_f_comps = prog.NewContinuousVariables(cfg.N - 1, 2, f"{name}_lambda_f")
bt3_lambda_f = bt3_lambda_f_comps[:, 1] - bt3_lambda_f_comps[:, 0]
f_v3_W = [np.array([f, n]) for n, f in zip(bt3_lambda_n, bt3_lambda_f)]
p_v3_B = box.vertices[3].flatten()
p_v3_W = [p + R @ p_v3_B for p, R in zip(p_WB, R_WB)]
bt3_phi = [p[1] for p in p_v3_W]

bt3_v_rel = np.array(
    [(p_next - p_curr)[0] for p_next, p_curr in zip(p_v3_W[1:], p_v3_W[:-1])]
)
bt3_v_rel_comps = np.vstack([-bt3_v_rel, bt3_v_rel]).T  # (N, 2)

# Box/table corner 2
name = "bt2"
bt2_gamma = prog.NewContinuousVariables(cfg.N - 1, f"{name}_gamma")
bt2_lambda_n = prog.NewContinuousVariables(cfg.N - 1, f"{name}_lambda_n")
bt2_lambda_f_comps = prog.NewContinuousVariables(cfg.N - 1, 2, f"{name}_lambda_f")
bt2_lambda_f = bt2_lambda_f_comps[:, 1] - bt2_lambda_f_comps[:, 0]
f_v2_W = [np.array([f, n]) for n, f in zip(bt2_lambda_n, bt2_lambda_f)]
p_v2_B = box.vertices[2].flatten()
p_v2_W = [p + R @ p_v2_B for p, R in zip(p_WB, R_WB)]
bt2_phi = [p[1] for p in p_v2_W]

bt2_v_rel = np.array(
    [(p_next - p_curr)[0] for p_next, p_curr in zip(p_v2_W[1:], p_v2_W[:-1])]
)
bt2_v_rel_comps = np.vstack([-bt2_v_rel, bt2_v_rel]).T  # (N, 2)

e = np.ones((2,))

for i in range(cfg.N - 1):
    # Box/finger
    # Add contact/non-contact complimentarity constraints
    lhs = bf_phi[i]
    rhs = bf_lambda_n[i]
    add_complimentarity_constraint(prog, lhs, rhs)

    # friction cone
    lhs = cfg.mu * bf_lambda_n[i] - np.sum(bf_lambda_f_comps[i])
    prog.AddLinearConstraint(lhs >= 0)

    # Box/table corner 3
    # Add sliding/sticking complimentarity constraints
    lhs_1 = bt3_gamma[i] * e + bt3_v_rel_comps[i]
    rhs_1 = bt3_lambda_f_comps[i]
    add_complimentarity_constraint(prog, lhs_1, rhs_1)

    lhs_2 = cfg.mu * bt3_lambda_n[i] - np.sum(bt3_lambda_f_comps[i])
    rhs_2 = bt3_gamma[i]
    add_complimentarity_constraint(prog, lhs_2, rhs_2)

    # Add contact/non-contact complimentarity constraints
    lhs = bt3_phi[i]
    rhs = bt3_lambda_n[i]
    add_complimentarity_constraint(prog, lhs, rhs)

    # Box/table corner 2
    # Add sliding/sticking complimentarity constraints
    lhs_1 = bt2_gamma[i] * e + bt2_v_rel_comps[i]
    rhs_1 = bt2_lambda_f_comps[i]
    add_complimentarity_constraint(prog, lhs_1, rhs_1)

    lhs_2 = cfg.mu * bt2_lambda_n[i] - np.sum(bt2_lambda_f_comps[i])
    rhs_2 = bt2_gamma[i]
    add_complimentarity_constraint(prog, lhs_2, rhs_2)

    # Add force balance constraint
    sum_forces = f_F_W[i] + f_v3_W[i] + f_v2_W[i]
    for c in sum_forces:
        prog.AddQuadraticConstraint(c, 0, 0)

    prog.AddQuadraticConstraint(cos_th[i] ** 2 + sin_th[i] ** 2, 1, 1)

# Initial conditions on box
th_I = 0
th_F = 0
prog.AddLinearConstraint(cos_th[0] == np.cos(th_I))
prog.AddLinearConstraint(sin_th[0] == np.sin(th_I))
prog.AddLinearConstraint(cos_th[cfg.N - 1] == np.cos(th_F))
prog.AddLinearConstraint(sin_th[cfg.N - 1] == np.sin(th_F))
prog.AddLinearConstraint(p_WB[0, 0] == 0)
prog.AddLinearConstraint(p_WB[0, 1] == box.height)
prog.AddLinearConstraint(p_WB[cfg.N - 1, 0] == 1)
prog.AddLinearConstraint(p_WB[cfg.N - 1, 1] == box.height)

# Initial conditions on finger
prog.AddLinearConstraint(p_BF_x[0] == -box.width - 0.2)
prog.AddLinearConstraint(p_BF_x[cfg.N - 1] == -box.width - 0.2)


prog.AddQuadraticCost(bt3_v_rel.T @ bt3_v_rel)
prog.AddQuadraticCost(bt2_v_rel.T @ bt2_v_rel)

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
# mosek_options.SetOption(CommonSolverOption.kPrintFileName, str(OUTPUT_DIR / "mosek_log.txt"))  # type: ignore
mosek_options.SetOption(CommonSolverOption.kPrintToConsole, True)
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
print(f"Rank(X): {np.linalg.matrix_rank(X_sol, tol=1e-3)}")
plot_eigvals(X_sol)

do_rounding = False
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


# First, retrieve the solutions for all your decision variables
p_WB_sol = result.GetSolution(p_WB)
p_BF_x_sol = result.GetSolution(p_BF_x)
cos_th_sol = result.GetSolution(cos_th)
sin_th_sol = result.GetSolution(sin_th)
bf_lambda_n_sol = result.GetSolution(bf_lambda_n)
bf_lambda_f_sol = evaluate_np_expressions_array(bf_lambda_f, result)
bt3_gamma_sol = result.GetSolution(bt3_gamma)
bt3_lambda_n_sol = result.GetSolution(bt3_lambda_n)
bt3_lambda_f_sol = evaluate_np_expressions_array(bt3_lambda_f, result)
bt3_v_rel_sol = evaluate_np_expressions_array(bt3_v_rel, result)
bt2_gamma_sol = result.GetSolution(bt2_gamma)
bt2_lambda_n_sol = result.GetSolution(bt2_lambda_n)
bt2_lambda_f_sol = evaluate_np_expressions_array(bt2_lambda_f, result)
bt2_v_rel_sol = evaluate_np_expressions_array(bt2_v_rel, result)

# Now, plot them
fig, axs = plt.subplots(
    7, 2, figsize=(10, 10)
)  # Adjust the number of subplots based on the number of variables

axs = axs.flatten()
# Plot p_WB
for i in range(p_WB_sol.shape[1]):
    axs[0].plot(p_WB_sol[:, i], label=f"p_WB_dim_{i}")
axs[0].set_title("p_WB")
axs[0].set_xlim(0, cfg.N - 1)
axs[0].legend()

# Plot p_BF_x
axs[1].plot(p_BF_x_sol)
axs[1].set_title("p_BF_x")
axs[1].set_xlim(0, cfg.N - 1)

# Plot cos_th
axs[2].plot(cos_th_sol)
axs[2].set_title("cos_th")
axs[2].set_xlim(0, cfg.N - 1)
axs[2].set_ylim(-1, 1)


# Plot sin_th
axs[3].plot(sin_th_sol)
axs[3].set_title("sin_th")
axs[3].set_xlim(0, cfg.N - 1)
axs[3].set_ylim(-1, 1)

# Plot bf_lambda_n
axs[4].plot(bf_lambda_n_sol)
axs[4].set_title("bf_lambda_n")
axs[4].set_xlim(0, cfg.N - 1)

# Plot bf_lambda_f
axs[5].plot(bf_lambda_f_sol)
axs[5].set_title("bf_lambda_f")
axs[5].set_xlim(0, cfg.N - 1)

# Plot bt3_gamma
axs[6].plot(bt3_gamma_sol)
axs[6].set_title("bt3_gamma")
axs[6].set_xlim(0, cfg.N - 1)

# Plot bt3_lambda_n
axs[7].plot(bt3_lambda_n_sol)
axs[7].set_title("bt3_lambda_n")
axs[7].set_xlim(0, cfg.N - 1)

# Plot bt3_lambda_f
axs[8].plot(bt3_lambda_f_sol)
axs[8].set_title("bt3_lambda_f")
axs[8].set_xlim(0, cfg.N - 1)

# Plot bt3_lambda_f
axs[9].plot(bt3_v_rel_sol)
axs[9].set_title("bt3_v_rel")
axs[9].set_xlim(0, cfg.N - 1)

# Plot bt2_gamma
axs[10].plot(bt2_gamma_sol)
axs[10].set_title("bt2_gamma")
axs[10].set_xlim(0, cfg.N - 1)

# Plot bt2_lambda_n
axs[11].plot(bt2_lambda_n_sol)
axs[11].set_title("bt2_lambda_n")
axs[11].set_xlim(0, cfg.N - 1)

# Plot bt2_lambda_f
axs[12].plot(bt2_lambda_f_sol)
axs[12].set_title("bt2_lambda_f")
axs[12].set_xlim(0, cfg.N - 1)

# Plot bt3_lambda_f
axs[13].plot(bt2_v_rel_sol)
axs[13].set_title("bt2_v_rel")
axs[13].set_xlim(0, cfg.N - 1)

plt.tight_layout()
plt.show()
