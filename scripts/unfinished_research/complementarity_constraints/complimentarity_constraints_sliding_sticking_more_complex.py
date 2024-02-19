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
    MathematicalProgramResult,
    MosekSolver,
    SnoptSolver,
    Solve,
    SolverOptions,
)
from pydrake.symbolic import Variable

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)

OUTPUT_DIR = Path("output/complimentarity_constraints/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dt = 0.1
N = 4
mu = 0.5
mass = 1.0

f_grav_W = np.repeat(np.array([0, -mass * 9.81]).reshape((1, -1)), N, axis=0)

prog = MathematicalProgram()

# NOTE: The knot point values represent the values at the END of a time step

# Finger variables
finger_gamma = prog.NewContinuousVariables(N, "finger_gamma")
finger_phi = prog.NewContinuousVariables(N, "finger_phi")
finger_lambda_n = prog.NewContinuousVariables(N, "finger_lambda_n")
finger_lambda_f_comps = prog.NewContinuousVariables(N, 2, "finger_lambda_f")
finger_lambda_f = finger_lambda_f_comps[:, 1] - finger_lambda_f_comps[:, 0]
finger_v_rel = prog.NewContinuousVariables(N, "finger_v_rel")
finger_v_rel_comps = np.vstack([-finger_v_rel, finger_v_rel]).T  # (N, 2)
p_BF = prog.NewContinuousVariables(N, 2, "p_BF")
p_WB = prog.NewContinuousVariables(N + 1, 2, "p_WB")
cos_th = prog.NewContinuousVariables(N + 1, "cos_th")
sin_th = prog.NewContinuousVariables(N + 1, "sin_th")
R_WB = [np.array([[c, -s], [s, c]]) for c, s in zip(cos_th, sin_th)]

f_F_B = [np.array([n, -f]) for n, f in zip(finger_lambda_n, finger_lambda_f)]
f_F_W = np.vstack([R @ f for R, f in zip(R_WB, f_F_B)])

box = Box2d(width=0.2, height=0.1)


# Box / table variables
gamma = prog.NewContinuousVariables(N, "gamma")
# First component is along negative x axis, second along positive x axis
cp1_phi = prog.NewContinuousVariables(N, "cp1_phi")
cp1_lambda_f_comps = prog.NewContinuousVariables(N, 2, "cp1_lambda_f")
cp1_lambda_n = prog.NewContinuousVariables(N, "cp1_lambda_n")
cp1_v_rel = prog.NewContinuousVariables(N, "cp1_v_rel")
cp1_v_rel_comps = np.vstack([-cp1_v_rel, cp1_v_rel]).T  # (N, 2)
cp1_lambda_f = cp1_lambda_f_comps[:, 1] - cp1_lambda_f_comps[:, 0]

p_cp1_B = box.vertices[3].flatten()
p_cp1_W = np.vstack([p + R @ p_cp1_B for p, R in zip(p_WB, R_WB)])
v_cp1_W = [
    (1 / dt) * (p_next - p_curr) for p_next, p_curr in zip(p_cp1_W[1:], p_cp1_W[:-1])
]

f_cp1_B = [np.array([f, n]) for f, n in zip(cp1_lambda_f, cp1_lambda_n)]
f_cp1_W = np.vstack([R @ f for R, f in zip(R_WB, f_cp1_B)])

cp2_phi = prog.NewContinuousVariables(N, "cp2_phi")
cp2_lambda_f_comps = prog.NewContinuousVariables(N, 2, "cp2_lambda_f")
cp2_lambda_n = prog.NewContinuousVariables(N, "cp2_lambda_n")
cp2_v_rel = prog.NewContinuousVariables(N, "cp2_v_rel")
cp2_v_rel_comps = np.vstack([-cp2_v_rel, cp2_v_rel]).T  # (N, 2)
cp2_lambda_f = cp2_lambda_f_comps[:, 1] - cp2_lambda_f_comps[:, 0]

p_cp2_B = box.vertices[2].flatten()
p_cp2_W = np.vstack([p + R @ p_cp2_B for p, R in zip(p_WB, R_WB)])
v_cp2_W = [
    (1 / dt) * (p_next - p_curr) for p_next, p_curr in zip(p_cp1_W[1:], p_cp1_W[:-1])
]

f_cp2_B = [np.array([f, n]) for f, n in zip(cp2_lambda_f, cp2_lambda_n)]
f_cp2_W = np.vstack([R @ f for R, f in zip(R_WB, f_cp2_B)])

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
    # Finger
    # Add sliding/sticking complimentarity constraints
    lhs = finger_gamma[i] * e + finger_v_rel_comps[i]
    rhs = finger_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * finger_lambda_n[i] - np.sum(finger_lambda_f_comps[i])
    rhs = finger_gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Contact point 1
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e + cp1_v_rel_comps[i]
    rhs = cp1_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * cp1_lambda_n[i] - np.sum(cp1_lambda_f_comps[i])
    rhs = gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Contact point 2
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e + cp2_v_rel_comps[i]
    rhs = cp2_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * cp2_lambda_n[i] - np.sum(cp2_lambda_f_comps[i])
    rhs = gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = finger_phi[i]
    rhs = finger_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = cp1_phi[i]
    rhs = cp1_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = cp2_phi[i]
    rhs = cp2_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Enforce v_rels equal
    prog.AddLinearConstraint(cp1_v_rel[i] == cp2_v_rel[i])

    # Enforce contact point velocities correspond to positions in world frame
    prog.AddLinearConstraint(v_cp1_W[i][0] == cp1_v_rel[i])
    prog.AddLinearConstraint(v_cp2_W[i][0] == cp2_v_rel[i])

    # Enforce sdf corresponds to negative x-component of finger position
    prog.AddLinearConstraint(finger_phi[i] == -p_BF[i, 0] - box.width / 2)
    prog.AddLinearConstraint(cp1_phi[i] == p_cp1_W[i][1])
    prog.AddLinearConstraint(cp2_phi[i] == p_cp2_W[i][1])

    # Add force balance constraint
    # x direction
    prog.AddLinearConstraint(
        cp1_lambda_f[i] + cp2_lambda_f[i] + finger_lambda_n[i] == 0
    )
    # y direction
    prog.AddLinearConstraint(cp1_lambda_n[i] + cp2_lambda_n[i] + f_grav_W[i][1] == 0)

    # SO(2)
    prog.AddQuadraticConstraint(cos_th[i] ** 2 + sin_th[i] ** 2, 1, 1)

    prog.AddLinearConstraint(cos_th[i] <= 1)
    prog.AddLinearConstraint(cos_th[i] >= -1)
    prog.AddLinearConstraint(sin_th[i] <= 1)
    prog.AddLinearConstraint(sin_th[i] >= -1)


# Initial conditions
prog.AddLinearConstraint(eq(p_BF[0], np.array([-0.1 - box.width / 2, 0])))
prog.AddLinearConstraint(eq(p_BF[N - 1], np.array([-0.1 - box.width / 2, 0])))
prog.AddLinearConstraint(eq(p_WB[0].flatten(), np.array([0, box.height / 2])))
end_x_pos = 0.5
prog.AddLinearConstraint(eq(p_WB[N].flatten(), np.array([end_x_pos, box.height / 2])))

th_I = 0
th_F = 0
prog.AddLinearConstraint(cos_th[0] == np.cos(th_I))
prog.AddLinearConstraint(sin_th[0] == np.sin(th_I))
prog.AddLinearConstraint(cos_th[N] == np.cos(th_F))
prog.AddLinearConstraint(sin_th[N] == np.sin(th_F))


cos_th_diffs = cos_th[1:] - cos_th[:-1]
sin_th_diffs = sin_th[1:] - sin_th[:-1]

prog.AddQuadraticCost(cos_th_diffs.T @ cos_th_diffs)
prog.AddQuadraticCost(sin_th_diffs.T @ sin_th_diffs)

finger_phi_diff = finger_phi[1:] - finger_phi[:-1]
cp1_phi_diff = cp1_phi[1:] - cp1_phi[:-1]
cp2_phi_diff = cp2_phi[1:] - cp2_phi[:-1]

# TODO: Can in principle replace these with penalizing velocity on contact points
# (or keypoints on object)
prog.AddQuadraticCost(finger_v_rel.T @ finger_v_rel)  # type: ignore
prog.AddQuadraticCost(cp1_v_rel.T @ cp1_v_rel)  # type: ignore
prog.AddQuadraticCost(cp2_v_rel.T @ cp2_v_rel)  # type: ignore
prog.AddQuadraticCost(finger_phi_diff.T @ finger_phi_diff)  # type: ignore
prog.AddQuadraticCost(cp1_phi_diff.T @ cp1_phi_diff)  # type: ignore
prog.AddQuadraticCost(cp2_phi_diff.T @ cp2_phi_diff)  # type: ignore
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
TOL = 1e-5
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
snopt_options.SetOption(snopt.solver_id(), "Major Feasibility Tolerance", TOL)
snopt_options.SetOption(snopt.solver_id(), "Major Optimality Tolerance", TOL)
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


def animate_vals(result):
    p_WB_sol = result.GetSolution(p_WB)[:-1, :]
    p_BF_sol = result.GetSolution(p_BF)
    f_F_W_sol = evaluate_np_expressions_array(f_F_W, result)
    f_cp1_W_sol = evaluate_np_expressions_array(f_cp1_W, result)
    f_cp2_W_sol = evaluate_np_expressions_array(f_cp2_W, result)
    p_cp1_W_sol = evaluate_np_expressions_array(p_cp1_W, result)[:-1, :]
    p_cp2_W_sol = evaluate_np_expressions_array(p_cp2_W, result)[:-1, :]

    R_WB_sol = [evaluate_np_expressions_array(R, result) for R in R_WB]
    rotation_traj = np.vstack([R.flatten() for R in R_WB_sol])

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]

    p_WF_sol = np.vstack([p + R @ f for p, R, f in zip(p_WB_sol, R_WB_sol, p_BF_sol)])

    viz_com_points = [
        VisualizationPoint2d(p_WB_sol, GRAVITY_COLOR),
        VisualizationPoint2d(p_WF_sol, FINGER_COLOR),
        VisualizationPoint2d(p_cp1_W_sol, CONTACT_COLOR),
        VisualizationPoint2d(p_cp2_W_sol, CONTACT_COLOR),
    ]  # type: ignore

    table = Box2d(width=2.0, height=0.2)
    table_com = np.repeat(np.array([0, -table.height / 2]).reshape((1, -1)), N, axis=0)
    table_rot = np.repeat(np.eye(2).flatten().reshape((1, -1)), N, axis=0)

    viz_contact_forces = [
        VisualizationForce2d(p_WF_sol, CONTACT_COLOR, f_F_W_sol),
        VisualizationForce2d(p_cp1_W_sol, CONTACT_COLOR, f_cp1_W_sol),
        VisualizationForce2d(p_cp2_W_sol, CONTACT_COLOR, f_cp2_W_sol),
        VisualizationForce2d(p_WB_sol, GRAVITY_COLOR, f_grav_W),
    ]
    viz_polygons = [
        VisualizationPolygon2d.from_trajs(
            p_WB_sol,
            rotation_traj,
            box,
            BOX_COLOR,
        ),
        VisualizationPolygon2d.from_trajs(
            table_com,
            table_rot,
            table,
            TABLE_COLOR,
        ),
    ]

    viz = Visualizer2d()
    viz.visualize(viz_com_points, viz_contact_forces, viz_polygons, 1.0, None)


show_plot = False
if show_plot:
    plot_eigvals(X_sol)

    fig, axs = plt.subplots(11, 2)
    fig.set_size_inches(16, 10)  # type: ignore

    def fill_plot_col(result, col_idx):
        phi_sol = result.GetSolution(finger_phi)
        finger_lambda_n_sol = result.GetSolution(finger_lambda_n)
        gamma_sol = result.GetSolution(gamma)
        cos_th_sol = result.GetSolution(cos_th)
        sin_th_sol = result.GetSolution(sin_th)

        cp1_lambda_f_comps_sol = result.GetSolution(cp1_lambda_f_comps)
        cp1_lambda_f_sol = evaluate_np_expressions_array(cp1_lambda_f, result)
        cp1_v_rel_sol = result.GetSolution(cp1_v_rel)

        cp2_lambda_f_comps_sol = result.GetSolution(cp2_lambda_f_comps)
        cp2_lambda_f_sol = evaluate_np_expressions_array(cp2_lambda_f, result)
        cp2_v_rel_sol = result.GetSolution(cp2_v_rel)

        axs[0, col_idx].plot(phi_sol)
        axs[0, col_idx].set_title("phi")
        axs[0, col_idx].set_ylim([0, max(phi_sol)])
        axs[0, col_idx].set_xlim(0, N)

        axs[1, col_idx].plot(finger_lambda_n_sol)
        axs[1, col_idx].set_title("finger_lambda_n")
        axs[1, col_idx].set_xlim(0, N)

        axs[2, col_idx].plot(gamma_sol)
        axs[2, col_idx].set_title("gamma")
        axs[2, col_idx].set_xlim(0, N)

        axs[3, col_idx].plot(cp1_v_rel_sol)  # type: ignore
        axs[3, col_idx].set_title("cp1_v_rel")
        axs[3, col_idx].set_xlim(0, N)

        axs[4, col_idx].plot(cp1_lambda_f_comps_sol[:, 0])
        axs[4, col_idx].set_title("cp1_lambda_f 1")
        axs[4, col_idx].set_xlim(0, N)

        axs[5, col_idx].plot(cp1_lambda_f_comps_sol[:, 1])
        axs[5, col_idx].set_title("cp1_lambda_f 2")
        axs[5, col_idx].set_xlim(0, N)

        axs[6, col_idx].plot(cp1_lambda_f_sol)
        axs[6, col_idx].set_title("cp1_lambda_f")
        axs[6, col_idx].set_xlim(0, N)

        axs[7, col_idx].plot(cp2_v_rel_sol)  # type: ignore
        axs[7, col_idx].set_title("cp2_v_rel")
        axs[7, col_idx].set_xlim(0, N)

        axs[8, col_idx].plot(cp2_lambda_f_comps_sol[:, 0])
        axs[8, col_idx].set_title("cp2_lambda_f 1")
        axs[8, col_idx].set_xlim(0, N)

        axs[9, col_idx].plot(cp2_lambda_f_comps_sol[:, 1])
        axs[9, col_idx].set_title("cp2_lambda_f 2")
        axs[9, col_idx].set_xlim(0, N)

        axs[10, col_idx].plot(cp2_lambda_f_sol)
        axs[10, col_idx].set_title("cp2_lambda_f")
        axs[10, col_idx].set_xlim(0, N)

        # # Plot cos_th
        # axs[11, col_idx].plot(cos_th_sol)
        # axs[11, col_idx].set_title("cos_th")
        # axs[11, col_idx].set_xlim(0, N)
        # axs[11, col_idx].set_ylim(-1.2, 1.2)
        #
        # # Plot sin_th
        # axs[12, col_idx].plot(sin_th_sol)
        # axs[12, col_idx].set_title("sin_th")
        # axs[12, col_idx].set_xlim(0, N)
        # axs[12, col_idx].set_ylim(-1.2, 1.2)

    fill_plot_col(result, 0)

    if do_rounding:
        fill_plot_col(feasible_result, 1)

        fig.suptitle("Relaxation | Feasible")
    else:
        fig.suptitle("Relaxation")

    plt.tight_layout()
    plt.show()

else:  # animate
    animate_vals(result)
