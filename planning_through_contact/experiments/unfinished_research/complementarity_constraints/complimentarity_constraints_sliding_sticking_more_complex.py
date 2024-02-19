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
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.tools.utils import evaluate_np_expressions_array
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


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


OUTPUT_DIR = Path("output/complimentarity_constraints/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

## Plan parameters
h = 0.1
N = 4
mu = 0.5
mass = 1.0

box = Box2d(width=0.2, height=0.1)

f_grav_W = np.repeat(np.array([0, -mass * 9.81]).reshape((1, -1)), N, axis=0)

## Initial conditions
p_WB_0 = np.array([0, box.height / 2])
p_BF_0 = np.array([-0.1 - box.width / 2, 0])
th_I = 0
cos_th_0 = np.cos(th_I)
sin_th_0 = np.sin(th_I)
R_WB_0 = np.array([[cos_th_0, -sin_th_0], [sin_th_0, cos_th_0]])

p_cp1_B = box.vertices[3].flatten()
p_cp2_B = box.vertices[2].flatten()
p_cp1_W_0 = p_WB_0 + R_WB_0 @ p_cp1_B
p_cp2_W_0 = p_WB_0 + R_WB_0 @ p_cp2_B

f_F_W_0 = np.array([0, 0])
f_cp1_W_0 = -f_grav_W / 2
f_cp2_W_0 = -f_grav_W / 2

# Finger sdf corresponds to negative x-component of finger position
finger_phi_0 = -p_BF_0[0] - box.width / 2
# CP SDFs corresponds to y coordinate in world frame
cp1_phi_0 = p_cp1_W_0[1]
cp2_phi_0 = p_cp2_W_0[1]


prog = MathematicalProgram()

# NOTE: The knot point values represent the values at the END of a time step

# Finger variables
finger_gamma = prog.NewContinuousVariables(N, "finger_gamma")
finger_lambda_n = prog.NewContinuousVariables(N, "finger_lambda_n")
finger_lambda_f_comps = prog.NewContinuousVariables(N, 2, "finger_lambda_f")

p_BF = prog.NewContinuousVariables(N, 2, "p_BF")
delta_p_BF = prog.NewContinuousVariables(N, 2, "delta_p_BF")
p_WB = prog.NewContinuousVariables(N, 2, "p_WB")
delta_p_WB = prog.NewContinuousVariables(N, 2, "delta_p_WB")
cos_th = prog.NewContinuousVariables(N, "cos_th")
sin_th = prog.NewContinuousVariables(N, "sin_th")
delta_cos_th = prog.NewContinuousVariables(N, "delta_cos_th")
delta_sin_th = prog.NewContinuousVariables(N, "delta_sin_th")
R_WB = [np.array([[c, -s], [s, c]]) for c, s in zip(cos_th, sin_th)]
delta_R_WB = [np.array([[c, -s], [s, c]]) for c, s in zip(delta_cos_th, delta_sin_th)]

finger_phi = -p_BF[:, 0] - box.width / 2

finger_delta_tang_pos = p_BF[:, 1]  # TODO: Replace this with general jacobian
finger_delta_tang_pos_comps = np.vstack(
    [-finger_delta_tang_pos, finger_delta_tang_pos]
).T  # (N, 2)
finger_lambda_f = finger_lambda_f_comps[:, 1] - finger_lambda_f_comps[:, 0]
f_F_B = [np.array([n, -f]) for n, f in zip(finger_lambda_n, finger_lambda_f)]
f_F_W = np.vstack([R @ f for R, f in zip(R_WB, f_F_B)])
f_grav_B = [R.T @ f for R, f in zip(R_WB, f_grav_W)]


# Shared box/table variables
gamma = prog.NewContinuousVariables(N, "gamma")

# Contact point 1
cp1_lambda_f_comps = prog.NewContinuousVariables(N, 2, "cp1_lambda_f")
cp1_lambda_n = prog.NewContinuousVariables(N, "cp1_lambda_n")
p_cp1_W = np.vstack([p + R @ p_cp1_B for p, R in zip(p_WB, R_WB)])
delta_p_cp1_W = np.vstack(
    [delta_p + delta_R @ p_cp1_B for delta_p, delta_R in zip(delta_p_WB, delta_R_WB)]
)
cp1_delta_tang_pos = delta_p_cp1_W[:, 0]
cp1_delta_tang_pos_comps = np.vstack(
    [-cp1_delta_tang_pos, cp1_delta_tang_pos]
).T  # (N, 2)
cp1_lambda_f = cp1_lambda_f_comps[:, 1] - cp1_lambda_f_comps[:, 0]

cp1_phi = p_cp1_W[:, 1]

f_cp1_B = [np.array([f, n]) for f, n in zip(cp1_lambda_f, cp1_lambda_n)]
f_cp1_W = np.vstack([R @ f for R, f in zip(R_WB, f_cp1_B)])

# Contact point 2
cp2_lambda_f_comps = prog.NewContinuousVariables(N, 2, "cp2_lambda_f")
cp2_lambda_n = prog.NewContinuousVariables(N, "cp2_lambda_n")
p_cp2_W = np.vstack([p + R @ p_cp2_B for p, R in zip(p_WB, R_WB)])
delta_p_cp2_W = np.vstack(
    [delta_p + delta_R @ p_cp2_B for delta_p, delta_R in zip(delta_p_WB, delta_R_WB)]
)
cp2_delta_tang_pos = delta_p_cp2_W[:, 0]
cp2_delta_tang_pos_comps = np.vstack(
    [-cp2_delta_tang_pos, cp2_delta_tang_pos]
).T  # (N, 2)

cp2_lambda_f = cp2_lambda_f_comps[:, 1] - cp2_lambda_f_comps[:, 0]
cp2_phi = p_cp2_W[:, 1]

f_cp2_B = [np.array([f, n]) for f, n in zip(cp2_lambda_f, cp2_lambda_n)]
f_cp2_W = np.vstack([R @ f for R, f in zip(R_WB, f_cp2_B)])

e = np.ones((2,))

sliding_comp_constraints = []
contact_comp_constraints = []

for i in range(N):
    # Finger
    # Add sliding/sticking complimentarity constraints
    lhs = finger_gamma[i] * e * h + finger_delta_tang_pos_comps[i]
    rhs = finger_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * finger_lambda_n[i] - np.sum(finger_lambda_f_comps[i])
    rhs = finger_gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = finger_phi[i]
    rhs = finger_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Contact point 1
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e * h + cp1_delta_tang_pos_comps[i]
    rhs = cp1_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * cp1_lambda_n[i] - np.sum(cp1_lambda_f_comps[i])
    rhs = gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = cp1_phi[i]
    rhs = cp1_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Contact point 2
    # Add sliding/sticking complimentarity constraints
    lhs = gamma[i] * e * h + cp2_delta_tang_pos_comps[i]
    rhs = cp2_lambda_f_comps[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    lhs = mu * cp2_lambda_n[i] - np.sum(cp2_lambda_f_comps[i])
    rhs = gamma[i]
    sliding_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add contact/non-contact complimentarity constraints
    lhs = cp2_phi[i]
    rhs = cp2_lambda_n[i]
    contact_comp_constraints.append(add_complimentarity_constraint(prog, lhs, rhs))

    # Add force balance constraint
    # x direction
    prog.AddLinearConstraint(
        cp1_lambda_f[i] + cp2_lambda_f[i] + finger_lambda_n[i] + f_grav_B[i][0] == 0
    )
    # y direction
    prog.AddLinearConstraint(cp1_lambda_n[i] + cp2_lambda_n[i] + f_grav_B[i][1] == 0)

    # Torque balance
    sum_torques = (
        cross_2d(p_BF[i], f_F_B[i])
        + cross_2d(p_cp1_B, f_cp1_B[i])
        + cross_2d(p_cp2_B, f_cp2_B[i])
    )
    prog.AddQuadraticConstraint(sum_torques, 0, 0)

    # SO(2)
    prog.AddQuadraticConstraint(cos_th[i] ** 2 + sin_th[i] ** 2, 1, 1)

    prog.AddLinearConstraint(cos_th[i] <= 1)
    prog.AddLinearConstraint(cos_th[i] >= -1)
    prog.AddLinearConstraint(sin_th[i] <= 1)
    prog.AddLinearConstraint(sin_th[i] >= -1)

    # Backward euler: x_next = x_curr + h * f(x_curr, u_curr)
    if i == 0:
        prog.AddLinearConstraint(eq(p_WB[i], p_WB_0 + h * delta_p_WB[i]))
        prog.AddLinearConstraint(eq(p_BF[i], p_BF_0 + h * delta_p_BF[i]))
        prog.AddLinearConstraint(cos_th[i] == cos_th_0 + h * delta_cos_th[i])
        prog.AddLinearConstraint(sin_th[i] == sin_th_0 + h * delta_sin_th[i])
    else:
        prog.AddLinearConstraint(eq(p_WB[i], p_WB[i - 1] + h * delta_p_WB[i]))
        prog.AddLinearConstraint(eq(p_BF[i], p_BF[i - 1] + h * delta_p_BF[i]))
        prog.AddLinearConstraint(cos_th[i] == cos_th[i - 1] + h * delta_cos_th[i])
        prog.AddLinearConstraint(sin_th[i] == sin_th[i - 1] + h * delta_sin_th[i])


# Final conditions
th_F = 0
end_x_pos = 0.5
prog.AddLinearConstraint(eq(p_BF[N - 1], p_BF_0))
prog.AddLinearConstraint(p_WB[N - 1][0] == end_x_pos)

prog.AddLinearConstraint(cos_th[N - 1] == np.cos(th_F))
prog.AddLinearConstraint(sin_th[N - 1] == np.sin(th_F))

prog.AddQuadraticCost(delta_cos_th.T @ delta_cos_th)
prog.AddQuadraticCost(delta_sin_th.T @ delta_sin_th)
prog.AddQuadraticCost(np.trace(delta_p_WB.T @ delta_p_WB))  # type: ignore
prog.AddQuadraticCost(np.trace(delta_p_BF.T @ delta_p_BF))  # type: ignore

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
TOL = 1e-6
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
    p_WB_sol = result.GetSolution(p_WB)
    p_BF_sol = result.GetSolution(p_BF)
    p_cp1_W_sol = evaluate_np_expressions_array(p_cp1_W, result)
    p_cp2_W_sol = evaluate_np_expressions_array(p_cp2_W, result)
    f_F_W_sol = evaluate_np_expressions_array(f_F_W, result)
    f_cp1_W_sol = evaluate_np_expressions_array(f_cp1_W, result)
    f_cp2_W_sol = evaluate_np_expressions_array(f_cp2_W, result)

    p_WB_sol = np.vstack([p_WB_0, p_WB_sol])
    p_BF_sol = np.vstack([p_BF_0, p_BF_sol])
    p_cp1_W_sol = np.vstack([p_cp1_W_0, p_cp1_W_sol])
    p_cp2_W_sol = np.vstack([p_cp2_W_0, p_cp2_W_sol])
    f_F_W_sol = np.vstack([f_F_W_0, f_F_W_sol])
    f_cp1_W_sol = np.vstack([f_cp1_W_0, f_cp1_W_sol])
    f_cp2_W_sol = np.vstack([f_cp2_W_0, f_cp2_W_sol])
    f_grav_W_sol = np.vstack([f_grav_W[0, :], f_grav_W])  # pad this for plotting

    R_WB_sol = [R_WB_0] + [evaluate_np_expressions_array(R, result) for R in R_WB]
    p_WF_sol = np.vstack([p + R @ f for p, R, f in zip(p_WB_sol, R_WB_sol, p_BF_sol)])

    rotation_traj = np.vstack([R.flatten() for R in R_WB_sol])

    CONTACT_COLOR = COLORS["dodgerblue4"]
    GRAVITY_COLOR = COLORS["blueviolet"]
    BOX_COLOR = COLORS["aquamarine4"]
    TABLE_COLOR = COLORS["bisque3"]
    FINGER_COLOR = COLORS["firebrick3"]

    viz_com_points = [
        VisualizationPoint2d(p_WB_sol, GRAVITY_COLOR),
        VisualizationPoint2d(p_WF_sol, FINGER_COLOR),
        VisualizationPoint2d(p_cp1_W_sol, CONTACT_COLOR),
        VisualizationPoint2d(p_cp2_W_sol, CONTACT_COLOR),
    ]  # type: ignore

    table = Box2d(width=2.0, height=0.2)
    table_com = np.repeat(
        np.array([0, -table.height / 2]).reshape((1, -1)), N + 1, axis=0
    )
    table_rot = np.repeat(np.eye(2).flatten().reshape((1, -1)), N + 1, axis=0)

    viz_contact_forces = [
        VisualizationForce2d(p_WF_sol, CONTACT_COLOR, f_F_W_sol),
        VisualizationForce2d(p_cp1_W_sol, CONTACT_COLOR, f_cp1_W_sol),
        VisualizationForce2d(p_cp2_W_sol, CONTACT_COLOR, f_cp2_W_sol),
        VisualizationForce2d(p_WB_sol, GRAVITY_COLOR, f_grav_W_sol),
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


def plot_vals(result: MathematicalProgramResult, title: str):
    # First, retrieve the solutions for all your decision variables
    p_WB_sol = result.GetSolution(p_WB)
    p_BF_sol = result.GetSolution(p_BF)
    cos_th_sol = result.GetSolution(cos_th)
    sin_th_sol = result.GetSolution(sin_th)
    finger_lambda_n_sol = result.GetSolution(finger_lambda_n)
    finger_lambda_f_sol = evaluate_np_expressions_array(finger_lambda_f, result)
    finger_lambda_f_comps_sol = result.GetSolution(finger_lambda_f_comps)
    finger_phi_sol = evaluate_np_expressions_array(finger_phi, result)  #  type: ignore
    gamma_sol = result.GetSolution(gamma)
    cp1_lambda_n_sol = result.GetSolution(cp1_lambda_n)
    cp1_lambda_f_sol = evaluate_np_expressions_array(cp1_lambda_f, result)
    cp1_lambda_f_comps_sol = result.GetSolution(cp1_lambda_f_comps)
    cp1_v_rel_sol = evaluate_np_expressions_array(cp1_delta_tang_pos, result)
    gamma_sol = result.GetSolution(gamma)
    cp1_phi_sol = evaluate_np_expressions_array(cp1_phi, result)
    cp2_lambda_n_sol = result.GetSolution(cp2_lambda_n)
    cp2_lambda_f_sol = evaluate_np_expressions_array(cp2_lambda_f, result)
    cp2_v_rel_sol = evaluate_np_expressions_array(cp2_delta_tang_pos, result)
    cp2_lambda_f_comps_sol = result.GetSolution(cp2_lambda_f_comps)
    cp2_phi_sol = evaluate_np_expressions_array(cp2_phi, result)

    # Now, plot them
    fig, axs = plt.subplots(
        7, 4, figsize=(15, 10)
    )  # Adjust the number of subplots based on the number of variables

    col = 0
    # Plot p_WB
    axs[0, col].plot(p_WB_sol[:, 0])
    axs[0, col].set_title("p_WB_x")
    axs[0, col].set_xlim(0, N)

    axs[1, col].plot(p_WB_sol[:, 1])
    axs[1, col].set_title("p_WB_y")
    axs[1, col].set_xlim(0, N)

    # Plot p_finger_x
    axs[2, col].plot(p_BF_sol[:, 0])
    axs[2, col].set_title("p_BF_x")
    axs[2, col].set_xlim(0, N)

    axs[3, col].plot(p_BF_sol[:, 1])
    axs[3, col].set_title("p_BF_y")
    axs[3, col].set_xlim(0, N)

    # Plot cos_th
    axs[4, col].plot(cos_th_sol)
    axs[4, col].set_title("cos_th")
    axs[4, col].set_xlim(0, N)
    axs[4, col].set_ylim(-1.2, 1.2)

    # Plot sin_th
    axs[5, col].plot(sin_th_sol)
    axs[5, col].set_title("sin_th")
    axs[5, col].set_xlim(0, N)
    axs[5, col].set_ylim(-1.2, 1.2)

    axs[6, col].axis("off")

    col = 1
    # Plot finger_lambda_n
    axs[0, col].plot(finger_lambda_n_sol)
    axs[0, col].set_title("finger_lambda_n")
    axs[0, col].set_xlim(0, N)

    # Plot finger_lambda_f
    axs[1, col].plot(finger_lambda_f_sol)
    axs[1, col].set_title("finger_lambda_f")
    axs[1, col].set_xlim(0, N)

    # Plot finger_phi
    axs[2, col].plot(finger_phi_sol)
    axs[2, col].set_title("finger_phi")
    axs[2, col].set_xlim(0, N)

    for i in [3, 4, 5, 6]:
        axs[i, col].axis("off")

    col = 2

    # Plot cp2_cp1_gamma
    axs[0, col].plot(gamma_sol)
    axs[0, col].set_title("cp2_cp1_gamma")
    axs[0, col].set_xlim(0, N)
    axs[0, col].set_ylim(0, max(max(gamma_sol) * 1.3, 0.1))

    # Plot cp1_lambda_n
    axs[1, col].plot(cp1_lambda_n_sol)
    axs[1, col].set_title("cp1_lambda_n")
    axs[1, col].set_xlim(0, N)
    axs[1, col].set_ylim(0, max(max(cp1_lambda_n_sol) * 1.3, 0.1))

    # Plot cp1_lambda_f_1
    axs[2, col].plot(cp1_lambda_f_comps_sol[:, 0])
    axs[2, col].set_title("cp1_lambda_f_1")
    axs[2, col].set_xlim(0, N)
    axs[2, col].set_ylim(0, max(max(cp1_lambda_f_comps_sol[:, 0]) * 1.3, 0.1))

    # Plot cp1_lambda_f_2
    axs[3, col].plot(cp1_lambda_f_comps_sol[:, 1])
    axs[3, col].set_title("cp1_lambda_f_2")
    axs[3, col].set_xlim(0, N)
    axs[3, col].set_ylim(0, max(max(cp1_lambda_f_comps_sol[:, 1]) * 1.3, 0.1))

    # Plot cp1_lambda_f
    axs[4, col].plot(cp1_lambda_f_sol)
    axs[4, col].set_title("cp1_lambda_f")
    axs[4, col].set_xlim(0, N)
    c = max(max(np.abs(cp1_lambda_f_sol)) * 1.3, 0.1)
    axs[4, col].set_ylim(-c, c)

    # Plot cp1_v_rel
    axs[5, col].plot(cp1_v_rel_sol)
    axs[5, col].set_title("cp1_v_rel")
    axs[5, col].set_xlim(0, N)
    c = max(max(cp1_v_rel_sol) * 1.3, 0.1)
    axs[5, col].set_ylim(-c, c)

    # Plot phi
    axs[6, col].plot(cp1_phi_sol)
    axs[6, col].set_title("cp1_phi")
    axs[6, col].set_xlim(0, N)
    c = max(max(cp1_phi_sol) * 1.3, 0.1)
    axs[6, col].set_ylim(-0.1, c)

    col = 3

    # Plot cp2_cp1_gamma
    axs[0, col].plot(gamma_sol)
    axs[0, col].set_title("cp2_cp1_gamma")
    axs[0, col].set_xlim(0, N)
    axs[0, col].set_ylim(0, max(max(gamma_sol) * 1.3, 0.1))

    # Plot cp2_lambda_n
    axs[1, col].plot(cp2_lambda_n_sol)
    axs[1, col].set_title("cp2_lambda_n")
    axs[1, col].set_xlim(0, N)
    axs[1, col].set_ylim(0, max(max(cp2_lambda_n_sol) * 1.3, 0.1))

    # Plot cp2_lambda_f_1
    axs[2, col].plot(cp2_lambda_f_comps_sol[:, 0])
    axs[2, col].set_title("cp2_lambda_f_1")
    axs[2, col].set_xlim(0, N)
    axs[2, col].set_ylim(0, max(max(cp2_lambda_f_comps_sol[:, 0]) * 1.3, 0.1))

    # Plot cp2_lambda_f_2
    axs[3, col].plot(cp2_lambda_f_comps_sol[:, 1])
    axs[3, col].set_title("cp2_lambda_f_2")
    axs[3, col].set_xlim(0, N)
    axs[3, col].set_ylim(0, max(max(cp2_lambda_f_comps_sol[:, 1]) * 1.3, 0.1))

    # Plot cp2_lambda_f
    axs[4, col].plot(cp2_lambda_f_sol)
    axs[4, col].set_title("cp2_lambda_f")
    axs[4, col].set_xlim(0, N)
    c = max(max(np.abs(cp2_lambda_f_sol)) * 1.3, 0.1)
    axs[4, col].set_ylim(-c, c)

    # Plot cp2_v_rel
    axs[5, col].plot(cp2_v_rel_sol)
    axs[5, col].set_title("cp2_v_rel")
    axs[5, col].set_xlim(0, N)
    c = max(max(cp2_v_rel_sol) * 1.3, 0.1)
    axs[5, col].set_ylim(-c, c)

    # Plot phi
    axs[6, col].plot(cp2_phi_sol)
    axs[6, col].set_title("cp2_phi")
    axs[6, col].set_xlim(0, N)
    c = max(max(cp2_phi_sol) * 1.3, 0.1)
    axs[6, col].set_ylim(-0.1, c)

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


show_plot = False
if show_plot:
    plot_eigvals(X_sol)

    plot_vals(result, "Relaxed")

    if do_rounding:
        plot_vals(result, "Rounded")


else:  # animate
    animate_vals(result)
