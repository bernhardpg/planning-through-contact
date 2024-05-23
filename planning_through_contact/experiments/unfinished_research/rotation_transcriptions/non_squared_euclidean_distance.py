import numpy as np
from pydrake.math import eq
from pydrake.solvers import (
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    Solve,
    SolverOptions,
)

from planning_through_contact.visualize.analysis import plot_cos_sine_trajs

NUM_CTRL_POINTS = 8
NUM_DIMS = 2

prog = MathematicalProgram()

r = prog.NewContinuousVariables(NUM_DIMS, NUM_CTRL_POINTS, "r")

# Constrain the points to lie on the unit circle
for i in range(NUM_CTRL_POINTS):
    r_i = r[:, i]
    so_2_constraint = r_i.T.dot(r_i) == 1
    prog.AddConstraint(so_2_constraint)

USE_SQ_EUCL_DISTANCE = False
# Minimize squared euclidean distances in rotation parameters
r_displacements = []
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_disp_i = r_next - r_i
    r_displacements.append(r_disp_i)

    rot_cost_i = r_disp_i.T.dot(r_disp_i)

    if USE_SQ_EUCL_DISTANCE:
        prog.AddCost(rot_cost_i)

# Initial conditions
th_initial = 0
th_final = np.pi / 2

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.AddConstraint(c)

for c in final_cond:
    prog.AddConstraint(c)

# Absolute value cost
# s = prog.NewContinuousVariables(NUM_CTRL_POINTS - 1, "s")
# prog.AddCost(10 * np.sum(s))
#
# for k in range(NUM_CTRL_POINTS - 1):
#     prog.AddLinearConstraint(s[k] >= th_dots[k])
#     prog.AddLinearConstraint(s[k] >= -th_dots[k])

for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_disp_i = r_next - r_i
    prog.AddQuadraticConstraint(r_disp_i.T @ r_disp_i, 0, 0.055)


# Solve SDP relaxation
relaxed_prog = MakeSemidefiniteRelaxation(prog)

# Is not tight
USE_EUCLIDEAN_DISTANCE = True
if USE_EUCLIDEAN_DISTANCE:
    s = relaxed_prog.NewContinuousVariables(NUM_CTRL_POINTS - 1, "s")
    relaxed_prog.AddCost(np.sum(s))

    for k in range(NUM_CTRL_POINTS - 1):
        mat = np.zeros((3, 3), dtype=object)
        mat[:2, :2] = np.eye(2) * s[k]
        mat[2, :2] = r_displacements[k].T
        mat[:2, 2] = r_displacements[k]
        mat[2, 2] = s[k]

        relaxed_prog.AddPositiveSemidefiniteConstraint(mat)


minimize_trace = True
if minimize_trace:
    EPS = 0.1

    # the first constraint is the PSD constraint we want
    X = relaxed_prog.positive_semidefinite_constraints()[0].variables()
    N = np.sqrt(len(X))
    assert int(N) == N
    X = X.reshape((int(N), int(N)))
    relaxed_prog.AddLinearCost(EPS * np.trace(X))


print("Finished formulating SDP relaxation")

solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

result = Solve(relaxed_prog, solver_options=solver_options)
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")

r_val = result.GetSolution(r)
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

plot_cos_sine_trajs(r_val.T)
