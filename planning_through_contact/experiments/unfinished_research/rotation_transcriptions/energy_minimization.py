import numpy as np
from pydrake.math import eq, le
from pydrake.solvers import MathematicalProgram, Solve

from planning_through_contact.convex_relaxation.sdp import create_sdp_relaxation
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

# Minimize squared euclidean distances in rotaion parameters
for i in range(NUM_CTRL_POINTS - 1):
    r_i = r[:, i]
    r_next = r[:, i + 1]
    r_dot_i = r_next - r_i

    rot_cost_i = r_dot_i.T.dot(r_dot_i)
    prog.AddCost(rot_cost_i)

minimize_squares = False
if minimize_squares:
    for i in range(NUM_CTRL_POINTS):
        r_i = r[:, i]
        prog.AddCost(r_i.T.dot(r_i))

# Initial conditions
th_initial = np.pi
th_final = 0 + 0.1

create_r_vec_from_angle = lambda th: np.array([np.cos(th), np.sin(th)])

initial_cond = eq(r[:, 0], create_r_vec_from_angle(th_initial))
final_cond = eq(r[:, -1], create_r_vec_from_angle(th_final))

for c in initial_cond:
    prog.AddConstraint(c)

for c in final_cond:
    prog.AddConstraint(c)

A = np.array([[1, -3], [-2, -6]])
b = np.array([2, 3])

for var in r.T:
    consts = le(A.dot(var), b)
    prog.AddConstraint(consts)

# Solve SDP relaxation
relaxed_prog, X, mon_basis = create_sdp_relaxation(prog)
# relaxed_prog = MakeSemidefiniteRelaxation(prog)
print("Finished formulating SDP relaxation")

# solver_options = SolverOptions()
# solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

result = Solve(relaxed_prog)
assert result.is_success()
print(f"Cost: {result.get_optimal_cost()}")

X_val = result.GetSolution(X)

tol = 1e-4
num_nonzero_eigvals = len(
    [val for val in np.linalg.eigvals(X_val) if np.abs(val) >= tol]
)
print(f"Rank of X: {num_nonzero_eigvals}")
print(f"cost: {result.get_optimal_cost()}")

x_val = X_val[:, 0]

r_val = x_val[1 : NUM_CTRL_POINTS * NUM_DIMS + 1]
r_val = r_val.reshape((NUM_DIMS, NUM_CTRL_POINTS), order="F")

# TODO continue here

plot_cos_sine_trajs(r_val.T, A, b)
