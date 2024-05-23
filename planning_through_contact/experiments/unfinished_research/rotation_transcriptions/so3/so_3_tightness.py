from itertools import combinations

import numpy as np
from pydrake.math import eq
from pydrake.solvers import MakeSemidefiniteRelaxation, MathematicalProgram, Solve

prog = MathematicalProgram()

######
# SO(2)
######

r = prog.NewContinuousVariables(2, 1, "r")

prog.AddConstraint(eq(r.T.dot(r), 1))

c = np.array([-1, -1]).reshape((2, 1))
prog.AddCost(c.T.dot(r).item())

relaxed_prog = MakeSemidefiniteRelaxation(prog)
result = Solve(relaxed_prog)

r_sol = result.GetSolution(r)
# r_sol = array([0.70710678, 0.70710678])

# check solution is tight
assert np.isclose(r_sol.T.dot(r_sol).item(), 1)

######
# SO(3)
######

R = prog.NewContinuousVariables(3, 3, "R")
r_1 = R[:, 0:1]
r_2 = R[:, 1:2]
r_3 = R[:, 2:3]

c = np.array([-1, -1, -1]).reshape((3, 1))

# unit length
for r_i in (r_1, r_2, r_3):
    prog.AddConstraint(eq(r_i.T.dot(r_i), 1))

cross = lambda a, b: np.cross(a, b, axisa=0, axisb=0).T

ADD_ALL_SO3_CONSTRAINTS = False
if ADD_ALL_SO3_CONSTRAINTS:
    # orthogonality
    for r_i, r_j in combinations((r_1, r_2, r_3), 2):
        prog.AddConstraint(eq(r_i.T.dot(r_j), 0))

    # cross-product constraint
    exprs = cross(r_1, r_2) - r_3
    for e in exprs.squeeze():
        prog.AddQuadraticConstraint(
            e, 0, 0
        )  # explicitly declare constraint as quadratic

for r_i in (r_1, r_2, r_3):
    prog.AddCost(c.T.dot(r_i).item())


relaxed_prog = MakeSemidefiniteRelaxation(prog)

result = Solve(relaxed_prog)

# check solution is tight
R_sol = result.GetSolution(R)
r_sol_1 = R_sol[:, 0:1]
r_sol_2 = R_sol[:, 1:2]
r_sol_3 = R_sol[:, 2:3]

# Unit vector constraints: these are tight when we don't add in the other constraints
for r_i in (r_sol_1, r_sol_2, r_sol_3):
    print(r_i.T.dot(r_i).item() - 1)
    # assert np.isclose(r_i.T.dot(r_i).item(), 1)

# These are unfortunately not tight
for r_i, r_j in combinations((r_sol_1, r_sol_2, r_sol_3), 2):
    print(r_i.T.dot(r_j).item())
    # assert np.isclose(r_i.T.dot(r_j).item(), 0)

# These are unfortunately not tight
print(cross(r_sol_1, r_sol_2) - r_sol_3)
# assert np.isclose(cross(r_sol_1, r_sol_2), r_sol_3)
