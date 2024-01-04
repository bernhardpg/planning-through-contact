import numpy as np
from pydrake.solvers import (
    ClarabelSolver,
    CommonSolverOption,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MosekSolver,
    Solve,
    SolverOptions,
)

prog = MathematicalProgram()
x = prog.NewContinuousVariables(1, "x")[0]
y = prog.NewContinuousVariables(1, "y")[0]

prog.AddCost(x)
prog.AddQuadraticConstraint(x * y, 1, 1)
prog.AddLinearConstraint(y <= 1)

result = Solve(prog, initial_guess=np.array([1, 1]))
print(f"is success: {result.is_success()}")
print(f"cost: {result.get_optimal_cost()}")
print(f"x: {result.GetSolution(x)}")
print(f"y: {result.GetSolution(y)}")

relaxed_prog = MakeSemidefiniteRelaxation(prog)

# solver = MosekSolver()
solver = ClarabelSolver()
solver_options = SolverOptions()
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

relaxed_result = solver.Solve(relaxed_prog, solver_options=solver_options)
print(f"is success: {relaxed_result.is_success()}")
print(f"cost: {relaxed_result.get_optimal_cost()}")

prog = MathematicalProgram()
X = prog.NewSymmetricContinuousVariables(3, "X")
prog.AddCost(X[0, 1])
prog.AddConstraint(X[0, 0] == 1)
prog.AddConstraint(X[1, 2] == 1)
prog.AddConstraint(X[0, 2] <= 1)
prog.AddConstraint(1 - 2 * X[0, 2] + X[2, 2] <= 1)
prog.AddPositiveSemidefiniteConstraint(X)

result = solver.Solve(prog, solver_options=solver_options)
print(f"is success: {result.is_success()}")
print(f"cost: {result.get_optimal_cost()}")
