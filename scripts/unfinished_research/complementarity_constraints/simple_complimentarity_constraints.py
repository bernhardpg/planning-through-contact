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

prog.AddCost(x**2)
prog.AddQuadraticConstraint(x * y, 0, 0)
prog.AddLinearConstraint(x >= 0)
prog.AddLinearConstraint(y >= 0)
prog.AddLinearConstraint(x + y == 1)

result = Solve(prog)

print(f"is success: {result.is_success()}")
print(f"cost: {result.get_optimal_cost()}")
print(f"x: {result.GetSolution(x)}")
print(f"y: {result.GetSolution(y)}")

relaxed_prog = MakeSemidefiniteRelaxation(prog)

solver = MosekSolver()
# solver = ClarabelSolver()
solver_options = SolverOptions()
# solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)  # type: ignore

result = solver.Solve(relaxed_prog, solver_options=solver_options)
print(f"is success: {result.is_success()}")
print(f"cost: {result.get_optimal_cost()}")
print(f"x: {result.GetSolution(x)}")
print(f"y: {result.GetSolution(y)}")

Y = relaxed_prog.positive_semidefinite_constraints()[0].variables().reshape((3, 3))
print("Y:")
print(result.GetSolution(Y))
