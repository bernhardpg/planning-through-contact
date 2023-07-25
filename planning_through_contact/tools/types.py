import numpy.typing as npt
import pydrake.symbolic as sym

# Define these types here to avoid typing error for every line.
# See the following question on StackOverflow:
# https://stackoverflow.com/questions/75068535/correct-typing-for-numpy-array-with-drake-expressions?noredirect=1#comment132578505_75068535

NpFormulaArray = npt.NDArray[sym.Formula]  # type: ignore
NpExpressionArray = npt.NDArray[sym.Expression]  # type: ignore
NpVariableArray = npt.NDArray[sym.Expression]  # type: ignore
NpMonomialArray = npt.NDArray[sym.Monomial]  # type: ignore
NpPolynomialArray = npt.NDArray[sym.Polynomial]  # type: ignore
