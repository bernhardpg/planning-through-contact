from dataclasses import dataclass

import numpy as np
import pydrake.symbolic as sym

from planning_through_contact.tools.types import NpExpressionArray, NpVariableArray


# TODO: perhaps this class can be unified with the other classes?
@dataclass
class ContinuityVariables:
    """
    A collection of the variables that continuity is enforced over
    """

    p_BF: NpVariableArray | NpExpressionArray
    p_WB: NpVariableArray
    cos_th: sym.Variable
    sin_th: sym.Variable

    @property
    def vector(self) -> NpVariableArray | NpExpressionArray:
        return np.concatenate(
            (self.p_BF.flatten(), self.p_WB.flatten(), (self.cos_th, self.sin_th))  # type: ignore
        )

    def get_pure_variables(self) -> NpVariableArray:
        """
        Function that returns a vector with only the symbolic variables (as opposed to having some be symbolic Expressions)
        """
        if not isinstance(self.p_BF[0, 0], sym.Expression):
            raise RuntimeError(
                "This function should only be called on instances that come from FaceContactMode"
            )

        # NOTE: Very specific way of picking out the variables
        # TODO: clean up this
        lam = list(self.p_BF[0, 0].GetVariables())[0]
        vars = np.concatenate(([lam], self.vector[2:]))
        return vars
