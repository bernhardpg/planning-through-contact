from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore


@dataclass
class ContactPoint:
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]
    friction_coeff: float

    def force_vec_from_symbols(
        self, normal_force: sym.Variable, friction_force: sym.Variable
    ) -> npt.NDArray[sym.Expression]:  # type: ignore
        return normal_force * self.normal_vec + friction_force * self.tangent_vec

    def force_vec_from_values(
        self, normal_force: float, friction_force: float
    ) -> npt.NDArray[np.float64]:
        force_vec = normal_force * self.normal_vec + friction_force * self.tangent_vec
        return force_vec

    def create_friction_cone_constraints(
        self, normal_force: sym.Variable, friction_force: sym.Variable
    ) -> npt.NDArray[sym.Formula]:  # type: ignore
        upper_bound = friction_force <= self.friction_coeff * normal_force
        lower_bound = -self.friction_coeff * normal_force <= friction_force
        normal_force_positive = normal_force >= 0
        return np.vstack([upper_bound, lower_bound, normal_force_positive])
