from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pydrake.symbolic as sym  # type: ignore
from pydrake.math import eq

from geometry.orientation.contact_point_2d import ContactPoint2d
from geometry.box import Box2d

@dataclass
class ContactPair2d:
    def __init__(
        self,
        pair_name: str,
        body_A: Box2d,
        contact_location_A: str,
        name_A: str,
        body_B: Box2d,
        contact_location_B: str,
        name_B: str,
        friction_coeff: float,
    ) -> None:
        self.contact_point_A = ContactPoint2d(
            body_A,
            contact_location_A,
            friction_coeff,
            name=f"{pair_name}_{name_A}",
        )
        self.contact_point_B = ContactPoint2d(
            body_B,
            contact_location_B,
            friction_coeff,
            name=f"{pair_name}_{name_B}",
        )

        cos_th = sym.Variable(f"{pair_name}_cos_th")
        sin_th = sym.Variable(f"{pair_name}_sin_th")
        self.R_AB = np.array([[cos_th, -sin_th], [sin_th, cos_th]])

        p_AB_A_x = sym.Variable(f"{pair_name}_p_AB_A_x")
        p_AB_A_y = sym.Variable(f"{pair_name}_p_AB_A_y")
        self.p_AB_A = np.array([p_AB_A_x, p_AB_A_y]).reshape((-1, 1))

        p_BA_B_x = sym.Variable(f"{pair_name}_p_BA_B_x")
        p_BA_B_y = sym.Variable(f"{pair_name}_p_BA_B_y")
        self.p_BA_B = np.array([p_BA_B_x, p_BA_B_y]).reshape((-1, 1))

    @property
    def variables(self) -> npt.NDArray[sym.Variable]:  # type: ignore
        return np.concatenate(
            [
                self.contact_point_A.variables,
                self.contact_point_B.variables,
                self.p_AB_A.flatten(),
                self.p_BA_B.flatten(),
            ]
        )

    def create_equal_contact_point_constraints(self) -> npt.NDArray[sym.Formula]:  # type: ignore
        p_Ac_A = self.contact_point_A.contact_position
        p_Bc_B = self.contact_point_B.contact_position

        p_Bc_A = self.R_AB.dot(p_Bc_B)
        eq_contact_point_in_A = eq(p_Ac_A, self.p_AB_A + p_Bc_A)

        p_Ac_B = self.R_AB.T.dot(p_Ac_A)
        eq_contact_point_in_B = eq(p_Bc_B, self.p_BA_B + p_Ac_B)

        return np.vstack(
            (
                eq_contact_point_in_A,
                eq_contact_point_in_B,
            )
        )

    def create_equal_rel_position_constraints(self) -> npt.NDArray[sym.Formula]: # type: ignore
        rel_pos_equal_in_A = eq(self.p_AB_A, -self.R_AB.dot(self.p_BA_B))
        rel_pos_equal_in_B = eq(self.p_BA_B, -self.R_AB.T.dot(self.p_AB_A))

        return np.vstack(
            (
                rel_pos_equal_in_A,
                rel_pos_equal_in_B,
            )
        )

    def create_newtons_third_law_force_constraints(self) -> npt.NDArray[sym.Formula]:  # type: ignore
        f_c_A = self.contact_point_A.contact_force
        f_c_B = self.contact_point_B.contact_force

        equal_and_opposite_in_A = eq(f_c_A, -self.R_AB.dot(f_c_B))
        equal_and_opposite_in_B = eq(f_c_B, -self.R_AB.T.dot(f_c_A))

        return np.vstack(
            (
                equal_and_opposite_in_A,
                equal_and_opposite_in_B,
            )
        )
