from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
import pydrake.symbolic as sym
from pydrake.math import eq
from pydrake.solvers import MathematicalProgram

from convex_relaxation.sdp import create_sdp_relaxation
from geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from geometry.rigid_body import RigidBody
from geometry.utilities import cross_2d
from tools.types import NpExpressionArray, NpVariableArray
from tools.utils import forward_differences


@dataclass
class PlanarPlanSpecs:
    num_knot_points_contact: int = 4
    num_knot_points_repositioning: int = 2
    time_in_contact: float = 2
    time_repositioning: float = 0.5


class AbstractContactMode(ABC):
    @abstractmethod
    def get_convex_set(self) -> opt.ConvexSet:
        pass

    @abstractmethod
    def get_boundary_variables(self) -> Tuple[NpExpressionArray]:
        pass


@dataclass
class FaceContactVariables:
    lams: NpVariableArray  # (num_knot_points, )
    normal_forces: NpVariableArray  # (num_knot_points, )
    friction_forces: NpVariableArray  # (num_knot_points, )
    cos_ths: NpVariableArray  # (num_knot_points, )
    sin_ths: NpVariableArray  # (num_knot_points, )
    p_WB_xs: NpVariableArray  # (num_knot_points, )
    p_WB_ys: NpVariableArray  # (num_knot_points, )

    time_in_mode: float
    dt: float

    pv1: npt.NDArray[np.float64]
    pv2: npt.NDArray[np.float64]
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]

    @property
    def R_WBs(self):
        Rs = [
            np.array([[cos, -sin], [sin, cos]])
            for cos, sin in zip(self.cos_ths, self.sin_ths)
        ]
        return Rs

    @property
    def p_WBs(self):
        return [
            np.array([x, y]).reshape((2, 1)) for x, y in zip(self.p_WB_xs, self.p_WB_ys)
        ]

    @property
    def f_c_Bs(self):
        return [
            c_n * self.normal_vec + c_f * self.tangent_vec
            for c_n, c_f in zip(self.normal_forces, self.friction_forces)
        ]

    @property
    def p_c_Bs(self):
        return [lam * self.pv1 + (1 - lam) * self.pv2 for lam in self.lams]

    @property
    def v_WBs(self):
        return forward_differences(self.p_WBs, self.dt)

    @property
    def cos_th_dots(self):
        return forward_differences(self.cos_ths, self.dt)

    @property
    def sin_th_dots(self):
        return forward_differences(self.sin_ths, self.dt)

    @property
    def v_c_Bs(self):
        return forward_differences(
            self.p_c_Bs, self.dt
        )  # NOTE: Not real velocity, only time differentiation of coordinates (not equal as B is not an inertial frame)!

    @property
    def omega_WBs(self):
        R_WB_dots = [
            np.array([[cos_dot, -sin_dot], [sin_dot, cos_dot]])
            for cos_dot, sin_dot in zip(self.cos_th_dots, self.sin_th_dots)
        ]
        # In 2D, omega_z = theta_dot will be at position (1,0) in R_dot * R'
        oms = [R_dot.dot(R.T)[1, 0] for R, R_dot in zip(self.R_WBs, R_WB_dots)]
        return oms

    @property
    def p_c_Ws(self) -> List[npt.NDArray[np.float64]]:
        return [
            p_WB + R_WB.dot(p_c_B)
            for p_WB, R_WB, p_c_B in zip(self.p_WBs, self.R_WBs, self.p_c_Bs)
        ]

    @property
    def f_c_Ws(self) -> List[npt.NDArray[np.float64]]:
        return [R_WB.dot(f_c_B) for f_c_B, R_WB in zip(self.f_c_Bs, self.R_WBs)]


@dataclass
class FaceContactMode(AbstractContactMode):
    num_knot_points: int
    time_in_mode: float
    contact_location: PolytopeContactLocation
    object: RigidBody

    @classmethod
    def create_from_spec(
        cls,
        contact_location: PolytopeContactLocation,
        specs: PlanarPlanSpecs,
        object: RigidBody,
    ) -> "FaceContactMode":
        return cls(
            specs.num_knot_points_contact,
            specs.time_in_contact,
            contact_location,
            object,
        )

    def __post_init__(self) -> None:
        self.prog = MathematicalProgram()
        self.variables = self._define_variables()
        self._define_constraints()
        self._define_costs()

    def _define_variables(
        self,
    ) -> FaceContactVariables:
        # Contact positions
        lams = self.prog.NewContinuousVariables(self.num_knot_points, "lam")
        pv1, pv2 = self.object.geometry.get_proximate_vertices_from_location(
            self.contact_location
        )

        # Contact forces
        normal_forces = self.prog.NewContinuousVariables(self.num_knot_points, "c_n")
        friction_forces = self.prog.NewContinuousVariables(self.num_knot_points, "c_f")
        (
            normal_vec,
            tangent_vec,
        ) = self.object.geometry.get_norm_and_tang_vecs_from_location(
            self.contact_location
        )

        # Rotations
        cos_ths = self.prog.NewContinuousVariables(self.num_knot_points, "cos_th")
        sin_ths = self.prog.NewContinuousVariables(self.num_knot_points, "sin_th")

        # Box position relative to world frame
        p_WB_xs = self.prog.NewContinuousVariables(self.num_knot_points, "p_WB_x")
        p_WB_ys = self.prog.NewContinuousVariables(self.num_knot_points, "p_WB_y")

        dt = self.time_in_mode / self.num_knot_points

        return FaceContactVariables(
            lams,
            normal_forces,
            friction_forces,
            cos_ths,
            sin_ths,
            p_WB_xs,
            p_WB_ys,
            self.time_in_mode,
            dt,
            pv1,
            pv2,
            normal_vec,
            tangent_vec,
        )

    def _define_constraints(self) -> None:
        # TODO: take this from drake simulation
        FRICTION_COEFF = 0.5

        for lam in self.variables.lams:
            self.prog.AddLinearConstraint(lam >= 0)
            self.prog.AddLinearConstraint(lam <= 1)

        # SO(2) constraints
        for c, s in zip(self.variables.cos_ths, self.variables.sin_ths):
            self.prog.AddConstraint(c**2 + s**2 == 1)

        # Friction cone constraints
        for c_n in self.variables.normal_forces:
            self.prog.AddLinearConstraint(c_n >= 0)
        for c_n, c_f in zip(
            self.variables.normal_forces, self.variables.friction_forces
        ):
            self.prog.AddLinearConstraint(c_f <= FRICTION_COEFF * c_n)
            self.prog.AddLinearConstraint(c_f >= -FRICTION_COEFF * c_n)

        # Quasi-static dynamics
        use_midpoint = True
        for k in range(self.num_knot_points - 1):
            v_WB = self.variables.v_WBs[k]
            omega_WB = self.variables.omega_WBs[k]

            # NOTE: We enforce dynamics at midway points as this is where the velocity is 'valid'
            if use_midpoint:
                f_c_B = self.get_midpoint(self.variables.f_c_Bs, k)
                p_c_B = self.get_midpoint(self.variables.p_c_Bs, k)
                R_WB = self.get_midpoint(self.variables.R_WBs, k)
            else:
                f_c_B = self.variables.f_c_Bs[k]
                p_c_B = self.variables.p_c_Bs[k]
                R_WB = self.variables.R_WBs[k]

            x_dot, dyn = self.quasi_static_dynamics(
                v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, self.object.mass
            )
            quasi_static_dynamic_constraint = eq(x_dot - dyn, 0)
            for row in quasi_static_dynamic_constraint:
                self.prog.AddConstraint(row)

        # Ensure sticking on the contact point
        for v_c_B in self.variables.v_c_Bs:
            # NOTE: This is not constraining the real velocity, but it does ensure sticking
            self.prog.AddLinearConstraint(eq(v_c_B, 0))

    def _define_costs(self) -> None:
        # Minimize kinetic energy through squared velocities
        sq_linear_vels = sum([v_WB.T.dot(v_WB) for v_WB in self.variables.v_WBs]).item()  # type: ignore
        self.prog.AddQuadraticCost(sq_linear_vels)

        sq_angular_vels = np.sum(
            [
                cos_dot**2 + sin_dot**2
                for cos_dot, sin_dot in zip(
                    self.variables.cos_th_dots, self.variables.sin_th_dots
                )
            ]
        )
        self.prog.AddQuadraticCost(sq_angular_vels)

    def get_convex_set(self) -> opt.Spectrahedron:
        import time

        start = time.time()
        print("Starting to create SDP relaxation...")
        self.relaxed_prog, self.X, _ = create_sdp_relaxation(self.prog)
        self.x = self.X[1:, 0]
        end = time.time()
        print(
            f"Finished formulating relaxed problem. Elapsed time: {end - start} seconds"
        )
        return opt.Spectrahedron(self.relaxed_prog)

    def get_boundary_variables(self) -> Tuple[NpExpressionArray]:
        breakpoint()

    @staticmethod
    def get_midpoint(vals, k: int):
        return vals[k] + vals[k + 1] / 2

    @staticmethod
    def quasi_static_dynamics(
        v_WB, omega_WB, f_c_B, p_c_B, R_WB, FRICTION_COEFF, OBJECT_MASS
    ):
        G = 9.81
        f_max = FRICTION_COEFF * G * OBJECT_MASS
        tau_max = f_max * 0.2  # TODO: change this!

        A = np.diag(
            [1 / f_max**2, 1 / f_max**2, 1 / tau_max**2]
        )  # Ellipsoidal Limit surface approximation

        # We need to add an entry for multiplication with the wrench, see paper "Reactive Planar Manipulation with Convex Hybrid MPC"
        R = np.zeros((3, 3), dtype="O")
        R[2, 2] = 1
        R[0:2, 0:2] = R_WB

        # Contact torques
        tau_c_B = cross_2d(p_c_B, f_c_B)

        x_dot = np.concatenate((v_WB, [[omega_WB]]))
        wrench_B = np.concatenate((f_c_B, [[tau_c_B]]))
        wrench_W = R.dot(wrench_B)
        dynamics = A.dot(
            wrench_W
        )  # Note: A and R are switched here compared to original paper, but A is diagonal so it makes no difference

        return x_dot, dynamics  # x_dot, f(x,u)


class NonCollisionMode:
    @classmethod
    def create_from_spec(
        cls, contact_location: PolytopeContactLocation, specs: PlanarPlanSpecs
    ) -> "NonCollisionMode":
        return cls()  # TODO
