import numpy as np
import numpy.typing as npt
from pydrake.math import cos, sin
from pydrake.systems.framework import Context, ContinuousState, LeafSystem_
from pydrake.systems.scalar_conversion import TemplateSystem

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.utilities import two_d_rotation_matrix_from_angle
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)


@TemplateSystem.define("SliderPusherSystem_")
def SliderPusherSystem_(T):
    class Impl(LeafSystem_[T]):  # type: ignore
        """
        Implements the quasi-dynamic slider-pusher system, as described in
        the paper:

        [1] F. R. Hogan, E. R. Grau, and A. Rodriguez,
        “Reactive Planar Manipulation with Convex Hybrid MPC,”
        in 2018 IEEE International Conference on Robotics and
        Automation (ICRA), May 2018, pp. 247–253.
        doi: 10.1109/ICRA.2018.8461175.

        state: x = [x, y, theta, lam]
        input: u = [c_n, c_f, lam_dot]

        lam is the relative position on the contact face, measured from 0 to 1.
        """

        def _construct(
            self,
            contact_location: PolytopeContactLocation,
            config: SliderPusherSystemConfig,
            converter=None,
        ) -> None:
            super().__init__(converter)

            self.contact_location = contact_location
            self.slider_geometry = config.slider.geometry
            self.config = config
            self.pusher_radius = self.config.pusher_radius

            (
                self.pv1,
                self.pv2,
            ) = self.slider_geometry.get_proximate_vertices_from_location(
                contact_location
            )
            (
                self.normal_vec,
                self.tangent_vec,
            ) = self.slider_geometry.get_norm_and_tang_vecs_from_location(
                contact_location
            )

            NUM_CONTACT_POINTS = 1
            NUM_SLIDER_STATES = 3  # x, y, theta
            state_index = self.DeclareContinuousState(
                NUM_SLIDER_STATES + NUM_CONTACT_POINTS
            )  # x, y, theta, lam
            self.output = self.DeclareStateOutputPort("y", state_index)  # y = x

            NUM_INPUTS = 3  # f_n, f_t, lam_dot
            self.input = self.DeclareVectorInputPort("u", NUM_INPUTS)

            c = self.config.limit_surface_const
            self.D = np.diag([1, 1, c])

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self,
                other.contact_location,
                other.config,
                converter=converter,
            )

        def _get_p_BP(self, lam: float) -> npt.NDArray[np.float64]:
            p_BP = self.slider_geometry.get_p_BP_from_lam(
                lam, self.contact_location, radius=self.pusher_radius
            )
            return p_BP

        def _get_p_Bc(self, lam: float) -> npt.NDArray[np.float64]:
            p_Bc = self.slider_geometry.get_p_Bc_from_lam(lam, self.contact_location)
            return p_Bc

        def _get_contact_jacobian(self, lam: float) -> npt.NDArray[np.float64]:
            p_Bc = self._get_p_Bc(lam).flatten()
            J_c = np.array([[1.0, 0.0, -p_Bc[1]], [0.0, 1.0, p_Bc[0]]])  # type: ignore
            return J_c

        def _get_pusher_jacobian(self, lam: float) -> npt.NDArray[np.float64]:
            p_BP = self._get_p_BP(lam).flatten()
            J_p = np.array([[1.0, 0.0, -p_BP[1]], [0.0, 1.0, p_BP[0]]])
            return J_p

        def _get_contact_force(self, c_n: float, c_f: float) -> npt.NDArray[np.float64]:
            return self.normal_vec * c_n + self.tangent_vec * c_f

        def _get_wrench(
            self, lam: float, c_n: float, c_f: float
        ) -> npt.NDArray[np.float64]:
            f_c_B = self._get_contact_force(c_n, c_f)
            J_c = self._get_contact_jacobian(lam)
            w = J_c.T.dot(f_c_B)
            return w

        def _get_twist(
            self,
            lam: float,
            c_n: float,
            c_f: float,
        ) -> npt.NDArray[np.float64]:
            w = self._get_wrench(lam, c_n, c_f)
            return self.D.dot(w)

        def _get_R(self, theta: float) -> npt.NDArray[np.float64]:
            R = np.array(
                [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
            )
            return R

        def get_state_from_planar_poses_by_projection(
            self,
            slider_pose: PlanarPose,
            pusher_pose: PlanarPose,
        ) -> npt.NDArray[np.float64]:
            """
            Gets the system state [x, y, theta, lambda] from the provided poses.
            NOTE: lambda is found by projecting the pusher pose onto the contact face.

            slider_pose: Slider pose in world frame
            pusher_pose: Pusher pose in world frame
            """
            R_WB = two_d_rotation_matrix_from_angle(slider_pose.theta)
            p_WP = pusher_pose.pos()
            p_WB = slider_pose.pos()
            p_BP = R_WB.T.dot(p_WP - p_WB)
            projected_lam = self.slider_geometry.get_lam_from_p_BP_by_projection(
                p_BP, self.contact_location
            )

            state = np.array(
                [slider_pose.x, slider_pose.y, slider_pose.theta, projected_lam]
            )
            return state

        def get_p_WP_from_state(
            self,
            state: npt.NDArray[np.float64],
            buffer: float = 0.0,
        ) -> npt.NDArray[np.float64]:
            """
            Returns the pusher planar pose in the world frame, i.e. p_WP.

            As the state only describes contact, the p_WP will always be such that the pusher
            is exactly in contact with the slider.
            """
            x, y, theta, lam = state

            p_BP = self.slider_geometry.get_p_BP_from_lam(
                lam, self.contact_location, radius=self.pusher_radius + buffer
            )

            R_WB = two_d_rotation_matrix_from_angle(theta)
            p_WB = np.array([x, y]).reshape((2, 1))

            p_WP = p_WB + R_WB.dot(p_BP)
            return p_WP

        def get_p_Wc_from_state(
            self,
            state: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """
            Returns the contact point the world frame, i.e. p_Wc
            """
            x, y, theta, lam = state

            p_Bc = self.slider_geometry.get_p_Bc_from_lam(lam, self.contact_location)

            R_WB = two_d_rotation_matrix_from_angle(theta)
            p_WB = np.array([x, y]).reshape((2, 1))

            p_Wc = p_WB + R_WB.dot(p_Bc)
            return p_Wc

        def get_control_from_contact_force(
            self, f_c_W: npt.NDArray[np.float64], slider_pose: PlanarPose
        ) -> npt.NDArray[np.float64]:
            lam_dot = 0  # We never plan to move the finger

            R_WB = two_d_rotation_matrix_from_angle(slider_pose.theta)

            f_c_B = R_WB.T.dot(f_c_W)
            c_n, c_f = self.slider_geometry.get_force_comps_from_f_c_B(
                f_c_B, self.contact_location
            )

            control = np.array([c_n, c_f, lam_dot])
            return control

        def calc_dynamics(
            self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            theta = x[2]
            lam = x[3]

            c_n = u[0]
            c_f = u[1]

            lam_dot = u[2]

            R = self._get_R(theta)
            t = self._get_twist(lam, c_n, c_f)

            x_dot = np.vstack((R.dot(t), [lam_dot]))
            return x_dot

        def DoCalcTimeDerivatives(
            self, context: Context, derivatives: ContinuousState
        ) -> None:
            x = context.get_continuous_state_vector()
            u = self.input.Eval(context)
            x_dot = self.calc_dynamics(x, u)  # type: ignore
            derivatives.get_mutable_vector().set_value(x_dot)  # type: ignore

        def get_slider_velocity(
            self, state: npt.NDArray[np.float64], control: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            x, y, theta, lam = state
            c_n, c_f, lam_dot = control

            # twist = v_WB_B, w_WB_B
            t = self._get_twist(lam, c_n, c_f)[:2]

            # Rotate into world frame
            R_WB = self._get_R(theta)
            # Get 2x2 upper left block of R_WB
            R_WB = R_WB[:2, :2]
            v_WB = R_WB @ t
            return v_WB

        def get_pusher_velocity(
            self,
            state: npt.NDArray[np.float64],
            control: npt.NDArray[np.float64],
            v_WB_W: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Equations 12-14 of the 2020 paper but modified to get pusher velocity in the world frame
            instead of contact point velocity in the body frame."""
            x, y, theta, lam = state
            c_n, c_f, lam_dot = control

            J_p = self._get_pusher_jacobian(lam)
            t = self._get_twist(lam, c_n, c_f)
            unnormalized_tangent_vec = self.pv2 - self.pv1
            v_BP_B = J_p.dot(t) + lam_dot * unnormalized_tangent_vec

            # Rotate into world frame
            R_WB = self._get_R(theta)
            # Get 2x2 upper left block of R_WB
            R_WB = R_WB[:2, :2]
            v_BP_W = R_WB.dot(v_BP_B)

            w_WB_W = np.array([0, 0, t[2, 0]])
            p_BP_B = self.slider_geometry.get_p_BP_from_lam(
                lam, self.contact_location, self.pusher_radius
            )
            p_BP_W = R_WB.dot(p_BP_B)
            v_WP_W = (
                v_WB_W + v_BP_W + np.cross(w_WB_W, p_BP_W.flatten())[:2].reshape((2, 1))
            )

            return v_WP_W

        def get_v_Wc(
            self,
            state: npt.NDArray[np.float64],
            control: npt.NDArray[np.float64],
            v_WB_W: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Gets the contact point velocity in the world frame."""
            x, y, theta, lam = state
            c_n, c_f, lam_dot = control

            J_c = self._get_contact_jacobian(lam)
            t = self._get_twist(lam, c_n, c_f)
            unnormalized_tangent_vec = self.pv2 - self.pv1
            v_Bc_B = lam_dot * unnormalized_tangent_vec
            v_Bc_B = J_c.dot(t) + lam_dot * unnormalized_tangent_vec

            # Rotate into world frame
            R_WB = self._get_R(theta)
            # Get 2x2 upper left block of R_WB
            R_WB = R_WB[:2, :2]
            v_Bc_W = R_WB.dot(v_Bc_B)
            w_WB_W = np.array([0, 0, t[2, 0]])
            p_Bc_B = self.slider_geometry.get_p_Bc_from_lam(lam, self.contact_location)
            p_Bc_W = R_WB.dot(p_Bc_B)
            v_Wc = (
                v_WB_W + v_Bc_W + np.cross(w_WB_W, p_Bc_W.flatten())[:2].reshape((2, 1))
            )
            return v_Wc

    return Impl


SliderPusherSystem = SliderPusherSystem_[None]  # type: ignore
