import numpy as np
import numpy.typing as npt
from pydrake.common.value import Value
from pydrake.geometry import (
    Box,
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    MakePhongIllustrationProperties,
    SceneGraph,
    Sphere,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix, cos, sin
from pydrake.systems.framework import (
    Context,
    ContinuousState,
    DiagramBuilder,
    LeafSystem,
    LeafSystem_,
    OutputPort,
)
from pydrake.systems.scalar_conversion import TemplateSystem

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.visualize.colors import COLORS


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
            slider_geometry: CollisionGeometry,
            contact_location: PolytopeContactLocation,
            converter=None,
        ) -> None:
            super().__init__(converter)

            self.contact_location = contact_location
            self.slider_geometry = slider_geometry
            self.pv1, self.pv2 = slider_geometry.get_proximate_vertices_from_location(
                contact_location
            )
            (
                self.normal_vec,
                self.tangent_vec,
            ) = slider_geometry.get_norm_and_tang_vecs_from_location(contact_location)

            NUM_CONTACT_POINTS = 1
            NUM_SLIDER_STATES = 3  # x, y, theta
            state_index = self.DeclareContinuousState(
                NUM_SLIDER_STATES + NUM_CONTACT_POINTS
            )  # x, y, theta, lam
            self.output = self.DeclareStateOutputPort("y", state_index)  # y = x

            NUM_INPUTS = 3  # f_n, f_t, lam_dot
            self.input = self.DeclareVectorInputPort("u", NUM_INPUTS)

            self.A = np.diag([0.1, 0.1, 0.01])  # TODO: change

        def _construct_copy(self, other, converter=None):
            Impl._construct(
                self, other.slider_geometry, other.contact_location, converter=converter
            )

        def _get_p_c_B(self, lam: float) -> npt.NDArray[np.float64]:
            return self.slider_geometry.get_p_c_B_from_lam(lam, self.contact_location)

        def _get_contact_jacobian(self, lam: float) -> npt.NDArray[np.float64]:
            p_c_B = self._get_p_c_B(lam).flatten()
            J_c = np.array([[1.0, 0.0, -p_c_B[1]], [0.0, 1.0, p_c_B[0]]])  # type: ignore
            return J_c

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
            return self.A.dot(w)

        def _get_R(self, theta: float) -> npt.NDArray[np.float64]:
            R = np.array(
                [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
            )
            return R

        def _calc_dynamics(
            self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            theta = x[2]
            # we can not move outside of the contact face
            lam = np.clip(x[3], 0.0, 1.0)
            c_n = u[0]
            c_f = u[1]

            lam_dot = u[2]
            if lam == 1.0:  # lam_dot must be nonpositive
                lam_dot = np.clip(lam_dot, None, 0.0)
            elif lam == 0.0:  # lam_dot must be nonnegative
                lam_dot = np.clip(lam_dot, 0.0, None)

            R = self._get_R(theta)
            t = self._get_twist(lam, c_n, c_f)

            x_dot = np.vstack((R.dot(t), [lam_dot]))
            return x_dot

        def DoCalcTimeDerivatives(
            self, context: Context, derivatives: ContinuousState
        ) -> None:
            x = context.get_continuous_state_vector()
            u = self.input.Eval(context)
            x_dot = self._calc_dynamics(x, u)  # type: ignore
            derivatives.get_mutable_vector().set_value(x_dot)  # type: ignore

    return Impl


SliderPusherSystem = SliderPusherSystem_[None]  # type: ignore


class SliderPusherGeometry(LeafSystem):
    def __init__(
        self,
        slider_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
    ) -> None:
        super().__init__()

        self.slider_geometry = slider_geometry
        self.contact_location = contact_location

        NUM_CONTACT_POINTS = 1
        NUM_SLIDER_STATES = 3  # x, y, theta

        self.DeclareVectorInputPort("x", NUM_SLIDER_STATES + NUM_CONTACT_POINTS)
        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()
        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider")
        )
        self.slider_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.slider_frame_id,
            GeometryInstance(RigidTransform.Identity(), Box(0.3, 0.3, 0.3), "slider"),
        )
        BOX_COLOR = COLORS["aquamarine4"]
        scene_graph.AssignRole(
            self.source_id,
            self.slider_geometry_id,
            MakePhongIllustrationProperties(BOX_COLOR.diffuse()),
        )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("pusher")
        )
        self.pusher_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.pusher_frame_id,
            GeometryInstance(RigidTransform.Identity(), Sphere(0.01), "pusher"),
        )
        FINGER_COLOR = COLORS["firebrick3"]
        scene_graph.AssignRole(
            self.source_id,
            self.pusher_geometry_id,
            MakePhongIllustrationProperties(FINGER_COLOR.diffuse()),
        )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        slider_pusher_output_port: OutputPort,
        slider_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
    ) -> "SliderPusherGeometry":
        slider_pusher_geometry = builder.AddNamedSystem(
            "slider_pusher_geometry",
            cls(slider_geometry, contact_location, scene_graph),
        )
        builder.Connect(
            slider_pusher_output_port, slider_pusher_geometry.get_input_port()
        )
        builder.Connect(
            slider_pusher_geometry.get_output_port(),
            scene_graph.get_source_pose_port(slider_pusher_geometry.source_id),
        )
        return slider_pusher_geometry

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        state: npt.NDArray[np.float64] = self.get_input_port().Eval(context)  # type: ignore

        p_x = state[0]
        p_y = state[1]
        theta = state[2]
        pose = RigidTransform(
            RollPitchYaw(np.array([0.0, 0.0, theta])), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.slider_frame_id, value=pose)  # type: ignore

        lam = state[3]
        p_c_B = self.slider_geometry.get_p_c_B_from_lam(lam, self.contact_location)
        pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_c_B.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.pusher_frame_id, value=pose)  # type: ignore
