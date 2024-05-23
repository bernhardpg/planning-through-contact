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
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.systems.framework import Context, DiagramBuilder, LeafSystem, OutputPort

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
    PolytopeContactLocation,
)
from planning_through_contact.visualize.colors import COLORS

# TODO(bernhardpg): This class should be unified with GeneralSliderPusherGeometry
# The only difference is that this class expects the state to be [x, y, th, lam],
# but this should just be converted by a conversion system in the diagram


class SliderPusherGeometry(LeafSystem):
    def __init__(
        self,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.slider_geometry = slider_geometry
        self.pusher_radius = pusher_radius
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
        assert isinstance(slider_geometry, Box2d)
        self.slider_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.slider_frame_id,
            GeometryInstance(
                RigidTransform.Identity(),
                Box(slider_geometry.width, slider_geometry.height, 0.05),
                "slider",
            ),
        )
        BOX_COLOR = COLORS["aquamarine4"]
        scene_graph.AssignRole(
            self.source_id,
            self.slider_geometry_id,
            MakePhongIllustrationProperties(BOX_COLOR.diffuse(alpha)),
        )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            source_id=self.source_id,
            parent_id=self.slider_frame_id,
            frame=GeometryFrame("pusher"),
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
            MakePhongIllustrationProperties(FINGER_COLOR.diffuse(alpha)),
        )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        slider_pusher_output_port: OutputPort,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
        name: str = "slider_pusher_geometry",
        alpha: float = 1.0,
    ) -> "SliderPusherGeometry":
        slider_pusher_geometry = builder.AddNamedSystem(
            name,
            cls(
                slider_geometry,
                pusher_radius,
                contact_location,
                scene_graph,
                alpha=alpha,
            ),
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
        pose = RigidTransform(RollPitchYaw(np.array([0.0, 0.0, theta])), np.array([p_x, p_y, 0.0]))  # type: ignore
        output.get_mutable_value().set_value(id=self.slider_frame_id, value=pose)  # type: ignore

        lam = state[3]
        p_BP = self.slider_geometry.get_p_BP_from_lam(
            lam, self.contact_location, self.pusher_radius
        )
        pose = RigidTransform(RotationMatrix.Identity(), np.concatenate((p_BP.flatten(), [0])))  # type: ignore
        output.get_mutable_value().set_value(id=self.pusher_frame_id, value=pose)  # type: ignore
