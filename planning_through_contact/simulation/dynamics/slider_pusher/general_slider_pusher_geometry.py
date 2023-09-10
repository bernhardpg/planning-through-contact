import numpy as np
import numpy.typing as npt
from pydrake.common.value import Value
from pydrake.geometry import (
    Box,
    Cylinder,
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
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.visualize.colors import COLORS


class GeneralSliderPusherGeometry(LeafSystem):
    def __init__(
        self,
        slider_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
        alpha: float = 1.0,
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
        BOX_COLOR = COLORS["aquamarine4"]
        if isinstance(slider_geometry, Box2d):
            DEFAULT_HEIGHT = 0.3
            box_geometry_id = scene_graph.RegisterGeometry(
                self.source_id,
                self.slider_frame_id,
                GeometryInstance(
                    RigidTransform.Identity(),
                    Box(slider_geometry.width, slider_geometry.height, DEFAULT_HEIGHT),
                    "slider",
                ),
            )
            scene_graph.AssignRole(
                self.source_id,
                box_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse(alpha)),
            )
        elif isinstance(slider_geometry, TPusher2d):
            DEFAULT_HEIGHT = 0.1
            boxes, transforms = slider_geometry.get_as_boxes(DEFAULT_HEIGHT / 2)
            box_geometry_ids = [
                scene_graph.RegisterGeometry(
                    self.source_id,
                    self.slider_frame_id,
                    GeometryInstance(
                        transform,
                        Box(box.width, box.height, DEFAULT_HEIGHT),
                        f"box_{idx}",
                    ),
                )
                for idx, (box, transform) in enumerate(zip(boxes, transforms))
            ]
            for box_geometry_id in box_geometry_ids:
                scene_graph.AssignRole(
                    self.source_id,
                    box_geometry_id,
                    MakePhongIllustrationProperties(BOX_COLOR.diffuse(alpha)),
                )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            source_id=self.source_id,
            parent_id=self.slider_frame_id,
            frame=GeometryFrame("pusher"),
        )
        CYLINDER_HEIGHT = 0.3
        self.pusher_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.pusher_frame_id,
            GeometryInstance(
                RigidTransform(
                    RotationMatrix.Identity(), np.array([0, 0, CYLINDER_HEIGHT / 2])  # type: ignore
                ),
                Cylinder(0.01, CYLINDER_HEIGHT),
                "pusher",
            ),
        )
        FINGER_COLOR = COLORS["firebrick3"]
        scene_graph.AssignRole(
            self.source_id,
            self.pusher_geometry_id,
            MakePhongIllustrationProperties(FINGER_COLOR.diffuse(alpha)),
        )

        TABLE_COLOR = COLORS["bisque3"]
        TABLE_HEIGHT = 0.1
        table_geometry_id = scene_graph.RegisterAnchoredGeometry(
            self.source_id,
            GeometryInstance(
                RigidTransform(
                    RotationMatrix.Identity(), np.array([0, 0, -TABLE_HEIGHT / 2])  # type: ignore
                ),
                Box(1.0, 1.0, TABLE_HEIGHT),
                "table",
            ),
        )
        scene_graph.AssignRole(
            self.source_id,
            table_geometry_id,
            MakePhongIllustrationProperties(TABLE_COLOR.diffuse(alpha)),
        )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        slider_pusher_output_port: OutputPort,
        slider_geometry: CollisionGeometry,
        contact_location: PolytopeContactLocation,
        scene_graph: SceneGraph,
        name: str = "slider_pusher_geometry",
        alpha: float = 1.0,
    ) -> "SliderPusherGeometry":
        slider_pusher_geometry = builder.AddNamedSystem(
            name,
            cls(slider_geometry, contact_location, scene_graph, alpha=alpha),
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
        p_BP = self.slider_geometry.get_p_BP_from_lam(lam, self.contact_location)
        pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_BP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.pusher_frame_id, value=pose)  # type: ignore
