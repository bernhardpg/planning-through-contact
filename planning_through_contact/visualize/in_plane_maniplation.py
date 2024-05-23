from dataclasses import dataclass
from typing import List

from pydrake.geometry import SceneGraph
from pydrake.systems.framework import LeafSystem

from planning_through_contact.geometry.rigid_body import RigidBody


# TODO move
@dataclass
class InPlaneTrajectory:
    objects: List[RigidBody]


class InPlaneGeometry(LeafSystem):
    def __init__(
        self,
        traj: InPlaneTrajectory,
        scene_graph: SceneGraph,
        visualize_knot_points: bool = False,
        visualize_goal: bool = True,
    ) -> None:
        super().__init__()

        self.traj = traj
        self.visualize_goal = visualize_goal
        self.visualize_knot_points = visualize_knot_points

        slider_geometry = self.traj.config.slider_geometry
        pusher_radius = self.traj.config.pusher_radius

        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()

        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider")
        )
        _add_slider_geometries(
            self.source_id, slider_geometry, scene_graph, self.slider_frame_id
        )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher"),
        )
        _add_pusher_geometry(
            self.source_id, pusher_radius, scene_graph, self.pusher_frame_id
        )

        GOAL_TRANSPARENCY = 0.3
        if self.visualize_goal:
            self.slider_goal_frame_id = scene_graph.RegisterFrame(
                self.source_id, GeometryFrame("slider_goal")
            )
            _add_slider_geometries(
                self.source_id,
                slider_geometry,
                scene_graph,
                self.slider_goal_frame_id,
                alpha=GOAL_TRANSPARENCY,
            )
            self.pusher_goal_frame_id = scene_graph.RegisterFrame(
                self.source_id,
                GeometryFrame("pusher_goal"),
            )
            _add_pusher_geometry(
                self.source_id,
                pusher_radius,
                scene_graph,
                self.pusher_goal_frame_id,
                alpha=GOAL_TRANSPARENCY,
            )

        # TODO: Shows table
        # TABLE_COLOR = COLORS["bisque3"]
        # TABLE_HEIGHT = 0.1
        # table_geometry_id = scene_graph.RegisterAnchoredGeometry(
        #     self.source_id,
        #     GeometryInstance(
        #         RigidTransform(
        #             RotationMatrix.Identity(), np.array([0, 0, -TABLE_HEIGHT / 2])  # type: ignore
        #         ),
        #         DrakeBox(1.0, 1.0, TABLE_HEIGHT),
        #         "table",
        #     ),
        # )
        # scene_graph.AssignRole(
        #     self.source_id,
        #     table_geometry_id,
        #     MakePhongIllustrationProperties(TABLE_COLOR.diffuse()),
        # )

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        traj: PlanarPushingTrajectory,
        scene_graph: SceneGraph,
        visualize_knot_points: bool = False,
        name: str = "traj_geometry ",
    ) -> "PlanarPushingTrajectory":
        traj_geometry = builder.AddNamedSystem(
            name,
            cls(
                traj,
                scene_graph,
                visualize_knot_points,
            ),
        )
        builder.Connect(
            traj_geometry.get_output_port(),
            scene_graph.get_source_pose_port(traj_geometry.source_id),
        )
        return traj_geometry

    def _set_outputs(
        self,
        slider_frame_id: FrameId,
        pusher_frame_id: FrameId,
        output: FramePoseVector,
        p_WB: npt.NDArray[np.float64],
        p_WP: npt.NDArray[np.float64],
        R_WB: npt.NDArray[np.float64],
    ):
        p_x = p_WB[0, 0]  # type: ignore
        p_y = p_WB[1, 0]  # type: ignore

        slider_pose = RigidTransform(RotationMatrix(R_WB), np.array([p_x, p_y, 0.0]))  # type: ignore
        output.get_mutable_value().set_value(id=slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0])))  # type: ignore
        output.get_mutable_value().set_value(id=pusher_frame_id, value=pusher_pose)  # type: ignore

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        t = context.get_time()

        if self.visualize_knot_points:
            # TODO force
            R_WB = self.traj.get_knot_point_value(t, "R_WB")
            p_WB = self.traj.get_knot_point_value(t, "p_WB")
            p_WP = self.traj.get_knot_point_value(t, "p_WP")
        else:
            # TODO force
            R_WB = self.traj.get_value(t, "R_WB")
            p_WB = self.traj.get_value(t, "p_WB")
            p_WP = self.traj.get_value(t, "p_WP")

        self._set_outputs(
            self.slider_frame_id,
            self.pusher_frame_id,
            output,
            p_WB,  # type: ignore
            p_WP,  # type: ignore
            R_WB,  # type: ignore
        )

        if self.visualize_goal:
            target_slider_planar_pose = self.traj.target_slider_planar_pose
            target_pusher_planar_pose = self.traj.target_pusher_planar_pose

            self._set_outputs(
                self.slider_goal_frame_id,
                self.pusher_goal_frame_id,
                output,
                target_slider_planar_pose.pos(),
                target_pusher_planar_pose.pos(),
                target_slider_planar_pose.rot_matrix(),
            )
