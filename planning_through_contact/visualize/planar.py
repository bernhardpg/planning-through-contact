from typing import Optional

import numpy as np
from pydrake.common.value import Value
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Cylinder as DrakeCylinder
from pydrake.geometry import (
    FramePoseVector,
    GeometryFrame,
    GeometryInstance,
    MakePhongIllustrationProperties,
    SceneGraph,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.all import Context, DiagramBuilder, LeafSystem
from pydrake.systems.analysis import Simulator
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer,
    PlanarSceneGraphVisualizer,
)

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


class PlanarPushingTrajectoryGeometry(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        scene_graph: SceneGraph,
        visualize_knot_points: bool = False,
    ) -> None:
        super().__init__()

        self.traj = traj
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
        BOX_COLOR = COLORS["aquamarine4"]
        DEFAULT_HEIGHT = 0.3
        if isinstance(slider_geometry, Box2d):
            box_geometry_id = scene_graph.RegisterGeometry(
                self.source_id,
                self.slider_frame_id,
                GeometryInstance(
                    RigidTransform.Identity(),
                    DrakeBox(
                        slider_geometry.width, slider_geometry.height, DEFAULT_HEIGHT
                    ),
                    "slider",
                ),
            )
            scene_graph.AssignRole(
                self.source_id,
                box_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse()),
            )
        elif isinstance(slider_geometry, TPusher2d):
            boxes, transforms = slider_geometry.get_as_boxes(DEFAULT_HEIGHT / 2)
            box_geometry_ids = [
                scene_graph.RegisterGeometry(
                    self.source_id,
                    self.slider_frame_id,
                    GeometryInstance(
                        transform,
                        DrakeBox(box.width, box.height, DEFAULT_HEIGHT),
                        f"box_{idx}",
                    ),
                )
                for idx, (box, transform) in enumerate(zip(boxes, transforms))
            ]
            for box_geometry_id in box_geometry_ids:
                scene_graph.AssignRole(
                    self.source_id,
                    box_geometry_id,
                    MakePhongIllustrationProperties(BOX_COLOR.diffuse()),
                )

        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher"),
        )
        CYLINDER_HEIGHT = 0.3
        self.pusher_geometry_id = scene_graph.RegisterGeometry(
            self.source_id,
            self.pusher_frame_id,
            GeometryInstance(
                RigidTransform(
                    RotationMatrix.Identity(), np.array([0, 0, CYLINDER_HEIGHT / 2])  # type: ignore
                ),
                DrakeCylinder(pusher_radius, CYLINDER_HEIGHT),
                "pusher",
            ),
        )
        FINGER_COLOR = COLORS["firebrick3"]
        scene_graph.AssignRole(
            self.source_id,
            self.pusher_geometry_id,
            MakePhongIllustrationProperties(FINGER_COLOR.diffuse()),
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

        p_x = p_WB[0, 0]  # type: ignore
        p_y = p_WB[1, 0]  # type: ignore

        slider_pose = RigidTransform(
            RotationMatrix(R_WB), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=self.pusher_frame_id, value=pusher_pose)  # type: ignore


def visualize_planar_pushing_trajectory(
    traj: PlanarPushingTrajectory,
    show: bool = False,
    save: bool = False,
    filename: Optional[str] = None,
    visualize_knot_points: bool = False,
):
    if save:
        assert filename is not None

    builder = DiagramBuilder()

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    traj_geometry = PlanarPushingTrajectoryGeometry.add_to_builder(
        builder,
        traj,
        scene_graph,
        visualize_knot_points,
    )

    # max_x_y = max(
    #     max(np.abs(traj.contact_pos_traj.flatten())),
    #     max(np.abs(traj.com_traj.flatten())),
    # )
    # max_geometry_dist = max(
    #     [max(np.abs(v.flatten())) for v in slider_geometry.vertices]
    # )
    # max_x_y_and_geometry = max_x_y + max_geometry_dist
    # LIM = max_x_y_and_geometry * 1.1
    LIM = 0.7

    def connect_planar_visualizer(
        builder: DiagramBuilder, scene_graph: SceneGraph
    ) -> PlanarSceneGraphVisualizer:
        T_VW = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        visualizer = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            T_VW=T_VW,
            xlim=[-LIM, LIM],
            ylim=[-LIM, LIM],
            show=show,
        )
        return visualizer

    visualizer = connect_planar_visualizer(builder, scene_graph)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Create the simulator, and simulate for 10 seconds.
    context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, context)

    visualizer.start_recording()  # type: ignore

    simulator.Initialize()
    # simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(traj.end_time)

    visualizer.stop_recording()  # type: ignore
    ani = visualizer.get_recording_as_animation()  # type: ignore

    if save:
        # Playback the recording and save the output.
        ani.save(f"{filename}.mp4", fps=30)

    return ani


def visualize_planar_pushing_trajectory_legacy(
    traj: OldPlanarPushingTrajectory,
    object_geometry: CollisionGeometry,
    pusher_radius: float,
    visualize_object_vel: bool = False,
    visualize_robot_base: bool = False,
) -> None:
    CONTACT_COLOR = COLORS["dodgerblue4"]
    BOX_COLOR = COLORS["aquamarine4"]
    FINGER_COLOR = COLORS["firebrick3"]
    TARGET_COLOR = COLORS["firebrick1"]

    flattened_rotation = np.vstack([R.flatten() for R in traj.R_WB])
    box_viz = VisualizationPolygon2d.from_trajs(
        traj.p_WB.T,
        flattened_rotation,
        object_geometry,
        BOX_COLOR,
    )

    # NOTE: I don't really need the entire trajectory here, but leave for now
    target_viz = VisualizationPolygon2d.from_trajs(
        traj.p_WB.T,
        flattened_rotation,
        object_geometry,
        TARGET_COLOR,
    )

    contact_point_viz = VisualizationPoint2d(traj.p_WP.T, FINGER_COLOR)
    contact_point_viz.change_radius(pusher_radius)

    contact_force_viz = VisualizationForce2d(traj.p_WP.T, CONTACT_COLOR, traj.f_c_W.T)
    contact_forces_viz = [contact_force_viz]

    if visualize_object_vel:
        # TODO(bernhardpg): functionality that is useful for debugging
        v_WB = (traj.p_WB[:, 1:] - traj.p_WB[:, :-1]) / 0.1
        object_vel_viz = VisualizationForce2d(traj.p_WB.T, CONTACT_COLOR, v_WB.T)
        contact_forces_viz.append(
            object_vel_viz
        )  # visualize vel as a force (with an arrow)

    viz = Visualizer2d()
    FRAMES_PER_SEC = 1 / traj.dt
    viz.visualize(
        [contact_point_viz],
        contact_forces_viz,
        [box_viz],
        FRAMES_PER_SEC,
        target_viz,
        draw_origin=visualize_robot_base,
    )
