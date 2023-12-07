from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.common.value import Value
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Cylinder as DrakeCylinder
from pydrake.geometry import (
    FrameId,
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
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.trajectory_builder import (
    OldPlanarPushingTrajectory,
)
from planning_through_contact.planning.planar.planar_pushing_planner import (
    PlanarPushingStartAndGoal,
)
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.visualize.visualizer_2d import (
    VisualizationForce2d,
    VisualizationPoint2d,
    VisualizationPolygon2d,
    Visualizer2d,
)


def _add_slider_geometries(
    source_id,
    slider_geometry: CollisionGeometry,
    scene_graph: SceneGraph,
    slider_frame_id: FrameId,
    alpha: float = 1.0,
) -> None:
    BOX_COLOR = COLORS["aquamarine4"]
    DEFAULT_HEIGHT = 0.3

    if isinstance(slider_geometry, Box2d):
        box_geometry_id = scene_graph.RegisterGeometry(
            source_id,
            slider_frame_id,
            GeometryInstance(
                RigidTransform.Identity(),
                DrakeBox(slider_geometry.width, slider_geometry.height, DEFAULT_HEIGHT),
                "slider",
            ),
        )
        scene_graph.AssignRole(
            source_id,
            box_geometry_id,
            MakePhongIllustrationProperties(BOX_COLOR.diffuse(alpha)),
        )
    elif isinstance(slider_geometry, TPusher2d):
        boxes, transforms = slider_geometry.get_as_boxes(DEFAULT_HEIGHT / 2)
        box_geometry_ids = [
            scene_graph.RegisterGeometry(
                source_id,
                slider_frame_id,
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
                source_id,
                box_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse(alpha)),
            )


def _add_pusher_geometry(
    source_id,
    pusher_radius: float,
    scene_graph: SceneGraph,
    pusher_frame_id: FrameId,
    alpha: float = 1.0,
) -> None:
    CYLINDER_HEIGHT = 0.3
    pusher_geometry_id = scene_graph.RegisterGeometry(
        source_id,
        pusher_frame_id,
        GeometryInstance(
            RigidTransform(
                RotationMatrix.Identity(), np.array([0, 0, CYLINDER_HEIGHT / 2])  # type: ignore
            ),
            DrakeCylinder(pusher_radius, CYLINDER_HEIGHT),
            "pusher",
        ),
    )
    pusher_COLOR = COLORS["firebrick3"]
    scene_graph.AssignRole(
        source_id,
        pusher_geometry_id,
        MakePhongIllustrationProperties(pusher_COLOR.diffuse(alpha)),
    )


class PlanarPushingStartGoalGeometry(LeafSystem):
    """
    Class used to visualize start and goal positions for a planar pushing task
    """

    def __init__(
        self,
        slider_initial_pose: PlanarPose,
        slider_target_pose: PlanarPose,
        pusher_initial_pose: PlanarPose,
        pusher_target_pose: PlanarPose,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        scene_graph: SceneGraph,
    ) -> None:
        super().__init__()

        self.slider_initial_pose = slider_initial_pose
        self.slider_target_pose = slider_target_pose
        self.pusher_initial_pose = self._rotate_to_world(
            pusher_initial_pose, slider_initial_pose
        )
        self.pusher_target_pose = self._rotate_to_world(
            pusher_target_pose, slider_target_pose
        )

        self.DeclareAbstractOutputPort(
            "geometry_pose",
            alloc=lambda: Value(FramePoseVector()),
            calc=self.calc_output,  # type: ignore
        )

        self.source_id = scene_graph.RegisterSource()

        TRANSPARENCY = 0.3

        self.slider_frame_id = scene_graph.RegisterFrame(
            self.source_id, GeometryFrame("slider_start")
        )
        _add_slider_geometries(
            self.source_id,
            slider_geometry,
            scene_graph,
            self.slider_frame_id,
            alpha=TRANSPARENCY,
        )
        self.pusher_frame_id = scene_graph.RegisterFrame(
            self.source_id,
            GeometryFrame("pusher_start"),
        )
        _add_pusher_geometry(
            self.source_id,
            pusher_radius,
            scene_graph,
            self.pusher_frame_id,
            alpha=TRANSPARENCY,
        )

    @staticmethod
    def _rotate_to_world(
        pusher_pose: PlanarPose, slider_pose: PlanarPose
    ) -> PlanarPose:
        p_WP = pusher_pose.pos()
        R_WB = slider_pose.two_d_rot_matrix()
        p_WB = slider_pose.pos()

        # We need to compute the pusher pos in the frame of the slider
        p_BP = R_WB.T @ (p_WP - p_WB)
        pusher_pose_world = PlanarPose(p_BP[0, 0], p_BP[1, 0], 0)

        return pusher_pose_world

    @classmethod
    def add_to_builder(
        cls,
        builder: DiagramBuilder,
        slider_initial_pose: PlanarPose,
        slider_target_pose: PlanarPose,
        pusher_initial_pose: PlanarPose,
        pusher_target_pose: PlanarPose,
        slider_geometry: CollisionGeometry,
        pusher_radius: float,
        scene_graph: SceneGraph,
        name: str = "start_goal_geometry",
    ) -> "PlanarPushingTrajectory":
        traj_geometry = builder.AddNamedSystem(
            name,
            cls(
                slider_initial_pose,
                slider_target_pose,
                pusher_initial_pose,
                pusher_target_pose,
                slider_geometry,
                pusher_radius,
                scene_graph,
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

        slider_pose = RigidTransform(
            RotationMatrix(R_WB), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
        output.get_mutable_value().set_value(id=pusher_frame_id, value=pusher_pose)  # type: ignore

    def _get_pusher_in_world(self, slider_pose, pusher_pose) -> PlanarPose:
        p_WP = slider_pose.pos() + slider_pose.two_d_rot_matrix().dot(pusher_pose.pos())
        return PlanarPose(p_WP[0, 0], p_WP[1, 0], 0)

    def get_pos_limits(
        self, slider_geometry, buffer
    ) -> Tuple[float, float, float, float]:
        def get_lims(vecs) -> Tuple[float, float, float, float]:
            vec_xs = [vec[0, 0] for vec in vecs]
            vec_ys = [vec[1, 0] for vec in vecs]

            vec_x_max = max(vec_xs)
            vec_x_min = min(vec_xs)
            vec_y_max = max(vec_ys)
            vec_y_min = min(vec_ys)

            return vec_x_min, vec_x_max, vec_y_min, vec_y_max

        def add_buffer_to_lims(lims, buffer) -> Tuple[float, float, float, float]:
            return (
                lims[0] - buffer,
                lims[1] + buffer,
                lims[2] - buffer,
                lims[3] + buffer,
            )

        def get_lims_from_two_lims(lim_a, lim_b) -> Tuple[float, float, float, float]:
            return (
                min(lim_a[0], lim_b[0]),
                max(lim_a[1], lim_b[1]),
                min(lim_a[2], lim_b[2]),
                max(lim_a[3], lim_b[3]),
            )

        p_WB_lims = get_lims(
            [self.slider_initial_pose.pos(), self.slider_target_pose.pos()]
        )
        object_radius = slider_geometry.max_dist_from_com
        obj_lims = add_buffer_to_lims(p_WB_lims, object_radius)
        p_WP_lims = get_lims(
            [
                self._get_pusher_in_world(
                    self.slider_initial_pose, self.pusher_initial_pose
                ).pos(),
                self._get_pusher_in_world(
                    self.slider_target_pose, self.pusher_target_pose
                ).pos(),
            ]
        )

        lims = get_lims_from_two_lims(obj_lims, p_WP_lims)
        return add_buffer_to_lims(lims, buffer)

    def calc_output(self, context: Context, output: FramePoseVector) -> None:
        t = context.get_time()
        if t <= 1.0:
            slider_pose = self.slider_initial_pose
            pusher_pose = self.pusher_initial_pose
        else:
            slider_pose = self.slider_target_pose
            pusher_pose = self.pusher_target_pose

        self._set_outputs(
            self.slider_frame_id,
            self.pusher_frame_id,
            output,
            slider_pose.pos(),
            self._get_pusher_in_world(slider_pose, pusher_pose).pos(),
            slider_pose.rot_matrix(),
        )


class PlanarPushingTrajectoryGeometry(LeafSystem):
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
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
        MIN_PUSHER_RADIUS_VIZ = 0.01  # we need some radius for the visualization
        if pusher_radius == 0:
            pusher_radius = MIN_PUSHER_RADIUS_VIZ

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

        slider_pose = RigidTransform(
            RotationMatrix(R_WB), np.array([p_x, p_y, 0.0])  # type: ignore
        )
        output.get_mutable_value().set_value(id=slider_frame_id, value=slider_pose)  # type: ignore

        pusher_pose = RigidTransform(
            RotationMatrix.Identity(), np.concatenate((p_WP.flatten(), [0]))  # type: ignore
        )
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


def visualize_planar_pushing_start_and_goal(
    slider_geometry: CollisionGeometry,
    pusher_radius: float,
    plan: PlanarPushingStartAndGoal,
    show: bool = False,
    save: bool = False,
    filename: Optional[str] = None,
):
    if save:
        assert filename is not None

    builder = DiagramBuilder()

    # Register geometry with SceneGraph
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    geometry = PlanarPushingStartGoalGeometry.add_to_builder(
        builder,
        plan.slider_initial_pose,
        plan.slider_target_pose,
        plan.pusher_initial_pose,
        plan.pusher_target_pose,
        slider_geometry,
        pusher_radius,
        scene_graph,
    )

    x_min, x_max, y_min, y_max = geometry.get_pos_limits(slider_geometry, buffer=0.1)

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
            xlim=np.array([x_min, x_max]),
            ylim=np.array([y_min, y_max]),
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
    simulator.AdvanceTo(2.0)

    visualizer.stop_recording()  # type: ignore
    ani = visualizer.get_recording_as_animation()  # type: ignore

    if save:
        # Playback the recording and save the output.
        ani.save(f"{filename}.mp4", fps=30)

    return ani


def visualize_planar_pushing_trajectory(
    traj: PlanarPushingTrajectory,
    show: bool = False,
    save: bool = False,
    filename: Optional[str] = None,
    visualize_knot_points: bool = False,
    lims: Optional[Tuple[float, float, float, float]] = None,
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

    if lims is None:
        x_min, x_max, y_min, y_max = traj.get_pos_limits(buffer=0.1)
    else:
        x_min, x_max, y_min, y_max = lims

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
            xlim=np.array([x_min, x_max]),
            ylim=np.array([y_min, y_max]),
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
    pusher_COLOR = COLORS["firebrick3"]
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

    contact_point_viz = VisualizationPoint2d(traj.p_WP.T, pusher_COLOR)
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
