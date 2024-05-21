from typing import List, Optional
import numpy as np
from manipulation.scenarios import AddMultibodyTriad
from pydrake.all import (
    Context,
    Diagram,
    DiagramBuilder,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    SceneGraph,
    StartMeshcat,
    MeshcatVisualizer,
    ModelInstanceIndex,
    GeometryInstance,
    RigidBody as DrakeRigidBody,
    Box as DrakeBox,
    Cylinder as DrakeCylinder,
    GeometryInstance,
    MakePhongIllustrationProperties,
    Rgba,
    RigidTransform,
    RollPitchYaw,
    Meshcat,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.systems.pusher_slider_pose_selector import (
    PusherSliderPoseSelector,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    AddSliderAndConfigureContact,
    AddRandomizedSliderAndConfigureContact,
    randomize_camera_config,
    randomize_pusher,
    randomize_table,
)
from planning_through_contact.simulation.state_estimators.plant_updater import (
    PlantUpdater,
)
from planning_through_contact.visualize.colors import COLORS


class StateEstimator(Diagram):
    """
    The State Estimator is the "internal" model that represents our knowledge of the real world and is not
    simulated. It contains a plant which is updated using a plant updater system. The
    plant itself is not part of the diagram while the updater system is.
    """

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
        robot_model_name: str,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._goal_geometries = []
        self._desired_pusher_geometry = None
        self.meshcat = meshcat
        builder = DiagramBuilder()

        # Create the multibody plant and scene graph
        self._plant = MultibodyPlant(time_step=sim_config.time_step)
        self._plant.set_name("state_estimator_plant")
        self._scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
        self._plant.RegisterAsSourceForSceneGraph(self._scene_graph)

        if not sim_config.domain_randomization:
            self.slider = AddSliderAndConfigureContact(
                sim_config=sim_config, plant=self._plant, scene_graph=self._scene_graph
            )
        else:
            table_grey = np.random.uniform(0.3, 0.95)
            pusher_grey = np.random.uniform(0.1, min(table_grey, 0.4))
            color_range = 0.025

            randomize_pusher()
            randomize_table(
                default_color=[table_grey, table_grey, table_grey],
                color_range=color_range,
            )
            self.slider = AddRandomizedSliderAndConfigureContact(
                default_color=[pusher_grey, pusher_grey, pusher_grey],
                color_range=color_range,
                sim_config=sim_config, 
                plant=self._plant, 
                scene_graph=self._scene_graph
            )

        # Add camera
        if sim_config.camera_configs is not None:
            from pydrake.systems.sensors import (
                ApplyCameraConfig
            )
            for camera_config in sim_config.camera_configs:
                if sim_config.randomize_camera:
                    camera_config = randomize_camera_config(camera_config)
                ApplyCameraConfig(
                    config=camera_config,
                    builder=builder,
                    plant=self._plant,
                    scene_graph=self._scene_graph,
                )

                builder.ExportOutput(
                    builder.GetSubsystemByName(
                        f"rgbd_sensor_{camera_config.name}"
                    ).color_image_output_port(),
                    f"rgbd_sensor_state_estimator_{camera_config.name}",
                )

        # Add system for updating the plant
        self._plant_updater: PlantUpdater = builder.AddNamedSystem(
            "plant_updater",
            PlantUpdater(
                plant=self._plant,
                robot_model_name=robot_model_name,
                object_model_name=sim_config.slider.name,
            ),
        )

        # Connect the plant to the scene graph
        mbp_position_to_geometry_pose: MultibodyPositionToGeometryPose = (
            builder.AddNamedSystem(
                "mbp_position_to_geometry_pose",
                MultibodyPositionToGeometryPose(self._plant),
            )
        )
        builder.Connect(
            self._plant_updater.get_position_output_port(),
            mbp_position_to_geometry_pose.get_input_port(),
        )
        builder.Connect(
            mbp_position_to_geometry_pose.get_output_port(),
            self._scene_graph.get_source_pose_port(self._plant.get_source_id()),
        )

        # Connect pusher slider planar pose selector
        slider_idx = self._plant.GetBodyByName(sim_config.slider.name).index()
        pusher_idx = self._plant.GetBodyByName("pusher").index()

        self._pusher_slider_pose_selector = builder.AddNamedSystem(
            "SliderPoseSelector", PusherSliderPoseSelector(slider_idx, pusher_idx)
        )
        builder.Connect(
            self._plant_updater.get_body_poses_output_port(),
            self._pusher_slider_pose_selector.GetInputPort("body_poses"),
        )

        # Export planar pose output ports
        builder.ExportOutput(
            self._pusher_slider_pose_selector.GetOutputPort("slider_pose"),
            "slider_pose_estimated",
        )
        builder.ExportOutput(
            self._pusher_slider_pose_selector.GetOutputPort("pusher_pose"),
            "pusher_pose_estimated",
        )

        # Export input ports
        builder.ExportInput(
            self._plant_updater.GetInputPort("robot_state"), "robot_state"
        )
        builder.ExportInput(
            self._plant_updater.GetInputPort("object_position"), "object_position"
        )

        # Export "cheat" ports
        builder.ExportOutput(self._scene_graph.get_query_output_port(), "query_object")
        builder.ExportOutput(
            self._plant_updater.get_state_output_port(), "plant_continuous_state"
        )
        builder.ExportOutput(
            self._plant_updater.get_body_poses_output_port(), "body_poses"
        )
        for i in range(self._plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            model_instance_name = self._plant.GetModelInstanceName(model_instance)
            builder.ExportOutput(
                self._plant_updater.get_state_output_port(model_instance),
                f"{model_instance_name}_state",
            )
        if self.meshcat and sim_config.visualize_desired:
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder, self._scene_graph.get_query_output_port(), self.meshcat
            )

            # Set up meshcat camera view
            zoom = 1.8
            camera_in_world = [sim_config.slider_goal_pose.x, 
                            (sim_config.slider_goal_pose.y-1)/zoom,
                            1.5/zoom]
            target_in_world = [sim_config.slider_goal_pose.x, sim_config.slider_goal_pose.x, 0]
            self.meshcat.SetCameraPose(camera_in_world, target_in_world)

            assert sim_config.slider_goal_pose is not None
            self._visualize_desired_slider_pose(sim_config.slider_goal_pose)

            if sim_config.draw_frames:
                # print(f"Drawing frames")
                for frame_name in [
                    # "iiwa_link_7",
                    # "pusher_base",
                    # "t_pusher",
                    "pusher_end",
                ]:
                    AddMultibodyTriad(
                        self._plant.GetFrameByName(frame_name),
                        self._scene_graph,
                        length=0.1,
                        radius=0.001,
                    )

        builder.BuildInto(self)

    def get_plant(self) -> MultibodyPlant:
        return self._plant

    def get_plant_context(self) -> Context:
        return self._plant_updater.get_plant_context()

    def get_scene_graph(self) -> SceneGraph:
        return self._scene_graph

    def _visualize_desired_slider_pose(
        self, desired_planar_pose: PlanarPose, time_in_recording: float = 0.0
    ) -> None:
        shapes = self.get_slider_shapes()
        poses = self.get_slider_shape_poses()

        heights = [shape.height() for shape in shapes]
        min_height = min(heights)
        desired_pose = desired_planar_pose.to_pose(
            min_height / 2, z_axis_is_positive=True
        )
        if len(self._goal_geometries) == 0:
            source_id = self._scene_graph.RegisterSource()
            BOX_COLOR = COLORS["emeraldgreen"]
            DESIRED_POSE_ALPHA = 0.4
            for idx, (shape, pose) in enumerate(zip(shapes, poses)):
                geom_instance = GeometryInstance(
                    desired_pose.multiply(pose),
                    shape,
                    f"shape_{idx}",
                )
                curr_shape_geometry_id = self._scene_graph.RegisterAnchoredGeometry(
                    source_id,
                    geom_instance,
                )
                self._scene_graph.AssignRole(
                    source_id,
                    curr_shape_geometry_id,
                    MakePhongIllustrationProperties(
                        BOX_COLOR.diffuse(DESIRED_POSE_ALPHA)
                    ),
                )
                geom_name = f"goal_shape_{idx}"
                self._goal_geometries.append(geom_name)
                self.meshcat.SetObject(
                    geom_name, shape, rgba=Rgba(*BOX_COLOR.diffuse(DESIRED_POSE_ALPHA))
                )
        else:
            for pose, geom_name in zip(poses, self._goal_geometries):
                self.meshcat.SetTransform(
                    geom_name, desired_pose.multiply(pose), time_in_recording
                )

    def _visualize_desired_pusher_pose(
        self, desired_planar_pose: PlanarPose, time_in_recording: float = 0.0
    ) -> None:
        shape = self.get_pusher_shape()
        shape = DrakeCylinder(shape.radius(), self._sim_config.pusher_z_offset)
        # height = 0.1 # height for the actuated cylinder robot
        height = shape.length() / 2
        # height = self._sim_config.pusher_z_offset +( shape.length())/2 # height for iiwa
        pose = self.get_pusher_shape_pose()
        desired_pose = desired_planar_pose.to_pose(height, z_axis_is_positive=True)
        if self._desired_pusher_geometry is None:
            source_id = self._scene_graph.RegisterSource()
            BOX_COLOR = COLORS["emeraldgreen"]
            DESIRED_POSE_ALPHA = 0.4
            geom_name = f"desired_pusher"
            geom_instance = GeometryInstance(
                desired_pose,  # desired_pose.multiply(pose),
                shape,
                geom_name,
            )
            curr_shape_geometry_id = self._scene_graph.RegisterAnchoredGeometry(
                source_id,
                geom_instance,
            )
            self._scene_graph.AssignRole(
                source_id,
                curr_shape_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse(DESIRED_POSE_ALPHA)),
            )

            self._desired_pusher_geometry = geom_name
            self.meshcat.SetObject(
                geom_name, shape, rgba=Rgba(*BOX_COLOR.diffuse(DESIRED_POSE_ALPHA))
            )
        else:
            self.meshcat.SetTransform(
                self._desired_pusher_geometry,
                desired_pose,  # desired_pose.multiply(pose),
                time_in_recording,
            )

    def get_slider_min_height(self) -> float:
        shapes = self.get_slider_shapes()
        heights = [shape.height() for shape in shapes]
        min_height = min(heights)
        return min_height

    def get_slider_body(self) -> DrakeRigidBody:
        slider_body = self._plant.GetUniqueFreeBaseBodyOrThrow(self.slider)
        return slider_body

    def get_slider_shapes(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
            slider_body
        )

        inspector = self._scene_graph.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

        # for now we only support Box shapes
        assert all([isinstance(shape, DrakeBox) for shape in shapes])

        return shapes

    def get_slider_shape_poses(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
            slider_body
        )

        inspector = self._scene_graph.model_inspector()
        poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

        return poses

    def get_pusher_shape(self) -> DrakeCylinder:
        pusher_body = self._plant.GetBodyByName("pusher")
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
            pusher_body
        )

        inspector = self._scene_graph.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

        # for now we only support Cylinder shapes
        assert all([isinstance(shape, DrakeCylinder) for shape in shapes])
        assert len(shapes) == 1, "Pusher should only have one shape"

        return shapes[0]

    def get_pusher_shape_pose(self) -> RigidTransform:
        pusher_body = self._plant.GetBodyByName("pusher")
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(
            pusher_body
        )

        inspector = self._scene_graph.model_inspector()
        pose = inspector.GetPoseInFrame(collision_geometries_ids[0])

        return pose
