from typing import List
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
    GeometryInstance,
    MakePhongIllustrationProperties,
)
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import PusherSliderPoseSelector, PlanarPushingSimConfig
from planning_through_contact.simulation.state_estimators.plant_updater import PlantUpdater
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
        environment, # TODO: Fix circular import issue
        add_visualizer: bool = False,
    ):
        super().__init__()

        builder = DiagramBuilder()

        # Create the multibody plant and scene graph
        self._plant = MultibodyPlant(time_step=sim_config.time_step)
        self._plant.set_name("state_estimator_plant")
        self._scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
        self._plant.RegisterAsSourceForSceneGraph(self._scene_graph)

        slider_name, self.slider = environment.add_all_directives(plant=self._plant, scene_graph=self._scene_graph)
        robot_model_name = "pusher" # TODO fix this, this will break when we transition to using the iiwa as the robot
        # Add system for updating the plant
        self._plant_updater: PlantUpdater = builder.AddNamedSystem(
            "plant_updater", PlantUpdater(plant=self._plant, robot_model_name=robot_model_name, object_model_name=slider_name)
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
        slider_idx = self._plant.GetBodyByName(slider_name).index()
        pusher_idx = self._plant.GetBodyByName("pusher").index()

        self._pusher_slider_pose_selector = builder.AddNamedSystem(
            "SliderPoseSelector", PusherSliderPoseSelector(slider_idx, pusher_idx)
        )
        builder.Connect(
            self._plant_updater.get_body_poses_output_port(),
            self._pusher_slider_pose_selector.GetInputPort("body_poses"),
        )

        # Export planar pose and velocity output ports
        builder.ExportOutput(
            self._pusher_slider_pose_selector.GetOutputPort("slider_pose"), "slider_pose"
        )
        builder.ExportOutput(
            self._pusher_slider_pose_selector.GetOutputPort("slider_spatial_velocity"),
            "slider_spatial_velocity",
        )
        builder.ExportOutput(
            self._pusher_slider_pose_selector.GetOutputPort("pusher_pose"),
            "pusher_pose",
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

        if add_visualizer:
            self.meshcat = StartMeshcat()  # type: ignore
            visualizer = MeshcatVisualizer.AddToBuilder(
                    builder, self._scene_graph.get_query_output_port(), self.meshcat
            )
            if sim_config.visualize_desired:
                assert sim_config.slider_goal_pose is not None
                self._visualize_desired_slider_pose(sim_config.slider_goal_pose)

        builder.BuildInto(self)

    def get_plant(self) -> MultibodyPlant:
        return self._plant

    def get_plant_context(self) -> Context:
        return self._plant_updater.get_plant_context()

    def get_scene_graph(self) -> SceneGraph:
        return self._scene_graph
    
    def _visualize_desired_slider_pose(self, desired_planar_pose: PlanarPose) -> None:
        source_id = self._scene_graph.RegisterSource()
        shapes = self.get_slider_shapes()
        poses = self.get_slider_shape_poses()

        heights = [shape.height() for shape in shapes]
        min_height = min(heights)
        desired_pose = desired_planar_pose.to_pose(
            min_height / 2, z_axis_is_positive=True
        )

        BOX_COLOR = COLORS["emeraldgreen"]
        DESIRED_POSE_ALPHA = 0.4
        for idx, (shape, pose) in enumerate(zip(shapes, poses)):
            curr_shape_geometry_id = self._scene_graph.RegisterAnchoredGeometry(
                source_id,
                GeometryInstance(
                    desired_pose.multiply(pose),
                    shape,
                    f"shape_{idx}",
                ),
            )
            self._scene_graph.AssignRole(
                source_id,
                curr_shape_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse(DESIRED_POSE_ALPHA)),
            )

    def get_slider_body(self) -> DrakeRigidBody:
        slider_body = self._plant.GetUniqueFreeBaseBodyOrThrow(self.slider)
        return slider_body

    def get_slider_shapes(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(slider_body)

        inspector = self._scene_graph.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

        # for now we only support Box shapes
        assert all([isinstance(shape, DrakeBox) for shape in shapes])

        return shapes
      
    def get_slider_shape_poses(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self._plant.GetCollisionGeometriesForBody(slider_body)

        inspector = self._scene_graph.model_inspector()
        poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

        return poses
