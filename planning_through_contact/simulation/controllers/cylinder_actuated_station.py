import numpy as np
from typing import List, Optional

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RollPitchYaw,
    AddDefaultVisualization,
    Meshcat,
    Box as DrakeBox,
    RigidBody as DrakeRigidBody,
    GeometryInstance,
    MakePhongIllustrationProperties,
    Rgba,
)

from pydrake.math import (
    RotationMatrix
)

from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)

from planning_through_contact.geometry.planar.planar_pose import PlanarPose

from .robot_system_base import RobotSystemBase
from planning_through_contact.simulation.sim_utils import (
    GetParser,
    AddSliderAndConfigureContact,
)

from planning_through_contact.visualize.colors import COLORS



class CylinderActuatedStation(RobotSystemBase):
    """Base controller class for an actuated floating cylinder robot."""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        self._pid_gains = dict(kp=1600, ki=100, kd=50)
        self._num_positions = 2  # Number of dimensions for robot position
        self._goal_geometries = []

        builder = DiagramBuilder()

        # "Internal" plant for the robot controller
        robot_controller_plant = MultibodyPlant(time_step=self._sim_config.time_step)
        parser = GetParser(robot_controller_plant)
        parser.AddModelsFromUrl(
            "package://planning_through_contact/pusher_floating_hydroelastic_actuated.sdf"
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        # "External" station plant
        self.station_plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._sim_config.time_step
        )
        self.slider = AddSliderAndConfigureContact(
            sim_config, self.station_plant, self._scene_graph
        )

        # self._meshcat.SetTransform(
        #     path="/Cameras/default",
        #     matrix=RigidTransform(
        #         RollPitchYaw([0.0, 0.0, np.pi / 2]),  # type: ignore
        #         np.array([1, 0, 0]),
        #     ).GetAsMatrix4(),
        # )
        # Set the initial camera pose
        zoom = 1.8
        camera_in_world = [0.5, -1/zoom, 1.5/zoom]
        target_in_world = [0.5, 0, 0]
        self._meshcat.SetCameraPose(camera_in_world, target_in_world)
        AddDefaultVisualization(builder, self._meshcat)

        ## Add Leaf systems

        robot_controller = builder.AddNamedSystem(
            "RobotController",
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._pid_gains["kp"]] * self._num_positions,
                ki=[self._pid_gains["ki"]] * self._num_positions,
                kd=[self._pid_gains["kd"]] * self._num_positions,
                has_reference_acceleration=False,
            ),
        )

        # Add system to convert desired position to desired position and velocity.
        desired_state_source = builder.AddNamedSystem(
            "DesiredStateSource",
            StateInterpolatorWithDiscreteDerivative(
                self._num_positions,
                self._sim_config.time_step,
                suppress_initial_transient=True,
            ),
        )

        if sim_config.camera_config is not None:
            from pydrake.systems.sensors import (
                ApplyCameraConfig
            )

            ApplyCameraConfig(
                config=sim_config.camera_config,
                builder=builder
            )

        ## Connect systems

        self._robot_model_instance = self.station_plant.GetModelInstanceByName(
            self.robot_model_name
        )
        builder.Connect(
            robot_controller.get_output_port_control(),
            self.station_plant.get_actuation_input_port(self._robot_model_instance),
        )

        builder.Connect(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            robot_controller.get_input_port_estimated_state(),
        )

        builder.Connect(
            desired_state_source.get_output_port(),
            robot_controller.get_input_port_desired_state(),
        )

        ## Export inputs and outputs

        builder.ExportInput(
            desired_state_source.get_input_port(),
            "planar_position_command",
        )

        builder.ExportOutput(
            self.station_plant.get_state_output_port(self._robot_model_instance),
            "robot_state_measured",
        )

        # Only relevant when use_hardware=False
        # If use_hardware=True, this info will be updated by the optitrack system in the state estimator directly
        builder.ExportOutput(
            self.station_plant.get_state_output_port(self.slider),
            "object_state_measured",
        )

        if self._sim_config.camera_config:
            builder.ExportOutput(
                builder.GetSubsystemByName(
                    "rgbd_sensor_overhead_camera"
                ).color_image_output_port(),
                "rgbd_sensor_overhead_camera",
            )

        builder.BuildInto(self)

        ## Set default position for the robot
        self.station_plant.SetDefaultPositions(
            self._robot_model_instance, self._sim_config.pusher_start_pose.pos()
        )

    @property
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        return "pusher"
    
    @property
    def slider_model_name(self) -> str:
        """The name of the robot model."""
        return "t_pusher"

    ## Visualization functions

    def get_slider_shapes(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self.station_plant.GetCollisionGeometriesForBody(
            slider_body
        )

        inspector = self._scene_graph.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

        # for now we only support Box shapes
        assert all([isinstance(shape, DrakeBox) for shape in shapes])

        return shapes
    
    def get_slider_shape_poses(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self.station_plant.GetCollisionGeometriesForBody(
            slider_body
        )

        inspector = self._scene_graph.model_inspector()
        poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

        return poses
    
    def get_slider_body(self) -> DrakeRigidBody:
        slider_body = self.station_plant.GetUniqueFreeBaseBodyOrThrow(self.slider)
        return slider_body
    
    def _visualize_desired_slider_pose(
        self, desired_planar_pose: PlanarPose, 
        time_in_recording: float = 0.0,
        scale_factor: float = 1.0
    ) -> None:
        actual_shapes = self.get_slider_shapes()
        actual_poses = self.get_slider_shape_poses()

        shapes = []
        poses = []
        if scale_factor != 1.0:
            for (shape, pose) in zip(actual_shapes, actual_poses):
                shapes.append(
                    DrakeBox(
                        shape.width() * scale_factor,
                        shape.depth() * scale_factor,
                        shape.height()
                    )
                )
                translation = pose.translation()
                new_translation = np.array([
                    translation[0] * scale_factor, 
                    translation[1] * scale_factor, 
                    translation[2]]
                )
                poses.append(RigidTransform(pose.rotation(), new_translation))
        else:
            shapes = actual_shapes
            poses = actual_poses

        heights = [shape.height() for shape in shapes]
        min_height = min(heights)
        desired_pose = desired_planar_pose.to_pose(
            min_height / 2, z_axis_is_positive=True
        )
        if len(self._goal_geometries) == 0:
            source_id = self._scene_graph.RegisterSource()
            BOX_COLOR = COLORS["emeraldgreen"]
            DESIRED_POSE_ALPHA = 0.3
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
                self._meshcat.SetObject(
                    geom_name, shape, rgba=Rgba(*BOX_COLOR.diffuse(DESIRED_POSE_ALPHA))
                )
        else:
            for pose, geom_name in zip(poses, self._goal_geometries):
                self._meshcat.SetTransform(
                    geom_name, desired_pose.multiply(pose), time_in_recording
                )