from typing import Tuple, Optional

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    Simulator,
    LogVectorOutput,
    ContactModel,
    LoadModelDirectives,
    ProcessModelDirectives,
    DiscreteContactApproximation,
    AddDefaultVisualization,
    Meshcat,
    RollPitchYaw,
    Demultiplexer,
    ModelInstanceIndex,
)
import numpy as np
from planning_through_contact.simulation.controllers.desired_position_source_base import (
    DesiredPositionSourceBase,
)

from planning_through_contact.simulation.controllers.position_controller_base import (
    PositionControllerBase,
)
from planning_through_contact.simulation.controllers.teleop_position_source import (
    TeleopPositionSource,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import (
    PlanarPushingSimConfig,
)
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.simulation.state_estimators.state_estimator import (
    StateEstimator,
)
from planning_through_contact.simulation.systems.rigid_transform_to_planar_pose_vector_system import (
    RigidTransformToPlanarPoseVectorSystem,
)
from planning_through_contact.visualize.analysis import (
    plot_planar_pushing_logs_from_pose_vectors,
)
from planning_through_contact.simulation.sim_utils import ConfigureParser, models_folder


class TableEnvironment:
    def __init__(
        self,
        desired_position_source: DesiredPositionSourceBase,
        position_controller: PositionControllerBase,
        sim_config: PlanarPushingSimConfig,
        meshcat: Optional[Meshcat] = None,
    ):
        self._desired_position_source = desired_position_source
        self._position_controller = position_controller
        self._sim_config = sim_config
        self._meshcat = None
        self._simulator = None

        if meshcat is None:
            self._meshcat = StartMeshcat()
        else:
            self._meshcat = meshcat

        builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._sim_config.time_step
        )
        slider_name, self.slider = self.add_all_directives(
            self._plant, self._scene_graph
        )

        self._meshcat.SetTransform(
            path="/Cameras/default",
            matrix=RigidTransform(
                RollPitchYaw([0.0, 0.0, np.pi / 2]),  # type: ignore
                np.array([1, 0, 0]),
            ).GetAsMatrix4(),
        )
        AddDefaultVisualization(builder, self._meshcat)

        # Set up state estimator
        self._state_estimator = builder.AddNamedSystem(
            "state_estimator",
            StateEstimator(
                sim_config=sim_config, environment=self, add_visualizer=True
            ),
        )

        # Set up position controller
        self._position_controller.add_meshcat(self._meshcat)
        desired_state_source = self._position_controller.AddToBuilder(
            builder=builder,
            state_estimator=self._state_estimator,
            station_plant=self._plant,
        )

        # Set up desired position source
        self._desired_position_source.add_meshcat(self._meshcat)
        desired_position_source_output_port = (
            self._desired_position_source.AddToBuilder(
                builder, state_estimator=self._state_estimator
            )
        )

        # This is only when not using hardware (fully simulated)
        # Note this name will be wrong when we transition to handling the iiwa as the robot as well
        robot_model_instance = self._plant.GetModelInstanceByName("pusher")
        # Connections to update the robot state within state estimator
        builder.Connect(
            self._plant.get_state_output_port(robot_model_instance),
            self._state_estimator.GetInputPort("robot_state"),
        )
        # Connections to update the object position within state estimator
        slider_demux = builder.AddSystem(
            Demultiplexer(
                [
                    self._plant.num_positions(self.slider),
                    self._plant.num_velocities(self.slider),
                ]
            )
        )
        builder.Connect(
            self._plant.get_state_output_port(self.slider),
            slider_demux.get_input_port(),
        )
        builder.Connect(
            slider_demux.get_output_port(0),
            self._state_estimator.GetInputPort("object_position"),
        )

        # Connection to update the desired position within the position controller
        builder.Connect(
            desired_position_source_output_port,
            desired_state_source.get_input_port(),
        )

        # Will break if save plots during teleop
        if sim_config.save_plots:
            assert not isinstance(
                self._desired_position_source, TeleopPositionSource
            ), "Cannot save plots during teleop"
            # Actual State Loggers
            pusher_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("pusher_pose"),
                pusher_pose_to_vector.get_input_port(),
            )
            pusher_pose_logger = LogVectorOutput(
                pusher_pose_to_vector.get_output_port(), builder
            )
            slider_pose_to_vector = builder.AddSystem(
                RigidTransformToPlanarPoseVectorSystem()
            )
            builder.Connect(
                self._state_estimator.GetOutputPort("slider_pose"),
                slider_pose_to_vector.get_input_port(),
            )
            slider_pose_logger = LogVectorOutput(
                slider_pose_to_vector.get_output_port(), builder
            )
            # Desired State Loggers
            pusher_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.planar_pose_pub.GetOutputPort(
                    "desired_pusher_planar_pose_vector"
                ),
                builder,
            )
            slider_pose_desired_logger = LogVectorOutput(
                self._desired_position_source.planar_pose_pub.GetOutputPort(
                    "desired_slider_planar_pose_vector"
                ),
                builder,
            )

            self._pusher_pose_logger = pusher_pose_logger
            self._slider_pose_logger = slider_pose_logger
            self._pusher_pose_desired_logger = pusher_pose_desired_logger
            self._slider_pose_desired_logger = slider_pose_desired_logger

        diagram = builder.Build()
        self._diagram = diagram

        self._simulator = Simulator(diagram)
        if sim_config.use_realtime:
            self._simulator.set_target_realtime_rate(1.0)

        self.context = self._simulator.get_mutable_context()
        self.mbp_context = self._plant.GetMyContextFromRoot(self.context)
        self.set_slider_planar_pose(self._sim_config.slider_start_pose)

        self._plant.SetDefaultPositions(
            robot_model_instance, self._sim_config.pusher_start_pose.pos()
        )
        self._plant.SetPositions(
            self.mbp_context,
            robot_model_instance,
            self._sim_config.pusher_start_pose.pos(),
        )
        self._state_estimator._plant.SetDefaultPositions(
            robot_model_instance, self._sim_config.pusher_start_pose.pos()
        )

    def export_diagram(self, filename: str):
        import pydot

        pydot.graph_from_dot_data(self._diagram.GetGraphvizString())[0].write_pdf(  # type: ignore
            filename
        )
        print(f"Saved diagram to: {filename}")

    def add_all_directives(self, plant, scene_graph) -> Tuple[str, ModelInstanceIndex]:
        parser = Parser(plant, scene_graph)
        ConfigureParser(parser)
        use_hydroelastic = self._sim_config.contact_model == ContactModel.kHydroelastic

        if not use_hydroelastic:
            raise NotImplementedError()

        directives = LoadModelDirectives(
            f"{models_folder}/{self._sim_config.scene_directive_name}"
        )
        ProcessModelDirectives(directives, plant, parser)  # type: ignore

        if isinstance(self._sim_config.slider.geometry, Box2d):
            body_name = "box"
            slider_sdf_url = "package://planning_through_contact/box_hydroelastic.sdf"
        elif isinstance(self._sim_config.slider.geometry, TPusher2d):
            body_name = "t_pusher"
            slider_sdf_url = "package://planning_through_contact/t_pusher.sdf"
        else:
            raise NotImplementedError(f"Body '{self._sim_config.slider}' not supported")

        (slider,) = parser.AddModels(url=slider_sdf_url)

        if use_hydroelastic:
            plant.set_contact_model(ContactModel.kHydroelastic)
            plant.set_discrete_contact_approximation(
                DiscreteContactApproximation.kLagged
            )

        plant.Finalize()
        return body_name, slider

    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = 0.05

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self._plant.SetPositions(self.mbp_context, self.slider, q)

    def simulate(self, timeout=1e8, save_recording_as: Optional[str] = None) -> None:
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        if save_recording_as:
            self._state_estimator.meshcat.StartRecording()
            self._meshcat.StartRecording()
        time_step = self._sim_config.time_step * 100
        if not isinstance(self._desired_position_source, TeleopPositionSource):
            for t in np.append(np.arange(0, timeout, time_step), timeout):
                self._simulator.AdvanceTo(t)
                # Hacky way of visualizing the desired slider pose
                context = (
                    self._desired_position_source.planar_pose_pub.GetMyContextFromRoot(
                        self.context
                    )
                )
                slider_desired_pose_vec = (
                    self._desired_position_source.planar_pose_pub.GetOutputPort(
                        "desired_slider_planar_pose_vector"
                    ).Eval(context)
                )
                self._state_estimator._visualize_desired_slider_pose(
                    PlanarPose(*slider_desired_pose_vec)
                )
        else:
            self._simulator.AdvanceTo(timeout)
        if save_recording_as:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()

            self._state_estimator.meshcat.StopRecording()
            self._state_estimator.meshcat.SetProperty(
                "/drake/contact_forces", "visible", False
            )
            self._state_estimator.meshcat.PublishRecording()
            res = self._state_estimator.meshcat.StaticHtml()
            with open(save_recording_as, "w") as f:
                f.write(res)
        if self._sim_config.save_plots:
            pusher_pose_log = self._pusher_pose_logger.FindLog(self.context)
            slider_pose_log = self._slider_pose_logger.FindLog(self.context)
            pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(
                self.context
            )
            slider_pose_desired_log = self._slider_pose_desired_logger.FindLog(
                self.context
            )
            plot_planar_pushing_logs_from_pose_vectors(
                pusher_pose_log,
                slider_pose_log,
                pusher_pose_desired_log,
                slider_pose_desired_log,
            )
