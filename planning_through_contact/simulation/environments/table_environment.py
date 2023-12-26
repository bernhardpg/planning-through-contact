import copy
from typing import List, Tuple, Optional

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    RotationMatrix,
    MeshcatVisualizer,
    Simulator,
    MeshcatVisualizerParams,
    Role,
    LogVectorOutput,
    Rgba,
    Box,
    ContactModel,
    LoadModelDirectives,
    ProcessModelDirectives,
    DiscreteContactSolver,
    AddDefaultVisualization,
    RollPitchYaw,
)
import numpy as np
import matplotlib.pyplot as plt
from planning_through_contact.simulation.controllers.desired_position_source_base import DesiredPositionSourceBase

from planning_through_contact.simulation.controllers.position_controller_base import PositionControllerBase
from planning_through_contact.simulation.planar_pushing.planar_pushing_diagram import PlanarPushingSimConfig
from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.t_pusher_2d import TPusher2d
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.rigid_body import RigidBody
from planning_through_contact.planning.planar.planar_plan_config import (
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.controllers.hybrid_mpc import HybridMpcConfig
from planning_through_contact.visualize.analysis import plot_planar_pushing_logs_from_pose_vectors
from planning_through_contact.visualize.colors import COLORS
from planning_through_contact.simulation.sim_utils import ConfigureParser, models_folder

class TableEnvironment():
    def __init__(
        self,
        desired_position_source: DesiredPositionSourceBase,
        position_controller: PositionControllerBase,
        sim_config: PlanarPushingSimConfig,
    ):
        self._desired_position_source = desired_position_source
        self._position_controller = position_controller
        self._sim_config = sim_config
        self._meshcat = None
        self._simulator = None

    def setup(self, meshcat=None) -> None:
        if meshcat is None:
            self._meshcat = StartMeshcat()
        else:
            self._meshcat = meshcat

        builder = DiagramBuilder()
        self.mbp, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._sim_config.time_step
        )
        self.parser = Parser(self.mbp, self.scene_graph)
        ConfigureParser(self.parser)
        use_hydroelastic = self._sim_config.contact_model == ContactModel.kHydroelastic
        
        if not use_hydroelastic:
            raise NotImplementedError()

        directives = LoadModelDirectives(f"{models_folder}/{self._sim_config.scene_directive_name}")
        ProcessModelDirectives(directives, self.mbp, self.parser)  # type: ignore

        if isinstance(self._sim_config.slider.geometry, Box2d):
            body_name = "box"
            slider_sdf_url = "package://planning_through_contact/box_hydroelastic.sdf"
        elif isinstance(self._sim_config.slider.geometry, TPusher2d):
            body_name = "t_pusher"
            slider_sdf_url = "package://planning_through_contact/t_pusher.sdf"
        else:
            raise NotImplementedError(f"Body '{self._sim_config.slider}' not supported")

        (self.slider,) = self.parser.AddModels(url=slider_sdf_url)

        if use_hydroelastic:
            self.mbp.set_contact_model(ContactModel.kHydroelastic)
            self.mbp.set_discrete_contact_solver(DiscreteContactSolver.kSap)

        self.mbp.Finalize()

        # self._meshcat.SetTransform(
        #     path="/Cameras/default",
        #     matrix=RigidTransform(
        #         # RollPitchYaw([-np.pi / 2 + 0.2, 0.0, np.pi]),  # type: ignore
        #         # np.array([0.0, 0.0, 0.0]),
        #         RollPitchYaw([-np.pi / 8, 0.0, np.pi]),  # type: ignore
        #         np.array([-1, -1.0, -1]),
        #     ).GetAsMatrix4(),
        # )
        AddDefaultVisualization(builder, self._meshcat)

        # Set up position controller
        self._position_controller.add_meshcat(self._meshcat)
        desired_state_source = self._position_controller.setup(builder, self.mbp)

        # Set up desired position source
        self._desired_position_source.add_meshcat(self._meshcat)
        desired_position_source = self._desired_position_source.setup(builder, self.mbp)

        builder.Connect(
            desired_position_source.get_output_port(),
            desired_state_source.get_input_port(),
        )

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        self.context = self._simulator.get_mutable_context()
        self.mbp_context = self.mbp.GetMyContextFromRoot(self.context)
        self.set_slider_planar_pose(self._sim_config.slider_start_pose)


    def _draw_object(
        self, name: str, x: np.array, color: Rgba = Rgba(0, 1, 0, 1.0)
    ) -> None:
        # Assumes x = [x, y]
        pose = RigidTransform(RotationMatrix(), [*x, 0])
        self._meshcat.SetObject(name, Box(1, 1, 0.3), rgba=color)
        self._meshcat.SetTransform(name, pose)
    
    def set_slider_planar_pose(self, pose: PlanarPose):
        min_height = 0.05

        # add a small height to avoid the box penetrating the table
        q = pose.to_generalized_coords(min_height + 1e-2, z_axis_is_positive=True)
        self.mbp.SetPositions(self.mbp_context, self.slider, q)

    def simulate(self, timeout=1e8, save_recording_as: Optional[str] = None) -> None:
        """
        :return: Returns a tuple of (success, simulation_time_s).
        """
        if save_recording_as:
            self._meshcat.StartRecording()
        self._simulator.AdvanceTo(timeout)
        if save_recording_as:
            self._meshcat.StopRecording()
            self._meshcat.SetProperty("/drake/contact_forces", "visible", False)
            self._meshcat.PublishRecording()
            res = self._meshcat.StaticHtml()
            with open(save_recording_as, "w") as f:
                f.write(res)
        # if self.config.save_plots:
        #     pusher_pose_log = self._pusher_pose_logger.FindLog(self.context)
        #     slider_pose_log = self._slider_pose_logger.FindLog(self.context)
        #     pusher_pose_desired_log = self._pusher_pose_desired_logger.FindLog(self.context)
        #     slider_pose_desired_log = self._slider_pose_desired_logger.FindLog(self.context)
        #     plot_planar_pushing_logs_from_pose_vectors(pusher_pose_log, slider_pose_log, pusher_pose_desired_log, slider_pose_desired_log)

    def _visualize_logs(self) -> None:
        context = self._simulator.get_mutable_context()
        action_log = self._action_logger.FindLog(context)
        state_log = self._state_logger.FindLog(context)
        self._plot_logs(state_log, action_log)

    # Not being used to generate data, only for debugging
    def _plot_logs(self, state_log, action_log) -> None:
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        axis = axs[0]
        axis.step(state_log.sample_times(), state_log.data().transpose()[:, :4])
        axis.legend([r"$q_{bx}$", r"$q_{by}$", r"$q_{fx}$", r"$q_{fy}$"])
        axis.set_ylabel("Positions")
        axis.set_xlabel("t")

        axis = axs[1]
        axis.step(state_log.sample_times(), state_log.data().transpose()[:, 4:])
        axis.legend([r"$v_{bx}$", r"$v_{by}$", r"$v_{fx}$", r"$v_{fy}$"])
        axis.set_ylabel("Velocities")
        axis.set_xlabel("t")

        axis = axs[2]
        axis.step(action_log.sample_times(), action_log.data().transpose())
        axis.legend([r"$u_x$", r"$u_y$"])
        axis.set_ylabel("u")
        axis.set_xlabel("t")
        plt.show()

        def __del__(self):
            self._meshcat.Delete()
            self._meshcat.DeleteAddedControls()
