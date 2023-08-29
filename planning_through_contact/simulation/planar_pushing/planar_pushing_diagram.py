from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt
from pydrake.all import (
    InverseDynamicsController,
    MultibodyPlant,
    StateInterpolatorWithDiscreteDerivative,
)
from pydrake.geometry import Box as DrakeBox
from pydrake.geometry import Cylinder as DrakeCylinder
from pydrake.geometry import (
    GeometryInstance,
    MakePhongIllustrationProperties,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    Parser,
    ProcessModelDirectives,
)
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    ContactModel,
    DiscreteContactSolver,
)
from pydrake.multibody.tree import Frame
from pydrake.multibody.tree import RigidBody as DrakeRigidBody
from pydrake.systems.framework import Context, Diagram, DiagramBuilder
from pydrake.systems.primitives import Adder, Demultiplexer, PassThrough
from pydrake.visualization import AddDefaultVisualization

from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.visualize.colors import COLORS


@dataclass
class PlanarPushingSimConfig:
    body: Literal["box", "t_pusher"] = "box"
    contact_model: ContactModel = ContactModel.kHydroelasticWithFallback
    visualize_desired: bool = False
    goal_pose: Optional[PlanarPose] = None
    start_pose: PlanarPose = field(
        default_factory=lambda: PlanarPose(x=0.0, y=0.5, theta=0.0)
    )
    default_joint_positions: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array(
            [0.666, 1.039, -0.7714, -2.0497, 1.3031, 0.6729, -1.0252]
        )
    )
    time_step: float = 1e-3


class PlanarPushingDiagram(Diagram):
    def __init__(
        self,
        add_visualizer: bool = False,
        config: PlanarPushingSimConfig = PlanarPushingSimConfig(),
    ):
        Diagram.__init__(self)

        builder = DiagramBuilder()
        self.mbp, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=config.time_step
        )
        self.parser = Parser(self.mbp, self.scene_graph)

        # Load ROS packages starting from current folder
        # NOTE: Depends on there being a `package.xml` file in the models/ folder in order to load all the models.
        # TODO: Is this really the best way to do something like this?
        self.models_folder = Path(__file__).parents[1] / "models"
        self.parser.package_map().PopulateFromFolder(str(self.models_folder))

        use_hydroelastic = config.contact_model == ContactModel.kHydroelastic
        plant_file = (
            "planar_pushing_iiwa_plant_hydroelastic.yaml"
            if use_hydroelastic
            else "planar_pushing_iiwa_plant.yaml"
        )
        if not use_hydroelastic:
            raise NotImplementedError()

        directives = LoadModelDirectives(str(self.models_folder / plant_file))
        ProcessModelDirectives(directives, self.mbp, self.parser)  # type: ignore

        if config.body == "box":
            slider_sdf_url = "package://planning_through_contact/box_hydroelastic.sdf"
        elif config.body == "t_pusher":
            slider_sdf_url = "package://planning_through_contact/t_pusher.sdf"
        else:
            raise NotImplementedError(f"Body '{config.body}' not supported")

        (self.slider,) = self.parser.AddModels(url=slider_sdf_url)

        if use_hydroelastic:
            self.mbp.set_contact_model(ContactModel.kHydroelastic)
            self.mbp.set_discrete_contact_solver(DiscreteContactSolver.kSap)

        self.mbp.Finalize()

        # TODO: rename these
        # Get model instances for pusher and robot
        self.pusher = self.mbp.GetModelInstanceByName("pusher")
        self.iiwa = self.mbp.GetModelInstanceByName("iiwa")

        if config.visualize_desired:
            assert config.goal_pose is not None
            self._visualize_desired_slider_pose(config.goal_pose)

        if add_visualizer:
            self.meshcat = StartMeshcat()  # type: ignore
            self.meshcat.SetTransform(
                path="/Cameras/default",
                matrix=RigidTransform(
                    RollPitchYaw([-np.pi / 2 + 0.2, 0.0, np.pi]),  # type: ignore
                    np.array([0.0, 0.0, 0.0]),
                    # RollPitchYaw([-np.pi / 8, 0.0, np.pi / 2]),  # type: ignore
                    # 0.01 * np.array([0.05, 0.0, 0.1]),
                ).GetAsMatrix4(),
            )
            # self.visualizer = MeshcatVisualizer.AddToBuilder(builder, self.scene_graph, self.meshcat)  # type: ignore
            AddDefaultVisualization(builder, self.meshcat)

        self.add_controller(builder)

        # Export states
        builder.ExportOutput(self.mbp.get_body_poses_output_port(), "body_poses")
        builder.BuildInto(self)

    def _visualize_desired_slider_pose(self, desired_planar_pose: PlanarPose) -> None:
        source_id = self.scene_graph.RegisterSource()
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
            curr_shape_geometry_id = self.scene_graph.RegisterAnchoredGeometry(
                source_id,
                GeometryInstance(
                    desired_pose.multiply(pose),
                    shape,
                    f"shape_{idx}",
                ),
            )
            self.scene_graph.AssignRole(
                source_id,
                curr_shape_geometry_id,
                MakePhongIllustrationProperties(BOX_COLOR.diffuse(DESIRED_POSE_ALPHA)),
            )

    def add_controller(self, builder):
        self.controller_plant = MultibodyPlant(1e-3)
        parser = Parser(self.controller_plant)
        parser.package_map().PopulateFromFolder(str(self.models_folder))
        directives = LoadModelDirectives(
            str(self.models_folder / "iiwa_controller_plant.yaml")
        )
        ProcessModelDirectives(directives, self.controller_plant, parser)  # type: ignore
        self.controller_plant.Finalize()
        kp = 800 * np.ones(7)
        ki = 100 * np.ones(7)
        kd = 2 * np.sqrt(kp)
        arm_controller = builder.AddSystem(
            InverseDynamicsController(self.controller_plant, kp, ki, kd, False)
        )
        adder = builder.AddSystem(Adder(2, 7))
        state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(7, 1e-3, True)
        )
        arm_command = builder.AddSystem(PassThrough(7))
        state_split = builder.AddSystem(Demultiplexer(14, 7))

        # Export positions command
        builder.ExportInput(arm_command.get_input_port(0), "iiwa_position")
        builder.ExportOutput(arm_command.get_output_port(0), "iiwa_position_commanded")

        # Export arm state ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa), state_split.get_input_port(0)
        )
        builder.ExportOutput(state_split.get_output_port(0), "iiwa_position_measured")
        builder.ExportOutput(state_split.get_output_port(1), "iiwa_velocity_estimated")
        builder.ExportOutput(
            self.mbp.get_state_output_port(self.iiwa), "iiwa_state_measured"
        )

        # Export controller stack ports
        builder.Connect(
            self.mbp.get_state_output_port(self.iiwa),
            arm_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            arm_controller.get_output_port_control(), adder.get_input_port(0)
        )
        builder.Connect(
            adder.get_output_port(0),
            self.mbp.get_actuation_input_port(self.iiwa),
        )
        builder.Connect(
            state_from_position.get_output_port(0),
            arm_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            arm_command.get_output_port(0), state_from_position.get_input_port(0)
        )

        builder.ExportInput(adder.get_input_port(1), "iiwa_feedforward_torque")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_commanded")
        builder.ExportOutput(adder.get_output_port(0), "iiwa_torque_measured")

        builder.ExportOutput(
            self.mbp.get_generalized_contact_forces_output_port(self.iiwa),
            "iiwa_torque_external",
        )

    def get_slider_body(self) -> DrakeRigidBody:
        slider_body = self.mbp.GetUniqueFreeBaseBodyOrThrow(self.slider)
        return slider_body

    def get_slider_shapes(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self.mbp.GetCollisionGeometriesForBody(slider_body)

        inspector = self.scene_graph.model_inspector()
        shapes = [inspector.GetShape(id) for id in collision_geometries_ids]

        # for now we only support Box shapes
        assert all([isinstance(shape, DrakeBox) for shape in shapes])

        return shapes

    def get_slider_shape_poses(self) -> List[DrakeBox]:
        slider_body = self.get_slider_body()
        collision_geometries_ids = self.mbp.GetCollisionGeometriesForBody(slider_body)

        inspector = self.scene_graph.model_inspector()
        poses = [inspector.GetPoseInFrame(id) for id in collision_geometries_ids]

        return poses

    def get_slider_planar_pose(self, context: Context):
        pose = self.mbp.GetFreeBodyPose(context, self.get_slider_body())
        planar_pose = PlanarPose.from_pose(pose)
        return planar_pose

    def get_pusher_shape(self) -> DrakeCylinder:
        pusher_body = self.mbp.GetBodyByName("pusher")
        collision_geometry_id = self.mbp.GetCollisionGeometriesForBody(pusher_body)[0]

        inspector = self.scene_graph.model_inspector()
        pusher_shape = inspector.GetShape(collision_geometry_id)

        return pusher_shape

    @property
    def pusher_frame(self) -> Frame:
        P = self.mbp.GetFrameByName("pusher_base")
        return P

    def get_pusher_planar_pose(self, context: Context):
        pose = self.mbp.CalcRelativeTransform(
            context, self.mbp.world_frame(), self.pusher_frame
        )
        planar_pose = PlanarPose.from_pose(pose)
        return planar_pose
