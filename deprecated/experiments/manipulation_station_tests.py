import numpy as np
import pydot
from pydrake.common import FindResourceOrThrow
from pydrake.examples import ManipulationStation
from pydrake.geometry import Meshcat, MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import FirstOrderLowPassFilter, VectorLogSink


class SchunkWsgButtons(LeafSystem):
    """
    Adds buttons to open/close the Schunk WSG gripper.

    .. pydrake_system::

        name: SchunkWsgButtons
        output_ports:
        - position
        - max_force
    """

    _BUTTON_NAME = "Open/Close Gripper"
    """The name of the button added to the meshcat UI."""

    def __init__(
        self, meshcat, open_position=0.107, closed_position=0.002, force_limit=40
    ):
        """ "
        Args:
            open_position:   Target position for the gripper when open.
            closed_position: Target position for the gripper when closed.
                             **Warning**: closing to 0mm can smash the fingers
                             together and keep applying force even when no
                             object is grasped.
            force_limit:     Force limit to send to Schunk WSG controller.
        """
        super().__init__()
        self.meshcat = meshcat
        self.DeclareVectorOutputPort("position", 1, self.CalcPositionOutput)
        self.DeclareVectorOutputPort("force_limit", 1, self.CalcForceLimitOutput)
        self._open_button = meshcat.AddButton(self._BUTTON_NAME)
        self._open_position = open_position
        self._closed_position = closed_position
        self._force_limit = force_limit

    def CalcPositionOutput(self, context, output):
        if self.meshcat.GetButtonClicks(name=self._BUTTON_NAME) % 2 == 0:
            output.SetAtIndex(0, self._open_position)
        else:
            output.SetAtIndex(0, self._closed_position)

    def CalcForceLimitOutput(self, context, output):
        output.SetAtIndex(0, self._force_limit)


def teleop():
    """
    Taken from https://github.com/RobotLocomotion/drake/blob/3fe033247b8db6fa6559ab61be28a250892f9268/examples/manipulation_station/joint_teleop.py#L67
    as an example for how to set up a manipulation station.
    """
    builder = DiagramBuilder()

    # NOTE: the meshcat instance is always created in order to create the
    # teleop controls (joint sliders and open/close gripper button).  When
    # args.hardware is True, the meshcat server will *not* display robot
    # geometry, but it will contain the joint sliders and open/close gripper
    # button in the "Open Controls" tab in the top-right of the viewing server.
    meshcat = Meshcat()

    station = builder.AddSystem(ManipulationStation())

    # Initializes the chosen station type.
    station.SetupManipulationClassStation()
    station.AddManipulandFromFile(
        "drake/examples/manipulation_station/models/" + "061_foam_brick.sdf",
        RigidTransform(RotationMatrix.Identity(), [0.6, 0, 0]),  # type: ignore
    )

    station.Finalize()

    geometry_query_port = station.GetOutputPort("geometry_query")
    meshcat_visualizer = MeshcatVisualizer.AddToBuilder(  # type: ignore
        builder=builder, query_object_port=geometry_query_port, meshcat=meshcat
    )

    teleop = builder.AddSystem(
        JointSliders(meshcat=meshcat, plant=station.get_controller_plant())
    )

    num_iiwa_joints = station.num_iiwa_joints()
    filter = builder.AddSystem(
        FirstOrderLowPassFilter(time_constant=2.0, size=num_iiwa_joints)
    )
    builder.Connect(teleop.get_output_port(0), filter.get_input_port(0))
    builder.Connect(filter.get_output_port(0), station.GetInputPort("iiwa_position"))

    wsg_buttons = builder.AddSystem(SchunkWsgButtons(meshcat=meshcat))
    builder.Connect(
        wsg_buttons.GetOutputPort("position"), station.GetInputPort("wsg_position")
    )
    builder.Connect(
        wsg_buttons.GetOutputPort("force_limit"),
        station.GetInputPort("wsg_force_limit"),
    )

    iiwa_velocities = builder.AddSystem(VectorLogSink(num_iiwa_joints))
    builder.Connect(
        station.GetOutputPort("iiwa_velocity_estimated"),
        iiwa_velocities.get_input_port(0),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context()
    )

    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(num_iiwa_joints)
    )

    # Eval the output port once to read the initial positions of the IIWA.
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(station_context)
    teleop.SetPositions(q0)
    filter.set_initial_output_value(
        diagram.GetMutableSubsystemContext(filter, simulator.get_mutable_context()), q0
    )

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(np.inf)


def simple_iiwa_and_brick():
    # Start meshcat
    meshcat = StartMeshcat()  # type: ignore
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()

    # Adds both MultibodyPlant and the SceneGraph, and wires them together.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    # Note that we parse into both the plant and the scene_graph here.
    iiwa_url = "package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
    iiwa_model = Parser(plant, scene_graph).AddModelsFromUrl(iiwa_url)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    brick_model = Parser(plant, scene_graph).AddModelsFromUrl(
        "package://drake/examples/manipulation_station/models/061_foam_brick.sdf"
    )[0]

    # Create a separate plant with only the robot for creating the controller
    control_plant = MultibodyPlant(time_step=1e-4)
    Parser(control_plant).AddModelsFromUrl(iiwa_url)
    control_plant.WeldFrames(
        control_plant.world_frame(), control_plant.GetFrameByName("iiwa_link_0")
    )

    plant.Finalize()
    control_plant.Finalize()

    # Adds the MeshcatVisualizer and wires it to the SceneGraph.
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)  # type: ignore

    # From:
    # https://deepnote.com/workspace/Manipulation-ac8201a1-470a-4c77-afd0-2cc45bc229ff/project/02-Lets-get-you-a-robot-8f86172b-b597-4ceb-9bad-92d11ac7a6cc/notebook/simulation-1ba6290623e34dbbb9d822a2180187c1
    num_positions = control_plant.num_positions()
    kp = np.array([100.0] * num_positions)
    ki = np.array([1.0] * num_positions)
    kd = np.array([20.0] * num_positions)
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(control_plant, kp, ki, kd, False)
    )

    iiwa_controller.set_name("iiwa_controller")

    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        iiwa_controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        iiwa_controller.get_output_port_control(), plant.get_actuation_input_port()
    )

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    iiwa_q0 = np.array([-1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])
    iiwa_x0 = np.hstack([iiwa_q0, np.zeros_like(iiwa_q0)])  # desired state

    brick_pos_0 = np.array([2, 2, 2])
    brick_rot_0 = RotationMatrix.MakeXRotation(np.pi / 2)

    def _quat_to_vec(quat):
        return np.array([quat.w(), quat.x(), quat.y(), quat.z()])

    def _rot_to_quat_vec(rot):
        return _quat_to_vec(rot.ToQuaternion())

    plant.SetPositions(plant_context, iiwa_model, iiwa_q0)
    plant.SetPositions(
        plant_context,
        brick_model,
        np.concatenate([_rot_to_quat_vec(brick_rot_0), brick_pos_0]),
    )
    iiwa_controller.GetInputPort("desired_state").FixValue(
        iiwa_controller.GetMyMutableContextFromRoot(context), iiwa_x0
    )

    pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_svg(
        "deleteme/graph_test.svg"
    )

    pydot.graph_from_dot_data(plant.GetTopologyGraphvizString())[0].write_svg(
        "deleteme/topology.svg"
    )

    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(5.0)

    while True:
        ...


# Good reference:
# https://github.com/RobotLocomotion/drake/blob/3fe033247b8db6fa6559ab61be28a250892f9268/examples/manipulation_station/joint_teleop.py#L67


def manipulation_station():
    # Start meshcat
    meshcat = StartMeshcat()  # type: ignore
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()

    station = builder.AddSystem(ManipulationStation(time_step=1e-4))
    plant = station.get_mutable_multibody_plant()
    scene_graph = station.get_mutable_scene_graph()

    # Add table
    table_url = (
        "package://drake/examples/manipulation_station/models/"
        "amazon_table_simplified.sdf"
    )
    table_model = Parser(plant, scene_graph).AddModelsFromUrl(table_url)[0]

    dx_table_center_to_robot_base = 0.3257
    dz_table_top_robot_base = 0.0127
    X_WT = RigidTransform(
        np.array([dx_table_center_to_robot_base, 0, -dz_table_top_robot_base])
    )

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("amazon_table"), X_WT)

    #
    breakpoint()

    # station.SetupManipulationClassStation()
    station.AddManipulandFromFile(
        "drake/examples/manipulation_station/models/" + "061_foam_brick.sdf",
        RigidTransform(RotationMatrix.Identity(), [0.6, 0, 0]),  # type: ignore
    )

    # I may want to use this instead at some point
    SETUP_MANUALLY = False
    if SETUP_MANUALLY:
        plant = station.get_mutable_multibody_plant()

        parser = Parser(plant, scene_graph)

        # Load iiwa model
        iiwa_model_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/iiwa7/" "iiwa7_no_collision.sdf"
        )
        (iiwa,) = parser.AddModels(iiwa_model_file)
        X_WI = RigidTransform.Identity()  # type: ignore

        # Weld iiwa to world frame
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa), X_WI
        )

        # Load wsg model
        wsg_model_file = FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/sdf/" "schunk_wsg_50.sdf"
        )
        (wsg,) = parser.AddModels(wsg_model_file)
        X_7G = RigidTransform.Identity()  # type: ignore

        # Weld gripper to iiwa
        plant.WeldFrames(
            plant.GetFrameByName("iiwa_link_7", iiwa),
            plant.GetFrameByName("body", wsg),
            X_7G,
        )

        # Register models for the controller
        station.RegisterIiwaControllerModel(
            iiwa_model_file,
            iiwa,
            plant.world_frame(),
            plant.GetFrameByName("iiwa_link_0", iiwa),
            X_WI,
        )
        station.RegisterWsgControllerModel(
            wsg_model_file,
            wsg,
            plant.GetFrameByName("iiwa_link_7", iiwa),
            plant.GetFrameByName("body", wsg),
            X_7G,
        )

    # Finalize
    station.Finalize()

    # Visualization
    geometry_query_port = station.GetOutputPort("geometry_query")
    visualizer = MeshcatVisualizer.AddToBuilder(builder, query_object_port=geometry_query_port, meshcat=meshcat)  # type: ignore

    diagram = builder.Build()

    simulator = Simulator(diagram)
    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context()
    )

    # Fix gripper input value
    station.GetInputPort("wsg_position").FixValue(station_context, 0)

    breakpoint()
    # pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].write_png(
    #     "deleteme/diagram.png"
    # )
    #
    # pydot.graph_from_dot_data(plant.GetTopologyGraphvizString())[0].write_png(
    #     "deleteme/topology.png"
    # )

    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(5.0)

    while True:
        ...
    breakpoint()
