import numpy as np
import pydot
from pydrake.common import FindResourceOrThrow
from pydrake.examples import ManipulationStation
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder


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


def manipulation_station():
    # Start meshcat
    meshcat = StartMeshcat()  # type: ignore
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    station = ManipulationStation()
    plant = station.get_mutable_multibody_plant()
    scene_graph = station.get_mutable_scene_graph()

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
    breakpoint()


if __name__ == "__main__":
    manipulation_station()
    # simple_iiwa_and_brick()
