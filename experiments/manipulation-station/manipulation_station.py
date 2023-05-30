import numpy as np
import pydot
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder

# Start meshcat
meshcat = StartMeshcat()  # type: ignore
meshcat.Delete()
meshcat.DeleteAddedControls()

builder = DiagramBuilder()

# Adds both MultibodyPlant and the SceneGraph, and wires them together.
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
# Note that we parse into both the plant and the scene_graph here.
iiwa_model = Parser(plant, scene_graph).AddModelsFromUrl(
    "package://drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
)[0]
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
plant.Finalize()

# Adds the MeshcatVisualizer and wires it to the SceneGraph.
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)  # type: ignore

# Adds an approximation of the iiwa controller.
# TODO(russt): replace this with the joint impedance controller.
kp = [100] * plant.num_positions()
ki = [1] * plant.num_positions()
kd = [20] * plant.num_positions()
iiwa_controller = builder.AddSystem(InverseDynamicsController(plant, kp, ki, kd, False))
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

q0 = np.array([-1.57, 0.1, 0.0, -1.2, 0.0, 1.6, 0.0])
x0 = np.hstack([q0, np.zeros_like(q0)])  # desired state

plant.SetPositions(plant_context, q0)
iiwa_controller.GetInputPort("desired_state").FixValue(
    iiwa_controller.GetMyMutableContextFromRoot(context), x0
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
