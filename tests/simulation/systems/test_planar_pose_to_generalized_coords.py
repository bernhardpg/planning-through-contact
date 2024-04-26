import pytest
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
)

from planning_through_contact.simulation.systems.planar_pose_to_generalized_coords import PlanarPoseToGeneralizedCoords
from planning_through_contact.geometry.planar.planar_pose import PlanarPose

@pytest.fixture
def planar_pose() -> PlanarPose:
    return PlanarPose(0.5, 0.25, 0.0)

def test_planar_position_to_rigid_transform(planar_pose: PlanarPose):
    # Build diagram and simulator
    builder = DiagramBuilder()
    
    system = builder.AddSystem(
        PlanarPoseToGeneralizedCoords(
            z_value=0.025, # Assumes objects are 5cm tall
            z_axis_is_positive=True,
        ),
    )
    pose_source = builder.AddSystem(
        ConstantVectorSource(planar_pose.vector())
    )
    
    builder.Connect(pose_source.get_output_port(0), system.get_input_port(0))
    builder.ExportOutput(system.get_output_port(0), "generalized_coords_output")
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    
    # Check output
    context = simulator.get_context()
    output = diagram.GetOutputPort("generalized_coords_output").Eval(context)
    desired_quat = [1.0, 0.0, 0.0, 0.0]
    desired_position = [0.5, 0.25, 0.025]
    desired_output = np.array(desired_quat + desired_position)
    
    assert output.shape == (7,)
    assert(np.allclose(output, desired_output))

