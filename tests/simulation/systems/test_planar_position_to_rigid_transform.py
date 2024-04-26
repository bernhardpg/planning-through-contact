import pytest
import numpy as np
import numpy.typing as npt

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
    RigidTransform,
    Quaternion,
)

from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import PlanarTranslationToRigidTransformSystem

@pytest.fixture
def planar_position() -> npt.NDArray[np.float64]:
    return np.array([0.5, 0.25])

def test_planar_position_to_rigid_transform(planar_position: np.ndarray):
    # Build diagram and simulator
    builder = DiagramBuilder()
    
    system = builder.AddSystem(
        PlanarTranslationToRigidTransformSystem(z_dist=0.02)
    )
    vector_source = builder.AddSystem(
        ConstantVectorSource(planar_position)
    )
    
    builder.Connect(vector_source.get_output_port(0), system.get_input_port(0))
    builder.ExportOutput(system.get_output_port(0), "output")
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
    
    # Run simulation
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    
    # Check output
    context = simulator.get_context()
    output = diagram.GetOutputPort("output").Eval(context)
    
    assert np.allclose(output.GetAsMatrix4(), output.GetAsMatrix4())

