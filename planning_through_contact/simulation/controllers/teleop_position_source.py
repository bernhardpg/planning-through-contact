from typing import Any, Dict

from manipulation.meshcat_utils import MeshcatSliders
from pydrake.all import DiagramBuilder, Meshcat, Multiplexer

from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)


class TeleopPositionSource(DesiredPlanarPositionSourceBase):
    """A teleop input source for desired positions"""

    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        teleop_config: Dict[str, Any],
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._teleop_config = teleop_config
        self._meshcat = meshcat
        input_limit = self._teleop_config["input_limit"]
        step = self._teleop_config["step_size"]
        robot_starting_translation = self._teleop_config["start_translation"]

        builder = DiagramBuilder()

        self._meshcat.AddSlider(
            "x",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=robot_starting_translation[0],
        )
        self._meshcat.AddSlider(
            "y",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=robot_starting_translation[1],
        )
        force_system = builder.AddSystem(MeshcatSliders(self._meshcat, ["x", "y"]))
        mux = builder.AddNamedSystem("teleop_mux", Multiplexer(2))
        builder.Connect(force_system.get_output_port(0), mux.get_input_port(0))
        builder.Connect(force_system.get_output_port(1), mux.get_input_port(1))

        ## Export outputs
        builder.ExportOutput(mux.get_output_port(), "planar_position_command")

        builder.BuildInto(self)
