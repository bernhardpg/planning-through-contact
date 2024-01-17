import numpy as np

from manipulation.station import (
    MakeHardwareStation,
    LoadScenario,
    Scenario,
    JointStiffnessDriver,
    IiwaDriver,
)

from pydrake.all import (
    DiagramBuilder,
    Context,
    StateInterpolatorWithDiscreteDerivative,
    ConstantValueSource,
    AbstractValue,
    Meshcat,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    PortSwitch,
    Demultiplexer,
    Multiplexer,
)
from planning_through_contact.simulation.planar_pushing.iiwa_planner import IiwaPlanner

from planning_through_contact.simulation.planar_pushing.inverse_kinematics import (
    solve_ik,
)
from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import (
    PlanarTranslationToRigidTransformSystem,
)
from .robot_system_base import RobotSystemBase
from planning_through_contact.simulation.sim_utils import (
    LoadRobotOnly,
    models_folder,
    package_xml_file,
    GetSliderUrl,
)
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)


class IiwaHardwareStation(RobotSystemBase):
    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        scenario_name = "demo"
        scenario_file_name = f"{models_folder}/planar_pushing_iiwa_scenario.yaml"
        scenario = LoadScenario(
            filename=scenario_file_name, scenario_name=scenario_name
        )

        def add_slider_to_parser(parser):
            slider_sdf_url = GetSliderUrl(sim_config)
            (slider,) = parser.AddModels(url=slider_sdf_url)
            return

        self._check_scenario_and_sim_config_consistent(scenario, sim_config)

        builder = DiagramBuilder()

        ## Add systems

        # Kuka station
        self.station = builder.AddSystem(
            MakeHardwareStation(
                scenario,
                meshcat=meshcat,
                package_xmls=[package_xml_file],
                hardware=sim_config.use_hardware,
                parser_prefinalize_callback=add_slider_to_parser,
            ),
        )
        if not sim_config.use_hardware:
            external_mbp = self.station.GetSubsystemByName("plant")
            self.station_plant = external_mbp
            self.slider = external_mbp.GetModelInstanceByName(sim_config.slider.name)

        # Iiwa Planer
        # Delay between starting the simulation and the iiwa starting to go to the home position
        INITIAL_DELAY = 1.0
        # Delay between the iiwa reaching the home position and the pusher starting to follow the planned pushing trajectory
        WAIT_PUSH_DELAY = 1.0
        assert sim_config.delay_before_execution > INITIAL_DELAY + WAIT_PUSH_DELAY
        self._planner = builder.AddNamedSystem(
            "IiwaPlanner",
            IiwaPlanner(
                sim_config=sim_config,
                robot_plant=LoadRobotOnly(
                    sim_config, robot_plant_file="iiwa_controller_plant.yaml"
                ),
                initial_delay=INITIAL_DELAY,
                wait_push_delay=WAIT_PUSH_DELAY,
            ),
        )

        # Diff IK
        EE_FRAME = "pusher_end"
        robot = LoadRobotOnly(sim_config, robot_plant_file="iiwa_controller_plant.yaml")
        ik_params = DifferentialInverseKinematicsParameters(
            robot.num_positions(), robot.num_velocities()
        )
        ik_params.set_time_step(sim_config.time_step)
        # True velocity limits for the IIWA14
        # (in rad, rounded down to the first decimal)
        IIWA14_VELOCITY_LIMITS = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        velocity_limit_factor = 0.7
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * IIWA14_VELOCITY_LIMITS,
                velocity_limit_factor * IIWA14_VELOCITY_LIMITS,
            )
        )
        ik_params.set_nominal_joint_position(self._sim_config.default_joint_positions)
        self._diff_ik = builder.AddNamedSystem(
            "DiffIk",
            DifferentialInverseKinematicsIntegrator(
                robot,
                robot.GetFrameByName(EE_FRAME),
                sim_config.time_step,
                ik_params,
            ),
        )
        const = builder.AddNamedSystem(
            "false", ConstantValueSource(AbstractValue.Make(False))
        )

        planar_translation_to_rigid_tranform = builder.AddSystem(
            PlanarTranslationToRigidTransformSystem(z_dist=sim_config.pusher_z_offset)
        )

        driver_config = scenario.model_drivers["iiwa"]
        if isinstance(driver_config, JointStiffnessDriver):
            # Turn desired position into desired state
            self._state_interpolator = builder.AddNamedSystem(
                "StateInterpolatorWithDiscreteDerivative",
                StateInterpolatorWithDiscreteDerivative(
                    robot.num_positions(),
                    time_step=sim_config.time_step,
                ),
            )

        # Switch for switching between planner output (for GoPushStart), and diff IK output (for pushing)
        switch = builder.AddNamedSystem("switch", PortSwitch(robot.num_positions()))

        if isinstance(driver_config, IiwaDriver):
            # Iiwa state estimated multiplexer
            iiwa_state_estimated_mux = builder.AddSystem(
                Multiplexer(input_sizes=[robot.num_positions(), robot.num_velocities()])
            )

        ## Connect systems

        # Inputs to diff IK
        builder.Connect(
            planar_translation_to_rigid_tranform.get_output_port(),
            self._diff_ik.GetInputPort("X_WE_desired"),
        )

        # builder.Connect(
        #     const.get_output_port(),
        #     self._diff_ik.GetInputPort("use_robot_state"),
        # )
        # Strangely, when we use the planner's reset_diff_ik port, which sets use_robot_state to True before the pushing phase and False during the pushing phase, we get persistent diff IK drift.
        builder.Connect(
            self._planner.GetOutputPort("reset_diff_ik"),
            self._diff_ik.GetInputPort("use_robot_state"),
        )

        if isinstance(driver_config, JointStiffnessDriver):
            # Inputs to the planner
            # Need an additional demultiplexer to split state_estimated into position and velocity
            demux = builder.AddSystem(
                Demultiplexer([robot.num_positions(), robot.num_velocities()])
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                demux.get_input_port(),
            )
            builder.Connect(
                demux.get_output_port(0),
                self._planner.GetInputPort("iiwa_position_measured"),
            )

            # Input to Diff IK
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                self._diff_ik.GetInputPort("robot_state"),
            )

            # Inputs to state interpolator
            builder.Connect(
                switch.get_output_port(),
                self._state_interpolator.get_input_port(),
            )

            # Inputs to station
            builder.Connect(
                self._state_interpolator.get_output_port(),
                self.station.GetInputPort("iiwa.desired_state"),
            )

        elif isinstance(driver_config, IiwaDriver):
            # Inputs to the planner
            builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                self._planner.GetInputPort("iiwa_position_measured"),
            )

            # Inputs to the state estimator multiplexer
            builder.Connect(
                self.station.GetOutputPort("iiwa.position_measured"),
                iiwa_state_estimated_mux.get_input_port(0),
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.velocity_estimated"),
                iiwa_state_estimated_mux.get_input_port(1),
            )

            # Input to Diff IK
            builder.Connect(
                iiwa_state_estimated_mux.get_output_port(),
                self._diff_ik.GetInputPort("robot_state"),
            )

            # Inputs to station
            builder.Connect(
                switch.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        # Inputs to switch
        builder.Connect(
            self._planner.GetOutputPort("iiwa_position_command"),
            switch.DeclareInputPort("planner_iiwa_position_command"),
        )
        builder.Connect(
            self._diff_ik.get_output_port(),
            switch.DeclareInputPort("open_loop_iiwa_position_cmd"),
        )
        builder.Connect(
            self._planner.GetOutputPort("control_mode"),
            switch.get_port_selector_input_port(),
        )

        ## Export inputs and outputs
        builder.ExportInput(
            planar_translation_to_rigid_tranform.get_input_port(),
            "planar_position_command",
        )

        if isinstance(driver_config, JointStiffnessDriver):
            builder.ExportOutput(
                self.station.GetOutputPort("iiwa.state_estimated"),
                "robot_state_measured",
            )
        elif isinstance(driver_config, IiwaDriver):
            builder.ExportOutput(
                iiwa_state_estimated_mux.get_output_port(),
                "robot_state_measured",
            )

        if not sim_config.use_hardware:
            # Only relevant when use_hardware=False
            # If use_hardware=True, this info will be updated by the optitrack system in the state estimator directly
            builder.ExportOutput(
                self.station.GetOutputPort(f"{sim_config.slider.name}_state"),
                "object_state_measured",
            )

        builder.BuildInto(self)

    def _check_scenario_and_sim_config_consistent(
        self, scenario: Scenario, sim_config: PlanarPushingSimConfig
    ):
        # Check that the duplicated config between scenario and sim config are consistent
        ...

    def pre_sim_callback(self, root_context: Context) -> None:
        ...

    @property
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        return "iiwa"
