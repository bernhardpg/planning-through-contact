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
)

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
            self._robot_model_instance = self.station_plant.GetModelInstanceByName(
                self.robot_model_name
            )
            # Set default joint positions for iiwa
            robot_only_plant = LoadRobotOnly(
                sim_config, robot_plant_file="iiwa_controller_plant.yaml"
            )
            desired_pose = self._sim_config.pusher_start_pose.to_pose(
                self._sim_config.pusher_z_offset
            )
            start_joint_positions = solve_ik(
                plant=robot_only_plant,
                pose=desired_pose,
                default_joint_positions=self._sim_config.default_joint_positions,
            )
            self.start_joint_positions = start_joint_positions
            self.station_plant.SetDefaultPositions(
                self._robot_model_instance, start_joint_positions
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

        ## Connect systems

        # Inputs to diff IK
        builder.Connect(
            planar_translation_to_rigid_tranform.get_output_port(),
            self._diff_ik.GetInputPort("X_WE_desired"),
        )
        builder.Connect(
            self.station.GetOutputPort("iiwa.state_estimated"),
            self._diff_ik.GetInputPort("robot_state"),
        )
        builder.Connect(
            const.get_output_port(),
            self._diff_ik.GetInputPort("use_robot_state"),
        )

        if isinstance(driver_config, JointStiffnessDriver):
            # Inputs to state interpolator
            builder.Connect(
                self._diff_ik.get_output_port(),
                self._state_interpolator.get_input_port(),
            )

            # Inputs to station
            builder.Connect(
                self._state_interpolator.get_output_port(),
                self.station.GetInputPort("iiwa.desired_state"),
            )
        elif isinstance(driver_config, IiwaDriver):
            # Inputs to station
            builder.Connect(
                self._diff_ik.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        ## Export inputs and outputs
        builder.ExportInput(
            planar_translation_to_rigid_tranform.get_input_port(),
            "planar_position_command",
        )

        builder.ExportOutput(
            self.station.GetOutputPort("iiwa.state_estimated"),
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
        # Set default joint positions for iiwa
        # Note this will break when using hardware=True
        # Are both of these necessary?
        self._diff_ik.get_mutable_parameters().set_nominal_joint_position(
            self.start_joint_positions
        )
        self._diff_ik.SetPositions(
            self._diff_ik.GetMyMutableContextFromRoot(root_context),
            self.start_joint_positions,
        )
        self.station_plant.SetDefaultPositions(
            self._robot_model_instance, self.start_joint_positions
        )
        # self.station_plant.SetPositions(
        #     self.station_plant.GetMyMutableContextFromRoot(root_context),
        #     self._robot_model_instance,
        #     self.start_joint_positions,
        # )

    @property
    def robot_model_name(self) -> str:
        """The name of the robot model."""
        return "iiwa"
