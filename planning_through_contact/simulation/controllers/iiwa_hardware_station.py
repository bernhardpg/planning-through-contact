import numpy as np
from lxml import etree
from manipulation.station import (
    IiwaDriver,
    JointStiffnessDriver,
    LoadScenario,
    MakeHardwareStation,
    Scenario,
)
from pydrake.all import (
    AbstractValue,
    ConstantValueSource,
    Context,
    Demultiplexer,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    Meshcat,
    Multiplexer,
    PortSwitch,
    StateInterpolatorWithDiscreteDerivative,
)

from planning_through_contact.simulation.planar_pushing.iiwa_planner import IiwaPlanner
from planning_through_contact.simulation.planar_pushing.planar_pushing_sim_config import (
    PlanarPushingSimConfig,
)
from planning_through_contact.simulation.sim_utils import (
    GetSliderUrl,
    LoadRobotOnly,
    clamp,
    models_folder,
    package_xml_file,
    randomize_camera_config,
    randomize_pusher,
    randomize_table,
)
from planning_through_contact.simulation.systems.joint_velocity_clamp import (
    JointVelocityClamp,
)
from planning_through_contact.simulation.systems.planar_translation_to_rigid_transform_system import (
    PlanarTranslationToRigidTransformSystem,
)

from .robot_system_base import RobotSystemBase


class IiwaHardwareStation(RobotSystemBase):
    def __init__(
        self,
        sim_config: PlanarPushingSimConfig,
        meshcat: Meshcat,
    ):
        super().__init__()
        self._sim_config = sim_config
        self._meshcat = meshcat
        self._num_positions = 7

        if sim_config.use_hardware:
            scenario_name = "speed-optimized"
        else:
            scenario_name = "accuracy-optimized"

        if not sim_config.domain_randomization:
            scenario_file_name = f"{models_folder}/planar_pushing_iiwa_scenario.yaml"

            def add_slider_to_parser(parser):
                slider_sdf_url = GetSliderUrl(sim_config)
                (slider,) = parser.AddModels(url=slider_sdf_url)
                return

        else:
            scenario_file_name = (
                f"{models_folder}/planar_pushing_iiwa_scenario_randomized.yaml"
            )

            table_grey = np.random.uniform(0.3, 0.95)
            pusher_grey = np.random.uniform(0.1, table_grey)
            color_range = 0.025

            randomize_pusher()
            randomize_table(
                default_color=[table_grey, table_grey, table_grey],
                color_range=color_range,
            )

            def add_slider_to_parser(parser):
                sdf_file = f"{models_folder}/t_pusher.sdf"
                safe_parse = etree.XMLParser(recover=True)
                tree = etree.parse(sdf_file, safe_parse)
                root = tree.getroot()

                diffuse_elements = root.xpath("//model/link/visual/material/diffuse")

                R = clamp(
                    pusher_grey + np.random.uniform(-color_range, color_range), 0.0, 1.0
                )
                G = clamp(
                    pusher_grey + np.random.uniform(-color_range, color_range), 0.0, 1.0
                )
                B = clamp(
                    pusher_grey + np.random.uniform(-color_range, color_range), 0.0, 1.0
                )
                A = 1  # assuming fully opaque

                new_diffuse_value = (
                    f"{R} {G} {B} {A}"  # Example: changing diffuse to white (R G B A)
                )
                for diffuse in diffuse_elements:
                    diffuse.text = new_diffuse_value

                sdf_as_string = etree.tostring(tree, encoding="utf8").decode()

                (slider,) = parser.AddModelsFromString(sdf_as_string, "sdf")

        scenario = LoadScenario(
            filename=scenario_file_name, scenario_name=scenario_name
        )
        # Add cameras to scenario
        if sim_config.camera_configs:
            for camera_config in sim_config.camera_configs:
                if sim_config.randomize_camera:
                    scenario.cameras[camera_config.name] = randomize_camera_config(
                        camera_config
                    )
                else:
                    scenario.cameras[camera_config.name] = camera_config

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
            self._scene_graph = self.station.scene_graph()
            self.slider = external_mbp.GetModelInstanceByName(sim_config.slider.name)

        # Iiwa Planer
        # Delay between starting the simulation and the iiwa starting to go to the home position
        INITIAL_DELAY = 0.5
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
        self.robot = robot
        ik_params = DifferentialInverseKinematicsParameters(
            robot.num_positions(), robot.num_velocities()
        )
        ik_params.set_time_step(sim_config.time_step)
        # True velocity limits for the IIWA14 and IIWA7
        # (in rad, rounded down to the first decimal)
        # IIWA14_VELOCITY_LIMITS = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        IIWA7_VELOCITY_LIMITS = np.array([1.7, 1.7, 1.7, 2.2, 2.4, 3.1, 3.1])
        velocity_limit_factor = 0.3
        ik_params.set_joint_velocity_limits(
            (
                -velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
                velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
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

        # Velocity clamp to prevent sudden spike when switching to diff IK
        joint_velocity_clamp = builder.AddNamedSystem(
            "JointVelocityClamp",
            JointVelocityClamp(
                num_positions=robot.num_positions(),
                joint_velocity_limits=velocity_limit_factor * IIWA7_VELOCITY_LIMITS,
            ),
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

            # Input to joint velocity clamp
            builder.Connect(
                switch.get_output_port(),
                joint_velocity_clamp.get_input_port(),
            )

            # Inputs to state interpolator
            builder.Connect(
                joint_velocity_clamp.get_output_port(),
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

            # Input to joint velocity clamp
            builder.Connect(
                switch.get_output_port(),
                joint_velocity_clamp.get_input_port(),
            )

            # Inputs to station
            builder.Connect(
                joint_velocity_clamp.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        # Inputs to switch
        builder.Connect(
            self._planner.GetOutputPort("iiwa_position_command"),
            switch.DeclareInputPort("planner_iiwa_position_command"),
        )
        builder.Connect(
            self._diff_ik.get_output_port(),
            switch.DeclareInputPort("diff_ik_iiwa_position_cmd"),
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
        builder.ExportOutput(
            joint_velocity_clamp.get_output_port(),
            "iiwa_position_command",
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

        if sim_config.camera_configs:
            for camera_config in self._sim_config.camera_configs:
                builder.ExportOutput(
                    self.station.GetOutputPort(f"{camera_config.name}.rgb_image"),
                    f"rgbd_sensor_{camera_config.name}",
                )

        # Set the initial camera pose
        zoom = 1.8
        camera_in_world = [
            sim_config.slider_goal_pose.x,
            (sim_config.slider_goal_pose.y - 1) / zoom,
            1.5 / zoom,
        ]
        target_in_world = [
            sim_config.slider_goal_pose.x,
            sim_config.slider_goal_pose.y,
            0,
        ]
        self._meshcat.SetCameraPose(camera_in_world, target_in_world)

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

    @property
    def slider_model_name(self) -> str:
        """The name of the robot model."""
        if self._sim_config.slider.name == "box":
            return "box"
        else:
            return "t_pusher"

    def num_positions(self) -> int:
        return self._num_positions

    def get_station_plant(self):
        return self.station_plant

    def get_scene_graph(self):
        return self._scene_graph

    def get_slider(self):
        return self.slider

    def get_meshcat(self):
        return self._meshcat
    
