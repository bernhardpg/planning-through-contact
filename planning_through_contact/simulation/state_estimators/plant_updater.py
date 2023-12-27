from functools import partial

from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    LeafSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    OutputPort,
    RigidTransform,
    State,
)


class PlantUpdater(LeafSystem):
    """
    Provides the API for updating and reading a plant context without simulating
    the plant (adding it to the diagram).
    """

    def __init__(self, plant: MultibodyPlant, robot_model_name: str, object_model_name: str):
        super().__init__()

        self._plant = plant

        self._plant_context = None
        self._robot_model_instance_index = plant.GetModelInstanceByName(robot_model_name)
        self._object_model_instance_index = plant.GetModelInstanceByName(object_model_name)
        # Input ports
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "robot_state",
            self._plant.num_positions(self._robot_model_instance_index)
            + self._plant.num_velocities(self._robot_model_instance_index),
        )
        self._object_position_input_port = self.DeclareVectorInputPort(
            "object_position",
            self._plant.num_positions(self._object_model_instance_index),
        )

        # Output ports
        self._position_output_port = self.DeclareVectorOutputPort(
            "position", self._plant.num_positions(), self._get_position
        )
        self._state_output_port = self.DeclareVectorOutputPort(
            "state",
            self._plant.num_positions() + self._plant.num_velocities(),
            self._get_state,
        )
        self._body_poses_output_port = self.DeclareAbstractOutputPort(
            "body_poses",
            lambda: AbstractValue.Make(
                [RigidTransform()]
            ),
            self._get_body_poses,
        )
        for i in range(self._plant.num_model_instances()):
            model_instance = ModelInstanceIndex(i)
            model_instance_name = self._plant.GetModelInstanceName(model_instance)
            self.DeclareVectorOutputPort(
                f"{model_instance_name}_state",
                self._plant.num_positions(model_instance)
                + self._plant.num_velocities(model_instance),
                partial(self._get_state, model_instance=model_instance),
            )

        self.DeclarePerStepUnrestrictedUpdateEvent(self._update_plant)


    def _update_plant(self, context: Context, state: State) -> None:
        if self._plant_context is None:
            self._plant_context = self._plant.CreateDefaultContext()

        # Update robot state
        self._plant.SetPositionsAndVelocities(
            self._plant_context,
            self._robot_model_instance_index,
            self._robot_state_input_port.Eval(context),
        )

        # Update object position
        self._plant.SetPositions(
            self._plant_context,
            self._object_model_instance_index,
            self._object_position_input_port.Eval(context),
        )

    def _get_position(self, context: Context, output: BasicVector) -> None:
        if self._plant_context is None:
            self._plant_context = self._plant.CreateDefaultContext()

        positions = self._plant.GetPositions(self._plant_context)
        output.set_value(positions)

    def get_position_output_port(self) -> OutputPort:
        return self._position_output_port

    def _get_state(
        self,
        context: Context,
        output: BasicVector,
        model_instance: ModelInstanceIndex = None,
    ) -> None:
        if self._plant_context is None:
            self._plant_context = self._plant.CreateDefaultContext()

        state = self._plant.GetPositionsAndVelocities(
            self._plant_context, model_instance
        )
        output.set_value(state)

    def get_state_output_port(
        self, model_instance: ModelInstanceIndex = None
    ) -> OutputPort:
        if model_instance is None:
            return self._state_output_port
        model_instance_name = self._plant.GetModelInstanceName(model_instance)
        return self.GetOutputPort(f"{model_instance_name}_state")

    def _get_body_poses(self, context: Context, output: AbstractValue) -> None:
        if self._plant_context is None:
            self._plant_context = self._plant.CreateDefaultContext()

        body_poses = []
        for model_idx in range(self._plant.num_model_instances()):
            model_instance = ModelInstanceIndex(model_idx)
            for body_idx in self._plant.GetBodyIndices(model_instance):
                body = self._plant.get_body(body_idx)
                pose = self._plant.CalcRelativeTransform(
                    context=self._plant_context,
                    frame_A=self._plant.world_frame(),
                    frame_B=body.body_frame(),
                )
                body_poses.append(pose)
        output.set_value(body_poses)

    def get_body_poses_output_port(self) -> OutputPort:
        return self._body_poses_output_port

    def get_plant_context(self) -> Context:
        if self._plant_context is None:
            self._plant_context = self._plant.CreateDefaultContext()
        return self._plant_context