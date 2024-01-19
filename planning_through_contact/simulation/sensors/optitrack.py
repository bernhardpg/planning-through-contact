# From https://github1s.com/nepfaff/iiwa_setup/blob/main/iiwa_setup/sensors/opitrack.py

from dataclasses import dataclass
import logging

from copy import copy
from typing import List, Optional

import numpy as np

from optitrack import optitrack_frame_t, optitrack_rigid_body_t
from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    DrakeLcm,
    LcmInterfaceSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    LeafSystem,
    ModelInstanceIndex,
    MultibodyPlant,
    Quaternion,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    State,
)

from planning_through_contact.simulation.state_estimators.state_estimator import StateEstimator

@dataclass
class OptitrackConfig:
    iiwa_id: int
    slider_id: int
    X_optitrackBody_plantBody: RigidTransform

class OptitrackFrameSource(LeafSystem):
    def __init__(
        self,
        optitrack_frames: List[optitrack_frame_t],
        optitrack_frame_times: List[float],
    ):
        """A system that publishes optitrack frames at the specified times.

        Args:
            optitrack_frames (List[optitrack_frame_t]): The frames to publish.
            optitrack_frame_times (List[float]): The times at which to publish the
            frames. Must be of the same length as `optitrack_frames`.
        """
        super().__init__()

        if len(optitrack_frames) != len(optitrack_frame_times):
            raise ValueError(
                "The number of optitrack frames must be equal to the number of "
                + "optitrack frame times."
            )
        self._optitrack_frames = optitrack_frames
        self._optitrack_frame_times = np.asarray(optitrack_frame_times)

        self._optitrack_frame_publisher = self.DeclareAbstractOutputPort(
            "optitrack_frame",
            lambda: AbstractValue.Make(optitrack_frame_t),
            self._get_optitrack_frame,
        )

    def _get_optitrack_frame(self, context: Context, output: AbstractValue) -> None:
        current_time = context.get_time()
        frame_idx = np.argmin(np.abs(self._optitrack_frame_times - current_time))
        output.set_value(self._optitrack_frames[frame_idx])


class PosesToOptitrackFrameConverter(LeafSystem):
    def __init__(self, optitrack_object_ids: List[int]):
        """A system that takes in a pose and outputs an Optitrack frame.

        Args:
            optitrack_object_ids (List[int]): The optitrack frame ids. The order of the
                ids must match the order of the poses.
        """
        super().__init__()

        self._optitrack_object_id = optitrack_object_ids

        self._pose_input_port = self.DeclareAbstractInputPort(
            "poses", AbstractValue.Make(List[RigidTransform])
        )
        self._optitrack_frame_publisher = self.DeclareAbstractOutputPort(
            "optitrack_frame",
            lambda: AbstractValue.Make(optitrack_frame_t),
            self._convert_poses_to_optitrack_frame,
        )

    def _convert_poses_to_optitrack_frame(
        self, context: Context, output: AbstractValue
    ) -> None:
        # Read poses
        poses: List[RigidTransform] = self._pose_input_port.Eval(context)
        assert len(poses) == len(
            self._optitrack_object_id
        ), "The number of poses must be equal to the number of Optitrack rigid bodies."

        # Construct optitrack frame
        rigid_bodies = []
        for pose, object_id in zip(poses, self._optitrack_object_id):
            rigid_body = optitrack_rigid_body_t()
            rigid_body.id = object_id
            rigid_body.xyz = pose.translation().tolist()
            # Convert quaternion from [w, x, y, z] to [x, y, z, w]
            drake_quat = pose.rotation().ToQuaternion()
            rigid_body.quat = [
                drake_quat.x(),
                drake_quat.y(),
                drake_quat.z(),
                drake_quat.w(),
            ]
            rigid_bodies.append(rigid_body)
        frame = optitrack_frame_t()
        frame.num_rigid_bodies = len(poses)
        frame.rigid_bodies = rigid_bodies

        output.set_value(frame)


class OptitrackObjectTransformUpdater(LeafSystem):
    """
    A system that updates the pose of `object_instance_idx` in the `plant` before
    every trajectory-advancing step using the optitrack measurements.

    NOTE: `set_plant_context` must be used to set the plant's context before the
    simulation is started. The plant context must be obtained from the context that
    is passes to the simulator.
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        object_instance_idx: ModelInstanceIndex,
        optitrack_iiwa_id: int,
        optitrack_body_id: int,
        X_optitrackBody_plantBody: RigidTransform,
        retain_z: bool,
        retain_roll: bool,
        retain_pitch: bool,
        X_world_iiwa: RigidTransform = RigidTransform.Identity(),
    ):
        """
        Args:
            plant (MultibodyPlant): The plant that contains the object whose pose should
            be set based on the optitrack measurements.
            object_instance_idx (ModelInstanceIndex): The model instance index of the
            object in the plant whose positions should be set.
            optitrack_iiwa_id (int): The optrirack body ID that corresponds to the iiwa
            base.
            optitrack_body_id (int): The optitrack body ID that corresponds to the
            object.
            X_optitrackBody_plantBody (RigidTransform): The trasform from the optitrack
            pose to the plant/ simulated pose.
            retain_z (bool): Whether to keep the current z position of the object rather
            than changing it to the optitrack measurement. This can be useful for planar
            pushing tasks, where we know that z doesn't change and small measurement
            errors can lead to the object falling through objects.
            retain_roll (bool): Similar to `retain_z`.
            retain_pitch (bool): Similar to `retain_z`.
            X_world_iiwa (RigidTransform): The pose of the iiwa base with respect ot the
            world frame.
        """
        super().__init__()

        self._plant = plant
        self._plant_context = None
        self._object_instance_idx = object_instance_idx
        self._optitrack_iiwa_id = optitrack_iiwa_id
        self._optitrack_body_id = optitrack_body_id
        self._X_optitrackBody_plantBody = X_optitrackBody_plantBody
        self._retain_z = retain_z
        self._retain_roll = retain_roll
        self._retain_pitch = retain_pitch
        self._X_world_iiwa = X_world_iiwa

        self._object_positions = np.array([np.nan] * 7)

        self._optitrack_frame_index = self.DeclareAbstractInputPort(
            "optitrack_frame", AbstractValue.Make(optitrack_frame_t)
        ).get_index()

        self.DeclareVectorOutputPort(
            "object_positions", 7, self._get_current_object_positions
        )

        # Update object pose before every trajectory-advancing step
        self.DeclarePerStepUnrestrictedUpdateEvent(self._update_object_pose)

    def set_plant_context(self, context: Context) -> None:
        """
        Sets the plant context. NOTE: The context must be set before simulation is
        started.
        """
        self._plant_context = context

    def _update_object_pose(self, context: Context, state: State) -> None:
        """Update the object pose in the plant based on the optitrack measurements."""
        assert (
            self._plant_context is not None
        ), "The plant context must be set before the simulation is started!"

        # Get optitrack measurement
        optitrack_frame: optitrack_frame_t = self.get_input_port(
            self._optitrack_frame_index
        ).Eval(context)
        optitrack_rigid_bodies: List[
            optitrack_rigid_body_t
        ] = optitrack_frame.rigid_bodies
        if len(optitrack_rigid_bodies) == 0:
            logging.warning("Skipping object pose update as no optitrack bodies found.")
            return

        optitrack_body_ids = [body.id for body in optitrack_rigid_bodies]
        iiwa_base_body = optitrack_rigid_bodies[
            optitrack_body_ids.index(self._optitrack_iiwa_id)
        ]
        object_body = optitrack_rigid_bodies[
            optitrack_body_ids.index(self._optitrack_body_id)
        ]

        # NOTE: The 'origin' frame refers to the optitrack world frame while the 'world'
        # frame refers to the plant/ simulated world frame that is of actual interest
        X_origin_iiwa = RigidTransform(
            self.get_quaternion_from_optitrack_rigid_body(iiwa_base_body),
            iiwa_base_body.xyz,
        )
        X_iiwa_origin = X_origin_iiwa.inverse()
        X_origin_optitrackBody = RigidTransform(
            RotationMatrix(self.get_quaternion_from_optitrack_rigid_body(object_body)),
            object_body.xyz,
        )

        # Find the body pose in the plant world frame
        X_world_optitrackBody: RigidTransform = (
            self._X_world_iiwa @ X_iiwa_origin @ X_origin_optitrackBody
        )

        X_world_plantBody: RigidTransform = (
            X_world_optitrackBody @ self._X_optitrackBody_plantBody
        )
        object_quaternion = copy(X_world_plantBody.rotation().ToQuaternion())
        object_translation = copy(X_world_plantBody.translation())

        # Optionally only update a subset of the positions
        current_object_positions = self._plant.GetPositions(
            self._plant_context, self._object_instance_idx
        )
        if self._retain_z:
            object_translation[2] = current_object_positions[-1]
        # Creating a Quaternion might throw an error if the input is not perfectly
        # normalized
        current_object_RPY = (
            RotationMatrix(
                Quaternion(
                    current_object_positions[:4]
                    / np.linalg.norm(current_object_positions[:4])
                )
            )
            .ToRollPitchYaw()
            .vector()
        )
        object_RPY = RotationMatrix(object_quaternion).ToRollPitchYaw().vector()
        if self._retain_roll:
            object_RPY[0] = current_object_RPY[0]
        if self._retain_pitch:
            object_RPY[1] = current_object_RPY[1]
        object_quaternion = RotationMatrix(RollPitchYaw(object_RPY)).ToQuaternion()

        new_object_positions = np.concatenate(
            (object_quaternion.wxyz(), object_translation)
        )
        self._plant.SetPositions(
            self._plant_context, self._object_instance_idx, new_object_positions
        )
        self._object_positions = new_object_positions

    def _get_current_object_positions(
        self, context: Context, output: BasicVector
    ) -> None:
        output.set_value(self._object_positions)

    @staticmethod
    def get_quaternion_from_optitrack_rigid_body(body: optitrack_rigid_body_t):
        return Quaternion(
            x=body.quat[0], y=body.quat[1], z=body.quat[2], w=body.quat[3]
        )


class OptitrackObjectTransformUpdaterDiagram(Diagram):
    """
    A diagram for updating the pose of `object_instance_idx` in the internal `plant`
    before every trajectory-advancing step using the optitrack measurements.

    NOTE: `set_plant_context` must be used to set the plant's context before the
    simulation is started. The plant context must be obtained from the context that
    is passes to the simulator.
    """

    def __init__(
        self,
        state_estimator: StateEstimator,
        optitrack_iiwa_id: int,
        optitrack_body_id: int,
        X_optitrackBody_plantBody: RigidTransform,
        retain_z: bool = False,
        retain_roll: bool = False,
        retain_pitch: bool = False,
        X_world_iiwa: RigidTransform = RigidTransform.Identity(),
        simulate: bool = False,
        lcm_publish_period: Optional[float] = None,
        optitrack_frames: Optional[List[optitrack_frame_t]] = None,
        optitrack_frame_times: Optional[List[float]] = None,
    ):
        """
        Args:
            station (IiwaHardwareStationDiagram): The iiwa hardware station.
            optitrack_iiwa_id (int): The optrirack body ID that corresponds to the iiwa
            base.
            object_name (str): The name of the object in the plant whose positions
            should be set.
            optitrack_body_id (int): The optitrack body ID that corresponds to the
            object.
            X_optitrackBody_plantBody (RigidTransform): The trasform from the optitrack
            pose to the plant/ simulated pose.
            retain_z (bool): Whether to keep the current z position of the object rather
            than changing it to the optitrack measurement. This can be useful for planar
            pushing tasks, where we know that z doesn't change and small measurement
            errors can lead to the object falling through objects.
            retain_roll (bool): Similar to `retain_z`.
            retain_pitch (bool): Similar to `retain_z`.
            X_world_iiwa (RigidTransform): The pose of the iiwa base with respect ot the
            world frame.
            simulate (bool): Whether to simulate optitrack messages.
            lcm_publish_period (Optional[float]): The period at which to publish
            simulated optitrack messages. Can be None if `simulate` is False.
            optitrack_frames (Optional[List[optitrack_frame_t]]): The simulated
            optitrack frames to publish. Can be None if `simulate` is False.
            optitrack_frame_times (Optional[List[float]]): The times at which to publish
            the simulated optitrack frames. Can be None if `simulate` is False.
        """
        super().__init__()

        builder = DiagramBuilder()

        lcm = DrakeLcm()
        lcm_system = builder.AddSystem(LcmInterfaceSystem(lcm))
        optitrack_frame_subscriber: LcmSubscriberSystem = builder.AddNamedSystem(
            "OptitrackFrameSubscriber",
            LcmSubscriberSystem.Make(
                channel="OPTITRACK_FRAMES",
                lcm_type=optitrack_frame_t,
                lcm=lcm_system,
                use_cpp_serializer=False,
                wait_for_message_on_initialization_timeout=0 if simulate else 10,
            ),
        )

        if simulate:
            if lcm_publish_period is None:
                raise ValueError(
                    "The lcm_publish_frequency must be specified if simulate is True."
                )
            optitrack_frame_publisher: LcmPublisherSystem = builder.AddNamedSystem(
                "OptitrackFramePublisher",
                LcmPublisherSystem.Make(
                    channel="OPTITRACK_FRAMES",
                    lcm_type=optitrack_frame_t,
                    lcm=lcm_system,
                    use_cpp_serializer=False,
                    publish_period=lcm_publish_period,
                ),
            )

            optitrack_frame_source: OptitrackFrameSource = builder.AddNamedSystem(
                "OptitrackFrameSource",
                OptitrackFrameSource(
                    optitrack_frames=optitrack_frames,
                    optitrack_frame_times=optitrack_frame_times,
                ),
            )
            builder.Connect(
                optitrack_frame_source.get_output_port(),
                optitrack_frame_publisher.get_input_port(),
            )

        plant = state_estimator.get_plant()
        manipuland_instance = state_estimator.slider
        self._optitrack_object_transform_updater: OptitrackObjectTransformUpdater = (
            builder.AddNamedSystem(
                "OptitrackObjectTransformUpdater",
                OptitrackObjectTransformUpdater(
                    plant=plant,
                    object_instance_idx=manipuland_instance,
                    optitrack_iiwa_id=optitrack_iiwa_id,
                    optitrack_body_id=optitrack_body_id,
                    X_optitrackBody_plantBody=X_optitrackBody_plantBody,
                    retain_z=retain_z,
                    retain_roll=retain_roll,
                    retain_pitch=retain_pitch,
                    X_world_iiwa=X_world_iiwa,
                ),
            )
        )
        builder.Connect(
            optitrack_frame_subscriber.get_output_port(),
            self._optitrack_object_transform_updater.get_input_port(),
        )
        builder.ExportOutput(
            self._optitrack_object_transform_updater.GetOutputPort("object_positions"),
            "object_positions",
        )

        builder.BuildInto(self)

    def set_plant_context(self, context: Context) -> None:
        """
        Sets the plant context. NOTE: The context must be set before simulation is
        started.
        """
        self._optitrack_object_transform_updater.set_plant_context(context)
