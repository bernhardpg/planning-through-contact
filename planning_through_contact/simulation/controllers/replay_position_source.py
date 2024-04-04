import numpy as np

from pydrake.systems.framework import Context, LeafSystem
from pydrake.all import (
    DiagramBuilder,
    ZeroOrderHold,
    Demultiplexer,
)

from planning_through_contact.simulation.controllers.desired_planar_position_source_base import (
    DesiredPlanarPositionSourceBase,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from planning_through_contact.geometry.planar.planar_pose import (
    PlanarPose,
)

class ReplayPublisher(LeafSystem):
    """
    Publishes the desired planar pose for the pusher and the slider.
    This system is similar to the PlanarPoseTrajPublisher,
    but it does not require a HybridMpcConfig and only outputs
    relevant information for data collection.
    """

    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        delay: float
    ):
        super().__init__()
        self.traj = traj
        self.delay = delay

        # Declare output ports
        self.DeclareVectorOutputPort(
            "desired_pusher_planar_pose_vector",
            3,
            self.DoCalcDesiredPusherPlanarPoseVectorOutput,
        )

        self.DeclareVectorOutputPort(
            "desired_slider_planar_pose_vector",
            3,
            self.DoCalcDesiredSliderPlanarPoseVectorOutput,
        )
    
    def _get_rel_t(self, t: float) -> float:
        return t - self.delay

    def _calc_pusher_pose(self, t: float) -> PlanarPose:
        p_WP = self.traj.get_value(t, "p_WP")

        # Avoid typing error
        assert isinstance(p_WP, type(np.array([])))

        planar_pose = PlanarPose(p_WP[0].item(), p_WP[1].item(), theta=0)
        return planar_pose

    def _calc_slider_pose(self, t: float) -> PlanarPose:
        p_WB = self.traj.get_value(t, "p_WB")
        theta = self.traj.get_value(t, "theta")

        # Avoid typing error
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0].item(), p_WB[1].item(), theta)
        return planar_pose
    
    def DoCalcDesiredPusherPlanarPoseVectorOutput(self, context: Context, output):
        curr_t = context.get_time()
        pusher_pose = self._calc_pusher_pose(self._get_rel_t(curr_t))
        output.SetFromVector(pusher_pose.vector())
    
    def DoCalcDesiredSliderPlanarPoseVectorOutput(self, context: Context, output):
        curr_t = context.get_time()
        slider_pose = self._calc_slider_pose(self._get_rel_t(curr_t))
        output.SetFromVector(slider_pose.vector())


class ReplayPositionSource(DesiredPlanarPositionSourceBase):
    """ Replays a given planar pushing trajectory. """
    def __init__(
        self,
        traj: PlanarPushingTrajectory,
        delay: float,
        dt: float = None, # throttle output speed
    ):
        super().__init__()
        self._traj = traj
        self._delay = delay

        # Add systems
        self._builder = builder = DiagramBuilder()
        self._replay_publisher = builder.AddSystem(ReplayPublisher(traj, delay))
        self._demux = builder.AddSystem(Demultiplexer([2,1]))
        
        # Connect Systems
        if dt is None:
            builder.Connect(
                self._replay_publisher.GetOutputPort("desired_pusher_planar_pose_vector"),
                self._demux.get_input_port(0)
            )
            pusher_pose_port = self._replay_publisher.GetOutputPort("desired_pusher_planar_pose_vector")
            slider_pose_port = self._replay_publisher.GetOutputPort("desired_slider_planar_pose_vector")
        else:
            self._pusher_zoh = builder.AddSystem(ZeroOrderHold(period_sec = dt, vector_size = 3))
            self._slider_zoh = builder.AddSystem(ZeroOrderHold(period_sec = dt, vector_size = 3))

            builder.Connect(
                self._replay_publisher.GetOutputPort("desired_pusher_planar_pose_vector"),
                self._pusher_zoh.get_input_port()
            )

            builder.Connect(
                self._replay_publisher.GetOutputPort("desired_slider_planar_pose_vector"),
                self._slider_zoh.get_input_port()
            )

            builder.Connect(
                self._pusher_zoh.get_output_port(),
                self._demux.get_input_port(0)
            )

            pusher_pose_port = self._pusher_zoh.get_output_port()
            slider_pose_port = self._slider_zoh.get_output_port()

        # Export ports
        builder.ExportOutput(
            self._demux.get_output_port(0),
            "planar_position_command"
        )

        builder.ExportOutput(
            pusher_pose_port,
            "desired_pusher_planar_pose_vector"
        )

        builder.ExportOutput(
            slider_pose_port,
            "desired_slider_planar_pose_vector"
        )

        builder.BuildInto(self)