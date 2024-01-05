import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt
from pydrake.math import RotationMatrix
from pydrake.trajectories import (
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
    Trajectory,
)

from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from planning_through_contact.geometry.planar.abstract_mode import AbstractModeVariables
from planning_through_contact.geometry.planar.face_contact import FaceContactVariables
from planning_through_contact.geometry.planar.non_collision import NonCollisionVariables
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.geometry.utilities import from_so2_to_so3
from planning_through_contact.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    SliderPusherSystemConfig,
)
from planning_through_contact.simulation.dynamics.slider_pusher.slider_pusher_system import (
    SliderPusherSystem,
)

GcsVertex = opt.GraphOfConvexSets.Vertex
GcsEdge = opt.GraphOfConvexSets.Edge


@dataclass
class So3TrajSegment:
    start_time: float
    end_time: float
    Rs: List[npt.NDArray[np.float64]]
    traj: PiecewiseQuaternionSlerp

    @classmethod
    def from_knot_points(
        cls,
        # NOTE: we don't really need the entire matrix for this,
        # but keep it around until we want to extend to SO(3)
        Rs: List[npt.NDArray],
        start_time: float,
        end_time: float,
    ) -> "So3TrajSegment":
        num_samples = len(Rs)
        if num_samples == 1:  # just a constant value throughout the traj time
            knot_point_times = np.array([start_time, end_time])
            samples = Rs * 2
        else:
            knot_point_times = np.linspace(start_time, end_time, num_samples)
            samples = Rs

        Rs_in_SO3 = [from_so2_to_so3(R) for R in samples]

        traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)  # type: ignore
        return cls(start_time, end_time, Rs, traj)

    def eval_theta(self, t: float) -> float:
        R = RotationMatrix(self.traj.orientation(t).rotation())
        theta = R.ToRollPitchYaw().yaw_angle()
        return theta

    def eval_omega(self, t: float) -> npt.NDArray[np.float64]:
        omega = self.traj.angular_velocity(t)
        return omega

    def eval_theta_dot(self, t: float) -> float:
        omega = self.eval_omega(t)
        Z_AXIS = 2
        return omega[Z_AXIS]

    def eval(self, t: float) -> npt.NDArray[np.float64]:
        """
        Returns the 3x3 rotation matrix associated with this trajectory segment at time t
        """
        R = self.traj.orientation(t).rotation()
        return R


@dataclass
class LinTrajSegment:
    num_dims: int
    start_time: float
    end_time: float
    knot_points: npt.NDArray[np.float64]
    traj: Trajectory

    @classmethod
    def from_knot_points(
        cls,
        knot_points: npt.NDArray[np.float64],
        start_time: float,
        end_time: float,
    ) -> "LinTrajSegment":
        if len(knot_points.shape) == 1:  # (NUM_SAMPLES, )
            knot_point_times = np.linspace(start_time, end_time, len(knot_points))
            samples = knot_points.reshape(
                (1, -1)
            )  # FirstOrderHold expects values to be two-dimensional
            num_dims = 1
        elif len(knot_points.shape) == 2:  # (NUM_DIMS, NUM_SAMPLES)
            num_dims, num_samples = knot_points.shape

            if num_samples == 1:  # just a constant value throughout the traj time
                knot_point_times = np.array([start_time, end_time])
                samples = np.repeat(knot_points, 2, axis=1)
            else:
                knot_point_times = np.linspace(start_time, end_time, num_samples)
                samples = knot_points
        else:
            raise ValueError("Invalid shape for knot points")

        traj = PiecewisePolynomial.FirstOrderHold(knot_point_times, samples)
        return cls(num_dims, start_time, end_time, knot_points, traj)

    def eval(self, t: float) -> float | npt.NDArray[np.float64]:
        if self.num_dims == 1:
            return self.traj.value(t).item()
        else:
            return self.traj.value(t)

    def make_derivative(self, derivative_order: int = 1) -> "LinTrajSegment":
        return LinTrajSegment(
            self.num_dims,
            self.start_time,
            self.end_time,
            self.knot_points,
            self.traj.MakeDerivative(derivative_order),
        )


# TODO(bernhardpg): Generalize
class PlanarPushingContactMode(Enum):
    NO_CONTACT = 0
    FACE_0 = 1
    FACE_1 = 2
    FACE_2 = 3
    FACE_3 = 4
    FACE_4 = 5
    FACE_5 = 6
    FACE_6 = 7
    FACE_7 = 8

    @classmethod
    def from_contact_location(
        cls,
        loc: PolytopeContactLocation,
    ) -> "PlanarPushingContactMode":
        return cls(1 + loc.idx)

    def to_contact_location(self) -> PolytopeContactLocation:
        if self == PlanarPushingContactMode.FACE_0:
            idx = 0
        elif self == PlanarPushingContactMode.FACE_1:
            idx = 1
        elif self == PlanarPushingContactMode.FACE_2:
            idx = 2
        elif self == PlanarPushingContactMode.FACE_3:
            idx = 3
        elif self == PlanarPushingContactMode.FACE_4:
            idx = 4
        elif self == PlanarPushingContactMode.FACE_5:
            idx = 5
        elif self == PlanarPushingContactMode.FACE_6:
            idx = 6
        elif self == PlanarPushingContactMode.FACE_7:
            idx = 7
        else:
            raise NotImplementedError()
        return PolytopeContactLocation(pos=ContactLocation.FACE, idx=idx)


class AbstractTrajSegment(ABC):
    @abstractmethod
    def get_p_WB(self, t: float) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_p_WP(self, t: float) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_p_BP(self, t: float) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def get_R_WB(self, t: float) -> npt.NDArray[np.float64]:
        pass


@dataclass
class FaceContactTrajSegment(AbstractTrajSegment):
    """
    A single trajectory segment for a FaceContact mode
    for the slider pusher planar pushing system, which
    only includes the state x = [p_WB_x, p_WB_y, theta, lam]^T
    and the input u = [c_n, c_f, lam_dot]^T
    """

    sys: SliderPusherSystem  # need a local copy of this so that we can compute system values
    mode: PlanarPushingContactMode
    start_time: float
    end_time: float
    p_WB_x: LinTrajSegment
    p_WB_y: LinTrajSegment
    R_WB: So3TrajSegment
    lam: LinTrajSegment
    c_n: LinTrajSegment
    c_f: LinTrajSegment
    lam_dot: LinTrajSegment
    f_B: LinTrajSegment

    @classmethod
    def from_knot_points(
        cls,
        knot_points: FaceContactVariables,
        start_time: float,
        end_time: float,
        config: SliderPusherSystemConfig,
    ) -> "FaceContactTrajSegment":
        p_WB_x = LinTrajSegment.from_knot_points(knot_points.p_WB_xs, start_time, end_time)  # type: ignore
        p_WB_y = LinTrajSegment.from_knot_points(knot_points.p_WB_ys, start_time, end_time)  # type: ignore
        lam = LinTrajSegment.from_knot_points(knot_points.lams, start_time, end_time)  # type: ignore
        R_WB = So3TrajSegment.from_knot_points(knot_points.R_WBs, start_time, end_time)  # type: ignore

        c_n = LinTrajSegment.from_knot_points(knot_points.normal_forces, start_time, end_time)  # type: ignore
        c_f = LinTrajSegment.from_knot_points(knot_points.friction_forces, start_time, end_time)  # type: ignore
        f_B = LinTrajSegment.from_knot_points(np.hstack(knot_points.f_c_Bs), start_time, end_time)  # type: ignore
        lam_dot = lam.make_derivative()

        sys = SliderPusherSystem(knot_points.contact_location, config)

        mode = PlanarPushingContactMode.from_contact_location(
            knot_points.contact_location
        )

        return cls(
            sys,
            mode,
            start_time,
            end_time,
            p_WB_x,
            p_WB_y,
            R_WB,
            lam,
            c_n,
            c_f,
            lam_dot,
            f_B,
        )

    def eval_state(self, t: float) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.p_WB_x.eval(t),
                self.p_WB_y.eval(t),
                self.R_WB.eval_theta(t),
                self.lam.eval(t),
            ]
        )

    def eval_control(self, t: float) -> npt.NDArray[np.float64]:
        if t <= self.end_time:
            return np.array([self.c_n.eval(t), self.c_f.eval(t), self.lam_dot.eval(t)])
        else:
            return np.array([0, 0, 0])

    def get_p_WB(self, t: float) -> npt.NDArray[np.float64]:
        return np.array([self.p_WB_x.eval(t), self.p_WB_y.eval(t)]).reshape((2, 1))

    def get_p_Wc(self, t: float) -> npt.NDArray[np.float64]:
        state = self.eval_state(t)
        p_Wc = self.sys.get_p_Wc_from_state(state)
        return p_Wc

    def get_p_WP(self, t: float) -> npt.NDArray[np.float64]:
        state = self.eval_state(t)
        p_WP = self.sys.get_p_WP_from_state(state)
        return p_WP

    def get_p_BP(self, t: float) -> npt.NDArray[np.float64]:
        state = self.eval_state(t)
        return self.sys._get_p_BP(state)

    def get_R_WB(self, t: float) -> npt.NDArray[np.float64]:
        return self.R_WB.eval(t)

    def get_f_B(self, t: float) -> npt.NDArray[np.float64]:
        f_B = self.f_B.eval(t)
        assert isinstance(f_B, type(np.array([])))
        return f_B

    def get_f_W(self, t: float) -> npt.NDArray[np.float64]:
        f_B = self.get_f_B(t)
        R_WB = self.get_R_WB(t)[:2, :2]  # 2x2 matrix
        return R_WB @ f_B


@dataclass
class NonCollisionTrajSegment(AbstractTrajSegment):
    """
    A single trajectory segment for either a NonCollision of FaceContact mode
    for the general planar pushing system.
    """

    start_time: float
    end_time: float
    p_WB: LinTrajSegment
    p_BP: LinTrajSegment
    R_WB: So3TrajSegment
    mode: PlanarPushingContactMode

    @classmethod
    def from_knot_points(
        cls, knot_points: NonCollisionVariables, start_time: float, end_time: float
    ) -> "NonCollisionTrajSegment":
        p_WB = LinTrajSegment.from_knot_points(
            np.vstack([knot_points.p_WB]),  # just one value
            start_time,
            end_time,
        )
        p_BP = LinTrajSegment.from_knot_points(
            np.hstack(knot_points.p_BPs), start_time, end_time
        )
        R_WB = So3TrajSegment.from_knot_points(
            [knot_points.R_WB],  # just one value
            start_time,
            end_time,
        )
        mode = PlanarPushingContactMode.NO_CONTACT

        return cls(start_time, end_time, p_WB, p_BP, R_WB, mode)

    def get_p_WB(self, t: float) -> npt.NDArray[np.float64]:
        p_WB = self.p_WB.eval(t)
        assert isinstance(p_WB, type(np.array([])))  # get rid of typing errors
        return p_WB

    def get_R_WB(self, t: float) -> npt.NDArray[np.float64]:
        R_WB = self.R_WB.eval(t)
        assert isinstance(R_WB, type(np.array([])))  # get rid of typing errors
        return R_WB

    def get_p_BP(self, t: float) -> npt.NDArray[np.float64]:
        p_BP = self.p_BP.eval(t)
        assert isinstance(p_BP, type(np.array([])))  # get rid of typing errors
        return p_BP

    def get_p_WP(self, t: float) -> npt.NDArray[np.float64]:
        p_WB = self.get_p_WB(t)
        p_BP = self.get_p_BP(t)
        R_WB = self.get_R_WB(t)[:2, :2]  # 2x2 matrix

        return p_WB + R_WB @ p_BP


class PlanarPushingTrajectory:
    def __init__(
        self,
        config: PlanarPlanConfig,
        path_knot_points: List[FaceContactVariables | NonCollisionVariables],
    ) -> None:
        self.config = config
        self.pusher_radius = config.pusher_radius
        self.path_knot_points = path_knot_points

        time_in_modes = [knot_points.time_in_mode for knot_points in path_knot_points]
        start_and_end_times = np.concatenate(([0], np.cumsum(time_in_modes)))
        self.start_times = start_and_end_times[:-1]
        self.end_times = start_and_end_times[1:]

        # Trajectory segments that only contain "global" states
        self.traj_segments = [
            NonCollisionTrajSegment.from_knot_points(p, start, end)
            if isinstance(p, NonCollisionVariables)
            else FaceContactTrajSegment.from_knot_points(
                p, start, end, config.dynamics_config
            )
            for p, start, end in zip(path_knot_points, self.start_times, self.end_times)
        ]

    @property
    def num_knot_points(self) -> int:
        return sum(knot_points.num_knot_points for knot_points in self.path_knot_points)

    def _get_curr_segment_idx(self, t: float) -> int:
        if t == self.end_time:
            # return the last element if we are at the end time
            return len(self.path_knot_points) - 1

        idx_of_curr_segment = np.where(t < self.end_times)[0][0]
        return idx_of_curr_segment

    def get_traj_segment_for_time(
        self, t: float
    ) -> NonCollisionTrajSegment | FaceContactTrajSegment:
        return self.traj_segments[self._get_curr_segment_idx(t)]

    def _t_or_end_time(self, t: float) -> float:
        if t > self.end_times[-1]:
            return self.end_times[
                -1
            ]  # repeat last element when we want trajectory after end time
        elif t <= 0:
            return 0
        else:
            return t

    def get_knot_point_value(
        self,
        t: float,
        traj_to_get: Literal["p_WB", "R_WB", "p_WP", "f_c_W"],
    ) -> npt.NDArray[np.float64] | float:
        t = self._t_or_end_time(t)
        segment_idx = self._get_curr_segment_idx(t)

        start_time = self.start_times[segment_idx]
        end_time = self.end_times[segment_idx]
        # We always want at least 2 knot points
        num_knot_points = max(self.path_knot_points[segment_idx].num_knot_points, 2)

        # Get the time that is exactly at the knot point
        ts = np.linspace(start_time, end_time, num_knot_points)
        t_idx = np.where(t <= ts)[0][0]

        val = self.get_value(ts[t_idx], traj_to_get)
        return val

    def get_value(
        self,
        t: float,
        traj_to_get: Literal[
            "p_WB", "R_WB", "p_WP", "f_c_W", "theta", "theta_dot", "p_BP", "state"
        ],
    ) -> npt.NDArray[np.float64] | float:
        t = self._t_or_end_time(t)
        seg = self.get_traj_segment_for_time(t)

        if traj_to_get == "p_WB":
            val = seg.get_p_WB(t)
        elif traj_to_get == "R_WB":
            val = seg.get_R_WB(t)
        elif traj_to_get == "p_WP":
            val = seg.get_p_WP(t)
        elif traj_to_get == "theta":
            val = seg.R_WB.eval_theta(t)
        elif traj_to_get == "theta_dot":
            val = seg.R_WB.eval_theta_dot(t)
        elif traj_to_get == "p_BP":
            val = seg.get_p_BP(t)
        elif traj_to_get == "state":
            assert isinstance(seg, FaceContactTrajSegment)
            val = seg.eval_state(t)
        elif traj_to_get == "f_c_W":
            if isinstance(seg, FaceContactTrajSegment):
                val = seg.get_f_W(t)
            else:  # NonCollisionTrajSegment
                val = np.zeros((2, 1))  # return 0 input force if we are not in contact
        else:
            raise NotImplementedError

        return val

    def get_mode(self, t: float) -> PlanarPushingContactMode:
        t = self._t_or_end_time(t)
        traj = self.get_traj_segment_for_time(t)
        return traj.mode

    def get_slider_planar_pose(self, t) -> PlanarPose:
        p_WB = self.get_value(t, "p_WB")
        theta = self.get_value(t, "theta")

        # avoid typing errors
        assert isinstance(p_WB, type(np.array([])))
        assert isinstance(theta, float)

        planar_pose = PlanarPose(p_WB[0, 0], p_WB[1, 0], theta)
        return planar_pose

    def get_pusher_planar_pose(self, t) -> PlanarPose:
        p_WP = self.get_value(t, "p_WP")

        # avoid typing errors
        assert isinstance(p_WP, type(np.array([])))

        planar_pose = PlanarPose(p_WP[0, 0], p_WP[1, 0], theta=0)
        return planar_pose

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            # NOTE: We save the config and path knot points, not this object, as some Drake objects are not serializable
            pickle.dump((self.config, self.path_knot_points), file)

    @classmethod
    def load(cls, filename: str) -> "PlanarPushingTrajectory":
        with open(Path(filename), "rb") as file:
            config, var_path = pickle.load(file)

            return cls(config, var_path)

    @property
    def start_time(self) -> float:
        return self.traj_segments[0].start_time

    @property
    def end_time(self) -> float:
        return self.traj_segments[-1].end_time

    @property
    def target_slider_planar_pose(self) -> PlanarPose:
        assert self.config.start_and_goal
        return self.config.start_and_goal.slider_target_pose

    @property
    def initial_slider_planar_pose(self) -> PlanarPose:
        assert self.config.start_and_goal
        return self.config.start_and_goal.slider_initial_pose

    @property
    def initial_pusher_planar_pose(self) -> Optional[PlanarPose]:
        assert self.config.start_and_goal
        assert self.config.start_and_goal.pusher_initial_pose
        return self.config.start_and_goal.pusher_initial_pose

    @property
    def target_pusher_planar_pose(self) -> Optional[PlanarPose]:
        assert self.config.start_and_goal
        return self.config.start_and_goal.pusher_target_pose

    def get_pos_limits(self, buffer: float) -> Tuple[float, float, float, float]:
        # We use a fixed timestep to quickly check all values of pos.
        # If the original resolution is finer than this, some values
        # might be missed
        FIXED_STEP = 0.01

        def get_lims(value_name: str) -> Tuple[float, float, float, float]:
            ts = np.arange(self.start_time, self.end_time, FIXED_STEP)
            vecs: List[npt.NDArray[np.float64]] = [self.get_knot_point_value(t, value_name) for t in ts]  # type: ignore
            vec_xs = [vec[0, 0] for vec in vecs]
            vec_ys = [vec[1, 0] for vec in vecs]

            vec_x_max = max(vec_xs)
            vec_x_min = min(vec_xs)
            vec_y_max = max(vec_ys)
            vec_y_min = min(vec_ys)

            return vec_x_min, vec_x_max, vec_y_min, vec_y_max

        def add_buffer_to_lims(lims, buffer) -> Tuple[float, float, float, float]:
            return (
                lims[0] - buffer,
                lims[1] + buffer,
                lims[2] - buffer,
                lims[3] + buffer,
            )

        def get_lims_from_two_lims(lim_a, lim_b) -> Tuple[float, float, float, float]:
            return (
                min(lim_a[0], lim_b[0]),
                max(lim_a[1], lim_b[1]),
                min(lim_a[2], lim_b[2]),
                max(lim_a[3], lim_b[3]),
            )

        p_WB_lims = get_lims("p_WB")
        object_radius = self.config.slider_geometry.max_dist_from_com
        obj_lims = add_buffer_to_lims(p_WB_lims, object_radius)
        p_WP_lims = get_lims("p_WP")

        lims = get_lims_from_two_lims(obj_lims, p_WP_lims)
        return add_buffer_to_lims(lims, buffer)
