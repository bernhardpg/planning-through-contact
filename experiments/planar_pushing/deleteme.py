class PlanarPushingContactModeType(Enum):
    FACE = 0
    NON_COLLISION = 1


class PlanarPushingVariableHandler(ABC):
    @abstractmethod
    def eval_result(result: MathematicalProgramResult) -> "ModeVars":
        ...

    def _get_traj(
        self,
        point_sequence: List[npt.NDArray[np.float64]],
        dt: float,
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:  # (N, 2)
        knot_points = np.hstack(point_sequence)  # (2, num_knot_points)
        if interpolate:
            return interpolate_w_first_order_hold(
                knot_points.T, 0, self.time_in_mode, dt
            )
        else:
            return knot_points.T

    @classmethod
    @abstractmethod
    def make(
        cls,
        prog: MathematicalProgram,
        slider: RigidBody2d,
        location: PolytopeContactLocation,
        num_knot_points: int,
        time_in_mode: float,
    ) -> "PlanarPushingContactModeVariables":
        ...


class FaceContactVariables(NamedTuple):
    # NOTE: These types are wrong
    lams: NpVariableArray  # (1, num_knot_points)
    normal_forces: NpVariableArray  # (1, num_knot_points)
    friction_forces: NpVariableArray  # (1, num_knot_points)
    cos_ths: NpVariableArray  # (1, num_knot_points)
    sin_ths: NpVariableArray  # (1, num_knot_points)
    p_WB_xs: NpVariableArray  # (1, num_knot_points)
    p_WB_ys: NpVariableArray  # (1, num_knot_points)

    time_in_mode: float

    pv1: npt.NDArray[np.float64]  # First vertex defining the contact surface
    pv2: npt.NDArray[np.float64]  # Second vertex defining the contact surface
    normal_vec: npt.NDArray[np.float64]
    tangent_vec: npt.NDArray[np.float64]

    # TODO: move together with time_in_mode
    dt: float


class FaceContactVariableHandler(FaceContactVariables, PlanarPushingVariableHandler):
    ...


class ContactMode(ABC):
    type: PlanarPushingContactModeType
    vars: PlanarPushingContactModeVariables


class GcsPlanarPushingPlanner:
    def __init__(self, slider: RigidBody):
        """
        A planner that generates motion plans for pushing an object (the "slider") with a point finger (the "pusher").
        The motion planner formulates the problem as a Graph-of-Convex-Sets problem, where each vertex in the graph
        corresponds to a contact mode.
        """
        self.gcs = opt.GraphOfConvexSets()
        self.slider = slider

    @staticmethod
    def _create_obj_pose(
        pos: npt.NDArray[np.float64], rot: float
    ) -> npt.NDArray[np.float64]:
        """
        Concatenates unactauted object config for source and target vertex
        """
        obj_pose = np.concatenate([pos.flatten(), [np.cos(rot), np.sin(rot)]])
        return obj_pose

    def set_source_pose(self, pos: npt.NDArray[np.float64], rot: float) -> None:
        pose = self._create_obj_pose(pos, rot)
        source_point = opt.Point(pose)
        self.source = self.gcs.AddVertex(source_point, name="source")

    def set_target_pose(self, pos: npt.NDArray[np.float64], rot: float) -> None:
        pose = self._create_obj_pose(pos, rot)
        target_point = opt.Point(pose)
        self.target = self.gcs.AddVertex(target_point, name="source")


#########
# TODO: move this


class TrajectoryBuilder:
    def __init__(self, vars: ContactModeVars):
        self.vars = vars

    # Need to handle R_traj as a special case due to List[(2x2)] structure
    def get_R_traj(
        self, dt: float, interpolate: bool = False
    ) -> List[npt.NDArray[np.float64]]:
        if interpolate:
            return interpolate_so2_using_slerp(self.R_WBs, 0, self.time_in_mode, dt)
        else:
            return self.R_WBs

    def _get_traj(
        self,
        point_sequence: List[npt.NDArray[np.float64]],
        dt: float,
        interpolate: bool = False,
    ) -> npt.NDArray[np.float64]:  # (N, 2)
        knot_points = np.hstack(point_sequence)  # (2, num_knot_points)
        if interpolate:
            return interpolate_w_first_order_hold(
                knot_points.T, 0, self.time_in_mode, dt
            )
        else:
            return knot_points.T

    def get_p_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_WBs, dt, interpolate)

    def get_p_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.p_c_Ws, dt, interpolate)

    def get_f_c_W_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        return self._get_traj(self.f_c_Ws, dt, interpolate)

    def get_v_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        # Pad with zero to avoid wrong length (velocities have one less element due to finite diffs)
        num_dims = 2
        return self._get_traj(self.v_WBs + [np.zeros((num_dims, 1))], dt, interpolate)  # type: ignore

    def get_omega_WB_traj(
        self, dt: float, interpolate: bool = False
    ) -> npt.NDArray[np.float64]:
        # Pad with zero to avoid wrong length (velocities have one less element due to finite diffs)
        return self._get_traj(self.omega_WBs + [0], dt, interpolate)  # type: ignore

    @staticmethod
    def interpolate_so2_using_slerp(
        Rs: List[npt.NDArray[np.float64]],
        start_time: float,
        end_time: float,
        dt: float,
    ) -> List[npt.NDArray[np.float64]]:
        """
        Assumes evenly spaced knot points R_matrices.

        @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
        """

        Rs_in_SO3 = [make_so3(R) for R in Rs]
        knot_point_times = np.linspace(start_time, end_time, len(Rs))
        quat_slerp_traj = PiecewiseQuaternionSlerp(knot_point_times, Rs_in_SO3)

        traj_times = np.arange(start_time, end_time, dt)
        R_traj_in_SO2 = [
            quat_slerp_traj.orientation(t).rotation()[0:2, 0:2] for t in traj_times
        ]

        return R_traj_in_SO2

    @staticmethod
    def interpolate_w_first_order_hold(
        values: npt.NDArray[np.float64],  # (NUM_SAMPLES, NUM_DIMS)
        start_time: float,
        end_time: float,
        dt: float,
    ) -> npt.NDArray[np.float64]:  # (NUM_POINTS, NUM_DIMS)
        """
        Assumes evenly spaced knot points.

        @return: trajectory evaluated evenly at every dt-th step, starting at start_time and ending at specified end_time.
        """

        knot_point_times = np.linspace(start_time, end_time, len(values))

        # Drake expects the values to be (NUM_DIMS, NUM_SAMPLES)
        first_order_hold = PiecewisePolynomial.FirstOrderHold(
            knot_point_times, values.T
        )
        traj_times = np.arange(start_time, end_time, dt)
        traj = np.hstack(
            [first_order_hold.value(t) for t in traj_times]
        ).T  # transpose to match format in rest of project

        return traj
