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
