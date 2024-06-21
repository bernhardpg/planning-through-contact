import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import yaml
from pydrake.geometry.optimization import GraphOfConvexSets, Spectrahedron
from pydrake.math import eq, ge
from pydrake.solvers import (
    LinearCost,
    MakeSemidefiniteRelaxation,
    MathematicalProgram,
    MathematicalProgramResult,
    PositiveSemidefiniteConstraint,
    QuadraticConstraint,
    QuadraticCost,
    SemidefiniteRelaxationOptions,
    SnoptSolver,
)
from pydrake.symbolic import DecomposeAffineExpressions, Expression, Variable, Variables

from planning_through_contact.convex_relaxation.band_sparse_semidefinite_relaxation import (
    BandSparseSemidefiniteRelaxation,
)
from planning_through_contact.convex_relaxation.convex_concave import (
    cross_product_2d_as_convex_concave,
)
from planning_through_contact.convex_relaxation.sdp import (
    add_trace_cost_on_psd_cones,
    approximate_sdp_cones_with_linear_cones,
    get_X_from_psd_constraint,
    get_X_from_semidefinite_relaxation,
    to_symmetric_matrix_from_lower_triangular_columns,
)
from planning_through_contact.geometry.planar.planar_pushing_trajectory import (
    LinTrajSegment,
    TrajType,
)
from planning_through_contact.geometry.utilities import cross_2d
from planning_through_contact.planning.footstep.footstep_plan_config import (
    FootstepPlanningConfig,
    PotatoRobot,
)
from planning_through_contact.planning.footstep.in_plane_terrain import (
    InPlaneSteppingStone,
    InPlaneTerrain,
)
from planning_through_contact.tools.gcs_tools import hash_gcs_edges
from planning_through_contact.tools.utils import evaluate_np_expressions_array

GcsVertex = GraphOfConvexSets.Vertex
GcsEdge = GraphOfConvexSets.Edge


@dataclass
class FootPlan:
    """
    A class to represent the knot points of a foot in a footstep plan.

    Attributes:
    foot_width: float
        The width of the foot.
    dt: float
        Time interval between knot points.
    p_WF: npt.NDArray[np.float64]
        Planned foot position in world frame. Shape: (num_knot_points, 2)
    f_F_Ws: List[npt.NDArray[np.float64]]
        Planned contact forces in world frame. List of arrays with shape: (num_knot_points, 2)
    tau_F_Ws: List[npt.NDArray[np.float64]]
        Planned contact torques in world frame. List of arrays with shape: (num_knot_points, )
    """

    foot_width: float
    dt: float
    p_WF: npt.NDArray[np.float64]  # (num_knot_points - 1, 2)
    f_F_Ws: List[
        npt.NDArray[np.float64]
    ]  # [(num_knot_points - 1, 2)] (one for each contact point)
    # This is the PLANNED torque (which might NOT be equal to p x f)
    tau_F_Ws: List[
        npt.NDArray[np.float64]
    ]  # [(num_knot_points - 1, )] (one for each contact point)
    # NOTE: p_WB is only used to compute torques and footstep positions relative to COM
    # and has one more knot point than the other values!
    p_WB: npt.NDArray[np.float64]  # (num_knot_points, 2)

    def __post_init__(self) -> None:
        self._initialize_trajectories()

    def __getstate__(self):
        # Exclude the trajectories that are not serializable
        state = {k: v for k, v in self.__dict__.items() if k != "trajectories"}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize_trajectories()

    def get_without_last_input_step(self) -> "FootPlan":
        new_p_WF = self.p_WF[:-1, :]
        new_f_F_Ws = [f[:-1, :] for f in self.f_F_Ws]
        new_tau_F_Ws = [tau[:-1] for tau in self.tau_F_Ws]

        new_plan = FootPlan(
            self.foot_width, self.dt, new_p_WF, new_f_F_Ws, new_tau_F_Ws, self.p_WB
        )
        new_plan._validate_shapes()
        return new_plan

    def _validate_shapes(self) -> None:
        num_inputs = self.num_knot_points - 1
        assert self.p_WF.shape == (num_inputs, 2), "p_WF shape mismatch"
        for f, tau in zip(self.f_F_Ws, self.tau_F_Ws):
            assert f.shape == (num_inputs, 2), "f_F_W shape mismatch"
            assert tau.shape == (num_inputs,), "tau_F_W shape mismatch"

        assert self.p_WB.shape == (self.num_knot_points, 2), "p_WB shape mismatch"

    def _initialize_trajectories(self) -> None:
        interpolation = "zero_order_hold"
        self.trajectories = {
            "p_WF": self._interpolate_segment(self.p_WF, interpolation),
            "f_F_Ws": [
                self._interpolate_segment(f, interpolation) for f in self.f_F_Ws
            ],
            "planned_tau_F_Ws": [
                self._interpolate_segment(tau, interpolation) for tau in self.tau_F_Ws
            ],
            "p_WB": self._interpolate_segment(self.p_WB, "first_order_hold"),
        }

    @property
    def num_knot_points(self) -> int:
        return self.p_WB.shape[0]

    @property
    def num_forces(self) -> int:
        return len(self.f_F_Ws)

    @property
    def end_time(self) -> float:
        return (self.num_knot_points - 1) * self.dt  # we start at time = 0

    def _get_p_WFcs(self, time: float) -> List[npt.NDArray[np.float64]]:
        """
        Planned contact force positions in world frame.
        """
        # TODO: Need to change this when we add rotations in the terrain!
        p_Fcs = [
            np.array([-self.foot_width / 2, 0]).reshape((2, 1)),
            np.array([self.foot_width / 2, 0]).reshape((2, 1)),
        ]
        p_WF = self.get(time, "p_WF")
        p_WFcs = [p_WF + p_Fc for p_Fc in p_Fcs]
        return p_WFcs

    def _get_p_BFc_Ws(self, time: float) -> List[npt.NDArray[np.float64]]:
        """
        Planned contact force positions relative to the body frame in world frame.
        """
        p_WFcs = self._get_p_WFcs(time)
        p_WB = self.get(time, "p_WB")
        return [p_WFc - p_WB for p_WFc in p_WFcs]

    def _get_computed_tau_F_Ws(self, time: float) -> List[npt.NDArray[np.float64]]:
        p_BFc_Ws = self.get(time, "p_BFc_Ws")
        f_F_Ws = self.get(time, "f_F_Ws")
        tau_Fc_Ws = [
            cross_2d(p_BFc_W, f_F_W) for p_BFc_W, f_F_W in zip(p_BFc_Ws, f_F_Ws)
        ]
        return tau_Fc_Ws

    def __add__(self, other: Optional["FootPlan"]) -> "FootPlan":
        if other is None:
            return self

        if not isinstance(other, FootPlan):
            return NotImplemented

        if self.foot_width != other.foot_width:
            raise ValueError("Cannot add FootPlan with different foot widths")

        new_p_WF = np.vstack((self.p_WF, other.p_WF))
        new_f_F_Ws = [
            np.vstack((self_f, other_f))
            for self_f, other_f in zip(self.f_F_Ws, other.f_F_Ws)
        ]
        new_tau_F_Ws = [
            np.hstack((self_tau, other_tau))
            for self_tau, other_tau in zip(self.tau_F_Ws, other.tau_F_Ws)
        ]
        new_p_WB = np.vstack((self.p_WB, other.p_WB))

        return FootPlan(
            foot_width=self.foot_width,
            dt=self.dt,
            p_WF=new_p_WF,
            f_F_Ws=new_f_F_Ws,
            tau_F_Ws=new_tau_F_Ws,
            p_WB=new_p_WB,
        )

    def _interpolate_segment(
        self, data: npt.NDArray[np.float64], interpolation: TrajType
    ) -> "LinTrajSegment":
        return LinTrajSegment.from_knot_points(
            data.T,
            start_time=0,
            end_time=self.end_time,
            traj_type=interpolation,
        )

    @classmethod
    def create_empty(
        cls, foot_width: float, dt: float, num_knot_points: int, num_forces: int
    ) -> "FootPlan":
        oned_shape = (num_knot_points,)
        twod_shape = (num_knot_points, 2)

        p_WF = np.full(twod_shape, np.nan)
        f_F_Ws = [np.full(twod_shape, np.nan) for _ in range(num_forces)]
        tau_F_Ws = [np.full(oned_shape, np.nan) for _ in range(num_forces)]
        p_WB = np.full(twod_shape, np.nan)

        return cls(foot_width, dt, p_WF, f_F_Ws, tau_F_Ws, p_WB)

    @classmethod
    def empty_like(cls, other: "FootPlan") -> "FootPlan":
        return cls.create_empty(
            other.foot_width, other.dt, other.num_knot_points, other.num_forces
        )

    def get(
        self, time: float, traj: str
    ) -> Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]:
        # these are computed values that are not pure trajectories
        if traj not in self.trajectories:
            if traj == "p_WFcs":
                return self._get_p_WFcs(time)
            elif traj == "p_BFc_Ws":
                return self._get_p_BFc_Ws(time)
            elif traj == "computed_tau_F_Ws":
                return self._get_computed_tau_F_Ws(time)
            else:
                raise NotImplementedError(f"Trajectory {traj} not implemented")

        else:
            trajectory = self.trajectories[traj]
            if isinstance(trajectory, list):
                return [t.eval(time) for t in trajectory]
            return trajectory.eval(time)


@dataclass
class FootstepPlan:
    """
    A class to represent the knot points of a footstep plan including the robot body pose.

    Attributes:
    dt: float
        Time interval between steps.
    p_WB: npt.NDArray[np.float64]
        Planned robot body position in world frame. Shape: (num_steps, 2)
    theta_WB: npt.NDArray[np.float64]
        Planned robot body orientation in world frame. Shape: (num_steps, )
    feet_knot_points: List[FootPlan]
        Knot points for the feet. List length can be 1 or 2.
    """

    dts: npt.NDArray[np.float64]
    p_WB: npt.NDArray[np.float64]  # (num_steps, 2)
    theta_WB: npt.NDArray[np.float64]  # (num_steps, )
    feet_knot_points: List[FootPlan]  # [num_feet]

    def __post_init__(self) -> None:
        self._validate_shapes()
        self._initialize_trajectories()

    def _validate_shapes(self) -> None:
        assert self.p_WB.shape == (self.num_states, 2), "p_WB shape mismatch"
        assert self.theta_WB.shape == (self.num_states,), "theta_WB shape mismatch"
        assert self.num_feet in [
            1,
            2,
        ], "Invalid number of feet knot points"
        if len(self.feet_knot_points) == 2:
            assert (
                self.feet_knot_points[0].num_knot_points
                == self.feet_knot_points[1].num_knot_points
            ), "Feet knot points mismatch"

    def _initialize_trajectories(self) -> None:
        interpolation = "first_order_hold"
        self.trajectories = {
            "p_WB": self._interpolate_segment(self.p_WB, interpolation),
            "theta_WB": self._interpolate_segment(self.theta_WB, interpolation),
        }

    @property
    def num_feet(self) -> int:
        return len(self.feet_knot_points)

    @property
    def both_feet(self) -> bool:
        return len(self.feet_knot_points) == 2

    @property
    def first_foot(self) -> FootPlan:
        return self.feet_knot_points[0]

    @property
    def second_foot(self) -> FootPlan:
        assert self.both_feet, "Only one foot knot point available"
        return self.feet_knot_points[1]

    @property
    def num_states(self) -> int:
        return self.p_WB.shape[0]

    @property
    def num_inputs(self) -> int:
        return self.feet_knot_points[0].num_knot_points

    @property
    def num_knot_points(self) -> int:
        return self.num_states

    @property
    def end_time(self) -> float:
        return sum(self.dts[:-1])  # the last dt is unused

    @property
    def tau_F_Ws(self) -> List[List[npt.NDArray[np.float64]]]:
        torques = [foot.tau_F_Ws for foot in self.feet_knot_points]
        return torques

    def _interpolate_segment(
        self, data: npt.NDArray[np.float64], interpolation: TrajType
    ) -> "LinTrajSegment":
        return LinTrajSegment.from_knot_points(
            data.T,  # this function expects the transpose of what we have
            start_time=0,
            end_time=self.end_time,
            traj_type=interpolation,
        )

    @classmethod
    def merge(cls, segments: List["FootstepPlan"]) -> "FootstepPlan":
        both_feet = np.array([s.both_feet for s in segments])
        # This is a quick way to check that the bool value changes for each element in the array
        feet_are_alternating = not np.any(both_feet[:-1] & both_feet[1:])
        if not feet_are_alternating:
            raise RuntimeError(
                "The provided segments do not have alternating modes and do not form a coherent footstep plan."
            )

        # NOTE: Here we just arbitrarily pick that we start with the left foot. Could just as well have picked the other foot
        gait_pattern = np.array([[1, 1], [1, 0], [1, 1], [0, 1]])
        start_idx = 0 if segments[0].both_feet else 1
        gait_schedule = np.array(
            [
                gait_pattern[(start_idx + i) % len(gait_pattern)]
                for i in range(len(segments))
            ]
        )

        p_WBs = np.vstack([k.p_WB for k in segments])
        theta_WBs = np.hstack([k.theta_WB for k in segments])

        dts = np.concatenate([k.dts for k in segments])

        first_foot, second_foot = None, None

        for segment, (first_active, last_active) in zip(segments, gait_schedule):
            both_active = bool(first_active and last_active)
            if both_active:
                first_foot = (
                    segment.first_foot
                    if first_foot is None
                    else first_foot + segment.first_foot
                )
                second_foot = (
                    segment.second_foot
                    if second_foot is None
                    else second_foot + segment.second_foot
                )
            else:
                # NOTE: These next lines look like they have a typo, but they don't.
                # When there is only one foot active, the values for this foot are
                # always stored in the "first" foot values (to avoid unnecessary optimization
                # variables)
                if first_active:
                    first_foot = (
                        segment.first_foot
                        if first_foot is None
                        else first_foot + segment.first_foot
                    )
                    second_foot = (
                        FootPlan.empty_like(segment.first_foot)
                        if second_foot is None
                        else second_foot + FootPlan.empty_like(segment.first_foot)
                    )
                else:
                    first_foot = (
                        FootPlan.empty_like(segment.first_foot)
                        if first_foot is None
                        else first_foot + FootPlan.empty_like(segment.first_foot)
                    )
                    second_foot = (
                        segment.first_foot
                        if second_foot is None
                        else second_foot + segment.first_foot
                    )

        assert (
            first_foot is not None and second_foot is not None
        ), "Foot knot points cannot be None"

        # we remove the last input step so we have N-1 input steps and N state steps
        # (in the GCS planning we don't know which mode shouldn't have a last step)
        first_foot = first_foot.get_without_last_input_step()
        second_foot = second_foot.get_without_last_input_step()

        return cls(dts, p_WBs, theta_WBs, [first_foot, second_foot])

    def __getstate__(self):
        # Exclude the trajectories that are not serializable
        state = {k: v for k, v in self.__dict__.items() if k != "trajectories"}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initialize_trajectories()

    def save(self, filename: str) -> None:
        with open(Path(filename), "wb") as file:
            pickle.dump(self.__getstate__(), file)

    @classmethod
    def load(cls, filename: str) -> "FootstepPlan":
        with open(Path(filename), "rb") as file:
            state = pickle.load(file)
            instance = cls.__new__(
                cls
            )  # Create a new instance without calling __init__
            instance.__setstate__(state)
            return instance

    def get(
        self, time: float, traj: str, derivative_order: int = 0
    ) -> Union[float, npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]:
        if traj not in self.trajectories:
            raise NotImplementedError(f"Trajectory {traj} not implemented")

        trajectory = self.trajectories[traj]
        if derivative_order != 0:
            breakpoint()
        if isinstance(trajectory, list):
            return [t.eval(time) for t in trajectory]
        return trajectory.eval(time)

    def get_foot(
        self, foot: int, time: float, traj: str
    ) -> Union[float, npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]:
        assert foot <= self.num_feet - 1
        return self.feet_knot_points[foot].get(time, traj)


@dataclass
class PlanMetrics:
    cost: float
    solve_time: float
    success: bool

    @classmethod
    def from_result(
        cls, result: MathematicalProgramResult, snopt_solve_time: Optional[float] = None
    ) -> "PlanMetrics":
        solver_name = result.get_solver_id().name()
        solver_details = result.get_solver_details()
        if solver_name == "Mosek":
            solve_time = solver_details.optimizer_time  # type: ignore
        elif solver_name == "SNOPT":
            assert (
                snopt_solve_time is not None
            ), "Must provide SNOPT solve time manually"
            solve_time = snopt_solve_time
        else:
            raise NotImplementedError

        cost = result.get_optimal_cost() if result.is_success() else np.inf

        return cls(cost, solve_time, result.is_success())

    def __str__(self) -> str:
        return f"cost: {self.cost:.4f}, solve_time: {self.solve_time:.2f} s, success: {self.success}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FootstepPlanResult:
    """
    A data structure that contains both the relaxed and rounded trajectory, as well as the metrics for each,
    and some helper functions for quickly evaluating the plans.
    """

    terrain: InPlaneTerrain
    config: FootstepPlanningConfig
    restriction_plan: FootstepPlan
    restriction_metrics: PlanMetrics
    rounded_plan: FootstepPlan  # NOTE(bernhardpg): It would not be hard to extend this with multiple rounded results
    rounded_metrics: PlanMetrics
    gcs_edge_flows: Optional[Dict[str, float]] = None
    gcs_metrics: Optional[PlanMetrics] = None

    @property
    def ub_relaxation_gap_pct(self) -> Optional[float]:
        if self.gcs_metrics is None:
            return None
        elif self.gcs_metrics.cost == 0:
            return np.NaN
        else:
            return (
                (self.rounded_metrics.cost - self.gcs_metrics.cost)
                / self.gcs_metrics.cost
            ) * 100

    @classmethod
    def from_results(
        cls,
        terrain: InPlaneTerrain,
        config: FootstepPlanningConfig,
        restriction_res: MathematicalProgramResult,
        restriction_plan: FootstepPlan,
        rounded_res: MathematicalProgramResult,
        rounded_plan: FootstepPlan,
        snopt_time: float,
        gcs_edge_flows: Optional[Dict[str, float]] = None,
        gcs_res: Optional[MathematicalProgramResult] = None,
    ) -> "FootstepPlanResult":
        restriction_metrics = PlanMetrics.from_result(restriction_res)
        rounded_metrics = PlanMetrics.from_result(
            rounded_res, snopt_solve_time=snopt_time
        )
        gcs_metrics = PlanMetrics.from_result(gcs_res) if gcs_res is not None else None
        return cls(
            terrain,
            config,
            restriction_plan,
            restriction_metrics,
            rounded_plan,
            rounded_metrics,
            gcs_edge_flows,
            gcs_metrics,
        )

    def to_metrics_dict(self) -> dict:
        return {
            "gcs_edge_flows": self.gcs_edge_flows,
            "gcs_metrics": (
                self.gcs_metrics.to_dict() if self.gcs_metrics is not None else None
            ),
            "restriction_metrics": self.restriction_metrics.to_dict(),
            "rounded_metrics": self.rounded_metrics.to_dict(),
            "ub_relaxation_gap_pct": self.ub_relaxation_gap_pct,
        }

    @property
    def gcs_active_edges(self) -> Optional[List[str]]:
        if self.gcs_edge_flows is None:
            return None
        else:
            return list(self.gcs_edge_flows.keys())

    @property
    def num_modes(self) -> int:
        if self.gcs_edge_flows is None:
            raise RuntimeError("Cannot get num_modes when there is no gcs result")

        return len(self.gcs_edge_flows)

    def save_metrics_to_yaml(self, file_path: str) -> None:
        with open(file_path, "w") as yaml_file:
            yaml.dump(self.to_metrics_dict(), yaml_file, indent=4, sort_keys=False)

    def _save_anim(self, plan: FootstepPlan, output_file: str) -> None:
        from planning_through_contact.visualize.footstep_visualizer import (
            animate_footstep_plan,
        )

        animate_footstep_plan(
            self.config.robot, self.terrain, plan, output_file=output_file
        )

    def _save_trajectories(self, plan: FootstepPlan, output_file: str) -> None:
        from planning_through_contact.visualize.footstep_visualizer import (
            visualize_feet_trajectories,
            visualize_footstep_plan_trajectories,
        )

        visualize_feet_trajectories(
            plan, filename=output_file.split(".")[0] + "_feet" + ".png"
        )

        visualize_footstep_plan_trajectories(
            self.config.robot, plan, filename=output_file
        )

    def save_relaxed_animation(self, output_file: str) -> None:
        self._save_anim(self.restriction_plan, output_file)

    def save_rounded_animation(self, output_file: str) -> None:
        self._save_anim(self.rounded_plan, output_file)

    def save_analysis_to_folder(self, folder: str) -> None:
        """
        Saves all the analysis data and the plans themselves, as well as animations,
        to the given folder
        """
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        self.save_metrics_to_yaml(str(path / "metrics.yaml"))
        self.config.save(str(path / "config.yaml"))
        self.terrain.save(str(path / "terrain.yaml"))

        if self.restriction_metrics.success:
            self.save_relaxed_animation(str(path / "relaxed_plan_video.mp4"))
            self._save_trajectories(
                self.restriction_plan, str(path / "relaxed_plan_trajectories.png")
            )
            self.restriction_plan.save(str(path / "relaxed_plan.pkl"))

        if self.rounded_metrics.success:
            self.save_rounded_animation(str(path / "rounded_plan_video.mp4"))
            self._save_trajectories(
                self.rounded_plan, str(path / "rounded_plan_trajectories.png")
            )
            self.rounded_plan.save(str(path / "rounded_plan.pkl"))

    def get_unique_gcs_name(self) -> str:
        """
        Assigns this result a unique name based on the GCS Edges that it traverses.
        Every path with the same sequence of edges will be given this name.
        """

        if self.gcs_active_edges is None:
            raise RuntimeError("Cannot assign name when GCS edges are not provided.")

        hash = hash_gcs_edges(self.gcs_active_edges)
        return hash


@dataclass
class FootstepPlanSegmentProgram:
    def __init__(
        self,
        stone: InPlaneSteppingStone,
        one_or_two_feet: Literal["one_foot", "two_feet"],
        robot: PotatoRobot,
        config: FootstepPlanningConfig,
        name: Optional[str] = None,
        stone_for_last_foot: Optional[InPlaneSteppingStone] = None,
        eq_num_input_state: bool = False,
    ) -> None:
        """
        A wrapper class for constructing a nonlinear optimization program for the
        motion within a specified mode.

        @param stones_per_foot: If passed, each foot is restriced to be in contact with their
                                respective stone, in the order (L,R). If passed, stone is disregarded.
        @param eq_num_input_state: Normally, we have N states and N - 1 inputs (due to
                                   Forward Euler). However, when chaining multiple segments together
                                   one often wants N states as well, which this flag accomplishes.
        """
        self.robot = robot

        # Save the costs that we want to add to GCS/relaxation
        self.costs_to_use_for_gcs = []

        if stone_for_last_foot:
            stone_first, stone_last = stone, stone_for_last_foot
            self.name = f"{stone_first.name}_and_{stone_last.name}"
        else:
            self.name = f"{stone.name}"

        if name is not None:
            self.name = f"{self.name}_{name}"

        self.config = config
        self.two_feet = one_or_two_feet == "two_feet"
        if stone_for_last_foot:
            self.stone_first, self.stone_last = stone, stone_for_last_foot
        else:
            self.stone_first = self.stone_last = stone

        self.num_steps = self.config.period_steps

        self.prog = MathematicalProgram()

        # declare states
        self.num_states = self.num_steps

        self.p_WB = self.prog.NewContinuousVariables(self.num_states, 2, "p_WB")
        self.v_WB = self.prog.NewContinuousVariables(self.num_states, 2, "v_WB")
        self.theta_WB = self.prog.NewContinuousVariables(self.num_states, "theta_WB")
        self.omega_WB = self.prog.NewContinuousVariables(self.num_states, "omega_WB")

        ### declare inputs
        # first foot
        if eq_num_input_state:
            self.num_inputs = self.num_states
        else:
            self.num_inputs = self.num_states - 1

        self.p_WF1_x = self.prog.NewContinuousVariables(self.num_inputs, "p_WF1_x")
        self.f_F1_1W = self.prog.NewContinuousVariables(self.num_inputs, 2, "f_F1_1W")
        self.f_F1_2W = self.prog.NewContinuousVariables(self.num_inputs, 2, "f_F1_2W")
        if self.two_feet:
            # second foot
            self.p_WF2_x = self.prog.NewContinuousVariables(self.num_inputs, "p_WF2_x")
            self.f_F2_1W = self.prog.NewContinuousVariables(
                self.num_inputs, 2, "f_F2_1W"
            )
            self.f_F2_2W = self.prog.NewContinuousVariables(
                self.num_inputs, 2, "f_F2_2W"
            )

        self.p_WF1 = np.vstack(
            [self.p_WF1_x, np.full(self.p_WF1_x.shape, self.stone_first.z_pos)]
        ).T  # (num_steps, 2)
        if self.two_feet:
            self.p_WF2 = np.vstack(
                [self.p_WF2_x, np.full(self.p_WF2_x.shape, self.stone_last.z_pos)]
            ).T  # (num_steps, 2)

        # compute the foot position
        self.p_BF1_W = self.p_WF1 - self.p_WB[: self.num_inputs]
        if self.two_feet:
            self.p_BF2_W = self.p_WF2 - self.p_WB[: self.num_inputs]

        # auxilliary vars
        # TODO(bernhardpg): we might be able to get around this once we
        # have SDP constraints over the edges
        self.tau_F1_1 = self.prog.NewContinuousVariables(self.num_inputs, "tau_F1_1")
        self.tau_F1_2 = self.prog.NewContinuousVariables(self.num_inputs, "tau_F1_2")
        if self.two_feet:
            self.tau_F2_1 = self.prog.NewContinuousVariables(
                self.num_inputs, "tau_F2_1"
            )
            self.tau_F2_2 = self.prog.NewContinuousVariables(
                self.num_inputs, "tau_F2_2"
            )

        # linear acceleration
        g = np.array([0, -9.81])
        self.a_WB = (1 / robot.mass) * self.config.force_scale * (
            self.f_F1_1W + self.f_F1_2W
        ) + g
        if self.two_feet:
            self.a_WB += (
                (1 / robot.mass)
                * self.config.force_scale
                * (self.f_F2_1W + self.f_F2_2W)
            )

        # angular acceleration
        self.omega_dot_WB = (1 / robot.inertia) * (self.tau_F1_1 + self.tau_F1_2)
        if self.two_feet:
            self.omega_dot_WB += (1 / robot.inertia) * (self.tau_F2_1 + self.tau_F2_2)

        # contact points positions relative to CoM
        self.p_BF1_1W = self.p_BF1_W + np.array([robot.foot_length / 2, 0])
        self.p_BF1_2W = self.p_BF1_W - np.array([robot.foot_length / 2, 0])
        if self.two_feet:
            self.p_BF2_1W = self.p_BF2_W + np.array([robot.foot_length / 2, 0])
            self.p_BF2_2W = self.p_BF2_W - np.array([robot.foot_length / 2, 0])

        # torque = arm x force
        self.non_convex_constraints = []
        self.convex_concave_slack_vars = []
        self.convex_concave_relaxations = []
        for k in range(self.num_inputs):
            if config.use_convex_concave:
                cs_for_knot_point = []
                slack_vars_for_knot_point = []
                cross_prod = cross_product_2d_as_convex_concave(
                    self.prog,
                    self.p_BF1_1W[k],
                    self.f_F1_1W[k],
                    cs_for_knot_point,
                    slack_vars_for_knot_point,
                )
                c = self.prog.AddLinearConstraint(self.tau_F1_1[k] == cross_prod)
                self.convex_concave_relaxations.append(c)

                cross_prod = cross_product_2d_as_convex_concave(
                    self.prog,
                    self.p_BF1_2W[k],
                    self.f_F1_2W[k],
                    cs_for_knot_point,
                    slack_vars_for_knot_point,
                )
                c = self.prog.AddLinearConstraint(self.tau_F1_2[k] == cross_prod)
                self.convex_concave_relaxations.append(c)

                if self.two_feet:
                    cross_prod = cross_product_2d_as_convex_concave(
                        self.prog,
                        self.p_BF2_1W[k],
                        self.f_F2_1W[k],
                        cs_for_knot_point,
                        slack_vars_for_knot_point,
                    )
                    c = self.prog.AddLinearConstraint(self.tau_F2_1[k] == cross_prod)
                    self.convex_concave_relaxations.append(c)

                    cross_prod = cross_product_2d_as_convex_concave(
                        self.prog,
                        self.p_BF2_2W[k],
                        self.f_F2_2W[k],
                        cs_for_knot_point,
                        slack_vars_for_knot_point,
                    )
                    c = self.prog.AddLinearConstraint(self.tau_F2_2[k] == cross_prod)
                    self.convex_concave_relaxations.append(c)

                self.convex_concave_slack_vars.append(slack_vars_for_knot_point)

            else:  # add quadratic equality constraints
                cs_for_knot_point = []
                c = self.prog.AddQuadraticConstraint(
                    self.tau_F1_1[k] - cross_2d(self.p_BF1_1W[k], self.f_F1_1W[k]), 0, 0
                )
                cs_for_knot_point.append(c)
                c = self.prog.AddQuadraticConstraint(
                    self.tau_F1_2[k] - cross_2d(self.p_BF1_2W[k], self.f_F1_2W[k]), 0, 0
                )
                cs_for_knot_point.append(c)
                if self.two_feet:
                    c = self.prog.AddQuadraticConstraint(
                        self.tau_F2_1[k] - cross_2d(self.p_BF2_1W[k], self.f_F2_1W[k]),
                        0,
                        0,
                    )
                    cs_for_knot_point.append(c)
                    c = self.prog.AddQuadraticConstraint(
                        self.tau_F2_2[k] - cross_2d(self.p_BF2_2W[k], self.f_F2_2W[k]),
                        0,
                        0,
                    )
                    cs_for_knot_point.append(c)

            self.non_convex_constraints.append(cs_for_knot_point)

        # Stay on the stepping stone
        for k in range(self.num_inputs):
            self.prog.AddLinearConstraint(
                self.stone_first.x_min <= self.p_WF1_x[k] - robot.foot_length / 2
            )
            self.prog.AddLinearConstraint(
                self.p_WF1_x[k] + robot.foot_length / 2 <= self.stone_first.x_max
            )
            if self.two_feet:
                self.prog.AddLinearConstraint(
                    self.stone_last.x_min <= self.p_WF2_x[k] - robot.foot_length / 2
                )
                self.prog.AddLinearConstraint(
                    self.p_WF2_x[k] + robot.foot_length / 2 <= self.stone_last.x_max
                )

        # Don't move the feet too far from the robot
        for k in range(self.num_inputs):
            self.prog.AddLinearConstraint(
                self.p_WB[k][0] - self.p_WF1_x[k] <= robot.max_step_dist_from_robot
            )
            self.prog.AddLinearConstraint(
                self.p_WB[k][0] - self.p_WF1_x[k] >= -robot.max_step_dist_from_robot
            )
            if self.two_feet:
                self.prog.AddLinearConstraint(
                    self.p_WB[k][0] - self.p_WF2_x[k] <= robot.max_step_dist_from_robot
                )
                self.prog.AddLinearConstraint(
                    self.p_WB[k][0] - self.p_WF2_x[k] >= -robot.max_step_dist_from_robot
                )

        # constrain feet to not move too far from each other:
        for k in range(self.num_inputs):
            if self.two_feet:
                first_last_foot_distance = self.p_WF1_x[k] - self.p_WF2_x[k]
                self.prog.AddLinearConstraint(
                    first_last_foot_distance <= robot.step_span
                )
                self.prog.AddLinearConstraint(
                    first_last_foot_distance >= -robot.step_span
                )

        # Friction cones
        for k in range(self.num_inputs):
            # TODO(bernhardpg): Friction cone must be formulated differently
            # when we have tilted ground
            mu = 0.5  # TODO: move friction coeff
            for f in (self.f_F1_1W, self.f_F1_2W):
                self.prog.AddLinearConstraint(f[k][1] >= 0)
                self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])
            if self.two_feet:
                for f in (self.f_F2_1W, self.f_F2_2W):
                    self.prog.AddLinearConstraint(f[k][1] >= 0)
                    self.prog.AddLinearConstraint(f[k][0] <= mu * f[k][1])
                    self.prog.AddLinearConstraint(f[k][0] >= -mu * f[k][1])

        # dynamics
        if self.config.use_variable_timing:
            self.dt = self.prog.NewContinuousVariables(1, "dt").item()
            self.prog.AddLinearConstraint(self.dt >= 0.1)

            if self.config.cost.sq_time is not None:
                self.prog.AddQuadraticCost(
                    self.num_steps * self.config.cost.sq_time * self.dt**2
                )
        else:
            self.dt = self.config.dt

        for k in range(self.num_steps - 1):
            s_next = self.get_state(k + 1)
            s_curr = self.get_state(k)
            f = self.get_dynamics(k)
            # forward euler
            dynamics = s_next - (s_curr + self.dt * f)

            if self.config.use_variable_timing:
                for d in dynamics:
                    self.prog.AddQuadraticConstraint(d, 0, 0)
            else:
                # This constraint is linear if we don't have variable timing
                self.prog.AddLinearConstraint(eq(dynamics, 0))

        # feet can't move during segment
        for k in range(self.num_inputs - 1):
            const = eq(self.p_WF1[k], self.p_WF1[k + 1])
            for c in const:
                self.prog.AddLinearEqualityConstraint(c)
            if self.two_feet:
                const = eq(self.p_WF2[k], self.p_WF2[k + 1])
                for c in const:
                    self.prog.AddLinearEqualityConstraint(c)

        self.costs = {
            "sq_forces": [],
            "sq_torques": [],
            "sq_acc_lin": [],
            "sq_acc_rot": [],
            "sq_lin_vel": [],
            "sq_rot_vel": [],
            "sq_nominal_pose": [],
            "sq_time": [],
        }

        cost = config.cost

        # Squared timej
        if type(self.dt) is Variable and self.config.cost.sq_time is not None:
            c = self.prog.AddQuadraticCost(
                self.num_steps * self.config.cost.sq_time * self.dt**2
            )
            self.costs["sq_time"].append(c)

        # squared forces
        if cost.sq_force is not None:
            for k in range(self.num_inputs):
                f1 = self.f_F1_1W[k]
                f2 = self.f_F1_2W[k]
                sq_forces = f1.T @ f1 + f2.T @ f2
                if self.two_feet:
                    f1 = self.f_F2_1W[k]
                    f2 = self.f_F2_2W[k]
                    sq_forces += f1.T @ f1 + f2.T @ f2
                c = self.prog.AddQuadraticCost(
                    cost.sq_force * self.config.force_scale**2 * sq_forces
                )
                self.costs["sq_forces"].append(c)

        if True:  # this causes the relaxation gap to be high
            pass
        else:
            # Note: This cost term enforces the convex concave slack variables
            # to be equal (because of tau = Q+ - Q-), causing a relaxation gap.
            if self.config.use_convex_concave:
                pass
            else:
                # squared torques
                for k in range(self.num_inputs):
                    tau1 = self.tau_F1_1[k]
                    tau2 = self.tau_F1_2[k]
                    sq_torques = tau1**2 + tau2**2
                    if self.two_feet:
                        tau3 = self.tau_F2_1[k]
                        tau4 = self.tau_F2_2[k]
                        sq_torques += tau3**2 + tau4**2
                    c = self.prog.AddQuadraticCost(cost_torque * sq_torques)
                    self.costs["sq_torques"].append(c)

        # TODO: do we need these? Potentially remove
        # squared accelerations
        if cost.sq_acc_lin is not None:
            for k in range(self.num_inputs):
                sq_acc = self.a_WB[k].T @ self.a_WB[k]
                c = self.prog.AddQuadraticCost(cost.sq_acc_lin * sq_acc)
                self.costs["sq_acc_lin"].append(c)

        if cost.sq_acc_rot is not None:
            for k in range(self.num_inputs):
                sq_rot_acc = self.omega_dot_WB[k] ** 2
                c = self.prog.AddQuadraticCost(cost.sq_acc_rot * sq_rot_acc)
                self.costs["sq_acc_rot"].append(c)

        # squared robot velocity
        if cost.sq_vel_lin is not None:
            for k in range(self.num_inputs):
                v = self.v_WB[k]
                sq_lin_vel = v.T @ v
                c = self.prog.AddQuadraticCost(cost.sq_vel_lin * sq_lin_vel)
                self.costs["sq_lin_vel"].append(c)

        if cost.sq_vel_rot is not None:
            for k in range(self.num_inputs):
                sq_rot_vel = self.omega_WB[k] ** 2
                c = self.prog.AddQuadraticCost(cost.sq_vel_rot * sq_rot_vel)
                self.costs["sq_rot_vel"].append(c)

        # squared distance from nominal pose
        # TODO: Use the mean stone height?
        if cost.sq_nominal_pose:
            pose_offset = np.array(
                [0, self.stone_first.height, 0]
            )  # offset the stone height
            for k in range(self.num_inputs):
                pose = self.get_robot_pose(k) - pose_offset
                diff = pose - robot.get_nominal_pose()
                sq_diff = diff.T @ diff
                c = self.prog.AddQuadraticCost(cost.sq_nominal_pose * sq_diff)  # type: ignore
                self.costs["sq_nominal_pose"].append(c)

    def get_state(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        return np.concatenate(
            [self.p_WB[k], [self.theta_WB[k]], self.v_WB[k], [self.omega_WB[k]]]
        )

    def get_input(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.config.period_steps - 1
        if self.two_feet:
            return np.concatenate(
                [
                    self.f_F1_1W[k],
                    self.f_F1_2W[k],
                    [self.p_WF1_x[k]],
                    self.f_F2_1W[k],
                    self.f_F2_2W[k],
                    [self.p_WF2_x[k]],
                ]
            )
        else:
            return np.concatenate([self.f_F1_1W[k], self.f_F1_2W[k], [self.p_WF1_x[k]]])

    def constrain_foot_pos_ge(self, foot: Literal["first", "last"], x: float) -> None:
        """
        Constrain the given foot to have a position more than the given treshold x
        """
        if foot == "first":
            p_WF = self.p_WF1
        else:  # last
            p_WF = self.p_WF2

        for k in range(self.num_inputs):
            self.prog.AddLinearConstraint(p_WF[k][0] >= x)

    def constrain_foot_pos_le(self, foot: Literal["first", "last"], x: float) -> None:
        """
        Constrain the given foot to have a position less than the given treshold x
        """
        if foot == "first":
            p_WF = self.p_WF1
        else:  # last
            p_WF = self.p_WF2

        for k in range(self.num_inputs):
            self.prog.AddLinearConstraint(p_WF[k][0] <= x)

    def get_foot_pos(self, foot: Literal["first", "last"], k: int) -> Variable:
        """
        Returns the decision variable for a given foot for a given knot point idx.
        If the segment has only one foot contact, it returns that one foot always.
        """
        if k == -1:
            k = self.num_inputs - 1
        if self.two_feet:
            if foot == "first":
                return self.p_WF1_x[k]
            else:  # last
                return self.p_WF2_x[k]
        else:  # if only one foot we return that one foot
            return self.p_WF1_x[k]

    def get_dynamics(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.num_inputs - 1
        return np.concatenate(
            [self.v_WB[k], [self.omega_WB[k]], self.a_WB[k], [self.omega_dot_WB[k]]]
        )

    def get_var_in_vertex(self, var: Variable, vertex_vars: npt.NDArray) -> Variable:
        idxs = self.relaxed_prog.FindDecisionVariableIndex(var)
        return vertex_vars[idxs]

    def get_vars_in_vertex(
        self, vars: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        shape = vars.shape
        idxs = self.relaxed_prog.FindDecisionVariableIndices(vars.flatten())
        return vertex_vars[idxs].reshape(shape)

    def get_lin_exprs_in_vertex(
        self, exprs: npt.NDArray, vertex_vars: npt.NDArray
    ) -> npt.NDArray:
        original_shape = exprs.shape
        if len(exprs.shape) > 1:
            exprs = exprs.flatten()

        # note: the dynamics are always linear (we introduced some aux vars to achieve this)
        vars = Variables()
        for e in exprs:
            if type(e) == Variable:
                vars.insert(e)
            elif type(e) == Expression:
                vars.insert(e.GetVariables())
            elif type(e) == float:
                continue  # no variables to add
            else:
                raise RuntimeError("Unknown type")
        vars = list(vars)

        A, b = DecomposeAffineExpressions(exprs, vars)  # type: ignore
        idxs = self.relaxed_prog.FindDecisionVariableIndices(vars)  # type: ignore

        x = vertex_vars[idxs]

        exprs_with_vertex_vars = A @ x + b

        exprs_with_vertex_vars = exprs_with_vertex_vars.reshape(original_shape)
        return exprs_with_vertex_vars

    def get_robot_pose(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.num_states - 1
        return np.concatenate([self.p_WB[k], [self.theta_WB[k]]])

    def get_robot_spatial_vel(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.num_states - 1
        return np.concatenate([self.v_WB[k], [self.omega_WB[k]]])

    def get_robot_spatial_acc(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.num_inputs - 1
        return np.concatenate([self.a_WB[k], [self.omega_dot_WB[k]]])

    def get_vars(self, k: int) -> npt.NDArray:
        if k == -1:
            k = self.num_inputs - 1
        if self.two_feet:
            return np.concatenate(
                (
                    self.get_state(k),
                    self.get_input(k),
                    (
                        self.tau_F1_1[k],
                        self.tau_F1_2[k],
                        self.tau_F2_1[k],
                        self.tau_F2_2[k],
                    ),
                )
            )
        else:
            return np.concatenate(
                (
                    self.get_state(k),
                    self.get_input(k),
                    (self.tau_F1_1[k], self.tau_F1_2[k]),
                )
            )

    def add_pose_constraint(
        self, k: int, p_WB: npt.NDArray[np.float64], theta_WB: float
    ) -> None:
        if k == -1:
            k = self.config.period_steps - 1
        self.prog.AddLinearConstraint(eq(self.p_WB[k], p_WB))
        self.prog.AddLinearConstraint(self.theta_WB[k] == theta_WB)

    def add_spatial_vel_constraint(
        self, k: int, v_WB: npt.NDArray[np.float64], omega_WB: float
    ) -> None:
        if k == -1:
            k = self.config.period_steps - 1
        self.prog.AddLinearConstraint(eq(self.v_WB[k], v_WB))
        # self.prog.AddLinearConstraint(self.omega_WB[k] == omega_WB)

    def add_spatial_acc_constraint(
        self, k: int, a_WB: npt.NDArray[np.float64], omega_dot_WB: float
    ) -> None:
        if k == -1:
            k = self.config.period_steps - 1
        self.prog.AddLinearConstraint(eq(self.a_WB[k], a_WB))
        self.prog.AddLinearConstraint(self.omega_dot_WB[k] == omega_dot_WB)

    def add_equilibrium_constraint(self, k: int) -> None:
        """
        Enforce that all accelerations are 0 for knot point k.
        """
        if k == -1:
            k = self.config.period_steps - 2  # N - 1 inputs, i.e. N - 1 accelerations!
        self.add_spatial_acc_constraint(k, np.zeros((2,)), 0)

    def make_relaxed_prog(
        self,
        use_lp_approx: bool = False,
        use_groups: bool = False,
        use_implied_constraints: bool = False,
        trace_cost: Optional[float] = False,
    ) -> MathematicalProgram:

        # When we use the convex-concave procedure we already have a convex program
        if self.config.use_convex_concave:
            self.relaxed_prog = self.prog
            return self.relaxed_prog

        options = SemidefiniteRelaxationOptions()
        if not use_implied_constraints:
            options.set_to_weakest()

        if use_groups:
            if self.num_states == self.num_inputs:
                variable_groups = [
                    Variables(np.concatenate([self.get_vars(k), self.get_vars(k + 1)]))
                    for k in range(self.num_states - 1)
                ]
            else:
                # We have N states and N - 1 inputs
                variable_groups = [
                    Variables(np.concatenate([self.get_vars(k), self.get_vars(k + 1)]))
                    for k in range(self.num_inputs - 1)
                ]
                variable_groups.append(
                    Variables(self.get_state(self.num_states - 1))
                )  # add the last state

            # Add dt as a variable in each group
            if self.config.use_variable_timing:
                assert type(self.dt) is Variable
                for v in variable_groups:
                    v.insert(self.dt)

            assert self.num_states - 1 == len(variable_groups)

            self.relaxed_prog = MakeSemidefiniteRelaxation(
                self.prog, variable_groups=variable_groups, options=options
            )

        else:
            self.relaxed_prog = MakeSemidefiniteRelaxation(self.prog)

        # Save the PSD constraints (i.e. before we remove them with LP approxixation)
        self.semidefinite_relaxation_constraints = (
            self.relaxed_prog.positive_semidefinite_constraints()
        )

        # We add all the original quadratic costs to GCS
        if self.config.use_linearized_cost:
            self.costs_to_use_for_gcs.extend(self.relaxed_prog.GetAllCosts())
        else:
            self.costs_to_use_for_gcs.extend(self.prog.GetAllCosts())

        if trace_cost is not None:
            trace_costs = add_trace_cost_on_psd_cones(self.relaxed_prog, eps=trace_cost)
            self.costs_to_use_for_gcs.extend(trace_costs)

        if use_lp_approx:
            approximate_sdp_cones_with_linear_cones(self.relaxed_prog)

            # Constrain all the square expressions from the cost to be nonnegative so the
            # naive LP relaxation is not unbounded
            if self.config.use_linearized_cost:
                original_vars = self.prog.decision_variables()

                def _pure_square(cost):
                    vars = Variables(cost.variables())
                    for var in original_vars:
                        if var in vars:
                            return False
                    return True

                for cost in self.relaxed_prog.GetAllCosts():
                    if type(cost.evaluator()) is not LinearCost:
                        raise NotImplementedError(
                            "We do not yet support nonlinear costs"
                        )

                    if not _pure_square(cost):
                        expr = (cost.evaluator().Eval(cost.variables())).item()
                        self.relaxed_prog.AddLinearConstraint(expr >= 0)

            if False:
                # TODO: remove if no longer needed
                # This code adds the cuts that the used squares in the cost must be positive.

                # def _find_correct_psd_constraint(vars):
                #     vars = Variables(vars)
                #     for c in self.semidefinite_relaxation_constraints:
                #         c_vars = Variables(c.variables())
                #         if vars.IsSubsetOf(c_vars):
                #             return c
                #
                #     raise RuntimeError(
                #         f"The variables {vars} does not belong to any one PSD constraint"
                #     )

                def _make_homogenous(Q, p, r):
                    """
                    Converts 0.5 xᵀQx + pᵀx + r
                    to 0.5 [x; 1]ᵀ[Q p; pᵀ 2r] [x; 1]
                    """
                    # Ensure p is a column vector
                    p = p.reshape((-1, 1))

                    # Construct the block matrix
                    Q_bar = np.block([[Q, p], [p.T, 2 * r]])

                    return Q_bar

                def _get_indices_in_vars(subset_vars, vars) -> List[int]:
                    temp = MathematicalProgram()
                    temp.AddDecisionVariables(vars)
                    return temp.FindDecisionVariableIndices(subset_vars)

                # Add the cuts that the squared costs must be nonnegative (otherwise
                # the LP relaxation is unbounded)
                for cost in self.prog.GetAllCosts():
                    e = cost.evaluator()
                    assert type(e) is QuadraticCost

                    # These cuts are already added
                    is_simple_square = np.allclose(e.b(), 0) and np.isclose(e.c(), 0)
                    if not is_simple_square:
                        vars = cost.variables()
                        N = len(vars)
                        psd_constraint = _find_correct_psd_constraint(vars)
                        Y = get_X_from_psd_constraint(psd_constraint)
                        # Y = [X x^T; x 1]
                        x = Y[-1, :]

                        # Retrieve the principal minor in Y corresponding to `vars`,
                        # i.e. we retrieve the matrix Y_bar = [X_bar vars; vars^T, 1] where
                        # X_bar corresponds to the higher order terms for `vars`.
                        indices_in_vars = _get_indices_in_vars(vars, x)
                        indices = np.concatenate([indices_in_vars, [len(x) - 1]])
                        Y_bar = Y[np.ix_(indices, indices)]

                        Q_bar = _make_homogenous(e.Q(), e.b(), e.c())
                        if not Q_bar.shape == Y_bar.shape:
                            raise RuntimeError("Something must be wrong")

                        sq_must_be_positive = 0.5 * np.sum(Q_bar * Y_bar) >= 0
                        self.relaxed_prog.AddLinearConstraint(sq_must_be_positive)

        return self.relaxed_prog

    def get_convex_set(
        self, use_lp_approx: bool = False, use_implied_constraints: bool = False
    ) -> Spectrahedron:
        relaxed_prog = self.make_relaxed_prog(
            use_lp_approx=use_lp_approx,
            use_implied_constraints=use_implied_constraints,
            trace_cost=self.config.relaxation_trace_cost,
            use_groups=self.config.use_variable_grouping,
        )

        spectrahedron = Spectrahedron(relaxed_prog)
        return spectrahedron

    def get_solution(
        self,
        vars: np.ndarray,
        result: MathematicalProgramResult,
        vertex_vars: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if vertex_vars is None:
            return result.GetSolution(vars)
        return result.GetSolution(self.get_vars_in_vertex(vars, vertex_vars))

    def evaluate_expressions(
        self,
        exprs: np.ndarray,
        result: MathematicalProgramResult,
        vertex_vars: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if vertex_vars is None:
            return evaluate_np_expressions_array(exprs, result)
        return evaluate_np_expressions_array(
            self.get_lin_exprs_in_vertex(exprs, vertex_vars), result
        )

    def evaluate_with_result(
        self,
        result: MathematicalProgramResult,
        vertex_vars: Optional[np.ndarray] = None,
    ) -> FootstepPlan:
        p_WB = self.get_solution(self.p_WB, result, vertex_vars)
        theta_WB = self.get_solution(self.theta_WB, result, vertex_vars)
        p_WF1 = self.evaluate_expressions(self.p_WF1, result, vertex_vars)
        f_F1_1W, f_F1_2W = self.config.force_scale * self.get_solution(
            self.f_F1_1W, result, vertex_vars
        ), self.config.force_scale * self.get_solution(
            self.f_F1_2W, result, vertex_vars
        )
        tau_F1_1, tau_F1_2 = self.get_solution(
            self.tau_F1_1, result, vertex_vars
        ), self.get_solution(self.tau_F1_2, result, vertex_vars)

        if type(self.dt) is Variable:
            dt = self.get_solution(np.array([self.dt]), result, vertex_vars).item()
        else:
            dt = self.dt

        # Repeat the one timestep length for the entire segment
        dts = np.array([dt] * len(p_WB))

        first_foot = FootPlan(
            self.robot.foot_length,
            dt,
            p_WF1,
            [f_F1_1W, f_F1_2W],
            [tau_F1_1, tau_F1_2],
            p_WB,
        )

        if self.two_feet:
            p_WF2 = self.evaluate_expressions(self.p_WF2, result, vertex_vars)
            f_F2_1W, f_F2_2W = self.config.force_scale * self.get_solution(
                self.f_F2_1W, result, vertex_vars
            ), self.config.force_scale * self.get_solution(
                self.f_F2_2W, result, vertex_vars
            )
            tau_F2_1, tau_F2_2 = self.get_solution(
                self.tau_F2_1, result, vertex_vars
            ), self.get_solution(self.tau_F2_2, result, vertex_vars)

            second_foot = FootPlan(
                self.robot.foot_length,
                dt,
                p_WF2,
                [f_F2_1W, f_F2_2W],
                [tau_F2_1, tau_F2_2],
                p_WB,
            )

            return FootstepPlan(dts, p_WB, theta_WB, [first_foot, second_foot])

        return FootstepPlan(dts, p_WB, theta_WB, [first_foot])

    def evaluate_with_vertex_result(
        self, result: MathematicalProgramResult, vertex_vars: npt.NDArray
    ) -> FootstepPlan:
        return self.evaluate_with_result(result, vertex_vars=vertex_vars)

    def round_result(
        self, result: MathematicalProgramResult
    ) -> MathematicalProgramResult:
        x = result.GetSolution(self.prog.decision_variables())

        if self.config.use_convex_concave:
            for c in self.prog.rotated_lorentz_cone_constraints():
                self.prog.RemoveConstraint(c)  # type: ignore

            for c in sum(self.non_convex_constraints, []):
                self.prog.AddConstraint(c.evaluator(), c.variables())

        snopt = SnoptSolver()
        rounded_result = snopt.Solve(self.prog, initial_guess=x)  # type: ignore
        assert rounded_result.is_success()

        return rounded_result

    def round_with_result(
        self, result: MathematicalProgramResult
    ) -> Tuple[FootstepPlan, MathematicalProgramResult]:
        rounded_result = self.round_result(result)
        knot_points = self.evaluate_with_result(rounded_result)
        return knot_points, rounded_result

    def evaluate_and_round_with_result(
        self, relaxed_result: MathematicalProgramResult
    ) -> FootstepPlanResult:
        """
        Creates a FootstepPlanResult for this segment only. Makes a terrain with the stone for this segment.
        """
        one_stone_terrain = InPlaneTerrain()
        one_stone_terrain.stepping_stones.append(self.stone_first)
        one_stone_terrain.stepping_stones.append(self.stone_last)
        relaxed_plan = self.evaluate_with_result(relaxed_result)

        import time

        curr_time = time.time()
        rounded_plan, rounded_result = self.round_with_result(relaxed_result)
        elapsed_time = time.time() - curr_time

        plan_result = FootstepPlanResult.from_results(
            one_stone_terrain,
            self.config,
            relaxed_result,
            relaxed_plan,
            rounded_result,
            rounded_plan,
            elapsed_time,
        )
        return plan_result

    def evaluate_costs_with_result(
        self, result: MathematicalProgramResult, vertex_vars: npt.NDArray
    ) -> Dict[str, List[float]]:
        cost_vals = {}
        for key, val in self.costs.items():
            cost_vals[key] = []

            for binding in val:
                vars = self.get_solution(
                    binding.variables(), result, vertex_vars=vertex_vars
                )
                cost_vals[key].append(binding.evaluator().Eval(vars))

        for key in cost_vals:
            cost_vals[key] = np.array(cost_vals[key])

        cost_vals["sum"] = np.sum([np.sum(vals) for vals in cost_vals.values()])

        return cost_vals

    def evaluate_non_convex_constraints_with_result(
        self, result: MathematicalProgramResult
    ) -> npt.NDArray[np.float64]:
        constraint_violations = []
        for cs in self.non_convex_constraints:
            violations_for_knot_point = []
            for c in cs:
                var_vals = result.GetSolution(c.variables())
                violation = np.abs(c.evaluator().Eval(var_vals))
                violations_for_knot_point.append(violation)
            constraint_violations.append(np.concatenate(violations_for_knot_point))

        return np.vstack(constraint_violations).T  # (num_steps, num_torques)
