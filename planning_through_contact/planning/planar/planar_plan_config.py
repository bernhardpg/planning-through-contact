from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt

from planning_through_contact.geometry.collision_geometry.box_2d import Box2d
from planning_through_contact.geometry.collision_geometry.collision_geometry import (
    CollisionGeometry,
)
from planning_through_contact.geometry.rigid_body import RigidBody


@dataclass
class BoxWorkspace:
    width: float = 0.5
    height: float = 0.5
    center: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )

    @property
    def x_min(self) -> float:
        return self.center[0] - self.width / 2

    @property
    def x_max(self) -> float:
        return self.center[0] + self.width / 2

    @property
    def y_min(self) -> float:
        return self.center[0] - self.height / 2

    @property
    def y_max(self) -> float:
        return self.center[0] + self.height / 2

    @property
    def bounds(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        lb = np.array([self.x_min, self.y_min], dtype=np.float64)
        ub = np.array([self.x_max, self.y_max], dtype=np.float64)
        return lb, ub


@dataclass
class PlanarPushingWorkspace:
    # Currently we don't use the workspace constraints for the slider, because
    # the full Semidefinite relaxation of the QCQP creates a ton of
    # constraints, which really slows down the optimization!
    slider: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(
            width=1.0, height=1.0, center=np.array([0.4, 0.0])
        )
    )
    pusher: BoxWorkspace = field(
        default_factory=lambda: BoxWorkspace(width=0.8, height=0.8)
    )


@dataclass
class SliderPusherSystemConfig:
    slider: RigidBody = field(
        default_factory=lambda: RigidBody(
            name="box", geometry=Box2d(width=0.15, height=0.15), mass=0.1
        )
    )
    pusher_radius: float = 0.015
    friction_coeff_table_slider: float = 0.5
    friction_coeff_slider_pusher: float = 0.5
    grav_acc: float = 9.81

    @cached_property
    def f_max(self) -> float:
        return self.friction_coeff_table_slider * self.grav_acc * self.slider.mass

    @cached_property
    def integration_constant(self) -> float:
        return 0.6

    @cached_property
    def max_contact_radius(self) -> float:
        geometry = self.slider.geometry
        if isinstance(geometry, Box2d):
            return np.sqrt((geometry.width / 2) ** 2 + (geometry.height) ** 2)
        else:
            raise NotImplementedError(
                f"Integration constant for {type(geometry)} is not implemented"
            )

    @cached_property
    def tau_max(self) -> float:
        return self.f_max * self.max_contact_radius * self.integration_constant

    @cached_property
    def ellipsoidal_limit_surface(self) -> npt.NDArray[np.float64]:
        D = np.diag([1 / self.f_max**2, 1 / self.f_max**2, 1 / self.tau_max**2])
        return D

    @cached_property
    def limit_surface_const(self) -> float:
        return (self.max_contact_radius * self.integration_constant) ** -2


@dataclass
class PlanarSolverParams:
    gcs_max_rounded_paths: int = 10
    # NOTE: Currently, there is no way to solve the MISDP, so this must be true
    gcs_convex_relaxation: bool = True
    print_flows: bool = False
    assert_determinants: bool = True
    nonlinear_traj_rounding: bool = True
    print_solver_output: bool = False
    measure_solve_time: bool = False
    print_path: bool = False
    print_cost: bool = False
    get_rounded_and_original_traj: bool = False


@dataclass
class PlanarCostFunctionTerms:
    # Non-collision
    cost_param_avoidance_lin: float = 0.1
    cost_param_avoidance_quad_dist: float = 0.2
    cost_param_avoidance_quad_weight: float = 0.4
    cost_param_avoidance_socp_weight: float = 0.001
    cost_param_eucl: float = 1.0
    # Face contact
    cost_param_lin_vels: float = 1.0
    cost_param_ang_vels: float = 1.0
    cost_param_forces: float = 1.0


@dataclass
class PlanarPlanConfig:
    num_knot_points_contact: int = 4
    num_knot_points_non_collision: int = 2
    time_in_contact: float = 2  # TODO: remove, no time
    time_non_collision: float = 0.5  # TODO: remove, there is no time
    avoid_object: bool = False
    allow_teleportation: bool = False
    avoidance_cost: Literal[
        "linear",
        "quadratic",
        "socp",
        "socp_single_mode",  # NOTE: The single mode is only used to test one non-collision mode at a time
    ] = "quadratic"
    minimize_squared_eucl_dist: bool = True
    use_eq_elimination: bool = False
    penalize_mode_transitions: bool = False
    use_entry_and_exit_subgraphs: bool = True
    no_cycles: bool = False
    workspace: PlanarPushingWorkspace = field(
        default_factory=lambda: PlanarPushingWorkspace()
    )
    dynamics_config: SliderPusherSystemConfig = field(
        default_factory=lambda: SliderPusherSystemConfig()
    )
    cost_terms: PlanarCostFunctionTerms = field(
        default_factory=lambda: PlanarCostFunctionTerms()
    )
    use_approx_exponential_map: bool = True

    @property
    def slider_geometry(self) -> CollisionGeometry:
        return self.dynamics_config.slider.geometry

    @property
    def pusher_radius(self) -> float:
        return self.dynamics_config.pusher_radius
