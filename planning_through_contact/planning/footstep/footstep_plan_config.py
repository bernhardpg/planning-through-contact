from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


# TODO(bernhardpg): Move to another location?
@dataclass
class PotatoRobot:
    # TODO(bernhardpg): set reasonable inertia properties
    # This is from a sphere with mass = 50 kg and all axis = 0.5 meter
    mass: float = 50.0  # kg
    # TODO(bernhardpg): compute inertia from dimensions
    inertia: float = 5.0  # kg m**2
    foot_length: float = 0.3  # m
    foot_height: float = 0.15  # m
    step_span: float = 0.8  # m
    max_step_dist_from_robot: float = 0.4  # m
    desired_com_height: float = 1.3  # m
    size: Tuple[float, float, float] = (0.5, 0.5, 1.0)

    @property
    def width(self) -> float:
        return self.size[0]

    @property
    def depth(self) -> float:
        return self.size[1]

    @property
    def height(self) -> float:
        return self.size[2]

    def get_nominal_pose(self) -> npt.NDArray:  # (3,)
        return np.array([0, self.desired_com_height, 0])


@dataclass
class FootstepCost:
    sq_force: Optional[float] = 1e-5
    sq_torque: Optional[float] = 1e-3
    sq_vel_lin: Optional[float] = 10.0
    sq_vel_rot: Optional[float] = 0.1
    sq_acc_lin: Optional[float] = 100.0
    sq_acc_rot: Optional[float] = 1.0
    sq_nominal_pose: Optional[float] = 5.0
    sq_time: Optional[float] = 1.0

    @classmethod
    def with_none(cls) -> "FootstepCost":
        return cls(
            sq_force=None,
            sq_torque=None,
            sq_vel_lin=None,
            sq_vel_rot=None,
            sq_acc_lin=None,
            sq_acc_rot=None,
            sq_nominal_pose=None,
        )


@dataclass
class FootstepPlanningConfig:
    cost: FootstepCost = field(default_factory=lambda: FootstepCost())
    robot: PotatoRobot = field(default_factory=lambda: PotatoRobot())
    period: float = 1.0
    period_steps: int = 3
    initial_is_equilibrium: bool = False
    use_lp_approx: bool = False
    use_convex_concave: bool = False
    use_linearized_cost: bool = True
    max_rounded_paths: int = 10
    use_implied_constraints: bool = False
    use_variable_timing: bool = False
    relaxation_trace_cost: Optional[float] = None
    use_variable_grouping: bool = True
    force_scale: float = 1

    @property
    def dt(self) -> float:
        return self.period / self.period_steps

    def to_dict(self) -> dict:
        data = asdict(self)
        data["robot"] = asdict(self.robot)  # Ensure robot is also converted to dict
        return data

    def save(self, file_path: str) -> None:
        import yaml

        with open(file_path, "w") as yaml_file:
            yaml.dump(
                self.to_dict(),
                yaml_file,
                default_flow_style=False,
                sort_keys=True,
            )

    @classmethod
    def from_dict(cls, data: dict) -> "FootstepPlanningConfig":
        robot_data = data.pop("robot")
        robot = PotatoRobot(**robot_data)
        config = cls(robot=robot, **data)
        return config

    @classmethod
    def load(cls, file_path: str) -> "FootstepPlanningConfig":
        import yaml

        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        return cls.from_dict(data)
