from dataclasses import dataclass


@dataclass
class PlanarPlanSpecs:
    num_knot_points_contact: int = 4
    num_knot_points_non_collision: int = 2
    time_in_contact: float = 2
    time_non_collision: float = 0.5
    pusher_radius: float = 0.01
