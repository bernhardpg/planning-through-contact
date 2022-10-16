import itertools
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy.typing as npt
import pydrake.symbolic as sym
from pydrake.geometry.optimization import ConvexSet

from geometry.polyhedron import PolyhedronFormulator


class ContactModeType(Enum):
    NO_CONTACT = 1
    ROLLING = 2
    SLIDING_POSITIVE = 3
    SLIDING_NEGATIVE = 3


class PositionModeType(Enum):
    LEFT = 1
    TOP_LEFT = 2
    TOP = 3
    TOP_RIGHT = 4
    RIGHT = 5
    BOTTOM_RIGHT = 6
    BOTTOM = 7
    BOTTOM_LEFT = 8


@dataclass(eq=True)
class ContactModeConfig:
    modes: Dict[str, ContactModeType]
    additional_constraints: Optional[npt.NDArray[sym.Formula]] = None

    def calculate_match(self, other) -> int:
        modes_self = list(self.modes.values())
        modes_other = list(other.modes.values())

        num_not_equal = sum([m1 != m2 for m1, m2 in zip(modes_self, modes_other)])
        return num_not_equal

    @staticmethod
    def switch_mode_for_pair(
        pair_name: str, old_modes: [str, ContactModeType]
    ) -> Dict[str, ContactModeType]:
        new_modes = dict(old_modes)
        if old_modes[pair_name] == ContactModeType.NO_CONTACT:
            new_modes[pair_name] = ContactModeType.ROLLING
        elif old_modes[pair_name] == ContactModeType.ROLLING:
            new_modes[pair_name] = ContactModeType.NO_CONTACT
        return new_modes

    @classmethod
    def create_adjacent_mode(
        cls, pair_to_switch: str, mode_cfg: "ContactModeConfig"
    ) -> "ContactModeConfig":
        new_modes = cls.switch_mode_for_pair(pair_to_switch, mode_cfg.modes)
        return cls(new_modes)

    @classmethod
    def create_all_adjacent_modes(
        cls, mode_cfg: "ContactModeConfig"
    ) -> List["ContactModeConfig"]:
        # Adjacent mode to mode with additional constraints is to first remove these
        if mode_cfg.additional_constraints is not None:
            return [cls(mode_cfg.modes)]
        else:
            new_modes = [
                ContactModeConfig.create_adjacent_mode(pair, mode_cfg)
                for pair in mode_cfg.modes.keys()
            ]
            return new_modes


@dataclass(order=True)
class PrioritizedContactModeConfig:
    priority: int
    item: ContactModeConfig = field(compare=False)


@dataclass
class ContactMode:
    pair_name: str
    constraints: List[npt.NDArray[sym.Formula]]
    all_vars: npt.NDArray[sym.Variable]
    type: ContactModeType
    polyhedron: ConvexSet = field(init=False, repr=False)
    name: str = field(init=False, repr=False)
    CONTACT_TYPE_ABBREVIATIONS: Dict[ContactModeType, str] = field(
        default_factory=lambda: {
            ContactModeType.NO_CONTACT: "NC",
            ContactModeType.ROLLING: "RL",
        }
    )

    def __post_init__(self):
        self.polyhedron = PolyhedronFormulator(self.constraints).formulate_polyhedron(
            variables=self.all_vars, make_bounded=True
        )
        self.name = f"{self.pair_name}[{self.CONTACT_TYPE_ABBREVIATIONS[self.type]}]"


# TODO make this part of collisionPairHandler?
def calc_intersection_of_contact_modes(
    modes: List[ContactMode],
) -> Tuple[bool, Optional[ConvexSet]]:
    pairwise_combinations = itertools.combinations(modes, 2)
    all_modes_intersect = all(
        map(
            lambda pair: pair[0].polyhedron.IntersectsWith(pair[1].polyhedron),
            pairwise_combinations,
        )
    )
    if all_modes_intersect:
        polys = [m.polyhedron for m in modes]
        intersection = reduce(lambda p1, p2: p1.Intersection(p2), polys)
        names = [m.name for m in modes]
        name = " | ".join(names)
        return (True, (name, intersection))
    else:
        return (False, (None, None))
