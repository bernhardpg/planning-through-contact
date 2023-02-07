from enum import Enum
from typing import NamedTuple


class ContactPosition(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING = 2


class PolytopeContactLocation(NamedTuple):
    pos: ContactPosition
    idx: int

