from enum import Enum


class ContactPosition(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING_LEFT = 2
    SLIDING_RIGHT = 3


class ContactType(Enum):
    ONE_SIDED_POINT_CONTACT = 1
    POINT_CONTACT = 2
    LINE_CONTACT = 3
