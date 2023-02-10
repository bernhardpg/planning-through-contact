from enum import Enum


class ContactLocation(Enum):
    FACE = 1
    VERTEX = 2


class ContactMode(Enum):
    ROLLING = 1
    SLIDING_LEFT = 2
    SLIDING_RIGHT = 3
