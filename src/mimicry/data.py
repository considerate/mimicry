from typing import NamedTuple
from dataclasses import dataclass

from torch import Tensor


@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

class Location(NamedTuple):
    x: float
    y: float

class Polar(NamedTuple):
    magnitude: float
    angle: float

Polygon = list[Location]

@dataclass
class Car:
    location: Location
    angle: float

Carry = tuple[Tensor, Tensor]
Carries = tuple[Carry, Carry, Carry]

@dataclass
class State:
    cars: list[Car]
    sensors: list[Location]
    history: list[list[tuple[Tensor, int, Carries]]]
    trails: list[list[list[Location]]]
