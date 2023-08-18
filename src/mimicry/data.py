from typing import NamedTuple
from dataclasses import dataclass


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

@dataclass
class State:
    cars: list[Car]
    sensors: list[Location]
