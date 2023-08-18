from collections.abc import Callable, Iterable
from pyglet import shapes
import numpy as np
from math import sin, cos, tau

from mimicry.data import Bounds, Location, Car

def random_car(
    rng: np.random.Generator,
    bounds: Bounds,
    on_track: Callable[[Location], bool],
) -> Car:
    angle = rng.random() * tau
    while True:
        x = bounds.xmin + rng.random() * (bounds.xmax - bounds.xmin)
        y = bounds.ymin + rng.random() * (bounds.ymax - bounds.ymin)
        location = Location(x, y)
        if on_track(location):
            return Car(location, angle)

def on_track(floor: Iterable[shapes.ShapeBase]):
    def check(location: Location) -> bool:
        for shape in floor:
            if location in shape:
                return True
        return False
    return check

def car_sensors(
    car: Car,
    sensor_field: Iterable[tuple[float, float]],
) -> list[Location]:
    x, y = car.location
    return [
        Location(
            x + d * cos(angle + car.angle),
            y + d * sin(angle + car.angle),
        )
        for (d, angle) in sensor_field
    ]
