from collections.abc import Callable, Iterable
from dataclasses import replace
from pyglet import shapes
import numpy as np
import numpy.typing as npt
from math import sin, cos, tau
import torch
from torch import Tensor

from mimicry.data import Bounds, Location, Car, Polar
from mimicry.network import Agent

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
    sensor_field: Iterable[Polar],
) -> list[Location]:
    x, y = car.location
    return [
        Location(
            x + d * cos(angle + car.angle),
            y + d * sin(angle + car.angle),
        )
        for (d, angle) in sensor_field
    ]

def sensor_values(
    sensors: Iterable[Location],
    inside: Callable[[Location], bool],
    device: torch.device,
) -> Tensor:
    values = [
        1.0 if inside(sensor) else 0.0
        for sensor in sensors
    ]
    return torch.tensor(
        values,
        device=device,
        dtype=torch.float32,
    )

def replicate_car(source: Car, target: Car):
    target.location = source.location
    target.angle = source.angle

def step_car(
    rng: np.random.Generator,
    car: Car,
    agent: Agent,
    sensor_field: Iterable[Polar],
    inside: Callable[[Location], bool],
    motor_values: npt.NDArray[np.float32],
    device: torch.device,
):
    sensors = car_sensors(car, sensor_field)
    values = sensor_values(sensors, inside, device)
    motors, carries = agent.model(values, agent.carries)
    agent.carries = carries
    motor_probs = torch.exp(motors).cpu().numpy()
    sampled = rng.choice(np.arange(len(motor_probs)), size=1, p=motor_probs)
    motor = float(motor_values[sampled])
    return move(turn(car, motor))

def turn(car: Car, turn: float, turn_rate = tau/40) -> Car:
    angle = (car.angle + turn*turn_rate) % tau
    return replace(car, angle=angle)

def move(car: Car, speed=0.05) -> Car:
    x, y = car.location
    location = Location(
        x + cos(car.angle) * speed,
        y + sin(car.angle) * speed,
    )
    return replace(car, location=location)
