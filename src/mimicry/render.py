from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from math import cos, sin, tau

import pyglet
from pyglet import shapes
from pyglet.window import Window
from pyglet.math import Vec3

from mimicry.data import Bounds, Car, Location, State


def draw_sensors(
    sensors: Iterable[Location],
    batch: shapes.Batch,
    radius: float = 0.01
) -> list[shapes.Circle]:
    blue = (50, 100, 255, 255)
    group = pyglet.graphics.Group(order=3)
    return [
        shapes.Circle(sensor.x, sensor.y, radius, batch=batch, color=blue, group=group)
        for sensor in sensors
    ]

def draw_car(car: Car, batch: shapes.Batch) -> shapes.Triangle:
    size = 0.02
    return shapes.Triangle(
        car.location.x + 4.0*size*cos(car.angle),
        car.location.y + 4.0*size*sin(car.angle),
        car.location.x + size*cos(car.angle + tau/3.0),
        car.location.y + size*sin(car.angle + tau/3.0),
        car.location.x + size*cos(car.angle - tau/3.0),
        car.location.y + size*sin(car.angle - tau/3.0),
        color=(255,0,0,255),
        batch=batch,
    )

def draw_cars(
    cars: Iterable[Car],
    batch: shapes.Batch,
) -> list[shapes.Triangle] :
    return [draw_car(car, batch) for car in cars]


def car_track_triangles(
    batch: shapes.Batch,
) -> tuple[Bounds, Iterable[shapes.Triangle]]:
    left = 0.0
    mid = 2.0
    right = 4.0
    top = 2.0
    bot = 0.0
    bounds = Bounds(-1.2, 5.2, -1.2, 3.2)
    return bounds, [
        # left
        shapes.Triangle(-0.2, -0.2, -0.2, 2.2, 0.2, -0.2, batch=batch),
        shapes.Triangle(0.2, -0.2, -0.2, 2.2, 0.2, 2.2, batch=batch),
        # right
        shapes.Triangle(right-0.2, -0.2, right-0.2, 2.2, right+0.2, -0.2, batch=batch),
        shapes.Triangle(right+0.2, -0.2, right-0.2, 2.2, right+0.2, 2.2, batch=batch),
        # bottom
        shapes.Triangle(0.2, -0.2, 0.2, 0.2, 3.8, -0.2, batch=batch),
        shapes.Triangle(3.8, -0.2, 0.2, 0.2, 3.8, 0.2, batch=batch),
        # top
        shapes.Triangle(0.2, top-0.2, 0.2, top+0.2, 3.8, top-0.2, batch=batch),
        shapes.Triangle(3.8, top-0.2, 0.2, top+0.2, 3.8, top+0.2, batch=batch),
        # top left
        shapes.Triangle(-1.0, top-0.2, -1.2, top+0.0, -1.0, top+0.2, batch=batch),
        shapes.Triangle(-1.0, top-0.2, -1.0, top+0.2, -0.2, top-0.2, batch=batch),
        shapes.Triangle(-0.2, top-0.2, -1.0, top+0.2, -0.2, top+0.2, batch=batch),
        # bottom left
        shapes.Triangle(-1.0, bot-0.2, -1.2, bot+0.0, -1.0, bot+0.2, batch=batch),
        shapes.Triangle(-1.0, bot-0.2, -1.0, bot+0.2, -0.2, bot-0.2, batch=batch),
        shapes.Triangle(-0.2, bot-0.2, -1.0, bot+0.2, -0.2, bot+0.2, batch=batch),
        # top right
        shapes.Triangle(5.0, top-0.2, 5.2, top+0.0, 5.0, top+0.2, batch=batch),
        shapes.Triangle(5.0, top-0.2, 4.2, top-0.2, 4.2, top+0.2, batch=batch),
        shapes.Triangle(4.2, top+0.2, 5.0, top+0.2, 5.0, top-0.2, batch=batch),
        # bottom right
        shapes.Triangle(5.0, bot-0.2, 5.2, bot+0.0, 5.0, bot+0.2, batch=batch),
        shapes.Triangle(5.0, bot-0.2, 4.2, bot-0.2, 4.2, bot+0.2, batch=batch),
        shapes.Triangle(4.2, bot+0.2, 5.0, bot+0.2, 5.0, bot-0.2, batch=batch),
        # left top
        shapes.Triangle(left-0.2, 2.2, left-0.2, 3.0, left+0.2, 2.2, batch=batch),
        shapes.Triangle(left+0.2, 2.2, left-0.2, 3.0, left+0.2, 3.0, batch=batch),
        shapes.Triangle(left-0.2, 3.0, left+0.0, 3.2, left+0.2, 3.0, batch=batch),
        # mid top
        shapes.Triangle(mid-0.2, 2.2, mid-0.2, 3.0, mid+0.2, 2.2, batch=batch),
        shapes.Triangle(mid+0.2, 2.2, mid-0.2, 3.0, mid+0.2, 3.0, batch=batch),
        shapes.Triangle(mid-0.2, 3.0, mid+0.0, 3.2, mid+0.2, 3.0, batch=batch),
        # right top
        shapes.Triangle(right-0.2, 2.2, right-0.2, 3.0, right+0.2, 2.2, batch=batch),
        shapes.Triangle(right+0.2, 2.2, right-0.2, 3.0, right+0.2, 3.0, batch=batch),
        shapes.Triangle(right-0.2, 3.0, right+0.0, 3.2, right+0.2, 3.0, batch=batch),
        # left bottom
        shapes.Triangle(left-0.2, -0.2, left-0.2, -1.0, left+0.2, -0.2, batch=batch),
        shapes.Triangle(left+0.2, -0.2, left-0.2, -1.0, left+0.2, -1.0, batch=batch),
        shapes.Triangle(left-0.2, -1.0, left+0.0, -1.2, left+0.2, -1.0, batch=batch),
        # mid bottom
        shapes.Triangle(mid-0.2, -0.2, mid-0.2, -1.0, mid+0.2, -0.2, batch=batch),
        shapes.Triangle(mid+0.2, -0.2, mid-0.2, -1.0, mid+0.2, -1.0, batch=batch),
        shapes.Triangle(mid-0.2, -1.0, mid+0.0, -1.2, mid+0.2, -1.0, batch=batch),
        # right bottom
        shapes.Triangle(right-0.2, -0.2, right-0.2, -1.0, right+0.2, -0.2, batch=batch),
        shapes.Triangle(right+0.2, -0.2, right-0.2, -1.0, right+0.2, -1.0, batch=batch),
        shapes.Triangle(right-0.2, -1.0, right+0.0, -1.2, right+0.2, -1.0, batch=batch),
        # bottom mid
        shapes.Triangle(mid-0.2, 0.2, mid-0.2, 0.8, mid+0.2, 0.8, batch=batch),
        shapes.Triangle(mid-0.2, 0.2, mid+0.2, 0.8, mid+0.2, 0.2, batch=batch),
        shapes.Triangle(mid+0.2, 0.8, mid-0.2, 0.8, mid+0.0, 0.9, batch=batch),
        # top mid
        shapes.Triangle(mid+0.2, 1.2, mid-0.2, 1.2, mid-0.2, 1.8, batch=batch),
        shapes.Triangle(mid+0.2, 1.2, mid-0.2, 1.8, mid+0.2, 1.8, batch=batch),
        shapes.Triangle(mid+0.0, 1.1, mid-0.2, 1.2, mid+0.2, 1.2, batch=batch),
    ]

@contextmanager
def scale_camera(window: Window, bounds: Bounds):
    # TODO preserve aspect ratio
    xscale = window.width / (bounds.xmax - bounds.xmin)
    yscale = window.height / (bounds.ymax - bounds.ymin)
    original_matrix = window.view
    try:
        view_matrix = window.view.scale(Vec3(xscale, yscale, 1.0))
        view_matrix = view_matrix.translate(Vec3(-bounds.xmin, -bounds.ymin, 0.0))
        window.view = view_matrix
        yield
    finally:
        window.view = original_matrix

def drawer(
    window: pyglet.window.Window,
    state: State,
    bounds: Bounds,
    background: shapes.Batch,
):
    batch = pyglet.graphics.Batch()
    sensors = pyglet.graphics.Batch()
    @window.event
    def on_draw():
        window.clear()
        with scale_camera(window, bounds):
            background.draw()
            _ = draw_cars(state.cars, batch)
            batch.draw()
            _ = draw_sensors(state.sensors, sensors)
            sensors.draw()