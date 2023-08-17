from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import NamedTuple
import pyglet
from pyglet import shapes
from dataclasses import dataclass
import time
import numpy as np
from math import sin, cos, tau
from pyglet.math import Vec3

from pyglet.window import Window


@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

class Location(NamedTuple):
    x: float
    y: float

Polygon = list[Location]

@dataclass
class Car:
    location: Location
    angle: float

class Arena(NamedTuple):
    outer: Polygon
    inner: Polygon
    bounds: Bounds

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
):
    return [draw_car(car, batch) for car in cars]


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

def random_car(
    rng: np.random.Generator,
    bounds: Bounds,
    on_track: Callable[[Location], bool],
) -> Car:
    angle = rng.random() * tau
    return Car(Location(0.0, 0.0), angle)

def on_track(floor: Iterable[shapes.ShapeBase]):
    def check(location: Location) -> bool:
        for shape in floor:
            if location in shape:
                return True
        return False
    return check

def main():
    window = pyglet.window.Window(1920, 1080, resizable=True)
    background = pyglet.graphics.Batch()
    bounds, floor = car_track_triangles(background)
    alive = on_track(floor)
    rng = np.random.default_rng()
    population = 100
    cars = [
        random_car(rng, bounds, alive)
        for _ in range(population)
    ]
    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        with scale_camera(window, bounds):
            background.draw()
            _ = draw_cars(cars, batch)
            batch.draw()
    running = True
    @window.event
    def on_close():
        nonlocal running
        running = False
    while running:
        pyglet.clock.tick()
        speed = 0.05
        for car in cars:
            x, y = car.location
            location = Location(
                x + cos(car.angle) * speed,
                y + sin(car.angle) * speed,
            )
            if alive(location):
                car.location = location
                car.angle = (car.angle + 0.1 * (0.5 - rng.random())) % tau
            else:
                # TODO replicate
                new_car = random_car(rng, bounds, alive)
                car.location = new_car.location
                car.angle = new_car.angle

        for win in pyglet.app.windows:
            win.switch_to()
            win.dispatch_events()
            win.dispatch_event('on_draw')
            win.flip()
        time.sleep(0.01)

if __name__ == '__main__':
    main()
