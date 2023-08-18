import pyglet
import numpy as np
from math import sin, cos, tau

from mimicry.render import car_track_triangles, drawer, State
from mimicry.car import car_sensors, on_track, random_car
from mimicry.data import Location


def update(rng, alive, bounds, sensor_field, state: State):
    speed = 0.05
    for car in state.cars:
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
    state.sensors = car_sensors(state.cars[0], sensor_field)

def main():
    rng = np.random.default_rng()
    sensor_field = [
        (d, a*tau)
        for d in (0.1, 0.2, 0.3, 0.4 ,0.5)
        for a in (0.25, 0.15, 0.05, -0.05, -0.15, -0.25)
    ]
    window = pyglet.window.Window(1920, 1080, resizable=True)
    background = pyglet.graphics.Batch()
    bounds, floor = car_track_triangles(background)
    alive = on_track(floor)
    population = 100
    cars = [
        random_car(rng, bounds, alive)
        for _ in range(population)
    ]
    car = cars[0]
    state = State(cars, car_sensors(car, sensor_field))
    drawer(window, state, bounds, background)
    running = True
    @window.event
    def on_close():
        nonlocal running
        running = False
    while running:
        pyglet.clock.tick()
        update(rng, alive, bounds, sensor_field, state)

        for win in pyglet.app.windows:
            win.switch_to()
            win.dispatch_events()
            win.dispatch_event('on_draw')
            win.flip()

if __name__ == '__main__':
    main()
