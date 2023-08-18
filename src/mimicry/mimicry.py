from collections.abc import Callable, Iterable, Sequence
import pyglet
import numpy as np
import numpy.typing as npt
from math import tau
import torch

from mimicry.render import car_track_triangles, drawer
from mimicry.car import car_sensors, on_track, random_car, replicate_car, step_car
from mimicry.data import Bounds, Location, Polar, State
from mimicry.network import Agent, create_agent


def update(
    rng: np.random.Generator,
    on_track: Callable[[Location], bool],
    bounds: Bounds,
    sensor_field: Iterable[Polar],
    motor_values: npt.NDArray[np.float32],
    agents: Sequence[Agent],
    state: State,
    device: torch.device,
):
    # move cars forward
    for i, car in enumerate(state.cars):
        agent = agents[i]
        step_car(rng, car, agent, sensor_field, on_track, motor_values, device)
    alives = [ on_track(car.location) for car in state.cars ]

    # if all cars are dead, create new random ones
    if all([not alive for alive in alives]):
        for car in state.cars:
            new_car = random_car(rng, bounds, on_track)
            car.location = new_car.location
            car.angle = new_car.angle
    else:
        # replace dead cars with alive ones
        n_cars = len(alives)
        for i, car in enumerate(state.cars):
            k = i
            while not alives[k]:
                k = (k - 1) % n_cars if rng.random() < 0.5 else (k + 1) % n_cars
            replicate_car(state.cars[k], car)
    state.sensors = car_sensors(state.cars[0], sensor_field)

def main():
    rng = np.random.default_rng()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sensor_field = [
        Polar(d, a*tau)
        for d in (0.1, 0.2, 0.3, 0.4 ,0.5)
        for a in (0.25, 0.15, 0.05, -0.05, -0.15, -0.25)
    ]
    motor_values = np.array([-1, -0.5, -0.1, 0, 0.1, 0.5, 1])
    window = pyglet.window.Window(1920, 1080, resizable=True)
    background = pyglet.graphics.Batch()
    bounds, floor = car_track_triangles(background)
    alive = on_track(floor)
    population = 100
    cars = [
        random_car(rng, bounds, alive)
        for _ in range(population)
    ]
    n_sensors = len(sensor_field)
    n_motors = len(motor_values)
    agents = [
        create_agent(n_sensors, n_motors, 40, 30, 20, device)
        for _ in cars
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
        update(
            rng,
            alive,
            bounds,
            sensor_field,
            motor_values,
            agents,
            state,
            device,
        )

        for win in pyglet.app.windows:
            win.switch_to()
            win.dispatch_events()
            win.dispatch_event('on_draw')
            win.flip()

if __name__ == '__main__':
    main()
