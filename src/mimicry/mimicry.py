from collections.abc import Callable, Iterable, Sequence

import pyglet
import numpy as np
import numpy.typing as npt
from copy import deepcopy
from math import tau
import ffmpeg
from pathlib import Path
from pyglet.window import Window
import torch
from datetime import datetime

from mimicry.render import car_track_triangles, drawer
from mimicry.car import (
    car_sensor_values,
    car_sensors,
    on_track,
    random_car,
    replicate_car,
    step_car
)
from mimicry.data import Bounds, Location, Polar, State
from mimicry.network import Agent, create_agent, train



def update(
    frame: int,
    rng: np.random.Generator,
    on_track: Callable[[Location], bool],
    bounds: Bounds,
    sensor_field: Iterable[Polar],
    motor_values: npt.NDArray[np.float32],
    agents: Sequence[Agent],
    state: State,
    device: torch.device,
    max_history: int = 100,
):
    sensor_values = [
        car_sensor_values(car, sensor_field, on_track, device)
        for car in state.cars
    ]
    # move cars forward
    for i, car in enumerate(state.cars):
        agent = agents[i]
        values = sensor_values[i]
        carries = agent.carries
        location = car.location
        _, sampled = step_car(rng, car, agent, values, motor_values)
        state.history[i].append((values, sampled, carries))
        state.trails[i][-1].append(location)
        while sum(len(section) for section in state.trails[i]) > max_history:
            while len(state.trails[i][0]) == 0:
                state.trails[i].pop(0)
            state.trails[i][0].pop(0)
        while len(state.history[i]) > max_history:
            state.history[i].pop(0)

    alives = [ on_track(car.location) for car in state.cars ]

    for i, car in enumerate(state.cars):
        if not alives[i]:
            continue
        if (frame + i) % (max_history // 10) == 0:
            train(agents[i], state.history[i])

    # if all cars are dead, create new random ones
    if all([not alive for alive in alives]):
        for i, car in enumerate(state.cars):
            new_car = random_car(rng, bounds, on_track)
            car.location = new_car.location
            car.angle = new_car.angle
            state.history[i].clear()
            state.trails[i].append([])
    else:
        # replace dead cars with alive ones
        n_cars = len(alives)
        for i, car in enumerate(state.cars):
            if alives[i]:
                continue
            k = i
            while not alives[k]:
                k = (k - 1) % n_cars if rng.random() < 0.5 else (k + 1) % n_cars
            replicate_car(state.cars[k], car)
            agents[i].model.load_state_dict(deepcopy(agents[k].model.state_dict()))
            agents[i].optimiser.load_state_dict(deepcopy(agents[k].optimiser.state_dict()))
            state.history[i].clear()
            state.trails[i].append([])
            for h in state.history[k]:
                state.history[i].append(h)
    state.sensors = car_sensors(state.cars[0], sensor_field)

def main(headless: bool = False):
    rng = np.random.default_rng()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    sensor_field = [
        Polar(d, a*tau)
        for d in (0.1, 0.2, 0.3, 0.4 ,0.5)
        for a in (0.25, 0.15, 0.05, -0.05, -0.15, -0.25)
    ]
    motor_values = np.tanh(np.array([-1, -0.5, -0.25 -0.1, 0, 0.1, 0.25, 0.5, 1]))
    width, height = 1920, 1080
    window = pyglet.window.Window(width, height, resizable=True)
    pyglet.gl.glClearColor(0.85,0.85,0.85,1.0)
    background = pyglet.graphics.Batch()
    bounds, floor = car_track_triangles(background)
    alive = on_track(floor)
    population = 150
    cars = [
        random_car(rng, bounds, alive)
        for _ in range(population)
    ]
    history = [[] for _ in cars]
    trails = [[[]] for _ in cars]
    n_sensors = len(sensor_field)
    n_motors = len(motor_values)
    agents = [
        create_agent(n_sensors, n_motors, 40, 30, 20, device)
        for _ in cars
    ]
    car = cars[0]
    state = State(cars, car_sensors(car, sensor_field), history, trails)
    drawer(window, state, bounds, background)
    running = True
    @window.event
    def on_close():
        nonlocal running
        running = False
    i = 0
    bufman = pyglet.image.get_buffer_manager()
    now = datetime.now().strftime('%Y-%m-%dT%H%m%S')
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    output_path = renders / f'render-{now}.mkv'
    stills = Path('stills') / f'{now}'
    stills.mkdir(exist_ok=True)
    if headless:
        writer = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .output(output_path.as_posix(), pix_fmt='yuv420p')
            .run_async(pipe_stdin=True)
        )
    else:
        writer = None
    try:
        while running:
            print('frame', i)
            pyglet.clock.tick()
            update(
                i,
                rng,
                alive,
                bounds,
                sensor_field,
                motor_values,
                agents,
                state,
                device,
            )

            windows: Iterable[Window] = pyglet.app.windows
            for win in windows:
                win.switch_to()
                win.dispatch_events()
                win.dispatch_event('on_draw')
                win.flip()
            depth_buffer = bufman.get_color_buffer()
            if writer is not None:
                image_data = depth_buffer.get_image_data()
                buffer = image_data.get_data(image_data.format, image_data.pitch)
                frame = np.asarray(buffer).reshape((
                    image_data.height,
                    image_data.width,
                    len(image_data.format),
                ))
                # drop alpha channel
                frame = frame[:,:,:3]
                # flip image
                frame = np.flipud(frame)
                writer.stdin.write(frame.tobytes())
            if (i - 1) % 20 == 0:
                image_data = depth_buffer.get_image_data()
                image_data.save(stills / f'frame-{i}.png')
            i += 1
    except Exception as e:
        print(e)
    finally:
        if writer is not None:
            writer.stdin.close()

if __name__ == '__main__':
    main()
