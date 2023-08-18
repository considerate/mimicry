from collections.abc import Callable, Iterable, Sequence

# The pyglet.options line needs to be directly after the import hence the weird
# placement
import pyglet
pyglet.options["headless"] = True

import numpy as np
import numpy.typing as npt
from math import tau
import ffmpeg
from pathlib import Path
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
from mimicry.network import Agent, Carries, create_agent, train



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
    history: list[list[tuple[torch.Tensor, int, Carries]]],
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
        _, sampled = step_car(rng, car, agent, values, motor_values)
        history[i].append((values, sampled, carries))
        while len(history[i]) > max_history:
            history[i].pop(0)
    alives = [ on_track(car.location) for car in state.cars ]

    for i, car in enumerate(state.cars):
        if not alives[i]:
            continue
        if (frame + i) % (max_history // 10) == 0:
            print('training agent', i)
            train(agents[i], history[i])

    # if all cars are dead, create new random ones
    if all([not alive for alive in alives]):
        for i, car in enumerate(state.cars):
            new_car = random_car(rng, bounds, on_track)
            car.location = new_car.location
            car.angle = new_car.angle
            history[i] = []
    else:
        # replace dead cars with alive ones
        n_cars = len(alives)
        for i, car in enumerate(state.cars):
            k = i
            while not alives[k]:
                k = (k - 1) % n_cars if rng.random() < 0.5 else (k + 1) % n_cars
            replicate_car(state.cars[k], car)
            history[i] = history[k]
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
    width, height = 1920, 1080
    window = pyglet.window.Window(width, height, resizable=True)
    background = pyglet.graphics.Batch()
    bounds, floor = car_track_triangles(background)
    alive = on_track(floor)
    population = 100
    cars = [
        random_car(rng, bounds, alive)
        for _ in range(population)
    ]
    history = [[] for _ in cars]
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
    i = 0
    bufman = pyglet.image.get_buffer_manager()
    now = datetime.now().strftime('%Y-%m-%dT%H%m%S')
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    output_path = renders / f'render-{now}.mkv'
    writer = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_path.as_posix(), pix_fmt='yuv420p')
        .run_async(pipe_stdin=True, quiet=True)
    )
    try:
        while running:
            print('frame', i)
            i += 1
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
                history,
            )

            for win in pyglet.app.windows:
                win.switch_to()
                win.dispatch_events()
                win.dispatch_event('on_draw')
                win.flip()
            depth_buffer = bufman.get_color_buffer()
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
    except Exception as e:
        print(e)
    finally:
        writer.stdin.close()

if __name__ == '__main__':
    main()
