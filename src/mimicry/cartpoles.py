import itertools
from typing import TypeAlias
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

import matplotlib.pyplot as plt
import ffmpeg
from datetime import datetime
from pathlib import Path
from stable_baselines3 import A2C
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import argparse
from copy import deepcopy
from mimicry.data import Carries
from mimicry.mimicry import copy_carries, replicate_params

from mimicry.network import Agent, create_agent, train

class SparseCartPole(CartPoleEnv):
    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = -1.0 if terminated else 0.0
        return observation, reward, terminated, truncated, info

def reinforcement_learning():
    model = A2C("MlpPolicy", SparseCartPole("rgb_array"), verbose=1)
    model.learn(total_timesteps=10_000)
    vec_env = model.get_env()
    assert vec_env is not None
    observation = vec_env.reset()

    frame = vec_env.render()
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)
    image = plt.imshow(frame)
    plt.show(block=False)
    now = datetime.now().strftime('%Y-%m-%dT%H%m%S')
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    output_path = renders / f'a2c-{now}.mkv'
    width, height = 600, 400
    writer = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_path.as_posix(), pix_fmt='yuv420p')
        .run_async(pipe_stdin=True)
    )
    for _ in range(1000):
        # agent policy that uses the observation and info
        action, _state = model.predict(observation, deterministic=True) # type: ignore
        observation, _reward, terminated, _info = vec_env.step(action)
        frame = vec_env.render("rgb_array")
        assert isinstance(frame, np.ndarray)
        if terminated:
            ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
            red = np.array([255,0,0],dtype=np.uint8)
            frame[ys,xs,:] = frame[ys,xs,:] * 0.8 + red * 0.2
        writer.stdin.write(frame.tobytes())
        image.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()
    vec_env.close()
    plt.close(fig)
    plt.close()

Observation: TypeAlias = npt.NDArray[np.float32]

def sample_agent(rng: np.random.Generator, sensors: Tensor, agent: Agent) -> int:
    motors, carries = agent.model(sensors, agent.carries)
    agent.carries = carries
    motor_probs = torch.exp(motors).cpu().detach().numpy()
    action = rng.choice(np.arange(len(motor_probs)), p=motor_probs, size=1)
    return action[0]



def random_walks(headless: bool):
    rng = np.random.default_rng()
    envs = [SparseCartPole("rgb_array") for _ in range(500)]
    history: list[list[tuple[Tensor, int, Carries]]] = [[] for _ in envs]
    n_sensors = 4
    n_motors = 2
    device = torch.device('cuda')
    agents = [
        create_agent(n_sensors, n_motors, 10, 10, 10, device)
        for _ in envs
    ]
    lives = [0 for _ in envs]
    max_history = 10
    observations = [env.reset()[0] for env in envs]
    def get_frame(env: SparseCartPole) -> npt.NDArray[np.uint8]:
        frame = env.render()
        if frame is None:
            raise ValueError("No rgb_array returned from env.render()")
        return frame.astype(np.uint8)
    frames = [get_frame(env) for env in envs[:10]]
    white = np.full_like(frames[0], 255, dtype=np.uint8)
    image = white.copy()
    for frame in frames:
        ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
        image[ys,xs,:] = image[ys,xs,:] * 0.8 + frame[ys,xs,:] * 0.2
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)
    if not headless:
        img = plt.imshow(image)
        plt.show(block=False)
    else:
        img = None
    width, height = 600, 400
    life_rate = 1.0

    now = datetime.now().strftime('%Y-%m-%dT%H%m%S')
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    output_path = renders / f'render-{now}.mkv'
    writer = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_path.as_posix(), pix_fmt='yuv420p')
        .run_async(pipe_stdin=True)
    )
    for iteration in itertools.count():
        actions = [ env.action_space.sample() for env in envs]
        with torch.no_grad():
            agent_motors = []
            for i, agent in enumerate(agents):
                observation = observations[i]
                if device.type == 'cuda':
                    stream = torch.cuda.Stream()
                    assert isinstance(stream, torch.cuda.Stream)
                    with torch.cuda.stream(stream):
                        sensors = (
                            torch.from_numpy(observation)
                            .to(device, non_blocking=True)
                        )
                        agent_motors.append(
                            (sensors, agent.model(sensors, agent.carries))
                        )
                else:
                    sensors = (
                        torch.from_numpy(observation).to(device)
                    )
                    agent_motors.append(agent.model(sensors, agent.carries))
        torch.cuda.synchronize()
        for i, (sensors, (motors, carries)) in enumerate(agent_motors):
            agent = agents[i]
            agent.carries = carries

            motor_probs = torch.exp(motors).cpu().detach().numpy()
            action = rng.choice(np.arange(len(motor_probs)), p=motor_probs, size=1)[0]
            step = (sensors, action, carries)
            history[i].append(step)
        for i, agent in enumerate(agents):
            if len(history[i]) > max_history:
                history[i].pop(0)
        alives: list[bool] = []
        for i, env in enumerate(envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            observations[i] = obs
            alives.append(not terminated)
        n = len(alives)
        alpha = 0.9
        life_rate = life_rate * alpha + (1.0 - alpha) * np.mean(alives)
        print(life_rate)
        if not any(alives):
            for i, env in enumerate(envs):
                env.reset()
                lives[i] = 0
                history[i].clear()
        else:
            for i in range(n):
                if alives[i]:
                    lives[i] += 1
                    continue
                k = i
                while not alives[k]:
                    k = (k - 1) % n if rng.random() < 0.5 else (k + 1) % n
                envs[i].state = deepcopy(envs[k].state)
                replicate_params(
                    agents[k].model.state_dict(),
                    agents[i].model.state_dict(),
                )
                # agents[i].optimiser.load_state_dict(deepcopy(agents[k].optimiser.state_dict()))
                history[i].clear()
                for (sensors, sampled, carries) in history[k]:
                    history[i].append((sensors.clone(), sampled, copy_carries(carries)))
                lives[i] = 0
        to_train = [
            i
            for i, _ in enumerate(agents)
            if (iteration + i) % (max_history // 10) == 0
        ]
        for i in to_train:
            stream = torch.cuda.Stream()
            assert isinstance(stream, torch.cuda.Stream)
            with torch.cuda.stream(stream):
                train(agents[i], history[i])
        frames = [get_frame(env) for env in envs[:10]]
        image = white.copy()
        for i, frame in enumerate(frames):
            if alives[i]:
                ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
                image[ys,xs,:] = image[ys,xs,:] * 0.8 + frame[ys,xs,:] * 0.2
            else:
                ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
                red = np.array([255,0,0],dtype=np.uint8)
                image[ys,xs,:] = image[ys,xs,:] * 0.8 + red * 0.2
        writer.stdin.write(image.tobytes())
        if img is not None:
            img.set_data(image)
            fig.canvas.draw()
            fig.canvas.flush_events()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(16)
    random_walks(args.headless)
    # reinforcement_learning()


if __name__ == '__main__':
    main()
