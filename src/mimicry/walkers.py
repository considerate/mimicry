import itertools
from math import log
from subprocess import Popen
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
import gymnasium

import matplotlib.pyplot as plt
import ffmpeg
from datetime import datetime
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from mimicry.data import Carries
from tqdm import tqdm

# from mimicry.network import Agent, create_agent, train
from mimicry.feedforward import create_agent
from mimicry.mimicry import copy_carries, replicate_params


def predict_motors(observations, agents, device):
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
            agent_motors.append((sensors, agent.model(sensors, agent.carries)))
    return agent_motors


def gaussian_log_negative_log_loss(
    means: Tensor,
    vars: Tensor,
    logvars: Tensor,
    sampled: Tensor,
    min_logvar: Tensor,
):
    logvars = torch.maximum(logvars, min_logvar)
    squared_deviations = (sampled-means) ** 2
    return 0.5*torch.sum(
        squared_deviations * vars + logvars
    )

def replicate_agents(rng, alives, envs, agents, history):
    n = len(alives)
    if not any(alives):
        for i, env in enumerate(envs):
            env.reset()
            history[i].clear()
    else:
        for i in range(n):
            if alives[i]:
                continue
            k = i
            while not alives[k]:
                k = (k - 1) % n if rng.random() < 0.5 else (k + 1) % n
            # envs[i].state = deepcopy(envs[k].state)
            envs[i].reset()
            replicate_params(
                agents[k].model.state_dict(),
                agents[i].model.state_dict(),
            )
            replicate_params(
                agents[k].optimiser.state_dict()["state"],
                agents[i].optimiser.state_dict()["state"]
            )
            history[i].clear()
            for (sensors, sampled, carries) in history[k]:
                history[i].append((sensors.clone(), sampled, copy_carries(carries)))


def step_agents(
    rng: np.random.Generator,
    agent_motors,
    agents,
    envs,
    observations,
    min_logvar,
    train_agents=False
):
    alives = []
    steps = []
    for i, (sensors, (motors, carries)) in enumerate(agent_motors):
        agent = agents[i]
        if train_agents:
            agent.optimiser.zero_grad()
        agent.carries = carries

        n_motors = len(motors)
        means = motors[0:n_motors//2]
        logvars = motors[n_motors//2:]
        vars = torch.exp(-logvars)
        mean_arr = means.detach().cpu().numpy()
        var_arr = vars.detach().cpu().numpy()
        actions = torch.tensor([
            np.clip(rng.normal(mean, max(0.01, np.sqrt(var))), -1.0, 1.0)
            for (mean, var) in zip(mean_arr, var_arr, strict=True)
        ])
        if train_agents:
            device_actions = actions.to(means.device)
            loss = gaussian_log_negative_log_loss(
                means, vars, logvars, device_actions, min_logvar
            )
            loss.backward()
            agent.optimiser.step()
        step = (sensors, actions, carries)
        steps.append(step)
        box = actions.numpy()
        obs, _, terminated, _, _ = envs[i].step(box)
        observations[i] = obs
        alives.append(not terminated)
    return alives, steps


def walkers(headless: bool):
    rng = np.random.default_rng()
    population = 20
    envs = [BipedalWalker("rgb_array") for _ in range(population)]
    history: list[list[tuple[Tensor, int, Carries]]] = [[] for _ in envs]
    n_sensors = 24
    n_motors = 4 * 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    agents = [
        create_agent(n_sensors, n_motors, 10, 10, device, lr=0.1)
        for _ in envs
    ]
    max_history = 1
    min_logvar = torch.scalar_tensor(2.0*log(0.01), device=device)
    observations = [env.reset()[0] for env in envs]
    width, height = 600, 400
    life_rate = 0.0

    now = datetime.now().strftime('%Y-%m-%dT%H%M%S')
    training_steps = 100_000
    try:
        bar = tqdm(range(training_steps))
        for iteration in bar:
            agent_motors = predict_motors(observations, agents, device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            alpha = 0.999
            alives, steps = step_agents(
                rng, agent_motors, agents, envs, observations,
                train_agents=True,
                min_logvar=min_logvar,
            )
            for i, step in enumerate(steps):
                history[i].append(step)
                if len(history[i]) > max_history:
                    history[i].pop(0)
            life_rate = life_rate * alpha + (1.0 - alpha) * np.mean(alives)
            expected_life = 1.0 / (1.0 - life_rate)
            bar.set_postfix({"life_rate": life_rate, "mean_life": expected_life})
            print(life_rate, flush=True)
            # replicate_agents(rng, alives, envs, agents, history)
    except KeyboardInterrupt:
        pass
    to_render = 10
    def get_frame(env: gymnasium.Env) -> npt.NDArray[np.uint8]:
        frame = env.render()
        if frame is None:
            raise ValueError("No rgb_array returned from env.render()")
        elif isinstance(frame, list):
            raise ValueError("Expected single frame in get_frame()")
        return frame.astype(np.uint8)

    def overlay_frames(frames: list[npt.NDArray[np.uint8]]):
        image = np.full_like(frames[0], fill_value=255, dtype=np.uint8)
        for frame in frames:
            ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
            image[ys,xs,:] = \
                image[ys,xs,:] * (1 - 1.0 / to_render) \
                + frame[ys,xs,:] * (1.0 / to_render)
        return image

    frames = [get_frame(env) for env in envs[:to_render]]
    fig = plt.figure()
    image = overlay_frames(frames)
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    output_path = renders / f'render-{now}.mkv'
    writer: Popen = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_path.as_posix(), pix_fmt='yuv420p')
        .run_async(pipe_stdin=True)
    )
    assert writer.stdin is not None
    writer.stdin.write(image.tobytes())
    assert isinstance(fig, plt.Figure)
    if not headless:
        img = plt.imshow(image)
        plt.show(block=False)
    else:
        img = None
    for iteration in itertools.count():
        with torch.no_grad():
            agent_motors = predict_motors(observations, agents[:to_render], device)
        with torch.no_grad():
            alives, _ = step_agents(
                rng, agent_motors, agents, envs, observations, min_logvar,
            )
        for i, alive in enumerate(alives):
            if not alive:
                envs[i].reset()
        frames = [get_frame(env) for env in envs[:to_render]]
        image = overlay_frames(frames)
        writer.stdin.write(image.tobytes())
        if img is not None:
            img.set_data(image)
            fig.canvas.draw()
            fig.canvas.flush_events()

def main():
    walkers(False)
