from collections.abc import Sequence
import itertools
from math import log
from subprocess import Popen
from Box2D import b2Body, b2RevoluteJoint
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
from mimicry.feedforward import Agent, create_agent
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
):
    squared_deviations = (sampled-means) ** 2
    return 0.5*torch.sum(
        squared_deviations * vars + logvars
    )

def replicate_joint(source: b2RevoluteJoint, target: b2RevoluteJoint):
    # https://github.com/openai/box2d-py/blob/647d6c66710cfe3ff81b3e80522ff92624a77b41/Box2D/Box2D_joints.i#L86
    target.motorSpeed = source.motorSpeed
    target.upperLimit = source.upperLimit
    target.lowerLimit = source.lowerLimit
    target.motorEnabled = source.motorEnabled
    target.limitEnabled = source.limitEnabled

def replicate_leg(source: b2Body, target: b2Body):
    # https://github.com/openai/box2d-py/blob/647d6c66710cfe3ff81b3e80522ff92624a77b41/Box2D/Box2D_bodyfixture.i#L326
    target.transform = (source.position.copy(), source.angle)
    target.linearVelocity = source.linearVelocity.copy()
    target.localCenter = source.localCenter.copy()
    target.inertia = source.inertia
    target.awake = source.awake
    target.angularVelocity = source.angularVelocity
    target.angularDamping = source.angularDamping
    target.linearDamping = source.linearDamping
    target.fixedRotation = source.fixedRotation

def replicate_hull(source: b2Body, target: b2Body):
    # there's not difference in replicating the hull to replicating a leg
    replicate_leg(source, target)

def replicate_agents(rng, alives, envs: list[BipedalWalker], agents, history):
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
            envs[i].reset()
            for j, joint in enumerate(envs[k].joints):
                replicate_joint(joint, envs[i].joints[j])
            for j, leg in enumerate(envs[k].legs):
                replicate_leg(leg, envs[i].legs[j])
                envs[i].legs[j].ground_contact = False # type: ignore
            envs[i].scroll = envs[k].scroll
            envs[i].game_over = envs[k].game_over
            envs[i].lidar_render = envs[k].lidar_render
            source_hull = envs[k].hull
            target_hull = envs[i].hull
            assert source_hull is not None
            assert target_hull is not None
            replicate_hull(source_hull, target_hull)
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
    agents: Sequence[Agent],
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
        logvars = torch.maximum(motors[n_motors//2:], min_logvar)
        vars = torch.exp(-logvars)
        mean_arr = means.detach().cpu().numpy()
        var_arr = vars.detach().cpu().numpy()
        actions = torch.tensor([
            np.clip(rng.normal(mean, np.sqrt(var)), -1.0, 1.0)
            for (mean, var) in zip(mean_arr, var_arr, strict=True)
        ])
        if train_agents:
            device_actions = actions.to(means.device)
            loss = gaussian_log_negative_log_loss(
                means, vars, logvars, device_actions,
            )
            loss.backward()
        step = (sensors, actions, carries)
        steps.append(step)
        box = actions.numpy()
        obs, _, terminated, _, _ = envs[i].step(box)
        alives.append(not terminated)
        observations[i] = obs
    # compute summed gradient for each parameter
    grad_accum: dict[str, torch.Tensor] = {}
    if train_agents:
        for agent in agents:
            for name, param in agent.model.named_parameters():
                grad = param.grad
                if grad is not None:
                    before = grad_accum.get(name)
                    if before is None:
                        grad_accum[name] = grad.clone()
                    else:
                        before.add_(grad)

    factor = 0.1 * 1.0 / len(agents)
    # penalize agent for doing the same as mean agent
    # by removing the
    for agent in agents:
        if train_agents:
            for name, param in agent.model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                accum = grad_accum.get(name)
                if accum is not None:
                    grad.sub_(accum, alpha=factor)
            agent.optimiser.step()
    return alives, steps

def get_frame(env: gymnasium.Env) -> npt.NDArray[np.uint8]:
    frame = env.render()
    if frame is None:
        raise ValueError("No rgb_array returned from env.render()")
    elif isinstance(frame, list):
        raise ValueError("Expected single frame in get_frame()")
    return frame.astype(np.uint8)

def overlay_frames(frames: list[npt.NDArray[np.uint8]], to_render: int):
    image = np.full_like(frames[0], fill_value=255, dtype=np.uint8)
    for frame in frames:
        ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
        image[ys,xs,:] = \
            image[ys,xs,:] * (1 - 1.0 / to_render) \
            + frame[ys,xs,:] * (1.0 / to_render)
    return image

def reset_in_place(env: BipedalWalker):
    scroll = env.scroll
    hull = env.hull
    assert hull is not None
    x = hull.position.x
    env.reset()
    env.scroll = scroll
    hull = env.hull
    assert hull is not None
    diff = x - hull.position.x
    hull.position.x += diff
    for leg in env.legs:
        leg.position.x += diff

def walkers(headless: bool, show_training: bool):
    rng = np.random.default_rng()
    population = 50
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
    renders = Path('renders')
    renders.mkdir(exist_ok=True)
    training_steps = 10_000
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)
    xs = plt.subplot(2, 2, 1)
    ys = plt.subplot(2, 2, 2)
    ax = plt.subplot(2, 1, 2)
    to_render = 5
    if not headless and show_training:
        frames = [get_frame(env) for env in envs[:to_render]]
        image = overlay_frames(frames, to_render)
        xs.hist([env.hull.position.x for env in envs if env.hull])
        ys.hist([env.hull.position.y for env in envs if env.hull])
        img = ax.imshow(image)
        plt.show(block=False)
    else:
        img = None
    try:
        with (renders / f'walker-{now}-training.log').open("w") as logfile:
            bar = tqdm(range(training_steps), ncols=90)
            for _ in bar:
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
                # kill the worst agents every frame
                # decile = len(envs) // 10 + 1
                # chaff = 5
                # worst = np.argsort([env.scroll for env in envs])
                # for bad in worst[:chaff]:
                #     alives[bad] = False
                # worst = np.argsort([env.hull.position.y for env in envs if env.hull])
                # for bad in worst[:chaff]:
                #     alives[bad] = False
                replicate_agents(rng, alives, envs, agents, history)
                # randomly reset one agent to standing position every frame
                if rng.uniform() < 0.01 * len(envs):
                    to_reset = rng.choice(len(envs), size=1)[0]
                    alives[to_reset] = True
                    reset_in_place(envs[to_reset])
                life_rate = life_rate * alpha + (1.0 - alpha) * np.mean(alives)
                expected_life = 1.0 / (1.0 - life_rate)
                bar.set_postfix({"life_rate": life_rate, "mean_life": expected_life})
                print(life_rate, file=logfile, flush=True)
                if show_training:
                    frames = [get_frame(env) for env in envs[:to_render]]
                    image = overlay_frames(frames, to_render)
                    if img is not None:
                        img.set_data(image)
                    xs.clear()
                    xs.hist([env.hull.position.x for env in envs if env.hull])
                    ys.clear()
                    ys.hist([env.hull.position.y for env in envs if env.hull])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
    except KeyboardInterrupt:
        pass

    observations = [env.reset()[0] for env in envs[:to_render]]
    frames = [get_frame(env) for env in envs[:to_render]]
    image = overlay_frames(frames, to_render)
    output_path = renders / f'walker-{now}.mkv'
    writer: Popen = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
        .output(output_path.as_posix(), pix_fmt='yuv420p')
        .run_async(pipe_stdin=True)
    )
    assert writer.stdin is not None
    writer.stdin.write(image.tobytes())
    if not headless:
        xs.clear()
        xs.hist([env.hull.position.x for env in envs if env.hull])
        ys.clear()
        ys.hist([env.hull.position.y for env in envs if env.hull])
        img = ax.imshow(image)
        plt.show(block=False)
    else:
        img = None
    for _ in itertools.count():
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
        image = overlay_frames(frames, to_render)
        if img is not None:
            xs.clear()
            xs.hist([env.hull.position.x for env in envs[:to_render] if env.hull])
            ys.clear()
            ys.hist([env.hull.position.y for env in envs[:to_render] if env.hull])
            img.set_data(image)
            fig.canvas.draw()
            fig.canvas.flush_events()
        writer.stdin.write(image.tobytes())

def headless():
    walkers(True, False)

def main():
    walkers(False, True)