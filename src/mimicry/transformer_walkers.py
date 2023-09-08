from collections.abc import Sequence
import json
from math import log
from subprocess import Popen
from typing import Any
from Box2D import b2Body, b2RevoluteJoint
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
import gymnasium
import argparse

import matplotlib.pyplot as plt
import ffmpeg
from datetime import datetime
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from tqdm import tqdm

# from mimicry.network import Agent, create_agent, train
from mimicry.transformer import Agent, create_agent
from mimicry.mimicry import replicate_params

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
            for (sensors, sampled) in history[k]:
                history[i].append((sensors.clone(), sampled))


def step_agents(
    rng: np.random.Generator,
    histories: list[list[tuple[Tensor, Tensor]]],
    agents: Sequence[Agent],
    envs,
    observations,
    min_logvar,
    max_logvar,
    device,
    train_agents=False
) -> tuple[list[bool], list[tuple[Tensor, Tensor]]]:
    alives = []
    steps = []
    one = torch.scalar_tensor(1.0,dtype=torch.float32).to(device)
    for i, history in enumerate(histories):
        agent = agents[i]
        current_sensor = torch.from_numpy(observations[i]).to(device)
        sensors = torch.stack(
            [sensor.unsqueeze(0) for (sensor, _) in history] +
            [current_sensor.unsqueeze(0)]
        )
        motors_sequence = agent.model(sensors)
        if train_agents:
            agent.optimiser.zero_grad()

        motors = motors_sequence[-1,0,:]
        n_motors = len(motors)
        means = torch.clip(motors[0:n_motors//2], -one, one)
        logvars = torch.minimum(
            max_logvar,
            torch.maximum(motors[n_motors//2:], min_logvar)
        )
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
        step = (current_sensor, actions)
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

    factor = 0.9 * 1.0 / len(agents)
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

def write_json_record(record: dict[str, Any], file) -> None:
    file.write("\x1e")
    json.dump(record, file)
    file.write("\n")

def dump_agents(agents: list[Agent], directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(
            agent.model.state_dict(),
            directory / f"agent-{i}-model.pt",
        )
        torch.save(
            agent.optimiser.state_dict(),
            directory / f"agent-{i}-optimiser.pt",
        )
def prune_chaff_x(envs: list[BipedalWalker], alives: list[bool], chaff: int):
    """
    kill the worst agents every frame
    """
    worst = np.argsort([env.scroll for env in envs])
    for bad in worst[:chaff]:
        alives[bad] = False

def prune_chaff_y(envs: list[BipedalWalker], alives: list[bool], chaff: int):
    worst = np.argsort([env.hull.position.y for env in envs if env.hull])
    for bad in worst[:chaff]:
        alives[bad] = False

def walkers(
    headless: bool,
    show_training: bool,
    population: int = 100,
    training_steps:int = 100_000,
    resume: Path | None = None,
):
    rng = np.random.default_rng()
    envs = [BipedalWalker("rgb_array") for _ in range(population)]
    history: list[list[tuple[Tensor, Tensor]]] = [[] for _ in envs]
    n_sensors = 24
    n_motors = 4 * 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    max_history = 50
    checkpoint_interval = 1000
    agents = [
        create_agent(n_sensors, n_motors, device, max_len=max_history+1, lr=0.1)
        for _ in envs
    ]
    if resume is not None:
        for i, agent in enumerate(agents):
            model_state = torch.load(resume / f"agent-{i}-model.pt")
            agent.model.load_state_dict(model_state)
            optim_state = torch.load(resume / f"agent-{i}-optimiser.pt")
            agent.optimiser.load_state_dict(optim_state)
    min_logvar = torch.scalar_tensor(2.0*log(0.01), device=device)
    max_logvar = torch.scalar_tensor(2.0*log(100.0), device=device)
    observations = [env.reset()[0] for env in envs]
    width, height = 600, 400
    life_rate = 0.0

    now = datetime.now().strftime('%Y-%m-%dT%H%M%S')
    renders = Path('renders')
    checkpoints = Path('checkpoints')
    renders.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)
    checkpoint_dir = checkpoints / f"walker-{now}"
    checkpoint_dir.mkdir()
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
    iteration = 0
    try:
        with (
            (renders / f'walker-{now}-training.log').open("w") as logfile,
            (renders / f'walker-{now}-training.json-seq').open("a") as jsonlog,
        ):
            bar = tqdm(range(training_steps), ncols=90)
            for it in bar:
                iteration = it
                if iteration % checkpoint_interval == 0:
                    dump_agents(agents, checkpoint_dir / f"{iteration}")
                alpha = 0.999
                alives, steps = step_agents(
                    rng, history, agents, envs, observations,
                    train_agents=True,
                    device=device,
                    min_logvar=min_logvar,
                    max_logvar=max_logvar,
                )
                for i, step in enumerate(steps):
                    history[i].append(step)
                    if len(history[i]) > max_history:
                        history[i].pop(0)
                prune_chaff_x(envs, alives, chaff=3)
                replicate_agents(rng, alives, envs, agents, history)
                # randomly reset one agent to standing position with 1%
                # probability per agent
                if rng.uniform() < 0.01 * len(envs):
                    to_reset = rng.choice(len(envs), size=1)[0]
                    alives[to_reset] = False
                    reset_in_place(envs[to_reset])
                life_rate = life_rate * alpha + (1.0 - alpha) * np.mean(alives)
                expected_life = 1.0 / (1.0 - life_rate)
                bar.set_postfix({"life_rate": life_rate, "mean_life": expected_life})
                print(life_rate, file=logfile, flush=True)
                write_json_record({
                    "life_rate": life_rate,
                    "alives": alives,
                    "hulls": [
                        {"pos": {"x": env.hull.position.x, "y": env.hull.position.y},
                         "vel": {"x": env.hull.linearVelocity.x, "y": env.hull.linearVelocity.y},
                         "ang": env.hull.angularVelocity,
                        }
                        for env in envs
                        if env.hull is not None
                    ],
                }, file=jsonlog)
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
    dump_agents(agents, checkpoint_dir / f"stopped-{iteration}")

    envs = envs[:to_render]
    agents = agents[:to_render]
    history = [[] for _ in range(to_render)]
    observations = [env.reset()[0] for env in envs]
    frames = [get_frame(env) for env in envs]
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
    four_minutes = 25*60*4
    for _ in range(four_minutes):
        with torch.no_grad():
            alives, _ = step_agents(
                rng, history, agents, envs, observations,
                min_logvar=min_logvar,
                max_logvar=max_logvar,
                device=device,
            )
        for i, alive in enumerate(alives):
            if not alive:
                envs[i].reset()
        frames = [get_frame(env) for env in envs]
        image = overlay_frames(frames, to_render)
        if img is not None:
            xs.clear()
            xs.hist([env.hull.position.x for env in envs if env.hull])
            ys.clear()
            ys.hist([env.hull.position.y for env in envs if env.hull])
            img.set_data(image)
            fig.canvas.draw()
            fig.canvas.flush_events()
        writer.stdin.write(image.tobytes())

def headless():
    walkers(True, False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train", type=int)
    parser.add_argument("--population", type=int)
    args = parser.parse_args()
    headless = args.headless
    show_training = not headless
    arguments: dict[str, Any] = {}
    if args.train is not None:
        arguments["training_steps"] = args.train
    if args.population is not None:
        arguments["population"] = args.population
    walkers(headless, show_training, resume=args.resume, **arguments)
