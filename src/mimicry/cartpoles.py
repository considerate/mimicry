from gymnasium.envs.classic_control.cartpole import CartPoleEnv

import matplotlib.pyplot as plt
from stable_baselines3 import A2C
import numpy as np
import numpy.typing as npt
import time

class SparseCartPole(CartPoleEnv):
    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = -1.0 if terminated else 0.0
        return observation, reward, terminated, truncated, info

def reinforcement_learning():
    model = A2C("MlpPolicy", SparseCartPole("rgb_array"), verbose=1)
    # model.learn(total_timesteps=10_000)
    vec_env = model.get_env()
    assert vec_env is not None
    observation = vec_env.reset()

    frame = vec_env.render()
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)
    image = plt.imshow(frame)
    plt.show(block=False)
    for _ in range(1000):
        # agent policy that uses the observation and info
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, _terminated, _info = vec_env.step(action)
        frame = vec_env.render("rgb_array")
        image.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()
    vec_env.close()
    plt.close(fig)
    plt.close()

def random_walks():
    envs = [SparseCartPole("rgb_array") for _ in range(10)]
    observations = [env.reset() for env in envs]
    frames = [env.render().astype(np.uint8) for env in envs]
    white = np.full_like(frames[0], 255, dtype=np.uint8)
    image = white.copy()
    for frame in frames:
        ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
        image[ys,xs,:] = image[ys,xs,:] * 0.8 + frame[ys,xs,:] * 0.2
    fig = plt.figure()
    assert isinstance(fig, plt.Figure)
    img = plt.imshow(image)
    plt.show(block=False)
    for _ in range(1000):
        actions = [ env.action_space.sample() for env in envs]
        for i, env in enumerate(envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            if terminated:
                env.reset()
        frames = [env.render().astype(np.uint8) for env in envs]
        image = white.copy()
        for frame in frames:
            ys, xs = np.where(np.all(frame != (255,255,255), axis=-1))
            image[ys,xs,:] = image[ys,xs,:] * 0.8 + frame[ys,xs,:] * 0.2
        img.set_data(image)
        fig.canvas.draw()
        fig.canvas.flush_events()

def main():
    random_walks()
    # reinforcement_learning()


if __name__ == '__main__':
    main()
