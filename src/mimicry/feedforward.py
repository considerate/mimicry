from dataclasses import dataclass
from math import exp
import torch
from torch import Tensor
import numpy as np


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_motors: int,
        hidden: list[int]
    ):
        super().__init__()
        dims = n_sensors
        encoders = []
        for h in hidden:
            encoders.append(torch.nn.Linear(dims, h))
            encoders.append(torch.nn.ReLU(inplace=True))
            dims = h
        self.encode = torch.nn.Sequential(*encoders)
        self.decode = torch.nn.Linear(dims, n_motors)

    def forward(self, x: Tensor, _) -> tuple[Tensor, int]:
        motors = self.decode(self.encode(x))
        motor_preds = torch.nn.functional.log_softmax(motors)
        return motor_preds, 0

@dataclass
class Agent:
    carries: int
    model: FeedForward
    optimiser: torch.optim.Optimizer

def train_one(rng: np.random.Generator, agent: Agent, sensors: Tensor) -> float:
    agent.optimiser.zero_grad()
    motors = agent.model(sensors)
    motor_preds = torch.exp(motors).cpu().detach().numpy()
    action = rng.choice(len(motor_preds),size=1)[0]
    loss = -motors[action]
    loss.backward()
    agent.optimiser.step()
    return loss.item()

def create_agent(n_sensors, n_motors, lstm_1, lstm_2, device) -> Agent:
    model = FeedForward(n_sensors, n_motors, [lstm_1, lstm_2]).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=exp(-4.0))
    return Agent(
        0,
        model,
        optimiser,
    )
