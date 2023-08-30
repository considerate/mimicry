from dataclasses import dataclass
import torch
from torch import Tensor


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

def create_agent(n_sensors, n_motors, lstm_1, lstm_2, device, lr=0.01) -> Agent:
    model = FeedForward(n_sensors, n_motors, [lstm_1, lstm_2]).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    return Agent(
        0,
        model,
        optimiser,
    )
