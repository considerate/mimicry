from collections.abc import Iterable
from dataclasses import dataclass
import torch
from torch import Tensor

Carries = tuple[Tensor, Tensor, Tensor]

class Network(torch.nn.Module):
    def __init__(
        self,
        n_sensors: int,
        n_motors: int,
        lstm_1: int,
        lstm_2: int,
        lstm_3: int,
    ):
        super().__init__()
        self.lstm_cell = torch.nn.LSTMCell(n_sensors, lstm_1)
        self.lstm_cell_2 = torch.nn.LSTMCell(lstm_1, lstm_2)
        self.lstm_cell_3 = torch.nn.LSTMCell(lstm_2, lstm_3)
        self.dense = torch.nn.Linear(n_sensors, lstm_3)
        self.motors = torch.nn.Linear(lstm_3, n_motors)

    def forward(self, x: Tensor, carries: Carries) -> tuple[Tensor, Carries]:
        carry, carry_2, carry_3 = carries
        y, new_carry = self.lstm_cell(x, carry)
        y_2, new_carry_2 = self.lstm_cell(y, carry_2)
        y_3, new_carry_3 = self.lstm_cell(y_2, carry_3)
        z = self.dense(x)
        mid = z + y_3
        motors = self.motors(mid)
        motor_preds = torch.nn.functional.log_softmax(motors)
        return motor_preds, (new_carry, new_carry_2, new_carry_3)

def sequence_loss(
    model: torch.nn.Module,
    initialcarry: Carries,
    sequence: Iterable[tuple[Tensor, int]],
):
    loss = torch.scalar_tensor(0.0, requires_grad=True)
    carry = initialcarry
    for (sensors, sampled) in sequence:
        motors, carry = model(sensors, carry)
        logloss = -motors[sampled]
        loss = loss + logloss
    return loss

@dataclass
class Agent:
    carries: Carries
    model: Network

def create_agent(n_sensors, n_motors, lstm_1, lstm_2, lstm_3, device) -> Agent:
    model = Network(n_sensors, n_motors, lstm_1, lstm_2, lstm_3).to(device)
    carry = torch.zeros(lstm_1, device=device)
    carry_2 = torch.zeros(lstm_2, device=device)
    carry_3 = torch.zeros(lstm_3, device=device)
    carries = carry, carry_2, carry_3
    return Agent(
        carries,
        model,
    )
