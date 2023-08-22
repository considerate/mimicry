from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import torch
from torch import Tensor

from mimicry.data import Carries


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
        y_2, new_carry_2 = self.lstm_cell_2(y, carry_2)
        y_3, new_carry_3 = self.lstm_cell_3(y_2, carry_3)
        z = self.dense(x)
        mid = z + y_3
        motors = self.motors(mid)
        motor_preds = torch.nn.functional.log_softmax(motors)
        return motor_preds, ((y, new_carry), (y_2, new_carry_2), (y_3, new_carry_3))

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
    optimiser: torch.optim.Optimizer

def train(
    agent: Agent,
    history: Sequence[tuple[Tensor, int, Carries]],
) -> float:
    if len(history) == 0:
        return 0.0
    _, _, carries = history[0]
    sequence = ( (sensors, sampled) for (sensors, sampled, _) in history )
    agent.optimiser.zero_grad()
    loss = sequence_loss(agent.model, carries, sequence)
    agent.optimiser.step()
    return loss.item()

def create_agent(n_sensors, n_motors, lstm_1, lstm_2, lstm_3, device) -> Agent:
    model = Network(n_sensors, n_motors, lstm_1, lstm_2, lstm_3).to(device)
    carry = torch.zeros(lstm_1, device=device), torch.zeros(lstm_1, device=device)
    carry_2 = torch.zeros(lstm_2, device=device), torch.zeros(lstm_2, device=device)
    carry_3 = torch.zeros(lstm_3, device=device), torch.zeros(lstm_3, device=device)
    carries = carry, carry_2, carry_3
    #torch.optim.AdamW(model.parameters(), lr=0.01)
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    return Agent(
        carries,
        model,
        optimiser,
    )
