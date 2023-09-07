from dataclasses import dataclass
import torch
from torch import Tensor
import math

class PositionalEncoding(torch.nn.Module):
    pe: Tensor

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        scale = -math.log(10000.0) / d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * scale)
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel2(torch.nn.Module):
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
        return motors, 0

class TransformerModel(torch.nn.Module):

    def __init__(
        self,
        n_sensors: int, n_motors: int,
        d_model: int, n_head: int, d_hid: int,
        n_layers: int, dropout: float = 0.5,
        max_len: int = 5000,
    ):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model, n_head, d_hid, dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)
        self.embedding = torch.nn.Linear(n_sensors, d_model)
        self.d_model = d_model
        self.linear = torch.nn.Linear(d_model, n_motors)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, n_sensors]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, n_motors]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src)
        output = self.linear(encoded)
        return output

@dataclass
class Agent:
    carries: int
    model: TransformerModel
    optimiser: torch.optim.Optimizer

def create_agent(n_sensors, n_motors, device, max_len, lr=0.01) -> Agent:
    model = TransformerModel(
        n_sensors, n_motors,
        d_model=128,
        n_head=4,
        d_hid=200,
        n_layers=2,
        max_len=max_len,
    ).to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    return Agent(
        0,
        model,
        optimiser,
    )
