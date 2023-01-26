import math
import dataclasses
from typing import Type

import torch
from torch import nn

from .. import utils
from . import Params
from .params import nn_lower


class MLP(nn.Module):
    @dataclasses.dataclass(kw_only=True)
    class Params(Params):
        # Training hyperparams
        batch_size: int = 64
        epochs: int = 50

        # Model hyperparams
        num_layers: int = 3
        hidden_size: int = 256
        dropout: float = 0.3

        # Optimizer hyperparams
        lr: float = 1e-3

        # Misc
        print_every: int = 5

    def __init__(self, params: "MLP.Params", N: int, C: int):
        super().__init__()

        self.params = params
        if params.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        elif params.num_layers == 1:
            self.net = nn.Linear(N, C)
        else:
            nets = [self._block(N, params.hidden_size)]
            for _ in range(params.num_layers - 2):
                nets.append(
                    self._block(params.hidden_size, params.hidden_size))
            nets.append(nn.Linear(params.hidden_size, C))
            self.net = nn.Sequential(*nets)

    def forward(self, X):
        return self.net(X)

    def _block(self, in_size, out_size):
        return nn.Sequential(nn.Linear(in_size, out_size),
                             self.params.Activation(),
                             nn.Dropout(self.params.dropout))


class SamePad2d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        size = (kernel_size - 1) // 2
        # If kernel_size is even, we need to make it odd by adding an extra
        # padding at the right and the bottom.
        # This behavior is taken from the PyTorch >= 1.9 implementation.
        if not kernel_size % 2:
            size = (size, size + 1, size, size + 1)
        self.padding = nn.ZeroPad2d(size)

    def forward(self, x):
        return self.padding(x)


class CNN(nn.Module):
    @dataclasses.dataclass(kw_only=True)
    class Params(Params):
        num_layers: int = 3
        input_kernel: int = 3
        hidden_channel: int = 32
        hidden_kernel: int = 5
        batchnorm: bool = True
        pooling: str = "max"
        pooling_size: int = 2

    def __init__(self, params: "CNN.Params", H: int, W: int, C: int):
        super().__init__()

        self.params = params
        if params.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Convolutional layers
        layers = [self._block(1, params.hidden_channel, params.input_kernel)]
        for _ in range(params.num_layers - 1):
            layers.append(self._block(params.hidden_channel,
                                      params.hidden_channel,
                                      params.hidden_kernel))

        # Flatten the output of the last convolutional layer
        layers.append(nn.Flatten())

        # Linear projection to the output
        out_H, out_W = self._output_size(H), self._output_size(W)
        layers.append(nn.Linear(out_H * out_W * params.hidden_channel, 10))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X.unsqueeze(1))

    def _block(self, in_channels, out_channels, kernel_size):
        layers = [SamePad2d(kernel_size),
                  nn.Conv2d(in_channels, out_channels, kernel_size)]

        if self.params.pooling is not None:
            layers.append(SamePad2d(self.params.pooling_size))
            if self.params.pooling == "max":
                layers.append(nn.MaxPool2d(self.params.pooling_size))
            elif self.params.pooling == "avg":
                layers.append(nn.AvgPool2d(self.params.pooling_size))
            else:
                raise ValueError("Unknown pooling method")

        # Activation usually after pooling
        layers.append(self.params.Activation())

        # Batchnorm usually after activation
        if self.params.batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def _output_size(self, input_size):
        def per_layer(input_size):
            return math.floor(
                (input_size - 1) / self.params.pooling_size + 1)

        out_size = input_size
        if self.params.pooling is not None:
            for _ in range(self.params.num_layers):
                out_size = per_layer(out_size)
        return out_size


class RNN(nn.Module):
    @dataclasses.dataclass(kw_only=True)
    class Params(Params):
        rnn: str = "gru"
        RNN: Type[nn.Module] = dataclasses.field(init=False, repr=False)

        embedding_dim: int = 32
        hidden_dim: int = 32
        num_layers: int = 3
        dropout: float = 0.2
        readout: str = "max"

        def __post_init__(self):
            super().__post_init__()
            try:
                self.RNN = nn_lower[self.rnn.lower()]  # type: ignore
            except KeyError:
                raise ValueError(f"Unknown RNN: {self.rnn}") from None

    def __init__(self, params: "RNN.Params", num_tokens: int):
        super().__init__()

        self.params = params
        if params.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Embedding layer
        self.embedding = nn.Embedding(
            num_tokens, params.embedding_dim, padding_idx=0)

        # RNN layers
        self.middle = params.RNN(params.embedding_dim, params.hidden_dim,
                                 num_layers=params.num_layers,
                                 dropout=params.dropout)

        # All pooling functions make (L, N, hidden_dim) -> (N, hidden_dim)
        if params.readout == "sum":
            self.readout = utils.masked_sum
        elif params.readout == "mean":
            self.readout = utils.masked_mean
        elif params.readout == "max":
            self.readout = utils.masked_max
        elif params.readout == "last":
            self.readout = utils.masked_last
        else:
            raise ValueError(f"Unknown readout: {params.readout}")

        # Linear projection to the output
        self.out = nn.Linear(self.params.hidden_dim, 2)

    def forward(self, X: torch.Tensor):
        # Shape: (N, L) -> (L, N, 1)
        mask = utils.get_mask(X).transpose(0, 1).unsqueeze_(-1)

        # Shape: (N, L) -> (L, N, embedding_dim)
        h = self.embedding(X).transpose(0, 1)
        # Shape: (L, N, embedding_dim) -> (L, N, hidden_dim)
        h, _ = self.middle(h)
        # Shape: (L, N, hidden_dim) -> (N, hidden_dim)
        h = self.readout(h, mask)
        # Shape: (N, hidden_dim) -> (N, 2)
        y_hat = self.out(h)
        return y_hat
