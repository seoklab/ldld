import dataclasses
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
from typing import Type

import torch
from torch import nn, optim

from ..utils import DEFAULT_DEVICE

nn_lower = {k.lower(): v for k, v in vars(nn).items()}
optim_lower = {k.lower(): v for k, v in vars(optim).items()}


@dataclasses.dataclass(kw_only=True)
class Params:
    # Training hyperparams
    batch_size: int = 128
    epochs: int = 20

    # Model hyperparams
    activation: str = "relu"
    Activation: Type[nn.Module] = dataclasses.field(init=False, repr=False)

    # Optimizer hyperparams
    optimizer: str = "adam"
    Optimizer: Type[optim.Optimizer] = dataclasses.field(
        init=False, repr=False)
    lr: float = 1e-4

    # Misc
    print_every: int = 4
    save_every: int = 1
    device: torch.device = DEFAULT_DEVICE
    save: Path = None

    def __post_init__(self):
        try:
            self.Activation = nn_lower[self.activation.lower()]
            self.Optimizer = optim_lower[self.optimizer.lower()]
        except KeyError:
            raise ValueError("Invalid activation or optimizer: "
                             f"{self.activation}, {self.optimizer}") from None

        self.device = torch.device(self.device)
        if self.save is None:
            self.save = Path("results", datetime.now().isoformat(sep="_"))

    @classmethod
    def from_args(cls, parser: ArgumentParser = None):
        if parser is None:
            parser = ArgumentParser()

        for field in dataclasses.fields(cls):
            if not field.init:
                continue

            if field.type in (int, float, Path):
                tp = field.type
            elif field.type is bool:
                def tp(x: str):
                    return x.lower() in ("true", "1")
            else:
                tp = str
            parser.add_argument(f"--{field.name}", type=tp, required=False)

        self = parser.parse_args(namespace=cls())
        self.__post_init__()
        return self
