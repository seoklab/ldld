from abc import ABC, abstractmethod
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm, trange

from ..models import Params


class Runner(ABC):
    def __init__(self, params: Params, model: nn.Module,
                 criterion: nn.Module, loader: DataLoader):
        self.params = params
        self.model = model
        self.criterion = criterion
        self.loader = loader
        self.device = params.device

    def run(self, epoch: int):
        self.on_epoch_begin()

        epoch_loss = []
        epoch_true = []
        epoch_pred = []

        for batch in self.loader:
            self.on_batch_begin()

            y, y_hat, loss = self.on_batch(batch)

            self.on_batch_end(loss)
            epoch_loss.append(loss.item())

            epoch_true.append(y.cpu().numpy())
            epoch_pred.append(y_hat.argmax(dim=1).cpu().numpy())

        epoch_true = np.concatenate(epoch_true)
        epoch_pred = np.concatenate(epoch_pred)

        epoch_loss = np.mean(epoch_loss)
        epoch_accu = metrics.accuracy_score(epoch_true, epoch_pred)
        epoch_f1 = self.calculate_f1_score(epoch_true, epoch_pred)

        self.on_epoch_end(epoch, epoch_loss)

        return epoch_loss, epoch_accu, epoch_f1

    @staticmethod
    def calculate_f1_score(y_true, y_pred):
        return metrics.f1_score(y_true, y_pred)

    def on_epoch_begin(self):
        pass

    def on_batch_begin(self):
        pass

    @abstractmethod
    def on_batch(self, batch) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def on_batch_end(self, loss: torch.Tensor):
        pass

    def on_epoch_end(self, epoch: int, epoch_loss: float):
        pass


class Trainer(Runner):
    def __init__(self, params: Params, model: nn.Module, criterion: nn.Module,
                 optim: optim.Optimizer, loader: DataLoader):
        super().__init__(params, model, criterion, loader)
        self.optim = optim

    def on_epoch_begin(self):
        self.model.train()

    def on_batch_begin(self):
        self.optim.zero_grad()

    def on_batch_end(self, loss: torch.Tensor):
        loss.backward()
        self.optim.step()


class Tester(Runner):
    def on_epoch_begin(self):
        self.model.eval()

    def run(self, epoch: int):
        with torch.no_grad():
            return super().run(epoch)


class Validator(Tester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_loss = None

    def on_epoch_end(self, epoch: int, epoch_loss: float):
        epoch_name = f"epoch-{epoch}.pth"
        is_best = self._best_loss is None or epoch_loss < self._best_loss

        if (is_best
                or not epoch % self.params.save_every
                or epoch in (1, self.params.epochs)):
            torch.save(self.model.state_dict(), self.params.save / epoch_name)

        if is_best:
            self._best_loss = epoch_loss
            best_file = self.params.save / "best.pth"
            best_file.unlink(missing_ok=True)
            best_file.symlink_to(epoch_name)


def load_best_state(params: Params):
    best_file = params.save / "best.pth"
    best_epoch = int(best_file.resolve(True).stem.split("-")[-1])
    return best_epoch, torch.load(best_file, map_location=params.device)


def main_loop(params: Params, trainer: Trainer, validator: Validator):
    stats = []
    for epoch in trange(1, params.epochs + 1):
        train_loss, train_acc, train_f1 = trainer.run(epoch)
        valid_loss, valid_acc, valid_f1 = validator.run(epoch)

        if not epoch % params.print_every or epoch in (1, params.epochs):
            tqdm.write(
                f"{train_loss = :.4f}, {train_acc = :.2%}, {train_f1 = :.4f}, "
                f"{valid_loss = :.4f}, {valid_acc = :.2%}, {valid_f1 = :.4f}")

        stats.append((epoch, train_loss, train_acc, train_f1,
                      valid_loss, valid_acc, valid_f1))

    stats = pd.DataFrame.from_records(stats, columns=[
        "epoch", "train_loss", "train_accu", "train_f1",
        "valid_loss", "valid_accu", "valid_f1"])
    return stats


def test(tester: Tester, best_epoch: int):
    test_loss, test_acc, test_f1 = tester.run(best_epoch)
    test_stats = pd.DataFrame.from_records(
        [(best_epoch, test_loss, test_acc, test_f1)],
        columns=["epoch", "test_loss", "test_accu", "test_f1"])
    return test_stats
