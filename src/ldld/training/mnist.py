from typing import Type, Union

import torch
from torch import nn
from sklearn import metrics

from .. import datasets, utils
from ..models import utils as model_utils, MLP, CNN
from . import common, utils as training_utils


class _MNISTRunner(common.Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_wrong = []

    def on_epoch_begin(self):
        super().on_epoch_begin()
        self.epoch_wrong.clear()

    def on_batch(self, batch):
        idxs, (x, y) = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        with torch.no_grad():
            y_hat_class = y_hat.argmax(dim=1)
            correct = y_hat_class == y
            for i, pred, target, res in zip(idxs, y_hat_class, y, correct):
                if not res:
                    self.epoch_wrong.append(
                        (pred, target, self.loader.dataset.imgs[i]))

        return y, y_hat, loss

    @staticmethod
    def calculate_f1_score(y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average="macro")


class _MLPRunner(_MNISTRunner):
    def on_batch(self, batch):
        idxs, (x, y) = batch
        x = torch.flatten(x, start_dim=1)
        return super().on_batch((idxs, (x, y)))


class MLPTrainer(_MLPRunner, common.Trainer):
    pass


class MLPValidator(_MLPRunner, common.Validator):
    pass


class CNNTrainer(_MNISTRunner, common.Trainer):
    pass


class CNNValidator(_MNISTRunner, common.Validator):
    pass


class MnistMLP(MLP):
    @classmethod
    def from_params(cls, params, H, W):
        return cls(params, H * W, 10)


class MnistCNN(CNN):
    @classmethod
    def from_params(cls, params, H, W):
        return cls(params, H, W, 10)


def _train_mnist(Model: Type[Union[MnistMLP, MnistCNN]],
                 Trainer: Type[common.Trainer],
                 Validator: Type[common.Validator]):
    utils.init_seaborn()
    utils.seed_everything()

    params = Model.Params.from_args()
    params.save.mkdir(parents=True, exist_ok=True)
    params.save.joinpath("params.txt").write_text(str(params) + "\n")

    train_set, valid_set = (
        datasets.MNIST_GPU(train).to(params.device)
        for train in [True, False])
    train_loader = datasets.to_dataloader(train_set, params.batch_size, True)
    valid_loader = datasets.to_dataloader(valid_set, params.batch_size, False)
    H, W = train_set[0][1][0].shape[-2:]

    model = Model.from_params(params, H, W).to(params.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = params.Optimizer(model.parameters(), lr=params.lr)
    model_utils.print_model_summary(model)

    trainer = Trainer(params, model, criterion, optimizer, train_loader)
    validator = Validator(params, model, criterion, valid_loader)

    stats = common.main_loop(params, trainer, validator)
    stats.to_csv(params.save / "stats.csv", index=False)

    training_utils.draw_wrong_predictions(
        trainer.epoch_wrong, params.save / "train_wrong.png")
    training_utils.draw_wrong_predictions(
        validator.epoch_wrong, params.save / "valid_wrong.png")


def train_mlp():
    _train_mnist(MnistMLP, MLPTrainer, MLPValidator)


def train_cnn():
    _train_mnist(MnistCNN, CNNTrainer, CNNValidator)
