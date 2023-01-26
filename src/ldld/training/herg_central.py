from typing import Type, Union

import torch
from torch import nn

from .. import datasets, utils
from ..models import (
    utils as model_utils, RNN, TransformerClassifier, GNNClassifier)
from . import common, utils as training_utils


class _HergCentralRunner(common.Runner):
    def on_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        return y, y_hat, loss


class HergCentralTrainer(_HergCentralRunner, common.Trainer):
    pass


class HergCentralValidator(_HergCentralRunner, common.Validator):
    pass


class HergCentralTester(_HergCentralRunner, common.Tester):
    pass


class _HergCentralSmiMixin:
    @classmethod
    def from_params(cls, params):
        num_tokens = len(datasets.HergCentralDatasetSmi.tok_to_idx) + 1
        return cls(params, num_tokens)


class HergCentralRNN(_HergCentralSmiMixin, RNN):
    pass


class HergCentralTransformer(_HergCentralSmiMixin, TransformerClassifier):
    pass


class HergCentralGNN(GNNClassifier):
    @classmethod
    def from_params(cls, params):
        atom_classes, bond_classes = datasets.get_feature_dims()
        return cls(params, atom_classes, bond_classes)


def _train_herg_central(
        Model: Type[Union[HergCentralRNN,
                          HergCentralTransformer,
                          HergCentralGNN]],
        Dataset: Type[Union[datasets.HergCentralDatasetSmi,
                            datasets.HergCentralDatasetGraph]]):
    utils.init_seaborn()
    utils.seed_everything()

    params = Model.Params.from_args()
    params.save.mkdir(parents=True, exist_ok=True)
    params.save.joinpath("params.txt").write_text(str(params) + "\n")

    split = datasets.hergcentral.split
    sets = {k: Dataset(df) for k, df in split.items()}
    train_loader, valid_loader, test_loader = (
        datasets.to_dataloader(
            s, params.batch_size, k == "train", collate_fn=Dataset.collate_fn)
        for k, s in sets.items())

    train_set = sets["train"]
    total = len(train_set)
    pos_freq = train_set.y.sum().item() / total
    w = torch.tensor([pos_freq, 1 - pos_freq], device=params.device)
    w *= len(w)

    model = Model.from_params(params).to(params.device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = params.Optimizer(model.parameters(), lr=params.lr)
    model_utils.print_model_summary(model)

    trainer = HergCentralTrainer(
        params, model, criterion, optimizer, train_loader)
    validator = HergCentralValidator(
        params, model, criterion, valid_loader)
    tester = HergCentralTester(params, model, criterion, test_loader)

    stats = common.main_loop(params, trainer, validator)

    best_epoch, best_state = common.load_best_state(params)
    model.load_state_dict(best_state)
    test_stats = common.test(tester, best_epoch)

    all_stats = stats.join(
        test_stats.set_index("epoch"), on="epoch", how="left")
    all_stats.to_csv(params.save / "stats.csv", index=False)

    training_utils.plot_stats(params, all_stats, params.save / "stats.png")
    training_utils.draw_confusion_matrix(
        params, model, test_loader, params.save / "confusion_matrix.png")


def train_rnn():
    _train_herg_central(HergCentralRNN, datasets.HergCentralDatasetSmi)


def train_trs():
    _train_herg_central(HergCentralTransformer, datasets.HergCentralDatasetSmi)


def train_gnn():
    _train_herg_central(HergCentralGNN, datasets.HergCentralDatasetGraph)
