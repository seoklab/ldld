import math
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn
from sklearn import metrics
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from ..models import Params


def _savefig(save_at: Path, dpi=300, bbox_inches="tight", transparent=True):
    plt.savefig(
        save_at, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    plt.close()


def draw_wrong_predictions(wrong: list, save_at: Path, col=7):
    if not wrong:
        print("No wrong predictions!")
        return

    row = math.ceil(len(wrong) / col)
    _, axes = plt.subplots(
        row, col, figsize=(col * 1.5, row * 1.5), gridspec_kw={"hspace": 0.5})
    for ax in axes.flatten():
        ax.axis("off")
    for ax, (pred, target, img) in zip(axes.flatten(), wrong):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{target} -> {pred}")

    _savefig(save_at)


def plot_stats(params: Params, stats: pd.DataFrame, save_at: Path):
    stats_long = stats.melt(
        id_vars="epoch", var_name="type", value_name="stat")
    stats_long["run"] = [tp.split("_")[0] for tp in stats_long["type"]]
    stats_long["type"] = [tp.split("_")[1] for tp in stats_long["type"]]

    epochs = params.epochs
    fg = sns.relplot(x="epoch", y="stat", data=stats_long,
                     hue="run", col="type", marker="o", kind="line",
                     col_order=["loss", "accu", "f1"],
                     facet_kws={"sharey": False, "xlim": (1, epochs + 1)})
    for ax in fg.axes[0, 1:]:
        ax.set_ylim(0, 1)

    xticks = list(range(1, epochs + 1, params.print_every))
    if epochs != xticks[-1]:
        xticks.append(epochs)
    fg.set(xticks=xticks)

    _savefig(save_at)


@torch.no_grad()
def draw_confusion_matrix(params: Params, model: nn.Module,
                          test_loader: DataLoader, save_at: Path):
    model.eval()

    y_true = []
    y_pred = []
    for X, y in test_loader:
        y_true.append(y.numpy())

        y_hat = model(X.to(params.device))
        y_pred.append(y_hat.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    _savefig(save_at)
