# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import random
import warnings

from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from opacus.validators import ModuleValidator
from torchvision.datasets import CIFAR10
from opacus import PrivacyEngine
import torch.nn as nn
import torch.optim as optim
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

from experiments.wandb_utils import init_wandb
from experiments.wandb_utils import run_with_wandb

import wandb


def default_cfg_tabular():
    cfg = config_dict.ConfigDict()
    cfg.dataset_name = "22_magic.gamma"
    cfg.MAX_GRAD_NORM = 1.2
    cfg.EPSILON = 1.0
    cfg.DELTA = 1e-5
    cfg.EPOCHS = 20
    cfg.LR = 1e-3
    cfg.BATCH_SIZE = 1000
    cfg.MAX_PHYSICAL_BATCH_SIZE = 2000
    cfg.log_wandb = "disabled"
    cfg.depth = 1
    cfg.model = "mlp"
    cfg.sweep_count = 0  # 0 means no limit.
    cfg.sweep_id = ""  # useful to resume a sweep.
    cfg.sweep_yaml_config = ""  # useful to load a sweep from a yaml file.
    cfg.width = 1
    return cfg


project = "ICLR_Opacus_Tabular"
cfg = default_cfg_tabular()
_CONFIG = config_flags.DEFINE_config_dict(
    "cfg", cfg
)  # for FLAGS parsing in command line.


def get_mlp(num_features):
    """Build multi-layer perceptron."""
    depth = cfg.depth
    width = cfg.width * 64
    layers = []
    last_width = num_features
    for i in range(depth):
        layers.append(nn.Linear(last_width, width))
        layers.append(nn.ReLU())
        last_width = width
    layers.append(nn.Linear(last_width, 1))
    return nn.Sequential(*layers)


def accuracy(preds, labels):
    return (preds == labels).mean()


def test(model, test_loader, epoch, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    top1_acc = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(torch.flatten(output), target)
            preds = np.ceil(output.detach().cpu().numpy())
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            all_labels.extend(labels)
            all_preds.extend(output.detach().cpu().numpy())

    top1_avg = np.mean(top1_acc)
    auroc = roc_auc_score(all_labels, all_preds)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
        f"AUROC: {auroc:.6f}"
    )
    res_test = {
        "val_epoch": epoch,
        "val_loss": np.mean(losses),
        "val_accuracy": np.mean(top1_acc),
        "val_auroc": auroc,
    }

    return res_test


def train(model, privacy_engine, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    top1_acc = []
    all_labels = []
    all_preds = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=cfg.MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        for i, (images, target) in (pbar := tqdm(enumerate(memory_safe_data_loader))):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(torch.flatten(output), target)

            preds = np.ceil(output.detach().cpu().numpy())
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            all_labels.extend(labels)
            all_preds.extend(output.detach().cpu().numpy())

            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                try:
                    epsilon = privacy_engine.get_epsilon(cfg.DELTA)
                except:
                    epsilon = float("nan")

            pbar.set_description(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {cfg.DELTA})"
            )

        try:
            epsilon = privacy_engine.get_epsilon(cfg.DELTA)
        except:
            epsilon = float("nan")
        res_train = {
            "epoch": epoch,
            "loss": np.mean(losses),
            "accuracy": np.mean(top1_acc),
            "auroc": roc_auc_score(all_labels, all_preds),
            "epsilon": np.mean(epsilon),
            "delta": cfg.DELTA,
        }
    return res_train


def download_adbench_datasets(dataset_dir: str):
    import os
    import fsspec

    fs = fsspec.filesystem("github", org="Minqi824", repo="ADBench")
    print(f"Downloading datasets from the remote github repo...")

    save_path = os.path.join(dataset_dir, "datasets", "Classical")
    print(f"Current saving path: {save_path}")

    os.makedirs(save_path, exist_ok=True)
    fs.get(fs.ls("adbench/datasets/" + "Classical"), save_path, recursive=True)


def load_adbench_data(
    dataset_name: str,
    dataset_dir: str,
    standardize: bool = True,
    redownload: bool = False,
):
    """Load a dataset from the adbench package."""
    if redownload:
        download_adbench_datasets(dataset_dir)

    data = np.load(
        f"{dataset_dir}/datasets/Classical/{dataset_name}.npz", allow_pickle=True
    )
    x_data, y_data = data["X"], data["y"]

    if standardize:
        x_data = (x_data - x_data.mean()) / x_data.std()

    return x_data, y_data


def train_dp_model():
    init_wandb(cfg=cfg, project=project)

    if cfg.BATCH_SIZE < cfg.MAX_PHYSICAL_BATCH_SIZE:
        cfg.MAX_PHYSICAL_BATCH_SIZE = cfg.BATCH_SIZE

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    x_data, y_data = load_adbench_data(
        cfg.dataset_name, dataset_dir="/data/datasets/adbench", standardize=True
    )

    print(f"x_data.shape = {x_data.shape}")
    print(f"y_data.shape = {y_data.shape} with labels {np.unique(y_data)}")

    random_state = random.randint(0, 1000)
    splits = train_test_split(
        x_data, y_data, test_size=0.2, random_state=random_state, stratify=y_data
    )
    x_train, x_test, y_train, y_test = splits

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).float(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    model = get_mlp(x_data.shape[1])

    errors = ModuleValidator.validate(model, strict=False)

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    print(
        f"Device = {torch.cuda.get_device_name(0)} and cuda={torch.cuda.is_available()}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=cfg.LR)

    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=cfg.EPOCHS,
        target_epsilon=cfg.EPSILON,
        target_delta=cfg.DELTA,
        max_grad_norm=cfg.MAX_GRAD_NORM,
    )

    print(f"Using sigma={optimizer.noise_multiplier} and C={cfg.MAX_GRAD_NORM}")

    for epoch in tqdm(range(cfg.EPOCHS), desc="Epoch", unit="epoch"):
        res = train(model, privacy_engine, train_loader, optimizer, epoch + 1, device)
        res_test = test(model, test_loader, epoch + 1, device)
        res.update(res_test)
        wandb.log(
            res,
            step=epoch,
        )
        with torch.no_grad():
            torch.cuda.empty_cache()

    del (
        model,
        train_loader,
        optimizer,
        test_loader,
        train_dataset,
        test_dataset,
        errors,
        privacy_engine,
    )
    with torch.no_grad():
        torch.cuda.empty_cache()


def main(_):
    run_with_wandb(cfg=cfg, train_function=train_dp_model, project=project)


if __name__ == "__main__":
    app.run(main)
