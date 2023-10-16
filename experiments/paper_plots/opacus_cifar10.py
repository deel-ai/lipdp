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
import warnings

from absl import app
from ml_collections import config_dict
from ml_collections import config_flags

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
from mlp_mixer_pytorch import MLPMixer

from experiments.wandb_utils import init_wandb
from experiments.wandb_utils import run_with_wandb

import wandb


def default_cfg_cifar10():
    cfg = config_dict.ConfigDict()
    cfg.MAX_GRAD_NORM = 1.2
    cfg.EPSILON = 20.0
    cfg.DELTA = 1e-5
    cfg.EPOCHS = 20
    cfg.LR = 1e-3
    cfg.BATCH_SIZE = 1000
    cfg.MAX_PHYSICAL_BATCH_SIZE = 200
    cfg.log_wandb = "disabled"
    cfg.model = "mlp_mixer"
    cfg.sweep_id = ""  # useful to resume a sweep.
    cfg.sweep_yaml_config = ""  # useful to load a sweep from a yaml file.
    return cfg


project = "ICLR_Opacus_Cifar10"
cfg = default_cfg_cifar10()
_CONFIG = config_flags.DEFINE_config_dict(
    "cfg", cfg
)  # for FLAGS parsing in command line.


def accuracy(preds, labels):
    return (preds == labels).mean()


def test(model, test_loader, epoch, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc: {top1_avg * 100:.6f} ")
    res_test = {
        "val_epoch": epoch,
        "val_loss": np.mean(losses),
        "val_accuracy": np.mean(top1_acc),
    }

    return res_test


def train(model, privacy_engine, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

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
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

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
            "epsilon": np.mean(epsilon),
            "delta": cfg.DELTA,
        }
    return res_train


def train_dp_model():
    init_wandb(cfg=cfg, project=project)

    # These values, specific to the CIFAR10 dataset, are assumed to be known.
    # If necessary, they can be computed with modest privacy budgets.
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ]
    )

    DATA_ROOT = "/data/datasets/pytorch/CIFAR10"

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
    )

    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    if cfg.model == "resnet18":
        model = models.resnet18(weights=None, num_classes=10)
    elif cfg.model == "mlp_mixer":
        model = MLPMixer(
            image_size=32,
            channels=3,
            patch_size=4,
            dim=64,
            depth=1,
            num_classes=10,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]

    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=True)
    assert not errors

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
