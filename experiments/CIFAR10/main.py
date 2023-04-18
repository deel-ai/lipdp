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
import os

import numpy as np
import tensorflow as tf
import tensorflow_privacy
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import (
    compute_noise,
)

import deel
import wandb
from deel.lip.activations import GroupSort
from deel.lip.layers import ScaledL2NormPooling2D
from deel.lip.layers import SpectralConv2D
from deel.lip.layers import SpectralDense
from deel.lip.losses import MulticlassHinge
from deel.lip.losses import MulticlassHKR
from deel.lip.losses import MulticlassKR
from deel.lip.losses import TauCategoricalCrossentropy
from deel.lipdp.losses import KCosineSimilarity
from deel.lipdp.model import DP_LipNet
from deel.lipdp.pipeline import load_data_cifar
from wandb.keras import WandbCallback
from wandb_sweeps.src_config.sweep_config import get_sweep_config

cfg = config_dict.ConfigDict()

cfg.alpha = 50.0
cfg.beta_1 = 0.9
cfg.beta_2 = 0.999
cfg.batch_size = 4096
cfg.condense = True
cfg.delta = 1e-5
cfg.epsilon = 500.0
cfg.input_clipping = 1.0
cfg.K = 1.0
cfg.learning_rate = 0.25
cfg.lip_coef = 1.0
cfg.loss = "TauCategoricalCrossentropy"
cfg.log_wandb = "disabled"
cfg.min_margin = 0.5
cfg.min_norm = 5.21
cfg.model_name = "No_name"
cfg.noise_multiplier = 0.0
cfg.optimizer = "Adam"
cfg.N = 50 * 1000
cfg.num_classes = 10
cfg.opt_iterations = 10
cfg.save = False
cfg.save_folder = os.getcwd()
cfg.steps = 300
cfg.sweep_id = ""
cfg.tau = 8.0
cfg.tag = "Default"

_CONFIG = config_flags.DEFINE_config_dict("cfg", cfg)


def create_model(cfg, InputUpperBound):
    # Using VGG architecture of Papernot et Al.
    model = DP_LipNet(
        [
            SpectralConv2D(
                filters=32,
                kernel_size=3,
                input_shape=(32, 32, 3),
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            SpectralConv2D(
                filters=32,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            ScaledL2NormPooling2D(pool_size=2, strides=2),
            SpectralConv2D(
                filters=64,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            SpectralConv2D(
                filters=64,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            ScaledL2NormPooling2D(pool_size=2, strides=2),
            SpectralConv2D(
                filters=128,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            SpectralConv2D(
                filters=128,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            ScaledL2NormPooling2D(pool_size=2, strides=2),
            SpectralConv2D(
                filters=256,
                kernel_size=3,
                kernel_initializer="orthogonal",
                activation=GroupSort(2),
                strides=1,
                use_bias=False,
            ),
            ScaledL2NormPooling2D(pool_size=4, strides=4),
            Flatten(),
            SpectralDense(4096, use_bias=False),
            SpectralDense(10, use_bias=False),
        ],
        k_coef_lip=1.0,
        name="hkr_model",
        X=InputUpperBound,
        cfg=cfg,
    )
    return model


def compile_model(model, cfg):
    # Choice of optimizer
    if cfg.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate,
            beta_1=cfg.beta_1,
            beta_2=cfg.beta_2,
            epsilon=1e-12,
        )
    else:
        raise TypeError("Illegal optimizer argument : ", cfg.optimizer)
    # Choice of loss function
    if cfg.loss == "MulticlassHKR":
        if cfg.optimizer == "SGD":
            cfg.learning_rate = cfg.learning_rate / cfg.alpha
        loss = MulticlassHKR(
            alpha=cfg.alpha,
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassHinge":
        loss = MulticlassHinge(
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassKR":
        loss = MulticlassKR(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif cfg.loss == "TauCategoricalCrossentropy":
        loss = TauCategoricalCrossentropy(
            cfg.tau, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "KCosineSimilarity":
        KX_min = cfg.K * cfg.min_norm
        loss = KCosineSimilarity(
            KX_min, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    else:
        raise ValueError(f"Illegal loss argument {cfg.loss}")
    # Compile model
    model.compile(
        # decreasing alpha and increasing min_margin improve robustness (at the cost of accuracy)
        # note also in the case of lipschitz networks, more robustness require more parameters.
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def train():
    if cfg.log_wandb == "run":
        wandb.init(project="dp-lipschitz", mode="online", config=cfg)

    elif cfg.log_wandb == "disabled":
        wandb.init(project="dp-lipschitz", mode="disabled", config=cfg)

    elif cfg.log_wandb.startswith("sweep_"):
        wandb.init()
        for key, value in wandb.config.items():
            cfg[key] = value

    num_epochs = cfg.steps // (cfg.N // cfg.batch_size)
    cfg.noise_multiplier = compute_noise(
        cfg.N, cfg.batch_size, cfg.epsilon, num_epochs, cfg.delta, 1e-6
    )

    x_train, x_test, y_train, y_test, InputUpperBound = load_data_cifar(
        cfg.input_clipping
    )
    model = create_model(cfg, InputUpperBound)
    model = compile_model(model, cfg)
    num_epochs = cfg.steps // (cfg.N // cfg.batch_size)
    callbacks = [
        WandbCallback(monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=15),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.9, min_delta=0.0001, patience=5
        ),
    ]
    hist = model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        batch_size=cfg.batch_size,
        callbacks=callbacks,
    )
    wandb.log(
        {
            "Accuracies": wandb.plot.line_series(
                xs=[
                    np.linspace(0, num_epochs, num_epochs + 1),
                    np.linspace(0, num_epochs, num_epochs + 1),
                ],
                ys=[hist.history["accuracy"], hist.history["val_accuracy"]],
                keys=["Train Accuracy", "Test Accuracy"],
                title="Train/Test Accuracy",
                xname="num_epochs",
            )
        }
    )
    if cfg.save:
        model.save(f"{cfg.save_folder}/{cfg.model_name}.h5")


def main(_):
    wandb.login()
    if cfg.log_wandb in ["run", "disabled"]:
        train()
    elif cfg.log_wandb.startswith("sweep_"):
        if cfg.sweep_id == "":
            sweep_config = get_sweep_config(cfg)
            sweep_id = wandb.sweep(sweep=sweep_config, project="dp-lipschitz")
        else:
            sweep_id = cfg.sweep_id
        wandb.agent(sweep_id, function=train, count=cfg.opt_iterations)


if __name__ == "__main__":
    app.run(main)
