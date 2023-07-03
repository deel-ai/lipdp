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
import yaml
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
from models_CIFAR import create_MLP_Mixer
from models_CIFAR import create_ResNet
from models_CIFAR import create_VGG
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import deel.lipdp.layers as DP_layers
import deel.lipdp.losses as DP_losses
import wandb
from deel.lipdp.model import AdaptiveLossGradientClipping
from deel.lipdp.model import DP_Accountant
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_clip_value
from deel.lipdp.pipeline import load_and_prepare_data
from deel.lipdp.sensitivity import get_max_epochs
from wandb.keras import WandbCallback
from wandb_sweeps.src_config.wandb_utils import init_wandb
from wandb_sweeps.src_config.wandb_utils import run_with_wandb

cfg = config_dict.ConfigDict()

cfg.architecture = "MLP_Mixer"  # resnet and VGG are also options
cfg.batch_size = 1000
cfg.delta = 1e-5
cfg.epsilon_max = 10.0
cfg.input_bound = 15.0
cfg.K = 0.99
cfg.learning_rate = 1e-3
cfg.log_wandb = "run"
cfg.opt_iterations = 10
cfg.noise_multiplier = 3.0
cfg.noisify_strategy = "global"
cfg.representation = "HSV"
cfg.optimizer = "Adam"
cfg.sweep_yaml_config = ""
cfg.tau = 8.0
cfg.sweep_id = ""
cfg.loss = "TauCategoricalCrossentropy"
cfg.alpha = 50
cfg.min_margin = 0.5
cfg.add_biases = False
cfg.clip_loss_gradient = True
cfg.layer_centering = False
cfg.patch_size = 3
cfg.hidden_size = 256
cfg.mlp_channel_dim = 256
cfg.mlp_seq_dim = 256
cfg.num_mixer_layers = 2
cfg.skip_connections = True

_CONFIG = config_flags.DEFINE_config_dict("cfg", cfg)


def train():
    init_wandb(cfg=cfg, project="CIFAR10_dynamic_clipping")

    # declare the privacy parameters
    dp_parameters = DPParameters(
        noisify_strategy=cfg.noisify_strategy,
        noise_multiplier=cfg.noise_multiplier,
        delta=cfg.delta,
    )

    ds_train, ds_test, dataset_metadata = load_and_prepare_data(
        "cifar10",
        batch_size=cfg.batch_size,
        colorspace=cfg.representation,
        drop_remainder=True,  # accounting assumes fixed batch size
        bound_fct=bound_clip_value(
            cfg.input_bound
        ),  # clipping preprocessing allows to control input bound
    )

    if "resnet" in cfg.architecture:
        model = create_ResNet(dp_parameters, dataset_metadata, cfg, cfg.input_bound)
    elif cfg.architecture == "MLP_Mixer":
        model = create_MLP_Mixer(dp_parameters, dataset_metadata, cfg, cfg.input_bound)
    elif "VGG" in cfg.architecture:
        model = create_VGG(dp_parameters, dataset_metadata, cfg, cfg.input_bound)
    else:
        print("Illegal architecture argument")

    # Choice of loss function
    if cfg.loss == "MulticlassHKR":
        cfg.learning_rate = cfg.learning_rate / cfg.alpha
        loss = DP_losses.DP_MulticlassHKR(
            alpha=cfg.alpha,
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassHinge":
        loss = DP_losses.DP_MulticlassHinge(
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassKR":
        loss = DP_losses.DP_MulticlassKR(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "TauCategoricalCrossentropy":
        loss = DP_losses.DP_TauCategoricalCrossentropy(
            cfg.tau, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "KCosineSimilarity":
        loss = DP_losses.DP_KCosineSimilarity(
            cfg.K, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "MAE":
        loss = DP_losses.DP_MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    else:
        raise ValueError(f"Illegal loss argument {cfg.loss}")

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        # accuracy metric is necessary for dynamic loss gradient clipping
        metrics=["accuracy"],
        run_eagerly=False,
    )

    num_epochs = get_max_epochs(cfg.epsilon_max, model)

    callbacks = [
        WandbCallback(save_model=False, monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=15),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.9, min_delta=0.001, patience=8
        ),
        DP_Accountant(),
    ]

    if cfg.clip_loss_gradient:
        callbacks.append(
            AdaptiveLossGradientClipping(
                ds_train=ds_train
            ),  # DO NOT USE THIS CALLBACK WHEN mode = "dynamic_svt"
        )

    hist = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
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


def main(_):
    run_with_wandb(cfg=cfg, train_function=train, project="CIFAR10_dynamic_clipping")


if __name__ == "__main__":
    app.run(main)
