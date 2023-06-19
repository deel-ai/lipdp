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
import wandb
from deel.lipdp import losses
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

cfg.batch_size = 5_000
cfg.delta = 1e-5
cfg.epsilon_max = 10.0
cfg.input_bound = 15.0
cfg.K = 0.99
cfg.learning_rate = 1e-3
cfg.log_wandb = "sweep_weekendboi"
cfg.opt_iterations = 10
cfg.noise_multiplier = 3.0
cfg.noisify_strategy = "global"
cfg.representation = "HSV"
cfg.optimizer = "Adam"
cfg.sweep_yaml_config = ""
cfg.tau = 8.0
cfg.sweep_id = ""
cfg.loss = "TauCategoricalCrossentropy"

_CONFIG = config_flags.DEFINE_config_dict("cfg", cfg)


def create_Mixer(dataset_metadata, dp_parameters):
    layers = [
        DP_layers.DP_BoundedInput(
            input_shape=dataset_metadata.input_shape,
            upper_bound=dataset_metadata.max_norm,
        )
    ]

    patch_size = 3
    num_mixer_layers = 2
    seq_len = (dataset_metadata.input_shape[0] // patch_size) * (
        dataset_metadata.input_shape[1] // patch_size
    )
    multiplier = 2
    mlp_seq_dim = multiplier * seq_len
    mlp_channel_dim = multiplier * seq_len
    hidden_size = multiplier * seq_len

    layers.append(
        DP_layers.DP_Lambda(
            tf.image.extract_patches,
            arguments=dict(
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            ),
        )
    )

    layers.append(
        DP_layers.DP_Reshape(
            (seq_len, (patch_size**2) * dataset_metadata.input_shape[-1])
        )
    )
    layers.append(
        DP_layers.DP_QuickSpectralDense(
            units=hidden_size, use_bias=False, kernel_initializer="identity"
        )
    )

    for _ in range(num_mixer_layers):
        to_add = [
            DP_layers.DP_Permute((2, 1)),
            DP_layers.DP_QuickSpectralDense(
                units=mlp_seq_dim, use_bias=False, kernel_initializer="identity"
            ),
        ]
        to_add.append(DP_layers.DP_GroupSort(2))
        to_add.append(DP_layers.DP_LayerCentering())
        to_add += [
            DP_layers.DP_QuickSpectralDense(
                units=seq_len, use_bias=False, kernel_initializer="identity"
            ),
            DP_layers.DP_Permute((2, 1)),
        ]
        layers += DP_layers.make_residuals("1-lip-add", to_add)
        to_add = [
            DP_layers.DP_QuickSpectralDense(
                units=mlp_channel_dim, use_bias=False, kernel_initializer="identity"
            ),
        ]
        to_add.append(DP_layers.DP_GroupSort(2))
        to_add.append(DP_layers.DP_LayerCentering())
        to_add.append(
            DP_layers.DP_QuickSpectralDense(
                units=hidden_size, use_bias=False, kernel_initializer="identity"
            )
        )
        layers += DP_layers.make_residuals("1-lip-add", to_add)

    layers.append(DP_layers.DP_Flatten())

    layers.append(
        DP_layers.DP_QuickSpectralDense(
            units=10, use_bias=False, kernel_initializer="identity"
        )
    )
    layers.append(DP_layers.DP_ClipGradient())

    model = DP_Model(
        layers,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
        nm_dynamic_clipping=100.0,
        name="mlp_mixer",
    )

    model.build(input_shape=(None, *dataset_metadata.input_shape))

    return model


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

    model = create_Mixer(dataset_metadata, dp_parameters)

    if cfg.loss == "TauCategoricalCrossentropy":
        loss = losses.DP_TauCategoricalCrossentropy(cfg.tau)
    elif cfg.loss == "KCosineSimilarity":
        loss = losses.DP_KCosineSimilarity(cfg.K)

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
        AdaptiveLossGradientClipping(),
    ]

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
