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
import pandas as pd
import tensorflow as tf
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags

import wandb
from deel.lipdp.dynamic import AdaptiveQuantileClipping
from deel.lipdp.layers import DP_AddBias
from deel.lipdp.layers import DP_BoundedInput
from deel.lipdp.layers import DP_ClipGradient
from deel.lipdp.layers import DP_Flatten
from deel.lipdp.layers import DP_GroupSort
from deel.lipdp.layers import DP_LayerCentering
from deel.lipdp.layers import DP_ScaledL2NormPooling2D
from deel.lipdp.layers import DP_SpectralConv2D
from deel.lipdp.layers import DP_SpectralDense
from deel.lipdp.losses import *
from deel.lipdp.model import DP_Accountant
from deel.lipdp.model import DP_Sequential
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_normalize
from deel.lipdp.pipeline import default_delta_value
from deel.lipdp.pipeline import load_and_prepare_images_data
from deel.lipdp.sensitivity import get_max_epochs
from experiments.wandb_utils import init_wandb
from experiments.wandb_utils import run_with_wandb
from wandb.keras import WandbCallback


def default_cfg_mnist():
    cfg = config_dict.ConfigDict()
    cfg.add_biases = True
    cfg.batch_size = 2_000
    cfg.clip_loss_gradient = None  # not required for dynamic clipping.
    cfg.dynamic_clipping = "quantiles"  # can be "fixed", "laplace", "quantiles". "fixed" requires a clipping value.
    cfg.dynamic_clipping_quantiles = (
        0.9  # crop to 90% of the distribution of gradient norm.
    )
    cfg.epsilon_max = 3.0
    cfg.input_clipping = 0.7
    cfg.learning_rate = 5e-3
    cfg.loss = "TauCategoricalCrossentropy"
    cfg.log_wandb = "disabled"
    cfg.noise_multiplier = 1.5
    cfg.noisify_strategy = "per-layer"
    cfg.optimizer = "Adam"
    cfg.opt_iterations = None
    cfg.save = False
    cfg.save_folder = os.getcwd()
    cfg.sweep_yaml_config = ""
    cfg.sweep_id = ""
    cfg.tau = 32.0
    return cfg


cfg = default_cfg_mnist()
_CONFIG = config_flags.DEFINE_config_dict("cfg", cfg)


def create_ConvNet(dp_parameters, dataset_metadata):
    norm_max = 1.0
    all_layers = [
        DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=dataset_metadata.max_norm),
        DP_SpectralConv2D(
            filters=16,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=norm_max),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_SpectralConv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        DP_AddBias(norm_max=norm_max),
        DP_GroupSort(2),
        DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        DP_LayerCentering(),
        DP_Flatten(),
        DP_SpectralDense(1024, use_bias=False, kernel_initializer="orthogonal"),
        DP_AddBias(norm_max=norm_max),
        DP_SpectralDense(10, use_bias=False, kernel_initializer="orthogonal"),
        DP_AddBias(norm_max=norm_max),
        DP_ClipGradient(
            clip_value=cfg.clip_loss_gradient,
            mode="dynamic",
        ),
    ]
    if not cfg.add_biases:
        all_layers = [
            layer for layer in all_layers if not isinstance(layer, DP_AddBias)
        ]
    model = DP_Sequential(
        all_layers,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
    )
    return model


def compile_model(model, cfg):
    # Choice of optimizer
    if cfg.optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    else:
        print("Illegal optimizer argument : ", cfg.optimizer)

    # Choice of loss function
    if cfg.loss == "MulticlassHKR":
        if cfg.optimizer == "SGD":
            cfg.learning_rate = cfg.learning_rate / cfg.alpha
        loss = DP_MulticlassHKR(
            alpha=50.0,
            min_margin=0.5,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassHinge":
        loss = DP_MulticlassHinge(
            min_margin=0.5,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassKR":
        loss = DP_MulticlassKR(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif cfg.loss == "TauCategoricalCrossentropy":
        loss = DP_TauCategoricalCrossentropy(
            cfg.tau, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "KCosineSimilarity":
        loss = DP_KCosineSimilarity(
            0.99, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "MAE":
        loss = DP_MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
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
    init_wandb(cfg=cfg, project="MNIST_ClipLess_SGD")

    ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
        "mnist",
        cfg.batch_size,
        colorspace="grayscale",
        drop_remainder=True,
        bound_fct=bound_normalize(),
    )

    model = create_ConvNet(
        DPParameters(
            noisify_strategy=cfg.noisify_strategy,
            noise_multiplier=cfg.noise_multiplier,
            delta=default_delta_value(dataset_metadata),
        ),
        dataset_metadata,
    )

    model = compile_model(model, cfg)
    model.summary()

    num_epochs = get_max_epochs(cfg.epsilon_max, model)

    adaptive = AdaptiveQuantileClipping(
        ds_train=ds_train,
        patience=1,
        noise_multiplier=cfg.noise_multiplier * 5,  # more noisy.
        quantile=cfg.dynamic_clipping_quantiles,
        learning_rate=1.0,
    )
    adaptive.set_model(model)
    callbacks = [
        WandbCallback(save_model=False, monitor="val_accuracy"),
        DP_Accountant(),
        adaptive,
    ]

    hist = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
    )


def main(_):
    run_with_wandb(cfg=cfg, train_function=train, project="ICLR_MNIST_acc")


if __name__ == "__main__":
    app.run(main)
