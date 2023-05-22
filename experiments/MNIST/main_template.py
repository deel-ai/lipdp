# -*- coding: utf-8 -*-
# Copyright anonymized et anonymized - All
# rights reserved. anonymized is a research program operated by anonymized, anonymized,
# anonymized and anonymized - https://www.anonymized.ai/
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
import wandb
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
from models_MNIST import create_ConvNet
from models_MNIST import create_Dense_Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from lipdp.losses import *
from lipdp.model import DPParameters
from lipdp.model import DP_Accountant
from lipdp.pipeline import bound_clip_value
from lipdp.pipeline import load_and_prepare_data
from lipdp.sensitivity import get_max_epochs
from wandb_sweeps.src_config.wandb_utils import init_wandb
from wandb_sweeps.src_config.wandb_utils import run_with_wandb

cfg = config_dict.ConfigDict()

cfg.add_biases = True
cfg.alpha = 50.0
cfg.architecture = "ConvNet"
cfg.batch_size = 8_192
cfg.condense = True
cfg.clip_loss_gradient = 1.0
cfg.delta = 1e-5
cfg.epsilon_max = 3.0
cfg.input_clipping = 0.7
cfg.K = 0.99
cfg.learning_rate = 1e-2
cfg.lip_coef = 1.0
cfg.loss = "TauCategoricalCrossentropy"
cfg.log_wandb = "disabled"
cfg.min_margin = 0.5
cfg.min_norm = 5.21
cfg.model_name = "No_name"
cfg.noise_multiplier = 5.0
cfg.noisify_strategy = "local"
cfg.optimizer = "Adam"
cfg.N = 50_000
cfg.num_classes = 10
cfg.opt_iterations = 10
cfg.run_eagerly = False
cfg.sweep_yaml_config = ""
cfg.save = False
cfg.save_folder = os.getcwd()
cfg.sweep_id = ""
cfg.tau = 1.0
cfg.tag = "Default"

_CONFIG = config_flags.DEFINE_config_dict("cfg", cfg)


def create_model(dp_parameters, dataset_metadata, cfg, upper_bound):
    if cfg.architecture == "ConvNet":
        model = create_ConvNet(dp_parameters, dataset_metadata, cfg, upper_bound)
    elif cfg.architecture == "Dense":
        model = create_Dense_Model(dp_parameters, dataset_metadata, cfg, upper_bound)
    else:
        raise ValueError(f"Invalid architecture argument {cfg.architecture}")
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
            alpha=cfg.alpha,
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassHinge":
        loss = DP_MulticlassHinge(
            min_margin=cfg.min_margin,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
    elif cfg.loss == "MulticlassKR":
        loss = DP_MulticlassKR(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    elif cfg.loss == "TauCategoricalCrossentropy":
        loss = DP_TauCategoricalCrossentropy(
            cfg.tau, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    elif cfg.loss == "KCosineSimilarity":
        KX_min = cfg.K * cfg.min_norm
        loss = DP_KCosineSimilarity(
            KX_min, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
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
        run_eagerly=cfg.run_eagerly,
    )
    return model


def train():
    init_wandb(cfg=cfg, project="MNIST_ClipLess_SGD")

    ds_train, ds_test, dataset_metadata = load_and_prepare_data(
        "mnist",
        cfg.batch_size,
        colorspace="RGB",
        drop_remainder=True,
        bound_fct=bound_clip_value(cfg.input_clipping),
    )
    model = create_model(
        DPParameters(
            noisify_strategy=cfg.noisify_strategy,
            noise_multiplier=cfg.noise_multiplier,
            delta=cfg.delta,
        ),
        dataset_metadata,
        cfg,
        upper_bound=dataset_metadata.max_norm,
    )
    model = compile_model(model, cfg)
    model.summary()
    num_epochs = get_max_epochs(cfg.epsilon_max, model)
    callbacks = [
        WandbCallback(save_model=False, monitor="val_accuracy"),
        EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=15),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.9, min_delta=0.0001, patience=5
        ),
        DP_Accountant(),
    ]
    hist = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
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
    run_with_wandb(cfg=cfg, train_function=train, project="MNIST_ClipLess_SGD")

if __name__ == "__main__":
    app.run(main)
