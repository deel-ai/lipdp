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

import numpy as np
import tensorflow as tf
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags
from sklearn.model_selection import train_test_split

import deel.lipdp.layers as DP_layers
from deel.lipdp import losses
from deel.lipdp.dynamic import AdaptiveQuantileClipping
from deel.lipdp.model import DP_Accountant
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_clip_value
from deel.lipdp.pipeline import default_delta_value
from deel.lipdp.pipeline import load_adbench_data
from deel.lipdp.pipeline import prepare_tabular_data
from deel.lipdp.sensitivity import get_max_epochs
from experiments.wandb_utils import init_wandb
from experiments.wandb_utils import run_with_wandb
from wandb.keras import WandbCallback


def default_cfg_cifar10():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 200
    cfg.clip_loss_gradient = None  # not required for dynamic clipping.
    cfg.depth = 2
    cfg.dataset_name = "22_magic.gamma"
    cfg.dynamic_clipping = "quantiles"  # can be "fixed", "laplace", "quantiles". "fixed" requires a clipping value.
    cfg.dynamic_clipping_quantiles = 0.9
    cfg.delta = 1e-5
    cfg.epsilon_max = 1.5  # budget!
    cfg.input_bound = None
    cfg.learning_rate = 8e-2  # works well for vanilla SGD.
    cfg.log_wandb = "disabled"
    cfg.loss = "TauBCE"
    cfg.multiplicity = 4
    cfg.noise_multiplier = 1.6
    cfg.noisify_strategy = "per-layer"
    cfg.optimizer = "SGD"
    cfg.sweep_id = ""  # useful to resume a sweep.
    cfg.sweep_yaml_config = ""  # useful to load a sweep from a yaml file.
    cfg.tau = 10.0  # temperature for the softmax.
    cfg.width_multiplier = 1
    return cfg


project = "ICLR_Tabular"
cfg = default_cfg_cifar10()
_CONFIG = config_flags.DEFINE_config_dict(
    "cfg", cfg
)  # for FLAGS parsing in command line.


def create_MLP(dataset_metadata, dp_parameters):
    layers = [
        DP_layers.DP_BoundedInput(
            input_shape=dataset_metadata.input_shape,
            upper_bound=dataset_metadata.max_norm,
        )
    ]

    width = 64 * cfg.width_multiplier
    for _ in range(cfg.depth):
        layers += [
            DP_layers.DP_QuickSpectralDense(
                units=width, use_bias=False, kernel_initializer="orthogonal"
            ),
            # DP_layers.DP_LayerCentering(),
            DP_layers.DP_GroupSort(2),
        ]

    layers.append(
        DP_layers.DP_QuickSpectralDense(
            units=1, use_bias=False, kernel_initializer="orthogonal"
        )
    )

    layers.append(
        DP_layers.DP_ClipGradient(
            clip_value=cfg.clip_loss_gradient,
            mode="dynamic",
        )
    )

    model = DP_Model(
        layers,
        dp_parameters=dp_parameters,
        dataset_metadata=dataset_metadata,
        name="mlp",
    )

    model.build(input_shape=(None, *dataset_metadata.input_shape))

    return model


def train():
    init_wandb(cfg=cfg, project=project)

    ##########################
    #### Dataset loading #####
    ##########################

    x_data, y_data = load_adbench_data(
        cfg.dataset_name, dataset_dir="/data/datasets/adbench", standardize=True
    )

    # clipping preprocessing allows to control input bound
    input_bound = cfg.input_bound
    if input_bound is None:
        norms = np.linalg.norm(x_data, axis=1)
        input_bound = float(np.max(norms))
    bound_fct = bound_clip_value(input_bound)

    random_state = random.randint(0, 1000)
    splits = train_test_split(x_data, y_data, test_size=0.2, random_state=random_state)

    ds_train, ds_test, dataset_metadata = prepare_tabular_data(
        *splits,
        batch_size=cfg.batch_size,
        drop_remainder=True,  # accounting assumes fixed batch size
        bound_fct=bound_fct,
    )

    ##########################
    #### Model definition ####
    ##########################

    # declare the privacy parameters
    dp_parameters = DPParameters(
        noisify_strategy=cfg.noisify_strategy,
        noise_multiplier=cfg.noise_multiplier,
        delta=default_delta_value(dataset_metadata),
    )

    model = create_MLP(dataset_metadata, dp_parameters)

    ##########################
    ######## Loss setup ######
    ##########################

    if cfg.loss == "TauBCE":
        loss = losses.DP_TauBCE(cfg.tau)

    ##########################
    ##### Optimizer setup ####
    ##########################

    if cfg.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "SGD":
        # geometric sequence: memory length ~= 1 / (1 - momentum)
        # memory length = nb_steps_per_epochs => momentum = 1 - (1./nb_steps_per_epochs)
        momentum = 1 - 1.0 / dataset_metadata.nb_steps_per_epochs
        momentum = max(0.5, min(0.99, momentum))  # reasonable range
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=cfg.learning_rate, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(from_logits=True),
        ],  # accuracy metric is necessary for dynamic loss gradient clipping with "laplace"
        run_eagerly=False,
    )

    callbacks = [
        WandbCallback(save_model=False, monitor="val_auc"),
        DP_Accountant(),
    ]

    ########################
    ### Dynamic clipping ###
    ########################

    if cfg.dynamic_clipping == "quantiles":
        adaptive = AdaptiveQuantileClipping(
            ds_train=ds_train,
            patience=1,
            noise_multiplier=cfg.noise_multiplier * 5,  # more noisy.
            quantile=cfg.dynamic_clipping_quantiles,
            learning_rate=1.0,
        )
        adaptive.set_model(model)
        callbacks.append(adaptive)
    else:
        raise ValueError(f"Unknown clipping strategy {cfg.dynamic_clipping}")

    ########################
    ### Training process ###
    ########################

    if cfg.epsilon_max is None:
        num_epochs = 50  # useful for debugging.
    else:
        # compute the max number of epochs to reach the budget.
        num_epochs = get_max_epochs(cfg.epsilon_max, model)

    hist = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
        callbacks=callbacks,
    )


def main(_):
    run_with_wandb(cfg=cfg, train_function=train, project=project)


if __name__ == "__main__":
    app.run(main)
