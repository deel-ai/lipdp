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
import numpy as np
import tensorflow as tf
from absl import app
from ml_collections import config_dict
from ml_collections import config_flags

import deel.lipdp.layers as DP_layers
from deel.lip.metrics import CategoricalProvableRobustAccuracy
from deel.lipdp import losses
from deel.lipdp.dynamic import AdaptiveQuantileClipping
from deel.lipdp.dynamic import LaplaceAdaptiveLossGradientClipping
from deel.lipdp.model import DP_Accountant
from deel.lipdp.model import DP_Model
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_clip_value
from deel.lipdp.pipeline import default_delta_value
from deel.lipdp.pipeline import load_and_prepare_images_data
from deel.lipdp.sensitivity import get_max_epochs
from deel.lipdp.utils import PrivacyMetrics
from deel.lipdp.utils import SignaltoNoiseAverage
from deel.lipdp.utils import SignaltoNoiseHistogram
from experiments.wandb_utils import init_wandb
from experiments.wandb_utils import run_with_wandb
from wandb.keras import WandbCallback


def default_cfg_cifar10():
    cfg = config_dict.ConfigDict()
    cfg.batch_size = 2_500  # 5% of the dataset.
    cfg.clip_loss_gradient = None  # not required for dynamic clipping.
    cfg.depth = 1
    cfg.dynamic_clipping = "quantiles"  # can be "fixed", "laplace", "quantiles". "fixed" requires a clipping value.
    cfg.dynamic_clipping_quantiles = (
        0.9  # crop to 90% of the distribution of gradient norm.
    )
    cfg.delta = 1e-5  # 1e-5 is the default value in the paper.
    cfg.epsilon_max = 8.0  # budget!
    cfg.input_bound = 3.0  # 15.0 works well in RGB non standardized.
    cfg.learning_rate = 8e-2  # works well for vanilla SGD.
    cfg.log_wandb = "disabled"
    cfg.loss = "TauCategoricalCrossentropy"
    cfg.mia = False
    cfg.multiplicity = 0  # 0 means no multiplicity.
    cfg.noise_multiplier = 3.0
    cfg.noisify_strategy = "per-layer"
    cfg.representation = "RGB_STANDARDIZED"  # "RGB", "RGB_STANDARDIZED", "HSV".
    cfg.optimizer = "SGD"
    cfg.signal_to_noise = "histogram"
    cfg.sweep_id = ""  # useful to resume a sweep.
    cfg.sweep_yaml_config = ""  # useful to load a sweep from a yaml file.
    cfg.tau = 20.0  # temperature for the softmax.
    cfg.use_residuals = False  # better without.
    cfg.width_multiplier = 1
    return cfg


project = "ICLR_Cifar10"
cfg = default_cfg_cifar10()
_CONFIG = config_flags.DEFINE_config_dict(
    "cfg", cfg
)  # for FLAGS parsing in command line.


def create_MLP_Mixer(dataset_metadata, dp_parameters):
    layers = [
        DP_layers.DP_BoundedInput(
            input_shape=dataset_metadata.input_shape,
            upper_bound=dataset_metadata.max_norm,
        )
    ]

    patch_size = 4
    num_mixer_layers = cfg.depth
    seq_len = (dataset_metadata.input_shape[0] // patch_size) * (
        dataset_metadata.input_shape[1] // patch_size
    )
    multiplier = cfg.width_multiplier
    mlp_seq_dim = multiplier * seq_len
    mlp_channel_dim = multiplier * seq_len
    hidden_size = multiplier * seq_len
    use_residuals = cfg.use_residuals

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
            units=hidden_size, use_bias=False, kernel_initializer="orthogonal"
        )
    )

    for _ in range(num_mixer_layers):
        to_add = [
            DP_layers.DP_Permute((2, 1)),
            DP_layers.DP_QuickSpectralDense(
                units=mlp_seq_dim, use_bias=False, kernel_initializer="orthogonal"
            ),
        ]
        to_add.append(DP_layers.DP_GroupSort(2))
        to_add.append(DP_layers.DP_LayerCentering())
        to_add += [
            DP_layers.DP_QuickSpectralDense(
                units=seq_len, use_bias=False, kernel_initializer="orthogonal"
            ),
            DP_layers.DP_Permute((2, 1)),
        ]

        if use_residuals:
            layers += DP_layers.make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

        to_add = [
            DP_layers.DP_QuickSpectralDense(
                units=mlp_channel_dim, use_bias=False, kernel_initializer="orthogonal"
            ),
        ]
        to_add.append(DP_layers.DP_GroupSort(2))
        to_add.append(DP_layers.DP_LayerCentering())
        to_add.append(
            DP_layers.DP_QuickSpectralDense(
                units=hidden_size, use_bias=False, kernel_initializer="orthogonal"
            )
        )

        if use_residuals:
            layers += DP_layers.make_residuals("1-lip-add", to_add)
        else:
            layers += to_add

    layers += [
        DP_layers.DP_Flatten(),
    ]

    layers.append(
        DP_layers.DP_QuickSpectralDense(
            units=dataset_metadata.nb_classes,
            use_bias=False,
            kernel_initializer="orthogonal",
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
        name="mlp_mixer",
    )

    model.build(input_shape=(None, *dataset_metadata.input_shape))

    return model


def get_cifar10_standardized(verbose=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 1, 3))
    CIFAR10_STD_DEV = np.array([0.2023, 0.1994, 0.2010]).reshape((1, 1, 3))
    x_train = (x_train - CIFAR10_MEAN) / CIFAR10_STD_DEV
    x_test = (x_test - CIFAR10_MEAN) / CIFAR10_STD_DEV
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    cifar10 = (x_train, y_train, x_test, y_test)
    all_norms = np.linalg.norm(x_train, axis=-1)
    if verbose:
        print(f"Dataset Max norm: {np.max(all_norms)}")
        print(f"Dataset Min norm: {np.min(all_norms)}")
        print(f"Dataset Mean norm: {np.mean(all_norms)}")
        print(f"Dataset Std norm: {np.std(all_norms)}")
        quantiles = [0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        print(f"Dataset Quantiles: {np.quantile(all_norms, quantiles)} at {quantiles}")
    # We assume no privacy loss to estimate the max norm, the mean pixel value and the std pixel value.
    # This is a reasonable assumption shared by many papers. If necessary, they can be estimated privately.
    max_norm = np.max(all_norms)
    max_norm = max_norm.astype(np.float32)
    return cifar10, max_norm


def certifiable_acc_metrics(epsilons):
    """Returns a list of metrics for certifiable accuracy at the given epsilons.

    Args:
        epsilons: list of epsilons to evaluate, assuming 8bits encoding.

    Returns:
        list of metrics.
    """
    metrics = []
    for epsilon_8bit in epsilons:
        name = f"certacc_{epsilon_8bit}"
        epsilon = epsilon_8bit / 255.0
        # dataset has been standardized so we take that into account:
        if cfg.representation == "RGB_STANDARDIZED":
            epsilon = epsilon / 0.2023  # maximum std dev: lower bound of radius.
        else:
            assert (
                cfg.representation == "RGB"
            ), "Certifiable accuracy only implemented for RGB and RGB_STANDARDIZED"
        metric = CategoricalProvableRobustAccuracy(
            epsilon=epsilon, disjoint_neurons=False, name=name
        )
        metrics.append(metric)
    return metrics


def train():
    init_wandb(cfg=cfg, project=project)

    ##########################
    #### Dataset loading #####
    ##########################

    # clipping preprocessing allows to control input bound
    input_bound = cfg.input_bound
    if cfg.representation == "RGB_STANDARDIZED":
        cifar10_standardized, max_norm_cifar10 = get_cifar10_standardized(verbose=True)
        if input_bound is None:
            input_bound = max_norm_cifar10
            print(f"Max norm set to {input_bound}")
    bound_fct = bound_clip_value(input_bound)

    ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
        "cifar10",
        batch_size=cfg.batch_size,
        colorspace=cfg.representation,
        drop_remainder=True,  # accounting assumes fixed batch size
        bound_fct=bound_fct,
        multiplicity=cfg.multiplicity,
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

    model = create_MLP_Mixer(dataset_metadata, dp_parameters)

    ##########################
    ######## Loss setup ######
    ##########################

    if cfg.loss == "TauCategoricalCrossentropy":
        loss = losses.DP_TauCategoricalCrossentropy(cfg.tau)
    elif cfg.loss == "MulticlassHKR":
        alpha = 200.0
        margin = 1.0
        loss = losses.DP_MulticlassHKR(alpha=alpha, min_margin=margin)
    elif cfg.loss == "KCosineSimilarity":
        K = 0.99
        loss = losses.DP_KCosineSimilarity(K=K)

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
            *certifiable_acc_metrics([1, 2, 4, 8, 16, 36]),
        ],  # accuracy metric is necessary for dynamic loss gradient clipping with "laplace"
        run_eagerly=False,
    )

    callbacks = [
        WandbCallback(save_model=False, monitor="val_accuracy"),
        DP_Accountant(),
    ]

    if cfg.signal_to_noise == "disabled":
        pass
    elif cfg.signal_to_noise == "average":
        batch_train = next(iter(ds_train))
        callbacks.append(SignaltoNoiseAverage(batch_train))
    elif cfg.signal_to_noise == "histogram":
        batch_train = next(iter(ds_train))
        callbacks.append(SignaltoNoiseHistogram(batch_train))
    else:
        raise ValueError(f"Unknown signal_to_noise {cfg.signal_to_noise}")

    ########################
    ### Dynamic clipping ###
    ########################

    if cfg.dynamic_clipping == "fixed":
        assert (
            cfg.clip_loss_gradient is not None
        ), "Fixed mode requires a clipping value"
    elif cfg.dynamic_clipping == "laplace":
        adaptive = LaplaceAdaptiveLossGradientClipping(
            ds_train=ds_train,
            patience=1,
            epsilon=1.0,
        )
        adaptive.set_model(model)
        callbacks.append(adaptive)
    elif cfg.dynamic_clipping == "quantiles":
        adaptive = AdaptiveQuantileClipping(
            ds_train=ds_train,
            patience=1,
            noise_multiplier=cfg.noise_multiplier * 2,  # more noisy.
            quantile=cfg.dynamic_clipping_quantiles,
            learning_rate=1.0,
        )
        adaptive.set_model(model)
        callbacks.append(adaptive)
    else:
        raise ValueError(f"Unknown clipping strategy {cfg.dynamic_clipping}")

    ########################
    ###### MIA attack ######
    ########################

    if cfg.mia:
        privacy_metrics = PrivacyMetrics(cifar10_standardized)
        callbacks.append(privacy_metrics)

    ########################
    ### Training process ###
    ########################

    if cfg.epsilon_max is None:
        num_epochs = 1  # useful for debugging.
    else:
        # compute the max number of epochs to reach the budget.
        num_epochs = get_max_epochs(cfg.epsilon_max, model, safe=True)

    hist = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
        callbacks=callbacks,
    )

    if cfg.mia:
        privacy_metrics.log_report()


def main(_):
    run_with_wandb(cfg=cfg, train_function=train, project=project)


if __name__ == "__main__":
    app.run(main)
