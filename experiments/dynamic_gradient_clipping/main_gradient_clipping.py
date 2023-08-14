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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

import deel.lipdp.layers as DP_layers
from deel.lipdp import losses
from deel.lipdp.model import AdaptiveLossGradientClipping
from deel.lipdp.model import DP_Accountant
from deel.lipdp.model import DP_Sequential
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_clip_value
from deel.lipdp.pipeline import load_and_prepare_data
from deel.lipdp.sensitivity import get_max_epochs

# declare the privacy parameters
dp_parameters = DPParameters(
    noisify_strategy="global",
    noise_multiplier=1.2,
    delta=1e-5,
)

ds_train, ds_test, dataset_metadata = load_and_prepare_data(
    "cifar10",
    batch_size=4096,
    colorspace="HSV",
    drop_remainder=True,  # accounting assumes fixed batch size
    bound_fct=bound_clip_value(
        10.0
    ),  # clipping preprocessing allows to control input bound
)

layers = [
    DP_layers.DP_BoundedInput(
        input_shape=dataset_metadata.input_shape,
        upper_bound=dataset_metadata.max_norm,
    ),
    DP_layers.DP_SpectralConv2D(
        filters=80, kernel_size=3, use_bias=False, kernel_initializer="orthogonal"
    ),
    DP_layers.DP_ScaledL2NormPooling2D(pool_size=2),
    DP_layers.DP_Flatten(),
    DP_layers.DP_SpectralDense(
        units=512, use_bias=False, kernel_initializer="orthogonal"
    ),
    DP_layers.DP_LayerCentering(),
    DP_layers.DP_GroupSort(2),
    DP_layers.DP_SpectralDense(
        units=10, use_bias=False, kernel_initializer="orthogonal"
    ),
    DP_layers.DP_ClipGradient(
        clip_value=None, epsilon=1, patience=5
    ),  # for fixed clipping use clip_value = cste
]

model = DP_Sequential(
    layers=layers, dp_parameters=dp_parameters, dataset_metadata=dataset_metadata
)

loss = losses.DP_TauCategoricalCrossentropy(14.5)

model.compile(
    loss=loss,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
    run_eagerly=False,
)

num_epochs = get_max_epochs(8.0, model)

callbacks = [
    DP_Accountant(log_fn="logging"),
    AdaptiveLossGradientClipping(
        ds_train=ds_train
    ),  # DO NOT USE THIS CALLBACK WHEN mode != "dynamic_svt"
    ReduceLROnPlateau(monitor="val_accuracy", factor=0.9, min_delta=0.01, patience=3),
]

hist = model.fit(
    ds_train,
    epochs=num_epochs,
    validation_data=ds_test,
    callbacks=callbacks,
)
