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
import tensorflow as tf

from lipdp import layers
from lipdp import losses
from lipdp.model import DP_Accountant
from lipdp.model import DP_Sequential
from lipdp.model import DPParameters
from lipdp.pipeline import bound_clip_value
from lipdp.pipeline import load_and_prepare_data
from lipdp.sensitivity import get_max_epochs


# declare the privacy parameters
dp_parameters = DPParameters(
    noisify_strategy="global",
    noise_multiplier=2.5,
    delta=1e-5,
)
# maximum epsilon will depends on dp_parameters and the number of epochs
# in order to control epsilon, we compute the adequate number of epochs
epsilon_max = 1.0

input_upper_bound = 1.0

# load the data
# data loader return dataset_metadata which allows to
# know the informations required for privacy accounting
# (dataset size, number of samples, max input bound...)
ds_train, ds_test, dataset_metadata = load_and_prepare_data(
    "mnist",
    500,
    colorspace="RGB",  # RGB leaves colors as-is
    drop_remainder=True,  # accounting assumes fixed batch size
    bound_fct=bound_clip_value(
        input_upper_bound
    ),  # clipping preprocessing allows to control input bound
)

# construct DP_Sequential
model = DP_Sequential(
    # works like usual sequential but requires DP layers
    layers=[
        # BoundedInput works like Input, but performs input clipping to guarantee input bound
        layers.DP_BoundedInput(
            input_shape=dataset_metadata.input_shape, upper_bound=input_upper_bound
        ),
        layers.DP_SpectralConv2D(
            filters=16,
            kernel_size=3,
            kernel_initializer="orthogonal",
            strides=1,
            use_bias=False,
        ),
        layers.DP_GroupSort(2),
        layers.DP_ScaledL2NormPooling2D(pool_size=2, strides=2),
        layers.DP_LayerCentering(),
        layers.DP_Flatten(),
        layers.DP_SpectralDense(512),
        layers.DP_GroupSort(),
        layers.DP_LayerCentering(),
        layers.DP_SpectralDense(dataset_metadata.nb_classes),
    ],
    dp_parameters=dp_parameters,
    dataset_metadata=dataset_metadata,
)

model.compile(
    # Compile model using DP loss
    loss=losses.DP_TauCategoricalCrossentropy(18.0),
    # this method is compatible with any first order optimizer
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    metrics=["accuracy"],
)
model.summary()
# now that we have the model, we can compute the max number of epochs
# to achieve epsilon max
num_epochs = get_max_epochs(epsilon_max, model)
hist = model.fit(
    ds_train,
    epochs=num_epochs,
    validation_data=ds_test,
    callbacks=[
        # accounting is done thanks to a callback
        DP_Accountant(log_fn="logging"),
    ],
)
