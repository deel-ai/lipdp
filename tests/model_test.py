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
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized

from deel.lipdp.dynamic import AdaptiveQuantileClipping
from deel.lipdp.layers import *
from deel.lipdp.losses import DP_TauCategoricalCrossentropy
from deel.lipdp.model import DP_Sequential
from deel.lipdp.model import DPParameters
from deel.lipdp.pipeline import bound_normalize
from deel.lipdp.pipeline import load_and_prepare_images_data


class ModelTest(parameterized.TestCase):
    def _get_mnist_cnn(self):
        ds_train, _, dataset_metadata = load_and_prepare_images_data(
            "mnist",
            batch_size=64,
            colorspace="grayscale",
            drop_remainder=True,
            bound_fct=bound_normalize(),
        )

        norm_max = 1.0
        all_layers = [
            DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=norm_max),
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
            DP_Flatten(),
            DP_SpectralDense(1024, use_bias=False, kernel_initializer="orthogonal"),
            DP_AddBias(norm_max=norm_max),
            DP_SpectralDense(10, use_bias=False, kernel_initializer="orthogonal"),
            DP_AddBias(norm_max=norm_max),
            DP_ClipGradient(
                clip_value=2.0**0.5,
                mode="dynamic",
            ),
        ]

        dp_parameters = DPParameters(
            noisify_strategy="per-layer",
            noise_multiplier=2.2,
            delta=1e-5,
        )

        model = DP_Sequential(
            all_layers,
            dp_parameters=dp_parameters,
            dataset_metadata=dataset_metadata,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = DP_TauCategoricalCrossentropy(
            tau=1.0, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return model, ds_train

    def test_forward_cnn(self):
        model, ds_train = self._get_mnist_cnn()
        batch_x, _ = ds_train.take(1).as_numpy_iterator().next()
        logits = model(batch_x)
        assert logits.shape == (len(batch_x), 10)

    def test_create_residuals(self):
        input_shape = (32, 32, 3)

        patch_size = 4
        seq_len = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        multiplier = 1
        mlp_seq_dim = multiplier * seq_len

        to_add = [
            DP_Permute((2, 1)),
            DP_QuickSpectralDense(
                units=mlp_seq_dim, use_bias=False, kernel_initializer="orthogonal"
            ),
        ]
        to_add.append(DP_GroupSort(2))
        to_add.append(DP_LayerCentering())
        to_add += [
            DP_QuickSpectralDense(
                units=seq_len, use_bias=False, kernel_initializer="orthogonal"
            ),
            DP_Permute((2, 1)),
        ]

        blocks = make_residuals("1-lip-add", to_add)
        input_bound = 1.0  # placeholder
        for layer in blocks[:-1]:
            input_bound = layer.propagate_inputs(input_bound)
            assert len(input_bound) == 2
        last = blocks[-1].propagate_inputs(input_bound)
        assert isinstance(last, float)

    def test_adaptive_clipping(self):
        num_steps_test_case = 3
        model, ds_train = self._get_mnist_cnn()
        ds_train = ds_train.take(num_steps_test_case)
        adaptive = AdaptiveQuantileClipping(
            ds_train=ds_train,
            patience=1,
            noise_multiplier=2.2,
            quantile=0.9,
            learning_rate=1.0,
        )
        adaptive.set_model(model)
        callbacks = [adaptive]
        model.fit(
            ds_train, epochs=2, callbacks=callbacks, steps_per_epoch=num_steps_test_case
        )


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    absltest.main()
