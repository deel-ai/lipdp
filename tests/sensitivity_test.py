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
from absl.testing import absltest
from absl.testing import parameterized

from deel.lipdp.dynamic import AdaptiveQuantileClipping
from deel.lipdp.layers import *
from deel.lipdp.losses import DP_TauCategoricalCrossentropy
from deel.lipdp.model import compute_gradient_bounds
from deel.lipdp.model import DP_Sequential
from deel.lipdp.model import DPParameters
from deel.lipdp.model import get_eps_delta
from deel.lipdp.pipeline import bound_normalize
from deel.lipdp.pipeline import load_and_prepare_images_data
from deel.lipdp.sensitivity import get_max_epochs


class SensitivityTest(parameterized.TestCase):
    def _get_small_mnist_cnn(self, dp_parameters, batch_size):
        ds_train, _, dataset_metadata = load_and_prepare_images_data(
            "mnist",
            batch_size=batch_size,
            colorspace="grayscale",
            drop_remainder=True,
            bound_fct=bound_normalize(),
        )

        norm_max = 1.0
        all_layers = [
            DP_BoundedInput(input_shape=(28, 28, 1), upper_bound=norm_max),
            DP_SpectralConv2D(
                filters=6,
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
            DP_SpectralDense(6, use_bias=False, kernel_initializer="orthogonal"),
            DP_AddBias(norm_max=norm_max),
            DP_SpectralDense(10, use_bias=False, kernel_initializer="orthogonal"),
            DP_AddBias(norm_max=norm_max),
            DP_ClipGradient(
                clip_value=2.0**0.5,
                mode="dynamic",
            ),
        ]

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

    @parameterized.parameters(
        ("per-layer", 0.8, 1e-5, 22.0, True),
        ("global", 1.2, 1e-6, 30.0, False),
    )
    def test_get_max_epochs(
        self, noisify_strategy, noise_multiplier, delta, epsilon_max, safe
    ):
        dp_parameters = DPParameters(
            noisify_strategy=noisify_strategy,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )

        model, _ = self._get_small_mnist_cnn(dp_parameters, batch_size=64)

        atol = 1e-2
        epochs = get_max_epochs(
            epsilon_max, model, epochs_max=None, safe=safe, atol=atol
        )

        if not safe:
            epochs += 1

        cur_epsilon, cur_delta = get_eps_delta(model, epochs)
        next_epsilon, _ = get_eps_delta(model, epochs + 1)

        self.assertLessEqual(cur_epsilon, epsilon_max + atol)
        self.assertGreaterEqual(next_epsilon + atol, epsilon_max)
        self.assertLessEqual(cur_delta, delta)

    def test_gradient_bounds(self):
        dp_parameters = DPParameters(
            noisify_strategy="per-layer",
            noise_multiplier=2.2,
            delta=1e-5,
        )

        batch_size = 16

        model, ds_train = self._get_small_mnist_cnn(
            dp_parameters, batch_size=batch_size
        )
        x, y = iter(ds_train.take(1)).next()

        loss_fn = DP_TauCategoricalCrossentropy(
            tau=1.0, reduction=tf.keras.losses.Reduction.NONE
        )

        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
            loss = tf.reshape(loss, (batch_size, 1))

        trainable_vars = model.trainable_variables
        gradient_per_variable = tape.jacobian(loss, trainable_vars)
        del tape

        gradient_bounds = compute_gradient_bounds(model)

        atol = 1e-5
        assert len(gradient_bounds) == len(gradient_per_variable)
        print(list(gradient_bounds.values()))
        for grad, bound in zip(gradient_per_variable, gradient_bounds.values()):
            grad = tf.reshape(grad, (grad.shape[0], -1))
            norm2 = tf.reduce_sum(grad**2, axis=-1) ** 0.5
            norm2 = tf.reduce_max(norm2)
            # correct for the batch size since reduction is None:
            bound = bound * batch_size
            self.assertLessEqual(norm2, bound + atol)


if __name__ == "__main__":
    absltest.main()
