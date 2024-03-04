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

from deel.lipdp.pipeline import bound_clip_value
from deel.lipdp.pipeline import bound_normalize
from deel.lipdp.pipeline import load_and_prepare_images_data
from deel.lipdp.pipeline import default_delta_value


class PipelineTest(parameterized.TestCase):

    def test_cifar10_common(self):
        batch_size = 64
        max_norm = 5e-2
        colorspace = "RGB"

        ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
            "cifar10",
            batch_size=batch_size,
            colorspace=colorspace,
            drop_remainder=True,  # accounting assumes fixed batch size.
            bound_fct=bound_clip_value(max_norm),
            multiplicity=0,  # no multiplicity for cifar10
        )

        self.assertEqual(dataset_metadata.nb_classes, 10)
        self.assertEqual(dataset_metadata.input_shape, (32, 32, 3))
        self.assertEqual(dataset_metadata.nb_samples_train, 50_000)
        self.assertEqual(dataset_metadata.nb_samples_test, 10_000)
        self.assertEqual(dataset_metadata.batch_size, batch_size)
        self.assertEqual(dataset_metadata.max_norm, max_norm)

        atol = 1e-7
        batch_x, batch_y = next(iter(ds_train))
        for x in batch_x:
            norm = tf.norm(x, axis=None)
            self.assertLessEqual(norm, max_norm + atol)
        self.assertEqual(batch_y.shape, (batch_size, 10))

        batch_sizes = [len(batch_x) for batch_x, batch_y in ds_train]
        self.assertEqual(batch_sizes[-1], batch_size)
        self.assertEqual(len(batch_sizes), 50_000 // batch_size)
        self.assertEqual(dataset_metadata.nb_steps_per_epochs, len(batch_sizes))
        delta_heuristic = default_delta_value(dataset_metadata)
        self.assertLessEqual(dataset_metadata.nb_samples_train, 1./delta_heuristic)

    @parameterized.parameters(("RGB",), ("grayscale",), ("HSV",))
    def test_cifar10_colorspace(self, colorspace):
        batch_size = 64
        max_norm = 5e-2

        ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
            "cifar10",
            batch_size=batch_size,
            colorspace=colorspace,
            drop_remainder=True,  # accounting assumes fixed batch size.
            bound_fct=bound_clip_value(max_norm),
            multiplicity=0,  # no multiplicity for cifar10
        )

        batch = next(iter(ds_test))
        if colorspace == "grayscale":
            self.assertEqual(batch[0].shape[-1], 1)
        else:
            self.assertEqual(batch[0].shape[-1], 3)

    @parameterized.parameters(
        (1,),
        (4,),
    )
    def test_cifar10_augmult(self, multiplicity: int):
        batch_size = 64
        max_norm = 5e-2
        colorspace = "grayscale"

        ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
            "cifar10",
            batch_size=batch_size,
            colorspace=colorspace,
            drop_remainder=True,  # accounting assumes fixed batch size.
            bound_fct=bound_clip_value(max_norm),
            multiplicity=multiplicity,
        )

        self.assertEqual(dataset_metadata.batch_size, batch_size)
        # multiplicity is not accounted in logical batch size for accounting.
        # Note: the DP_ClipGradient must reduce over the multiplicity for this to work.

        batch_sizes = [len(batch_x) for batch_x, batch_y in ds_train]
        for physical_batch_x, _ in ds_train:
            self.assertEqual(len(physical_batch_x), batch_size * multiplicity)
            # multiplicity is accounted in physical batch size.
        self.assertEqual(dataset_metadata.nb_samples_train, 50_000)
        self.assertEqual(len(batch_sizes), 50_000 // batch_size)
        self.assertEqual(dataset_metadata.nb_steps_per_epochs, len(batch_sizes))

    def test_mnist_normalize(self):
        batch_size = 64

        ds_train, ds_test, dataset_metadata = load_and_prepare_images_data(
            "mnist",
            colorspace="grayscale",
            batch_size=batch_size,
            drop_remainder=True,  # accounting assumes fixed batch size.
            bound_fct=bound_normalize(),
            multiplicity=0,  # no multiplicity for mnist
        )

        self.assertEqual(dataset_metadata.nb_classes, 10)
        self.assertEqual(dataset_metadata.input_shape, (28, 28, 1))
        self.assertEqual(dataset_metadata.nb_samples_train, 60_000)
        self.assertEqual(dataset_metadata.nb_samples_test, 10_000)
        self.assertEqual(dataset_metadata.batch_size, batch_size)
        self.assertEqual(dataset_metadata.max_norm, 1.0)

        atol = 1e-5
        batch_x, batch_y = next(iter(ds_train))
        for x in batch_x:
            norm = tf.norm(x, axis=None)
            self.assertAlmostEqual(norm, 1.0, delta=atol)
        self.assertEqual(batch_y.shape, (batch_size, 10))

        batch_sizes = [len(batch_x) for batch_x, batch_y in ds_train]
        self.assertEqual(batch_sizes[-1], batch_size)
        self.assertEqual(len(batch_sizes), 60_000 // batch_size)
        self.assertEqual(dataset_metadata.nb_steps_per_epochs, len(batch_sizes))


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    absltest.main()
